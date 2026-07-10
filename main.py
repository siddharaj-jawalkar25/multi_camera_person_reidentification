"""
Multi-Camera Person Re-Identification POC
YOLOv8 (detection) + ByteTrack (per-camera tracking) + OSNet (ReID embeddings)
+ Hungarian matching (cross-camera association) + Union-Find (global ID fusion)

ACCURACY ADDITIONS (v2):
- Blur detection (Laplacian variance) — reject motion-blurred crops
- Size / aspect-ratio gating — reject partial-body / tiny detections
- CLAHE illumination normalization — reduce cross-camera lighting drift
- Quality-weighted top-K gallery — average the K best crops per track
  instead of a plain running average, so one bad frame can't drag the
  embedding off target

USAGE:
    python multi_camera_reid.py --videos cam1.mp4 cam2.mp4 --output_dir outputs

REQUIREMENTS:
    pip install ultralytics torchreid opencv-python scipy torch torchvision --break-system-packages

NOTES FOR JAY:
- Fresh build, not pulled from your enterprise_reid_env pipeline. Sanity-check
  all thresholds below against what you already tuned there.
- OSNet weights auto-download unless you pass --reid_weights pointing at your
  own MSMT17 checkpoint (do this if you have it — bigger accuracy win than
  anything in this file).
- Known limitation: OSNet trained on side-view data struggles on overhead
  cameras. CLAHE and quality gating help but don't fix this fundamentally —
  homography-based zone matching is still the correct long-term fix.
"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

try:
    import torchreid
except ImportError:
    raise ImportError(
        "torchreid not installed. Run: pip install torchreid --break-system-packages"
    )


# ----------------------------- Config ----------------------------------- #

MATCH_THRESHOLD = 0.35      # cosine distance threshold for accepting a cross-cam match (tune this)
MATCH_INTERVAL = 15         # run cross-camera matching every N frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Crop quality gating ---
MIN_BOX_AREA = 1500         # pixels; rejects tiny/far-away detections
MIN_ASPECT_RATIO = 1.2      # height/width; rejects wide partial-body boxes
MAX_ASPECT_RATIO = 4.5      # height/width; rejects unnaturally tall slivers
BLUR_VAR_THRESHOLD = 45.0   # Laplacian variance; below this = too blurry to trust
GALLERY_CONF_FLOOR = 0.55   # stricter than detection --conf; only update gallery above this

# --- Quality-weighted gallery ---
TOP_K_CROPS = 5             # keep best K embeddings per track, average them

# --- Feature fusion (deep embedding + color histogram) ---
EMBEDDING_WEIGHT = 0.75      # weight for OSNet cosine distance in fused cost
HISTOGRAM_WEIGHT = 0.25      # weight for HSV histogram Bhattacharyya distance

# --- Same-camera re-identification (handles occlusion -> new track ID) ---
SAME_CAMERA_MATCH_THRESHOLD = 0.30   # stricter than cross-camera: same lighting/angle expected
RELINK_WINDOW_FRAMES = 200           # how far back to look for a track to re-link to

# --- Tracker ---
# bytetrack with an extended track_buffer (see bytetrack_extended.yaml,
# shipped alongside this script) — holds a lost track's identity longer
# through occlusion before giving up and minting a new local track ID.
# Combined with the same-camera re-link safety net below for cases that
# exceed even this buffer.
TRACKER_CONFIG = "bytetrack_extended.yaml"


# ----------------------------- Union-Find --------------------------------- #

class UnionFind:
    """Manages global identity fusion across (camera_id, local_track_id) pairs."""

    def __init__(self):
        self.parent = {}
        self.next_global_id = 0
        self.global_id_of_root = {}
        self.frozen_display_id = {}   # key -> global_id, LOCKED once first shown

    def _make(self, key):
        if key not in self.parent:
            self.parent[key] = key

    def find(self, key):
        self._make(key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, key_a, key_b):
        root_a, root_b = self.find(key_a), self.find(key_b)
        if root_a != root_b:
            self.parent[root_b] = root_a

    def global_id(self, key):
        root = self.find(key)
        if root not in self.global_id_of_root:
            self.global_id_of_root[root] = self.next_global_id
            self.next_global_id += 1
        return self.global_id_of_root[root]

    def display_id(self, key):
        """
        Returns a STABLE id for on-screen display. Once a key has been shown
        with a given global id, that id is locked forever for this key —
        even if a later cross-camera merge would technically reassign it.
        This trades a small amount of cross-camera consistency for visual
        stability: no more IDs flickering/jumping mid-video.
        """
        if key not in self.frozen_display_id:
            self.frozen_display_id[key] = self.global_id(key)
        return self.frozen_display_id[key]


# ----------------------------- Crop Quality Filter -------------------------- #

def blur_score(crop):
    """Laplacian variance — higher = sharper. Cheap and effective motion-blur proxy."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def crop_quality_check(crop, box, det_conf):
    """
    Returns (is_valid, quality_score). quality_score combines detection
    confidence and sharpness, used to rank crops for the top-K gallery.
    Rejects: too small, wrong aspect ratio (partial body), too blurry,
    low detection confidence.
    """
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    area = w * h
    aspect = h / w

    if area < MIN_BOX_AREA:
        return False, 0.0
    if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
        return False, 0.0
    if det_conf < GALLERY_CONF_FLOOR:
        return False, 0.0

    sharpness = blur_score(crop)
    if sharpness < BLUR_VAR_THRESHOLD:
        return False, 0.0

    # normalize sharpness roughly to 0-1 range (empirical cap at 300) and blend with conf
    sharpness_norm = min(sharpness / 300.0, 1.0)
    quality_score = 0.5 * det_conf + 0.5 * sharpness_norm
    return True, quality_score


def apply_clahe(bgr_crop):
    """
    Illumination normalization: CLAHE on the L channel in LAB space.
    Helps reduce embedding drift caused by cameras with different exposure/lighting
    (e.g. one camera near a window, another under fluorescent lighting).
    """
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extract_color_histogram(bgr_crop):
    """
    Cheap complementary feature to the deep OSNet embedding: an HSV color
    histogram of the person crop (clothing color/texture signal). Deep
    features can get fooled by pose/viewpoint changes across cameras;
    color histograms are largely pose-invariant but lighting-sensitive —
    since we run this AFTER CLAHE normalization, that weakness is reduced.
    Fusing both gives two independent signals instead of relying on one.
    """
    clahe_crop = apply_clahe(bgr_crop)
    hsv = cv2.cvtColor(clahe_crop, cv2.COLOR_BGR2HSV)
    # focus on H and S channels (V/brightness is the least reliable across cameras)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def histogram_distance(hist_a, hist_b):
    """Bhattacharyya distance: 0 = identical, 1 = totally different. Good for histograms."""
    return cv2.compareHist(
        hist_a.astype(np.float32), hist_b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
    )


# ----------------------------- ReID Embedder ------------------------------- #

class OSNetEmbedder:
    def __init__(self, weights_path=None):
        self.model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )
        if weights_path and os.path.exists(weights_path):
            torchreid.utils.load_pretrained_weights(self.model, weights_path)
            print(f"[OSNet] Loaded custom weights: {weights_path}")
        else:
            print("[OSNet] Using default torchreid pretrained weights "
                  "(swap in your MSMT17 checkpoint with --reid_weights for best accuracy)")
        self.model.eval().to(DEVICE)
        self.input_size = (256, 128)  # (H, W) standard for OSNet

    @torch.no_grad()
    def extract(self, bgr_crop):
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        crop = apply_clahe(bgr_crop)
        img = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        feat = self.model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy()[0]


# ----------------------------- Track Gallery ------------------------------- #

class TrackGallery:
    """
    Holds up to TOP_K_CROPS best (quality_score, embedding) pairs per
    (camera_id, local_track_id). The representative embedding is the
    normalized average of the kept top-K — far more robust to occasional
    bad detections than a plain running average.
    """

    def __init__(self):
        self.crops = defaultdict(list)   # key -> list of (quality_score, embedding, histogram)
        self.last_seen = {}              # key -> frame_idx, for staleness pruning
        self.first_seen = {}             # key -> frame_idx, first quality crop accepted

    def update(self, cam_id, track_id, embedding, histogram, quality_score, frame_idx):
        key = (cam_id, track_id)
        bucket = self.crops[key]
        bucket.append((quality_score, embedding, histogram))
        bucket.sort(key=lambda x: x[0], reverse=True)
        if len(bucket) > TOP_K_CROPS:
            del bucket[TOP_K_CROPS:]
        if key not in self.first_seen:
            self.first_seen[key] = frame_idx
        self.last_seen[key] = frame_idx

    def representative_embedding(self, key):
        bucket = self.crops.get(key)
        if not bucket:
            return None
        embs = np.stack([e for _, e, _ in bucket])
        avg = embs.mean(axis=0)
        avg /= np.linalg.norm(avg) + 1e-8
        return avg

    def representative_histogram(self, key):
        bucket = self.crops.get(key)
        if not bucket:
            return None
        hists = np.stack([h for _, _, h in bucket])
        return hists.mean(axis=0)

    def prune_stale(self, frame_idx, max_age=90):
        stale = [k for k, v in self.last_seen.items() if frame_idx - v > max_age]
        for k in stale:
            del self.crops[k]
            del self.last_seen[k]

    def active_keys_by_camera(self):
        by_cam = defaultdict(list)
        for (cam_id, track_id) in self.crops:
            by_cam[cam_id].append((cam_id, track_id))
        return by_cam


# ----------------------------- Cross-Camera Matching ------------------------ #

def try_relink_same_camera(gallery: TrackGallery, union_find: UnionFind, cam_id, new_key, frame_idx):
    """
    Handles the #1 cause of ID hopping: a person gets briefly occluded, the
    tracker loses them, and on reappearance assigns a BRAND NEW local track
    ID — even though it's the same person in the same camera. Cross-camera
    matching alone never catches this since it only compares across
    different cameras.

    When a new local track appears, check its appearance against other
    recently-active tracks in the SAME camera. If one matches closely
    (stricter threshold than cross-camera, since lighting/angle are
    identical), union them so the displayed ID carries over instead of
    minting a new one.
    """
    new_emb = gallery.representative_embedding(new_key)
    new_hist = gallery.representative_histogram(new_key)
    if new_emb is None:
        return

    best_key, best_cost = None, SAME_CAMERA_MATCH_THRESHOLD

    for key in list(gallery.crops.keys()):
        if key == new_key or key[0] != cam_id:
            continue
        if frame_idx - gallery.last_seen.get(key, -1) > RELINK_WINDOW_FRAMES:
            continue

        emb = gallery.representative_embedding(key)
        hist = gallery.representative_histogram(key)
        if emb is None:
            continue

        cost = (EMBEDDING_WEIGHT * (1.0 - float(np.dot(new_emb, emb)))
                + HISTOGRAM_WEIGHT * histogram_distance(new_hist, hist))

        if cost < best_cost:
            best_cost = cost
            best_key = key

    if best_key is not None:
        # best_key first -> its root/identity is preserved, new_key inherits it
        union_find.union(best_key, new_key)


def cross_camera_match(gallery: TrackGallery, union_find: UnionFind, num_cameras: int):
    """
    For every pair of cameras, build a cosine-distance cost matrix between
    active tracks' representative (top-K averaged) embeddings and solve
    optimal assignment with the Hungarian algorithm. Matches under
    MATCH_THRESHOLD get unioned into the same global ID.
    """
    by_cam = gallery.active_keys_by_camera()

    for cam_a in range(num_cameras):
        for cam_b in range(cam_a + 1, num_cameras):
            keys_a = by_cam.get(cam_a, [])
            keys_b = by_cam.get(cam_b, [])
            if not keys_a or not keys_b:
                continue

            emb_a = np.stack([gallery.representative_embedding(k) for k in keys_a])
            emb_b = np.stack([gallery.representative_embedding(k) for k in keys_b])
            hist_a = [gallery.representative_histogram(k) for k in keys_a]
            hist_b = [gallery.representative_histogram(k) for k in keys_b]

            # cosine distance = 1 - cosine similarity (embeddings are already L2-normalized)
            embedding_cost = 1.0 - emb_a @ emb_b.T

            # histogram distance computed pairwise (cv2.compareHist has no batched form)
            histogram_cost = np.zeros_like(embedding_cost)
            for i, ha in enumerate(hist_a):
                for j, hb in enumerate(hist_b):
                    histogram_cost[i, j] = histogram_distance(ha, hb)

            cost_matrix = EMBEDDING_WEIGHT * embedding_cost + HISTOGRAM_WEIGHT * histogram_cost

            row_idx, col_idx = linear_sum_assignment(cost_matrix)

            for r, c in zip(row_idx, col_idx):
                if cost_matrix[r, c] < MATCH_THRESHOLD:
                    union_find.union(keys_a[r], keys_b[c])


# ----------------------------- Main Pipeline -------------------------------- #

def run(video_paths, output_dir, detector_weights, reid_weights, conf_thresh, display=True, imgsz=640):
    os.makedirs(output_dir, exist_ok=True)
    num_cameras = len(video_paths)

    # CRITICAL: one detector PER CAMERA, not shared. ultralytics' tracker
    # keeps internal motion-prediction state tied to the detector object
    # when persist=True. Sharing one detector across cameras means the
    # tracker sees frames alternating between two unrelated video feeds
    # every call — it loses tracks constantly and reassigns new IDs on
    # recovery. This was the root cause of boxes vanishing / IDs hopping.
    detectors = [YOLO(detector_weights) for _ in range(num_cameras)]
    embedder = OSNetEmbedder(reid_weights)
    gallery = TrackGallery()
    union_find = UnionFind()

    caps = [cv2.VideoCapture(p) for p in video_paths]

    # --- Diagnostics: catch silent failures immediately instead of running
    # the whole loop and producing nothing ---
    for i, (cap, path) in enumerate(zip(caps, video_paths)):
        if not cap.isOpened():
            raise RuntimeError(
                f"cam{i}: FAILED TO OPEN '{path}'. Check the path is correct "
                f"relative to where you're running the script, and that the "
                f"file/codec is valid (try opening it in VLC)."
            )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[cam{i}] opened OK: '{path}' | {w}x{h} | {total_frames} frames")
        if total_frames == 0:
            print(f"[cam{i}] WARNING: reports 0 frames — file may be corrupt or unreadable")

    writers = []
    for i, cap in enumerate(caps):
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        out_path = os.path.join(output_dir, f"cam{i}_global_id.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers.append(cv2.VideoWriter(out_path, fourcc, fps, (w, h)))

    frame_idx = 0
    active = [True] * num_cameras
    rejected_crops = 0
    raw_detections_seen = 0
    latest_frames = [None] * num_cameras  # for live side-by-side display

    while any(active):
        frame_idx += 1

        for cam_id in range(num_cameras):
            if not active[cam_id]:
                continue
            ok, frame = caps[cam_id].read()
            if not ok:
                active[cam_id] = False
                continue

            # per-camera detection + tracking (persistent IDs via ByteTrack) —
            # each camera has its own detector instance, so tracker state
            # stays isolated and motion prediction sees a continuous stream
            results = detectors[cam_id].track(
                frame,
                classes=[0],              # person class only
                conf=conf_thresh,
                imgsz=imgsz,
                persist=True,
                tracker=TRACKER_CONFIG,
                verbose=False,
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                raw_detections_seen += len(results.boxes)
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                for box, local_id, det_conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    crop = frame[y1:y2, x1:x2]

                    # draw box regardless (visual continuity), but only feed
                    # good-quality crops into the ReID gallery
                    is_valid, quality_score = crop_quality_check(crop, (x1, y1, x2, y2), det_conf)

                    if is_valid:
                        is_new_key = (cam_id, local_id) not in gallery.crops
                        embedding = embedder.extract(crop)
                        histogram = extract_color_histogram(crop)
                        if embedding is not None:
                            gallery.update(cam_id, local_id, embedding, histogram, quality_score, frame_idx)
                            if is_new_key:
                                try_relink_same_camera(gallery, union_find, cam_id, (cam_id, local_id), frame_idx)
                    else:
                        rejected_crops += 1

                    key = (cam_id, local_id)
                    if key in gallery.last_seen:
                        age = frame_idx - gallery.first_seen[key]
                        if age >= MATCH_INTERVAL or key in union_find.frozen_display_id:
                            g_id = union_find.display_id(key)
                            label = f"ID {g_id}"
                            color = (0, 255, 0)
                        else:
                            # track is real but too young — cross-camera matching
                            # hasn't had a chance to run yet, so don't mint/freeze
                            # an id now (that's what caused ids to jump before)
                            label = f"track {local_id} (settling)"
                            color = (0, 165, 255)
                    else:
                        # tracked but no quality crop yet contributed to gallery
                        label = f"track {local_id} (pending)"
                        color = (0, 165, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                    )

            latest_frames[cam_id] = frame
            writers[cam_id].write(frame)

        if display and all(f is not None for f in latest_frames):
            # resize all frames to a common height, then stitch side by side
            target_h = 480
            resized = []
            for f in latest_frames:
                scale = target_h / f.shape[0]
                resized.append(cv2.resize(f, (int(f.shape[1] * scale), target_h)))
            combined = cv2.hconcat(resized)
            cv2.imshow("Multi-Camera ReID (press 'q' to quit)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed — stopping early.")
                active = [False] * num_cameras

        if frame_idx % MATCH_INTERVAL == 0:
            cross_camera_match(gallery, union_find, num_cameras)
            gallery.prune_stale(frame_idx)

        if frame_idx % 50 == 0:
            print(f"[frame {frame_idx}] active tracks: {len(gallery.crops)}, "
                  f"global IDs assigned: {union_find.next_global_id}, "
                  f"raw detections seen so far: {raw_detections_seen}, "
                  f"crops rejected so far: {rejected_crops}")

    for cap in caps:
        cap.release()
    for w in writers:
        w.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\nDone. Outputs written to: {output_dir}")
    print(f"Total global identities discovered: {union_find.next_global_id}")
    print(f"Total raw detections seen: {raw_detections_seen}")
    print(f"Total low-quality crops rejected: {rejected_crops}")
    if raw_detections_seen == 0:
        print("\n*** raw_detections_seen == 0: YOLO never detected a person in any frame. ***")
        print("Check: are these actually person-visible videos? Try lowering --conf (e.g. 0.2),")
        print("or run a quick standalone test: `yolo predict model=yolov8n.pt source=1.mp4 show=True`")


# ----------------------------- CLI ------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-camera person ReID POC")
    parser.add_argument("--videos", nargs="+", required=True,
                         help="Paths to camera video files, e.g. cam1.mp4 cam2.mp4")
    parser.add_argument("--output_dir", default="reid_outputs")
    parser.add_argument("--detector_weights", default="yolov8n.pt",
                         help="YOLOv8 weights (yolov8n/s/m.pt or custom)")
    parser.add_argument("--reid_weights", default=None,
                         help="Path to custom OSNet checkpoint (e.g. your MSMT17 weights)")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--no_display", action="store_true",
                         help="Disable the live side-by-side preview window")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Detection inference resolution — lower (e.g. 480) is faster but "
                              "less accurate on small/far detections")
    args = parser.parse_args()

    run(args.videos, args.output_dir, args.detector_weights, args.reid_weights,
        args.conf, display=not args.no_display, imgsz=args.imgsz)