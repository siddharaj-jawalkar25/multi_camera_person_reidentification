"""
OSNet-based cross-view person embedding and similarity scoring.

Given a frame number, a track_id in View A, and a track_id in View B,
this looks up both bounding boxes from detections.csv, crops the person
from each frame, runs both crops through OSNet, and returns the cosine
similarity between their embeddings.

This is a TIEBREAKER signal, not a primary matching signal -- intended
to be called only for the small number of geometrically-ambiguous pairs
flagged by hungarian_match.py, not for every detection in every frame.

Setup:
    pip install torchreid torch torchvision --break-system-packages

    torchreid will auto-download osnet_x1_0 pretrained weights
    (trained on Market1501) on first use if no --weights path is given.

Usage as a script (quick manual check):
    python reid_embedder.py \
        --detections ../detections.csv \
        --frames_dir_a ../View_001 --frames_dir_b ../View_005 \
        --view_a View_001 --view_b View_005 \
        --frame 120 --track_id_a 3 --track_id_b 7

Usage as a module (for hungarian_match_with_reid.py):
    from reid_embedder import ReIDEmbedder
    embedder = ReIDEmbedder()
    similarity = embedder.compare(
        detections_csv="../detections.csv",
        frames_dir_a="../View_001", frames_dir_b="../View_005",
        view_a="View_001", view_b="View_005",
        frame=120, track_id_a=3, track_id_b=7,
    )
"""

import argparse
import csv
import os

import numpy as np
import cv2


def foot_point(x, y, w, h):
    return (x + w / 2.0, y + h)


class ReIDEmbedder:
    """
    Thin wrapper around an OSNet model for cross-view person re-id
    embedding + similarity. Loads the model once; reuse the same
    instance across many comparisons rather than constructing a new
    one per call -- model loading is the slow part.
    """

    def __init__(self, model_name="osnet_x1_0", weights_path=None, device=None):
        # Imports kept inside __init__ rather than at module level so
        # that importing THIS FILE for its helper functions (e.g. in
        # the validation script) doesn't hard-require torch/torchreid
        # to be installed if you only need the CSV/box-lookup utilities.
        import torch
        import torchreid

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # num_classes is required by torchreid's model builder but is
        # irrelevant here -- we only ever use the embedding (feature
        # vector) output, never a classification head.
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=(weights_path is None),
        )

        if weights_path is not None:
            torchreid.utils.load_pretrained_weights(self.model, weights_path)

        self.model.eval()
        self.model.to(self.device)

        # Standard ReID preprocessing: OSNet's torchreid weights were
        # trained on 256x128 (h x w) crops normalized with ImageNet
        # stats. Using different input stats than training time will
        # quietly degrade embedding quality without throwing an error,
        # so don't change these without a reason.
        self.input_size = (256, 128)  # (height, width)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, crop_bgr):
        """crop_bgr: a cv2 image crop (BGR, uint8). Returns a model-ready tensor."""
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))  # cv2 wants (w, h)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        tensor = self.torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        return tensor

    def embed_crop(self, crop_bgr):
        """
        Returns a 1D numpy embedding vector for a single person crop.
        Returns None if the crop is empty/invalid (e.g. box landed
        outside the frame, near-zero width/height after clamping).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
            # Crop is essentially a sliver -- almost certainly a bad
            # box rather than a usable person image. Flag rather than
            # silently embed garbage.
            return None

        tensor = self._preprocess(crop_bgr)
        with self.torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()

    @staticmethod
    def cosine_similarity(emb1, emb2):
        if emb1 is None or emb2 is None:
            return None
        a = emb1 / (np.linalg.norm(emb1) + 1e-8)
        b = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(a, b))

    def compare(self, detections_csv, frames_dir_a, frames_dir_b,
                view_a, view_b, frame, track_id_a, track_id_b,
                frame_pattern="frame_{:04d}.jpg"):
        """
        High-level convenience method: look up boxes, crop, embed, and
        return cosine similarity in one call. Returns None (with a
        printed reason) if anything along the way fails -- a tiebreaker
        that silently returns 0.0 on failure is worse than one that
        tells you it couldn't produce a number at all.
        """
        box_a = lookup_box(detections_csv, view_a, frame, track_id_a)
        box_b = lookup_box(detections_csv, view_b, frame, track_id_b)

        if box_a is None:
            print(f"No detection found for {view_a}, frame {frame}, track_id {track_id_a}")
            return None
        if box_b is None:
            print(f"No detection found for {view_b}, frame {frame}, track_id {track_id_b}")
            return None

        crop_a = crop_person(frames_dir_a, frame, box_a, frame_pattern)
        crop_b = crop_person(frames_dir_b, frame, box_b, frame_pattern)

        emb_a = self.embed_crop(crop_a)
        emb_b = self.embed_crop(crop_b)

        if emb_a is None:
            print(f"Could not embed crop for {view_a}, frame {frame}, track_id {track_id_a} "
                  f"(crop missing or too small -- check the box coordinates)")
            return None
        if emb_b is None:
            print(f"Could not embed crop for {view_b}, frame {frame}, track_id {track_id_b} "
                  f"(crop missing or too small -- check the box coordinates)")
            return None

        return self.cosine_similarity(emb_a, emb_b)


def lookup_box(detections_csv, view, frame, track_id):
    """
    Scans detections.csv for a matching (view, frame, track_id) row.
    For a one-off lookup this is fine; if you're calling this in a tight
    loop over many pairs, load the CSV once into a dict instead (see
    load_detections_lookup in analyze_rejected_matches.py for that pattern)
    rather than re-reading the file from disk every call.
    """
    with open(detections_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["view"] == view and int(row["frame"]) == frame
                    and int(row["track_id"]) == track_id):
                return (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
    return None


def crop_person(frames_dir, frame, box, frame_pattern="frame_{:04d}.jpg"):
    """
    Loads the frame image and crops the person box out of it.
    Clamps the box to image bounds rather than failing outright if a
    box slightly overhangs the frame edge (common for people near the
    edge of the camera's field of view).
    """
    path = os.path.join(frames_dir, frame_pattern.format(frame))
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load frame image: {path}")

    x, y, w, h = box
    img_h, img_w = img.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))

    return img[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--frames_dir_a", required=True)
    parser.add_argument("--frames_dir_b", required=True)
    parser.add_argument("--view_a", required=True)
    parser.add_argument("--view_b", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--track_id_a", type=int, required=True)
    parser.add_argument("--track_id_b", type=int, required=True)
    parser.add_argument("--model_name", default="osnet_x1_0")
    parser.add_argument("--weights", default=None,
                         help="Optional path to custom OSNet weights. "
                              "If omitted, torchreid downloads pretrained "
                              "Market1501 weights automatically.")
    args = parser.parse_args()

    embedder = ReIDEmbedder(model_name=args.model_name, weights_path=args.weights)

    similarity = embedder.compare(
        detections_csv=args.detections,
        frames_dir_a=args.frames_dir_a, frames_dir_b=args.frames_dir_b,
        view_a=args.view_a, view_b=args.view_b,
        frame=args.frame, track_id_a=args.track_id_a, track_id_b=args.track_id_b,
    )

    if similarity is not None:
        print(f"\nCosine similarity: {similarity:.4f}")
        print("(higher = more visually similar; 1.0 = identical embedding, "
              "-1.0 = maximally dissimilar, though in practice ReID "
              "embeddings rarely go very negative)")
    else:
        print("\nCould not compute similarity -- see error above.")


if __name__ == "__main__":
    main()
