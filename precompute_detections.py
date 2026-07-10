"""
Precompute Detections Cache

Runs YOLO + the tracker ONCE per camera and saves every
(frame_idx, track_id, x1, y1, x2, y2, confidence) to a .npz file.

WHY: almost all iteration on this project happens on the ReID/matching side
(quality thresholds, fusion weights, gallery size, matching thresholds) —
none of that touches detection at all. Re-running YOLO from scratch every
time you tweak one of those wastes the most expensive part of the pipeline
for no reason. Generate the cache once, then iterate on main.py freely with
--detections_cache_dir pointed at it — detection is skipped entirely.

IMPORTANT: the cache is only valid for the EXACT combination of
detector_weights, --conf, --imgsz, and --tracker used here. If you change
any of those, delete the cache and regenerate it — otherwise main.py will
silently use stale/mismatched detections.

USAGE:
    python precompute_detections.py --videos cam1.mp4 cam2.mp4 \
        --output_dir detections_cache --detector_weights yolo26x.pt \
        --conf 0.2 --imgsz 960 --tracker bytetrack_extended.yaml

Then in main.py:
    python main.py --videos cam1.mp4 cam2.mp4 --output_dir outputs \
        --reid_weights weights\\osnet_x1_0_msmt17.pth \
        --detections_cache_dir detections_cache
"""

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO


def precompute(video_paths, output_dir, detector_weights, conf_thresh, imgsz, tracker_config):
    os.makedirs(output_dir, exist_ok=True)

    for cam_id, video_path in enumerate(video_paths):
        print(f"[cam{cam_id}] loading detector for '{video_path}' ...")
        # fresh detector instance per camera — same reasoning as main.py:
        # persist=True tracker state must not be shared across video sources
        detector = YOLO(detector_weights)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cam{cam_id}: failed to open '{video_path}'")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[cam{cam_id}] {total_frames} frames to process")

        rows = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = detector.track(
                frame,
                classes=[0],
                conf=conf_thresh,
                imgsz=imgsz,
                persist=True,
                tracker=tracker_config,
                verbose=False,
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                for box, tid, conf in zip(boxes, track_ids, confs):
                    rows.append([frame_idx, tid, box[0], box[1], box[2], box[3], conf])

            if frame_idx % 100 == 0:
                print(f"[cam{cam_id}] frame {frame_idx}/{total_frames}, "
                      f"detections so far: {len(rows)}")

        cap.release()

        arr = np.array(rows, dtype=np.float32) if rows else np.zeros((0, 7), dtype=np.float32)
        out_path = os.path.join(output_dir, f"cam{cam_id}_detections.npz")
        np.savez_compressed(
            out_path,
            detections=arr,
            meta=np.array(
                [detector_weights, str(conf_thresh), str(imgsz), tracker_config, video_path],
                dtype=object,
            ),
        )
        print(f"[cam{cam_id}] saved {len(rows)} detections across {frame_idx} frames "
              f"to '{out_path}'\n")

    print("Done. Run main.py with --detections_cache_dir pointed at this folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute YOLO+tracker detections for caching")
    parser.add_argument("--videos", nargs="+", required=True,
                         help="Same video paths/order you'll pass to main.py")
    parser.add_argument("--output_dir", default="detections_cache")
    parser.add_argument("--detector_weights", default="yolo26x.pt",
                         help="Must match what you intend to use in main.py conceptually — "
                              "though once cached, main.py doesn't re-run this model at all")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--tracker", default="bytetrack_extended.yaml")
    args = parser.parse_args()

    precompute(args.videos, args.output_dir, args.detector_weights,
               args.conf, args.imgsz, args.tracker)