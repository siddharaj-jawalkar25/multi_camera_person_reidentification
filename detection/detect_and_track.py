"""
Step 1: Detection + Tracking (Person A)
-----------------------------------------
Run YOLOv8 person detection + ByteTrack tracking on a video.
Saves cropped person images per track ID into data/crops/<cam_id>_track<id>/

Usage:
    python detect_and_track.py --source data/videos/cam1.mp4 --cam_id cam1

Settings (model name, tracker config, paths) are pulled from config.py —
change values there, not in this file, so Person B's scripts stay in sync.
"""

import argparse
import os
import sys
import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_MODEL, TRACKER_CONFIG, DETECT_CLASSES, CROPS_DIR, ANNOTATED_DIR


def run(source: str, cam_id: str, output_dir: str = CROPS_DIR, show: bool = False, save_video: bool = True):
    model = YOLO(YOLO_MODEL)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_video:
        os.makedirs(ANNOTATED_DIR, exist_ok=True)
        out_path = f"{ANNOTATED_DIR}/{cam_id}_tracked.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    track_counts = {}

    # model.track returns a generator of Results when stream=True
    results_gen = model.track(
        source=source,
        classes=DETECT_CLASSES,
        tracker=TRACKER_CONFIG,
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results_gen:
        frame = result.orig_img
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), tid, conf in zip(xyxy, track_ids, confs):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                track_dir = os.path.join(output_dir, f"{cam_id}_track{tid}")
                os.makedirs(track_dir, exist_ok=True)

                count = track_counts.get(tid, 0)
                # save every 5th frame per track to avoid thousands of near-duplicate crops
                if count % 5 == 0:
                    crop_path = os.path.join(track_dir, f"frame{frame_idx}.jpg")
                    cv2.imwrite(crop_path, crop)
                track_counts[tid] = count + 1

                # draw box + id on frame for visual confirmation
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"[{cam_id}] Done. Processed {frame_idx} frames.")
    print(f"[{cam_id}] Track IDs found: {sorted(track_counts.keys())}")
    print(f"[{cam_id}] Crops saved under: {output_dir}/{cam_id}_track*/")
    if save_video:
        print(f"[{cam_id}] Annotated video saved to: {ANNOTATED_DIR}/{cam_id}_tracked.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--cam_id", required=True, help="Camera identifier, e.g. cam1")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    run(source=args.source, cam_id=args.cam_id, show=args.show)
