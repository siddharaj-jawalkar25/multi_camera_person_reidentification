"""
Export YOLOv8 + ByteTrack tracking output to CSV, in the format expected
by batch_check_projection.py:

    frame, view, track_id, x, y, w, h

where (x, y) is the TOP-LEFT corner of the box and (w, h) is width/height
-- this matches what foot_point() in the projection scripts expects
(bottom-center = (x + w/2, y + h)).

This version runs directly on a VIDEO FILE (cam1.mp4, cam2.mp4), matching
a model.track(source="cam1.mp4", tracker="bytetrack.yaml", persist=True)
style call -- letting Ultralytics process the video in one continuous
pass internally, which is both faster and more reliable for ID
persistence than calling track() once per individual frame image.

IMPORTANT -- frame numbering caveat:
    The "frame" column here is the video's internal frame INDEX (0, 1, 2...
    in playback order), NOT necessarily the original frame_XXXX.jpg number
    from the source dataset. These are only the same thing if your video
    was created from the frame folder with zero drops/duplicates and in
    the correct order. RUN verify_frame_alignment.py FIRST and confirm a
    clean match before trusting that frame N in this CSV corresponds to
    frame_{N:04d}.jpg in the original folder. If they don't match 1:1,
    you'll need a mapping step between video frame index and original
    filename before the homography/projection scripts will line up correctly.

Run this ONCE PER VIEW (once per video file), then merge the two CSVs:

Usage:
    python export_tracks_to_csv.py \
        --video cam1.mp4 \
        --view View_001 \
        --weights yolov8n.pt \
        --out tracks_view001.csv

    python export_tracks_to_csv.py \
        --video cam2.mp4 \
        --view View_005 \
        --weights yolov8n.pt \
        --out tracks_view005.csv

Then merge:
    cat tracks_view001.csv > detections.csv
    tail -n +2 tracks_view005.csv >> detections.csv

(tail -n +2 skips the header row on the second file so you don't get
 a duplicate header in the middle of the combined CSV)

Notes:
    - classes=[0] restricts detection to the "person" class only (class 0
      in the standard COCO-trained YOLOv8 weights). Drop this filter if
      you're using a custom-trained model with different class indices.
    - persist=True maintains consistent track_ids as Ultralytics processes
      the video frame-by-frame internally in this single call.
    - stream=True is used so frames are processed and yielded one at a
      time rather than all loaded into memory at once -- matters once
      you're running this on full-length sequences rather than a handful
      of test frames.
"""

import argparse
import csv
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to cam1.mp4 / cam2.mp4")
    parser.add_argument("--view", required=True,
                         help="View label to write into the CSV, e.g. View_001")
    parser.add_argument("--weights", default="yolov8n.pt",
                         help="YOLOv8 weights file (default: yolov8n.pt, downloads if missing)")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         help="Tracker config, default uses Ultralytics' built-in ByteTrack")
    parser.add_argument("--conf", type=float, default=0.3,
                         help="Confidence threshold for detections")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    model = YOLO(args.weights)

    results_generator = model.track(
        source=args.video,
        tracker=args.tracker,
        persist=True,
        classes=[0],          # person class only -- drop this for custom models
        conf=args.conf,
        stream=True,          # yields results one frame at a time, doesn't load whole video into memory
        verbose=False,
    )

    rows = []
    frame_num = -1

    for frame_num, result in enumerate(results_generator):
        if result.boxes is None or result.boxes.id is None:
            # No detections, or none assigned a track ID yet (can happen
            # on the first couple frames while ByteTrack initializes).
            continue

        boxes_xywh = result.boxes.xywh.cpu().numpy()   # center_x, center_y, w, h
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for (cx, cy, w, h), track_id in zip(boxes_xywh, track_ids):
            # Convert center-based xywh (Ultralytics default) to
            # top-left-based xywh (what the projection scripts expect)
            top_left_x = cx - w / 2.0
            top_left_y = cy - h / 2.0

            rows.append({
                "frame": frame_num,
                "view": args.view,
                "track_id": track_id,
                "x": round(float(top_left_x), 1),
                "y": round(float(top_left_y), 1),
                "w": round(float(w), 1),
                "h": round(float(h), 1),
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "view", "track_id", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} detection rows to {args.out}")
    print(f"Processed {frame_num + 1} video frames total.")

    if not rows:
        print("WARNING: zero rows written. Check your --conf threshold, "
              "--classes filter, and that the video actually contains people.")


if __name__ == "__main__":
    main()
