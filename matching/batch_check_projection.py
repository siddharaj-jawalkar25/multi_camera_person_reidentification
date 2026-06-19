"""
Batch cross-view projection check.

Runs the same single-frame projection check across MANY frames at once,
so you can flip through a folder of outputs and look for patterns --
e.g. does drift increase as the person moves farther from the camera,
or is it roughly constant across the whole walk?

Two ways to feed it detections:

1. CSV mode (recommended): point it at a CSV with your YOLO/ByteTrack
   output already in it. Expected columns:
       frame, view, track_id, x, y, w, h
   It will pair up matching frame numbers between view_a and view_b
   automatically.

2. Manual mode: if you don't have a CSV yet, you can still batch-test
   by pointing at a folder of frames and supplying box_a only (no box_b)
   -- useful for a quick look at where projections land before your
   detection CSV is ready. In this mode you pass a single box_a and it
   reuses it across frames just to sanity check the homography itself
   (e.g. projecting a fixed ground point across the sequence). For real
   per-frame person projection you want CSV mode.

Usage (CSV mode):
    python batch_check_projection.py csv \
        --detections detections.csv \
        --view_a View_001 --view_b View_005 \
        --frames_dir_a path/to/View_001 \
        --frames_dir_b path/to/View_005 \
        --homography ../calibration/homography_matrix.npy \
        --out_dir ../outputs/batch_check \
        --frame_step 10

    frame_step controls sampling -- with frame_step=10 it checks every
    10th frame instead of all of them, useful for a quick first look
    across the full walk without generating hundreds of images.

Output:
    One image per checked frame in out_dir, named like
    frame_0123_check.jpg, plus a summary.csv logging pixel distance
    per frame (logged for your own reference -- still no pass/fail
    threshold applied).
"""

import argparse
import os
import csv
import numpy as np
import cv2
from homography import load_homography, project_point


def foot_point(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h)


def draw_box_and_point(img, box, color, label):
    x, y, w, h = box
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    fx, fy = foot_point(box)
    cv2.circle(img, (int(fx), int(fy)), 6, color, -1)
    cv2.putText(img, label, (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_projected_point(img, point_xy, label="projected"):
    x, y = point_xy
    cv2.drawMarker(img, (int(x), int(y)), (0, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3)
    cv2.putText(img, label, (int(x) + 10, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def load_detections_csv(path):
    """
    Returns dict keyed by (view, frame) -> list of (track_id, x, y, w, h)
    so multiple people per frame are supported even though right now
    you only have one.
    """
    detections = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["view"], int(row["frame"]))
            box = (
                int(row["track_id"]),
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            )
            detections.setdefault(key, []).append(box)
    return detections


def frame_path(frames_dir, frame_num, pattern="frame_{:04d}.jpg"):
    return os.path.join(frames_dir, pattern.format(frame_num))


def stack_side_by_side(img_a, img_b):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    target_h = max(h_a, h_b)
    if h_a != target_h:
        img_a = cv2.resize(img_a, (int(w_a * target_h / h_a), target_h))
    if h_b != target_h:
        img_b = cv2.resize(img_b, (int(w_b * target_h / h_b), target_h))
    return np.hstack([img_a, img_b])


def run_csv_mode(args):
    H = load_homography(args.homography)
    detections = load_detections_csv(args.detections)
    os.makedirs(args.out_dir, exist_ok=True)

    # Find all frame numbers present in BOTH views
    frames_a = {f for (v, f) in detections if v == args.view_a}
    frames_b = {f for (v, f) in detections if v == args.view_b}
    common_frames = sorted(frames_a & frames_b)

    if not common_frames:
        print("No overlapping frame numbers found between the two views "
              "in the detections CSV. Check your 'view' column values "
              f"match --view_a ({args.view_a}) and --view_b ({args.view_b}) exactly.")
        return

    sampled_frames = common_frames[::args.frame_step]
    print(f"{len(common_frames)} overlapping frames found, "
          f"checking {len(sampled_frames)} at step={args.frame_step}.")

    summary_rows = []

    for frame_num in sampled_frames:
        path_a = frame_path(args.frames_dir_a, frame_num)
        path_b = frame_path(args.frames_dir_b, frame_num)

        img_a = cv2.imread(path_a)
        img_b = cv2.imread(path_b)
        if img_a is None or img_b is None:
            print(f"  Skipping frame {frame_num}: could not load image "
                  f"({path_a} or {path_b}).")
            continue

        dets_a = detections.get((args.view_a, frame_num), [])
        dets_b = detections.get((args.view_b, frame_num), [])

        if not dets_a:
            print(f"  Skipping frame {frame_num}: no detection in view A.")
            continue

        # Single-person case: just take the first detection.
        # (When you move to multi-person, this is the spot that needs
        # to loop over all dets_a and match against all dets_b instead
        # of assuming one person.)
        track_id, x, y, w, h = dets_a[0]
        box_a = (x, y, w, h)
        img_a_draw = draw_box_and_point(img_a.copy(), box_a, (0, 0, 255), f"id{track_id} (A)")

        foot_a = foot_point(box_a)
        projected_xy = project_point(H, foot_a)
        img_b_draw = draw_projected_point(img_b.copy(), projected_xy)

        dist = None
        if dets_b:
            _, bx, by, bw, bh = dets_b[0]
            box_b = (bx, by, bw, bh)
            img_b_draw = draw_box_and_point(img_b_draw, box_b, (0, 0, 255), "person (B)")
            actual_foot_b = foot_point(box_b)
            dist = float(np.linalg.norm(np.array(projected_xy) - np.array(actual_foot_b)))

        combined = stack_side_by_side(img_a_draw, img_b_draw)
        out_path = os.path.join(args.out_dir, f"frame_{frame_num:04d}_check.jpg")
        cv2.imwrite(out_path, combined)

        summary_rows.append({
            "frame": frame_num,
            "track_id": track_id,
            "projected_x": round(projected_xy[0], 1),
            "projected_y": round(projected_xy[1], 1),
            "pixel_distance": round(dist, 1) if dist is not None else "no_detection_in_b",
        })

        dist_str = f"{dist:.1f}px" if dist is not None else "n/a (no detection in B)"
        print(f"  Frame {frame_num}: projected to {projected_xy}, distance={dist_str}")

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "track_id", "projected_x",
                                                 "projected_y", "pixel_distance"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nDone. {len(summary_rows)} frames checked.")
    print(f"Images saved to {args.out_dir}/")
    print(f"Summary log saved to {summary_path}")
    print("\nNo pass/fail threshold has been applied -- inspect the images "
          "and the pixel_distance column yourself to look for patterns "
          "(e.g. does distance grow for frames where the person is farther "
          "from the camera or near the edge of the frame).")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    csv_p = sub.add_parser("csv", help="Batch check using a detections CSV")
    csv_p.add_argument("--detections", required=True)
    csv_p.add_argument("--view_a", required=True)
    csv_p.add_argument("--view_b", required=True)
    csv_p.add_argument("--frames_dir_a", required=True)
    csv_p.add_argument("--frames_dir_b", required=True)
    csv_p.add_argument("--homography", default="../calibration/homography_matrix.npy")
    csv_p.add_argument("--out_dir", default="../outputs/batch_check")
    csv_p.add_argument("--frame_step", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "csv":
        run_csv_mode(args)


if __name__ == "__main__":
    main()
