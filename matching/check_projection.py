"""
Visual cross-view matching check (manual inspection, no threshold yet).

For a given frame index, takes the YOLO/ByteTrack detection in View A,
projects the foot-point into View B using the homography, and draws:
    - actual detection box + foot-point in View A (red)
    - projected foot-point in View B (green X)
    - actual detection box + foot-point in View B (red), if one exists

Stack the two frames side by side and save to outputs/ so you can flip
through them and visually judge: does the green X land close to the red
detection in View B, or does it drift? Does drift get worse for people
farther from the camera (this is the depth/slope issue flagged in the
PETS2009 literature)?

This intentionally does NOT compute a pass/fail distance threshold yet --
that comes after you've seen the error pattern with your own eyes.

Usage:
    python check_projection.py \
        --frame_a path/to/view1/frame_0123.jpg \
        --frame_b path/to/view6/frame_0123.jpg \
        --box_a 410,260,40,90 \
        --box_b 512,300,38,85 \
        --homography ../calibration/homography_matrix.npy \
        --out ../outputs/check_frame_0123.jpg

    box_a / box_b are optional -- pass box_a only if you just want to see
    where the projection lands without comparing to a real detection in B yet.
"""

import argparse
import numpy as np
import cv2
from homography import load_homography, project_point


def foot_point(box):
    """box = (x, y, w, h) -> bottom-center point, used as the ground-plane anchor."""
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


def parse_box(box_str):
    return tuple(float(v) for v in box_str.split(","))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_a", required=True)
    parser.add_argument("--frame_b", required=True)
    parser.add_argument("--box_a", required=True, help="x,y,w,h of detection in View A")
    parser.add_argument("--box_b", default=None, help="x,y,w,h of detection in View B (optional)")
    parser.add_argument("--homography", default="../calibration/homography_matrix.npy")
    parser.add_argument("--out", default="../outputs/check_projection.jpg")
    args = parser.parse_args()

    img_a = cv2.imread(args.frame_a)
    img_b = cv2.imread(args.frame_b)
    if img_a is None or img_b is None:
        raise FileNotFoundError("Could not load one of the input frames.")

    H = load_homography(args.homography)
    box_a = parse_box(args.box_a)

    # Draw the real detection in View A
    img_a = draw_box_and_point(img_a, box_a, (0, 0, 255), "person (A)")

    # Project the foot-point into View B
    foot_a = foot_point(box_a)
    projected_xy = project_point(H, foot_a)
    img_b = draw_projected_point(img_b, projected_xy)

    # If we also have the real detection in B, draw it too for comparison
    if args.box_b:
        box_b = parse_box(args.box_b)
        img_b = draw_box_and_point(img_b, box_b, (0, 0, 255), "person (B)")
        actual_foot_b = foot_point(box_b)
        dist = np.linalg.norm(np.array(projected_xy) - np.array(actual_foot_b))
        print(f"Projected foot-point: {projected_xy}")
        print(f"Actual foot-point in B: {actual_foot_b}")
        print(f"Pixel distance (for your own reference, no threshold applied): {dist:.1f} px")
    else:
        print(f"Projected foot-point in View B: {projected_xy}")
        print("No box_b provided -- just eyeball where the green X lands.")

    # Stack side by side, resizing to same height if needed
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    target_h = max(h_a, h_b)
    if h_a != target_h:
        img_a = cv2.resize(img_a, (int(w_a * target_h / h_a), target_h))
    if h_b != target_h:
        img_b = cv2.resize(img_b, (int(w_b * target_h / h_b), target_h))

    combined = np.hstack([img_a, img_b])
    cv2.imwrite(args.out, combined)
    print(f"Saved visual check to {args.out}")


if __name__ == "__main__":
    main()
