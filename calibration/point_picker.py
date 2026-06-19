"""
Interactive correspondence point picker for two-view homography.

Usage:
    python point_picker.py --img1 path/to/view1_frame.jpg --img2 path/to/view2_frame.jpg --out correspondence_points.json

How it works:
    - Two windows open side by side: View A and View B.
    - Click a point in View A, then click the SAME real-world point in View B.
    - Points must be clicked in matching order (1st in A <-> 1st in B, etc).
    - Aim for points spread across the FULL DEPTH of the shared ground plane,
      not clustered close to the camera -- homography accuracy degrades fastest
      at the edges of your point spread.
    - Good targets on PETS2009: painted ground-line intersections, cone bases,
      corners of the parking bay markings -- anything flat on the ground.
    - Press 's' to save once you have at least 4 points in each view.
    - Press 'u' to undo the last point in the currently active window.
    - Press 'q' to quit without saving.
"""

import cv2
import json
import argparse
import numpy as np


class PointPicker:
    def __init__(self, img1_path, img2_path):
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)
        if self.img1 is None:
            raise FileNotFoundError(f"Could not load image: {img1_path}")
        if self.img2 is None:
            raise FileNotFoundError(f"Could not load image: {img2_path}")

        self.points1 = []
        self.points2 = []

        self.win1 = "View A (click ground points)"
        self.win2 = "View B (click SAME points, same order)"

    def _draw(self, img, points):
        disp = img.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return disp

    def _mouse_cb_1(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points1.append((x, y))
            print(f"View A point {len(self.points1)}: ({x}, {y})")
            cv2.imshow(self.win1, self._draw(self.img1, self.points1))

    def _mouse_cb_2(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points2.append((x, y))
            print(f"View B point {len(self.points2)}: ({x}, {y})")
            cv2.imshow(self.win2, self._draw(self.img2, self.points2))

    def run(self):
        cv2.namedWindow(self.win1)
        cv2.namedWindow(self.win2)
        cv2.setMouseCallback(self.win1, self._mouse_cb_1)
        cv2.setMouseCallback(self.win2, self._mouse_cb_2)

        cv2.imshow(self.win1, self.img1)
        cv2.imshow(self.win2, self.img2)

        print("\nClick matching ground points in BOTH views, same order.")
        print("Press 's' to save, 'u' to undo last point in either window, 'q' to quit.\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('u'):
                if self.points1 and len(self.points1) >= len(self.points2):
                    self.points1.pop()
                    cv2.imshow(self.win1, self._draw(self.img1, self.points1))
                elif self.points2:
                    self.points2.pop()
                    cv2.imshow(self.win2, self._draw(self.img2, self.points2))
            elif key == ord('s'):
                if len(self.points1) != len(self.points2):
                    print(f"Mismatch: {len(self.points1)} points in A, "
                          f"{len(self.points2)} in B. Must be equal.")
                    continue
                if len(self.points1) < 4:
                    print(f"Need at least 4 point pairs, have {len(self.points1)}.")
                    continue
                cv2.destroyAllWindows()
                return self.points1, self.points2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", required=True, help="Path to View A reference frame")
    parser.add_argument("--img2", required=True, help="Path to View B reference frame")
    parser.add_argument("--out", default="correspondence_points.json")
    args = parser.parse_args()

    picker = PointPicker(args.img1, args.img2)
    result = picker.run()

    if result is None:
        print("Cancelled, nothing saved.")
        return

    points1, points2 = result
    data = {
        "view_a_image": args.img1,
        "view_b_image": args.img2,
        "points_a": points1,
        "points_b": points2,
    }
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(points1)} point pairs to {args.out}")


if __name__ == "__main__":
    main()
