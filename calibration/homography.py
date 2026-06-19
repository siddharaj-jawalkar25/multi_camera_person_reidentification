"""
Compute and apply a homography between two camera views using saved
correspondence points.

Usage:
    # Step 1: compute and save the homography matrix
    python homography.py compute --points correspondence_points.json --out homography_matrix.npy

    # Step 2 (used by other scripts): load and apply
    from homography import load_homography, project_point
    H = load_homography("homography_matrix.npy")
    projected_xy = project_point(H, (x, y))
"""

import json
import argparse
import numpy as np
import cv2


def compute_homography(points_a, points_b):
    """
    points_a, points_b: lists of (x, y) tuples, same length, same order.
    Returns the 3x3 homography matrix H such that:
        point_in_b ≈ H @ point_in_a   (in homogeneous coordinates)

    Uses RANSAC so a couple of mis-clicked points won't wreck the whole
    homography -- but with only ~4-8 points total, RANSAC has little
    room to reject anything, so double-check your clicks are accurate
    rather than relying on RANSAC to save you here.
    """
    pts_a = np.array(points_a, dtype=np.float32)
    pts_b = np.array(points_b, dtype=np.float32)

    if len(pts_a) < 4:
        raise ValueError("Need at least 4 point correspondences for a homography.")

    H, mask = cv2.findHomography(pts_a, pts_b, method=cv2.RANSAC)

    if H is None:
        raise RuntimeError(
            "Homography computation failed. Points may be collinear "
            "or too few/noisy. Re-pick points, ideally not all in a line."
        )

    inliers_used = int(mask.sum()) if mask is not None else len(pts_a)
    print(f"Homography computed. {inliers_used}/{len(pts_a)} points used as inliers.")

    return H


def project_point(H, point_xy):
    """
    Project a single (x, y) point from view A into view B's coordinate space
    using homography H.
    """
    pt = np.array([point_xy[0], point_xy[1], 1.0], dtype=np.float64)
    projected = H @ pt
    projected /= projected[2]  # de-homogenize
    return float(projected[0]), float(projected[1])


def project_points(H, points_xy):
    """Vectorized version for projecting many points at once."""
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H)
    return projected.reshape(-1, 2)


def load_homography(path):
    return np.load(path)


def save_homography(H, path):
    np.save(path, H)
    print(f"Homography matrix saved to {path}")


def reprojection_error(H, points_a, points_b):
    """
    Sanity check: for each correspondence pair used to BUILD the homography,
    project point_a and measure pixel distance to the real point_b.
    This tells you how well the homography fits its own training points --
    it does NOT tell you how well it generalizes to new points (e.g. the
    person's foot position), which is what visual inspection in the next
    script is for.
    """
    projected = project_points(H, points_a)
    actual = np.array(points_b, dtype=np.float32)
    errors = np.linalg.norm(projected - actual, axis=1)
    for i, err in enumerate(errors):
        print(f"  Point {i+1}: reprojection error = {err:.2f} px")
    print(f"Mean reprojection error: {errors.mean():.2f} px, Max: {errors.max():.2f} px")
    return errors


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    compute_p = sub.add_parser("compute")
    compute_p.add_argument("--points", required=True, help="Path to correspondence_points.json")
    compute_p.add_argument("--out", default="homography_matrix.npy")

    args = parser.parse_args()

    if args.command == "compute":
        with open(args.points) as f:
            data = json.load(f)

        H = compute_homography(data["points_a"], data["points_b"])
        print("\nHomography matrix (View A -> View B):")
        print(H)

        print("\nReprojection error on the correspondence points themselves:")
        reprojection_error(H, data["points_a"], data["points_b"])

        save_homography(H, args.out)


if __name__ == "__main__":
    main()
