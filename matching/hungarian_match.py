"""
Cross-view identity matching using the Hungarian algorithm on projected
ground-plane distances.

This REPLACES the naive "first detection vs first detection" comparison
in batch_check_projection.py. For every frame that has detections in BOTH
views, it:

    1. Projects every person's foot-point from View A into View B's
       coordinate space using the homography.
    2. Builds a cost matrix: cost[i][j] = pixel distance between
       projected person i (from A) and actual person j (in B).
    3. Solves the assignment problem with scipy's linear_sum_assignment
       (Hungarian algorithm) to find the lowest-total-cost one-to-one
       pairing -- NOT just "closest pair greedily," which can produce
       worse global results when multiple people are near each other.
    4. Optionally rejects matches above --max_distance as "no match"
       rather than forcing a pairing when nobody is actually close.

This is geometry-only matching. No ReID yet -- the point of this step is
to see HOW WELL pure geometry performs before deciding where ReID is
actually needed as a tiebreaker (look at which frames/matches have high
cost or get rejected -- that's your evidence for where ReID would help).

Usage:
    python hungarian_match.py \
        --detections ../detections.csv \
        --view_a View_001 --view_b View_005 \
        --homography ../calibration/homography_matrix.npy \
        --out matched_identities.csv \
        --max_distance 150

    --max_distance is a REJECTION threshold, not a pass/fail grade on
    homography quality -- pairs costing more than this are left unmatched
    rather than force-assigned. Start generous (e.g. 150-200px) since you
    don't yet know the "normal" cost distribution for true matches across
    a full multi-person scene; tighten it once you've looked at the output.
"""

import argparse
import csv
from collections import defaultdict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.optimize import linear_sum_assignment

from calibration.homography import load_homography, project_point


def foot_point(x, y, w, h):
    return (x + w / 2.0, y + h)


def load_detections_by_frame(path, view_a, view_b):
    """
    Returns two dicts: {frame_num: [(track_id, x, y, w, h), ...]}
    one for view_a, one for view_b.
    """
    dets_a = defaultdict(list)
    dets_b = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            entry = (
                int(row["track_id"]),
                float(row["x"]), float(row["y"]),
                float(row["w"]), float(row["h"]),
            )
            if row["view"] == view_a:
                dets_a[frame].append(entry)
            elif row["view"] == view_b:
                dets_b[frame].append(entry)

    return dets_a, dets_b


def match_frame(H, people_a, people_b, max_distance):
    """
    people_a, people_b: lists of (track_id, x, y, w, h)

    Returns a list of dicts describing matches (and unmatched people on
    both sides), for this single frame.
    """
    if not people_a or not people_b:
        # Nobody to match against on one side -- everyone is unmatched.
        results = []
        for tid, x, y, w, h in people_a:
            results.append({"track_id_a": tid, "track_id_b": None,
                             "cost_px": None, "status": "unmatched_a_no_b_detections"})
        for tid, x, y, w, h in people_b:
            results.append({"track_id_a": None, "track_id_b": tid,
                             "cost_px": None, "status": "unmatched_b_no_a_detections"})
        return results

    # Project every person in A into B's coordinate space
    projected_a = []
    for tid, x, y, w, h in people_a:
        fx, fy = foot_point(x, y, w, h)
        projected_a.append(project_point(H, (fx, fy)))

    feet_b = [foot_point(x, y, w, h) for (_, x, y, w, h) in people_b]

    # Cost matrix: rows = people in A (projected), cols = people in B (actual)
    cost_matrix = np.zeros((len(projected_a), len(feet_b)))
    for i, p_proj in enumerate(projected_a):
        for j, p_actual in enumerate(feet_b):
            cost_matrix[i, j] = np.linalg.norm(
                np.array(p_proj) - np.array(p_actual)
            )

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    results = []
    matched_a_indices = set()
    matched_b_indices = set()

    for i, j in zip(row_idx, col_idx):
        cost = cost_matrix[i, j]
        tid_a = people_a[i][0]
        tid_b = people_b[j][0]

        if cost <= max_distance:
            results.append({"track_id_a": tid_a, "track_id_b": tid_b,
                             "cost_px": round(float(cost), 1), "status": "matched"})
            matched_a_indices.add(i)
            matched_b_indices.add(j)
        else:
            # Hungarian found this as the best pairing, but it's still
            # too far to trust -- reject rather than force-assign.
            results.append({"track_id_a": tid_a, "track_id_b": tid_b,
                             "cost_px": round(float(cost), 1),
                             "status": "rejected_too_far"})
            matched_a_indices.add(i)
            matched_b_indices.add(j)

    # Anyone Hungarian didn't even pair up at all (only happens when
    # len(people_a) != len(people_b))
    for i, (tid, x, y, w, h) in enumerate(people_a):
        if i not in matched_a_indices:
            results.append({"track_id_a": tid, "track_id_b": None,
                             "cost_px": None, "status": "unmatched_a_extra"})
    for j, (tid, x, y, w, h) in enumerate(people_b):
        if j not in matched_b_indices:
            results.append({"track_id_a": None, "track_id_b": tid,
                             "cost_px": None, "status": "unmatched_b_extra"})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--view_a", required=True)
    parser.add_argument("--view_b", required=True)
    parser.add_argument("--homography", default="../calibration/homography_matrix.npy")
    parser.add_argument("--out", default="matched_identities.csv")
    parser.add_argument("--max_distance", type=float, default=150.0,
                         help="Pixel distance above which a match is rejected, "
                              "not force-assigned. Start generous, tighten later.")
    args = parser.parse_args()

    H = load_homography(args.homography)
    dets_a, dets_b = load_detections_by_frame(args.detections, args.view_a, args.view_b)

    common_frames = sorted(set(dets_a.keys()) | set(dets_b.keys()))
    print(f"Processing {len(common_frames)} frames with detections in at least one view...")

    all_results = []
    status_counts = defaultdict(int)

    for frame in common_frames:
        people_a = dets_a.get(frame, [])
        people_b = dets_b.get(frame, [])
        frame_results = match_frame(H, people_a, people_b, args.max_distance)

        for r in frame_results:
            r["frame"] = frame
            status_counts[r["status"]] += 1
            all_results.append(r)

    fieldnames = ["frame", "track_id_a", "track_id_b", "cost_px", "status"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nWrote {len(all_results)} rows to {args.out}")
    print("\nStatus breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")

    matched_costs = [r["cost_px"] for r in all_results
                     if r["status"] == "matched" and r["cost_px"] is not None]
    if matched_costs:
        print(f"\nFor 'matched' pairs only:")
        print(f"  Mean cost: {np.mean(matched_costs):.1f} px")
        print(f"  Median cost: {np.median(matched_costs):.1f} px")
        print(f"  Max cost: {np.max(matched_costs):.1f} px")
        print("\nThis is your real geometry-only performance signal -- look at "
              "'rejected_too_far' rows next to see exactly which frames/people "
              "geometry struggles with. Those are your candidates for where "
              "ReID would actually need to step in as a tiebreaker.")


if __name__ == "__main__":
    main()
