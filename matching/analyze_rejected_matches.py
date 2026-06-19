"""
Inspect 'rejected_too_far' cases from hungarian_match.py output, and
visualize the actual frames involved -- so you can tell, by eye, whether
a high-cost case is:

    (a) a genuine homography/projection problem (the projected point
        lands nowhere near ANY real person in View B), or
    (b) a cross-view ambiguity / occlusion / missed-detection issue
        (the projected point is reasonably close to a plausible person,
        but Hungarian picked a worse pairing, or the right person simply
        isn't visible in one view at that frame), or
    (c) a detection problem upstream (YOLO missed someone, or boxed the
        wrong thing entirely) that has nothing to do with geometry or
        ReID at all.

This distinction matters: (a) means "go back and improve homography",
(b) means "this is exactly where ReID-as-tiebreaker should help", and
(c) means "fix detection first, nothing downstream can compensate."

What it does:
    1. Reads matched_identities.csv, pulls out all 'rejected_too_far' rows.
    2. Ranks them by cost (worst first) and also samples some mid-range
       ones, since the absolute worst cases are often boring/obvious
       (e.g. one camera had zero detections) while mid-range rejections
       are usually the more informative "almost-but-not-quite" cases.
    3. For each selected case, looks up the ACTUAL box coordinates for
       track_id_a and track_id_b from detections.csv (since
       matched_identities.csv only stores IDs + cost, not coordinates).
    4. Draws both views side by side: the real detection in A, the
       projected point in B (green X), and the real detection in B that
       Hungarian rejected as a match (red box) -- plus, if there's a
       CLOSER unrejected/unused detection in B nearby, flags that too,
       since "the right match might have been available but wasn't
       picked" is a different failure mode than "no plausible match
       existed at all."
    5. Saves annotated images + a text summary you can read without
       opening every image.

Usage:
    python analyze_rejected_matches.py \
        --matched ../matched_identities.csv \
        --detections ../detections.csv \
        --view_a View_001 --view_b View_005 \
        --frames_dir_a path/to/View_001 \
        --frames_dir_b path/to/View_005 \
        --homography ../calibration/homography_matrix.npy \
        --out_dir ../outputs/rejected_analysis \
        --top_n 15 \
        --sample_midrange 10
"""

import argparse
import os
import csv
from collections import defaultdict

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from calibration.homography import load_homography, project_point


def foot_point(x, y, w, h):
    return (x + w / 2.0, y + h)


def draw_box_and_point(img, box, color, label):
    x, y, w, h = box
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
    fx, fy = foot_point(x, y, w, h)
    cv2.circle(img, (int(fx), int(fy)), 6, color, -1)
    cv2.putText(img, label, (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def draw_projected_point(img, point_xy, label="projected"):
    x, y = point_xy
    cv2.drawMarker(img, (int(x), int(y)), (0, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3)
    cv2.putText(img, label, (int(x) + 10, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return img


def stack_side_by_side(img_a, img_b):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    target_h = max(h_a, h_b)
    if h_a != target_h:
        img_a = cv2.resize(img_a, (int(w_a * target_h / h_a), target_h))
    if h_b != target_h:
        img_b = cv2.resize(img_b, (int(w_b * target_h / h_b), target_h))
    return np.hstack([img_a, img_b])


def frame_path(frames_dir, frame_num, pattern="frame_{:04d}.jpg"):
    return os.path.join(frames_dir, pattern.format(frame_num))


def load_matched_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_detections_lookup(path):
    """
    Returns dict: (view, frame, track_id) -> (x, y, w, h)
    and a second dict: (view, frame) -> [list of all (track_id, x, y, w, h)]
    The second is needed to check "was there a closer person available
    that Hungarian didn't pick."
    """
    by_id = {}
    by_frame = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view = row["view"]
            frame = int(row["frame"])
            tid = int(row["track_id"])
            box = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
            by_id[(view, frame, tid)] = box
            by_frame[(view, frame)].append((tid, *box))
    return by_id, by_frame


def select_cases(rejected_rows, top_n, sample_midrange):
    """
    Worst-cost cases are often trivially explained (e.g. only one person
    visible in one view, nothing to match against). Mid-range cases are
    usually more diagnostically useful -- "almost a match, but not quite"
    is exactly where the geometry-vs-ReID question gets interesting.
    """
    sorted_rows = sorted(rejected_rows, key=lambda r: float(r["cost_px"]), reverse=True)

    worst = sorted_rows[:top_n]

    remaining = sorted_rows[top_n:]
    if remaining and sample_midrange > 0:
        step = max(1, len(remaining) // sample_midrange)
        midrange = remaining[::step][:sample_midrange]
    else:
        midrange = []

    return worst, midrange


def analyze_case(row, H, by_id, by_frame, view_a, view_b,
                  frames_dir_a, frames_dir_b, out_dir, case_label):
    frame = int(row["frame"])
    tid_a = int(row["track_id_a"]) if row["track_id_a"] else None
    tid_b = int(row["track_id_b"]) if row["track_id_b"] else None
    cost = float(row["cost_px"])

    if tid_a is None or tid_b is None:
        return {"frame": frame, "case_label": case_label, "cost_px": cost,
                "diagnosis": "incomplete_row_missing_id", "note": "skipped, no image generated"}

    box_a = by_id.get((view_a, frame, tid_a))
    box_b = by_id.get((view_b, frame, tid_b))

    if box_a is None or box_b is None:
        return {"frame": frame, "case_label": case_label, "cost_px": cost,
                "diagnosis": "coordinate_lookup_failed",
                "note": "could not find box in detections.csv, skipped"}

    img_a = cv2.imread(frame_path(frames_dir_a, frame))
    img_b = cv2.imread(frame_path(frames_dir_b, frame))
    if img_a is None or img_b is None:
        return {"frame": frame, "case_label": case_label, "cost_px": cost,
                "diagnosis": "image_load_failed", "note": "could not load source frames"}

    fx, fy = foot_point(*box_a)
    projected_xy = project_point(H, (fx, fy))

    img_a_draw = draw_box_and_point(img_a.copy(), box_a, (0, 0, 255), f"id{tid_a} (A)")
    img_b_draw = draw_projected_point(img_b.copy(), projected_xy, "projected from A")
    img_b_draw = draw_box_and_point(img_b_draw, box_b, (0, 140, 255),
                                     f"id{tid_b} (B, rejected match)")

    # Check whether a CLOSER person existed in B that Hungarian didn't
    # assign here -- distinguishes "no good option existed" from
    # "a better option existed but went to someone else."
    all_people_b = by_frame.get((view_b, frame), [])
    closest_alt = None
    closest_alt_dist = None
    for other_tid, ox, oy, ow, oh in all_people_b:
        if other_tid == tid_b:
            continue
        other_foot = foot_point(ox, oy, ow, oh)
        dist = float(np.linalg.norm(np.array(projected_xy) - np.array(other_foot)))
        if closest_alt_dist is None or dist < closest_alt_dist:
            closest_alt_dist = dist
            closest_alt = (other_tid, ox, oy, ow, oh)

    diagnosis_notes = []
    if closest_alt is not None and closest_alt_dist < cost:
        other_tid = closest_alt[0]
        img_b_draw = draw_box_and_point(img_b_draw, closest_alt[1:], (255, 0, 255),
                                         f"id{other_tid} (closer, unused)")
        diagnosis_notes.append(
            f"a closer unused detection (id{other_tid}) existed at {closest_alt_dist:.0f}px "
            f"-- worth checking if Hungarian's global optimum took id{other_tid} elsewhere "
            f"this frame, or if it's a genuinely closer wrong person (occlusion/clutter)"
        )
        diagnosis = "possible_ambiguity_closer_option_existed"
    elif cost > 400:
        diagnosis_notes.append(
            "no plausible person near the projected point at all -- "
            "check this frame for a missed YOLO detection in view B, "
            "or a genuine homography/projection failure in this region of the scene"
        )
        diagnosis = "likely_homography_or_detection_gap"
    else:
        diagnosis_notes.append(
            "moderate distance, no closer alternative found -- could be "
            "residual homography imprecision in this part of the scene"
        )
        diagnosis = "moderate_geometry_imprecision"

    combined = stack_side_by_side(img_a_draw, img_b_draw)
    out_path = os.path.join(out_dir, f"{case_label}_frame{frame:04d}_id{tid_a}-{tid_b}.jpg")
    cv2.imwrite(out_path, combined)

    return {"frame": frame, "case_label": case_label, "track_id_a": tid_a,
            "track_id_b": tid_b, "cost_px": round(cost, 1),
            "diagnosis": diagnosis, "note": " | ".join(diagnosis_notes),
            "image": out_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", required=True, help="Path to matched_identities.csv")
    parser.add_argument("--detections", required=True, help="Path to detections.csv")
    parser.add_argument("--view_a", required=True)
    parser.add_argument("--view_b", required=True)
    parser.add_argument("--frames_dir_a", required=True)
    parser.add_argument("--frames_dir_b", required=True)
    parser.add_argument("--homography", default="../calibration/homography_matrix.npy")
    parser.add_argument("--out_dir", default="../outputs/rejected_analysis")
    parser.add_argument("--top_n", type=int, default=15,
                         help="Number of worst-cost rejected cases to visualize")
    parser.add_argument("--sample_midrange", type=int, default=10,
                         help="Number of mid-range-cost rejected cases to sample "
                              "(often more diagnostically useful than the absolute worst)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    H = load_homography(args.homography)
    matched_rows = load_matched_csv(args.matched)
    by_id, by_frame = load_detections_lookup(args.detections)

    rejected_rows = [r for r in matched_rows if r["status"] == "rejected_too_far"]
    print(f"Found {len(rejected_rows)} 'rejected_too_far' rows out of {len(matched_rows)} total.")

    if not rejected_rows:
        print("No rejected_too_far cases found -- nothing to analyze. "
              "Either geometry-only matching is working very well, or "
              "check that matched_identities.csv has this status value at all.")
        return

    worst, midrange = select_cases(rejected_rows, args.top_n, args.sample_midrange)
    print(f"Visualizing {len(worst)} worst-cost cases and {len(midrange)} mid-range cases.")

    summary = []
    for row in worst:
        result = analyze_case(row, H, by_id, by_frame, args.view_a, args.view_b,
                               args.frames_dir_a, args.frames_dir_b, args.out_dir, "worst")
        summary.append(result)
    for row in midrange:
        result = analyze_case(row, H, by_id, by_frame, args.view_a, args.view_b,
                               args.frames_dir_a, args.frames_dir_b, args.out_dir, "midrange")
        summary.append(result)

    summary_path = os.path.join(args.out_dir, "rejected_analysis_summary.csv")
    fieldnames = ["frame", "case_label", "track_id_a", "track_id_b",
                  "cost_px", "diagnosis", "note", "image"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSaved {len(summary)} annotated images to {args.out_dir}/")
    print(f"Summary CSV: {summary_path}")

    diagnosis_counts = defaultdict(int)
    for row in summary:
        diagnosis_counts[row.get("diagnosis", "unknown")] += 1

    print("\nDiagnosis breakdown across visualized cases:")
    for diag, count in sorted(diagnosis_counts.items(), key=lambda x: -x[1]):
        print(f"  {diag}: {count}")

    print(
        "\nHow to read this:\n"
        "  - 'likely_homography_or_detection_gap' cases: open these images first.\n"
        "    If the projected green X is nowhere near any real person, and there's\n"
        "    no missed detection nearby either, that's a real homography weak spot\n"
        "    in that part of the scene.\n"
        "  - 'possible_ambiguity_closer_option_existed' cases: this is your direct\n"
        "    evidence for where ReID-as-tiebreaker would help -- geometry had a\n"
        "    real decision to make between plausible candidates and didn't have\n"
        "    enough signal to get it right alone.\n"
        "  - 'moderate_geometry_imprecision' cases: borderline, worth a second look\n"
        "    but not yet alarming on their own.\n"
        "This is your evidence base -- don't generalize from 1-2 images, look at\n"
        "the diagnosis breakdown and a representative sample before concluding\n"
        "anything about where the architecture needs to change."
    )


if __name__ == "__main__":
    main()
