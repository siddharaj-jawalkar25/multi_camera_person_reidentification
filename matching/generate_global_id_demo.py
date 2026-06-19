"""
Generates the final demo: a side-by-side video of View_001 and View_005
with GlobalID labels drawn on each tracked person, using the mapping
from build_global_ids.py.

Handles ID PERSISTENCE: if a track has low match coverage (matched in
only some frames, per diagnose_pairing_stability.py), this draws the
GlobalID on every frame where the underlying track_id appears at all
(from detections.csv), not just frames with a fresh 'matched' row --
otherwise the label flickers on/off every time geometry momentarily
fails to confirm the match, which looks broken even when the underlying
identity is actually still correct and persistent.

This persistence is BOUNDED, not unlimited: each GlobalID's mapping
carries a first_frame/last_frame range (from build_global_ids.py,
covering only the dominant pairing's actually-confirmed span, plus a
small --persistence_buffer on each side). Outside that range, even if
the raw track_id is still technically alive in detections.csv, the
label reverts to the neutral gray track_id display -- because showing
"P1" 300 frames past the last confirmed cross-camera match would
overstate confidence the data doesn't support, especially given known
tracking fragmentation in this dataset.

People with NO assigned GlobalID (excluded for low pairing purity, or
simply never matched) are still drawn with their raw track_id in a
neutral color, so the video doesn't look like detection is failing --
it's just clear which people have a confirmed cross-camera identity link.

Usage:
    python generate_global_id_demo.py \
        --detections ../detections.csv \
        --mapping ../outputs/global_id_mapping.json \
        --frames_dir_a ../View_001 --frames_dir_b ../View_005 \
        --view_a View_001 --view_b View_005 \
        --out ../outputs/global_id_demo.mp4 \
        --fps 7 \
        --start_frame 0 --end_frame 400 \
        --persistence_buffer 5
"""

import argparse
import csv
import json
from collections import defaultdict

import cv2
import numpy as np


# Distinct colors per GlobalID so the same person is visually consistent
# across both camera views in the demo -- cycles if more people than colors.
GLOBAL_ID_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]
UNMATCHED_COLOR = (160, 160, 160)  # neutral gray for people with no GlobalID


def load_detections(path):
    """Returns dict: (view, frame) -> [(track_id, x, y, w, h), ...]"""
    by_frame = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            view = row["view"]
            frame = int(row["frame"])
            entry = (int(row["track_id"]), float(row["x"]), float(row["y"]),
                      float(row["w"]), float(row["h"]))
            by_frame[(view, frame)].append(entry)
    return by_frame


def color_for_global_id(global_id):
    if global_id is None:
        return UNMATCHED_COLOR
    idx = int(global_id.lstrip("P")) % len(GLOBAL_ID_COLORS)
    return GLOBAL_ID_COLORS[idx]


def draw_person(img, box, label, color):
    x, y, w, h = box
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Label background for readability against busy frame content
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def frame_path(frames_dir, frame_num, pattern="frame_{:04d}.jpg"):
    import os
    return os.path.join(frames_dir, pattern.format(frame_num))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--mapping", required=True, help="global_id_mapping.json")
    parser.add_argument("--frames_dir_a", required=True)
    parser.add_argument("--frames_dir_b", required=True)
    parser.add_argument("--view_a", required=True)
    parser.add_argument("--view_b", required=True)
    parser.add_argument("--out", default="../outputs/global_id_demo.mp4")
    parser.add_argument("--fps", type=float, default=7.0,
                         help="PETS2009 source footage is ~7fps -- match it "
                              "so playback speed looks natural, not sped up/slowed down.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None,
                         help="If omitted, runs to the last frame found in detections.csv")
    parser.add_argument("--persistence_buffer", type=int, default=5,
                         help="Frames of slack added before first_frame and after "
                              "last_frame of a GlobalID's confirmed range, so the "
                              "label doesn't blink off one frame before/after the "
                              "literal last confirmed match. Keep small (a handful "
                              "of frames) -- this is cosmetic smoothing, not license "
                              "to extend the claimed identity span.")
    args = parser.parse_args()

    with open(args.mapping) as f:
        mapping = json.load(f)

    view_a_to_global = mapping["view_a_to_global"]
    view_b_to_global = mapping["view_b_to_global"]
    global_id_info = mapping["global_id_info"]

    # Per-GlobalID valid frame range (bounded persistence)
    valid_range = {
        gid: (info["first_frame"] - args.persistence_buffer,
              info["last_frame"] + args.persistence_buffer)
        for gid, info in global_id_info.items()
    }

    def resolve_label_and_color(track_id, view_to_global, frame_num):
        global_id = view_to_global.get(str(track_id))
        if global_id is None:
            return f"id{track_id}", UNMATCHED_COLOR
        lo, hi = valid_range[global_id]
        if lo <= frame_num <= hi:
            return global_id, color_for_global_id(global_id)
        # Outside the confirmed-pairing span -- don't claim the link here
        return f"id{track_id}", UNMATCHED_COLOR

    detections = load_detections(args.detections)

    all_frames = [f for (v, f) in detections.keys()]
    end_frame = args.end_frame if args.end_frame is not None else max(all_frames)

    print(f"Rendering frames {args.start_frame} to {end_frame}...")

    writer = None

    for frame_num in range(args.start_frame, end_frame + 1):
        img_a = cv2.imread(frame_path(args.frames_dir_a, frame_num))
        img_b = cv2.imread(frame_path(args.frames_dir_b, frame_num))

        if img_a is None or img_b is None:
            print(f"  Skipping frame {frame_num}: could not load source image(s)")
            continue

        people_a = detections.get((args.view_a, frame_num), [])
        people_b = detections.get((args.view_b, frame_num), [])

        for tid, x, y, w, h in people_a:
            label, color = resolve_label_and_color(tid, view_a_to_global, frame_num)
            img_a = draw_person(img_a, (x, y, w, h), label, color)

        for tid, x, y, w, h in people_b:
            label, color = resolve_label_and_color(tid, view_b_to_global, frame_num)
            img_b = draw_person(img_b, (x, y, w, h), label, color)

        # Stack side by side, equal heights
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        target_h = max(h_a, h_b)
        if h_a != target_h:
            img_a = cv2.resize(img_a, (int(w_a * target_h / h_a), target_h))
        if h_b != target_h:
            img_b = cv2.resize(img_b, (int(w_b * target_h / h_b), target_h))

        # Frame number burned in for easy reference when reviewing the demo
        combined = np.hstack([img_a, img_b])
        cv2.putText(combined, f"frame {frame_num}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer is None:
            h_out, w_out = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w_out, h_out))

        writer.write(combined)

    if writer is not None:
        writer.release()
        print(f"\nSaved demo video to {args.out}")
    else:
        print("\nNo frames were rendered -- check your frame range and paths.")


if __name__ == "__main__":
    main()
