"""
Builds a GlobalID mapping from matched_identities.csv.

For each track_id_a, finds its DOMINANT track_id_b pairing (the one it
matched most often) and assigns both a single shared GlobalID. This
handles both clean cases (track always pairs with the same partner) and
drifting cases (track occasionally mismatches) by trusting the majority
vote rather than the first or last row seen.

Output: a JSON mapping usable by the video annotation script:
{
  "view_a_to_global": {"9": "P1", "12": "P2", ...},
  "view_b_to_global": {"4": "P1", "7": "P2", ...},
  "global_id_info": {
      "P1": {"track_id_a": 9, "track_id_b": 4, "purity_pct": 100.0,
             "matched_frame_count": 210, "first_frame": 5, "last_frame": 312}
  }
}

Usage:
    python build_global_ids.py \
        --matched ../matched_identities.csv \
        --out ../outputs/global_id_mapping.json \
        --min_purity 70
"""

import argparse
import csv
import json
from collections import defaultdict, Counter


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", required=True)
    parser.add_argument("--out", default="../outputs/global_id_mapping.json")
    parser.add_argument("--min_purity", type=float, default=70.0,
                         help="Minimum %% of matched frames that must agree on the "
                              "dominant track_id_b pairing for a track_id_a to be "
                              "assigned a GlobalID at all. Tracks below this are "
                              "excluded -- their pairing is too inconsistent to trust.")
    parser.add_argument("--min_coverage", type=int, default=10,
                         help="Minimum number of frames the DOMINANT pairing must "
                              "have been matched on. Without this, a track matched "
                              "only 3 times (2 dominant + 1 other = 66%% purity) could "
                              "pass a purity check on pure chance -- coverage protects "
                              "against small-sample false confidence, purity protects "
                              "against genuine drift. Both are needed.")
    args = parser.parse_args()

    rows = load_rows(args.matched)

    pairing_counter = defaultdict(Counter)
    frames_per_pair = defaultdict(list)  # (tid_a, tid_b) -> [frame, frame, ...]

    for row in rows:
        if row["status"] != "matched" or not row["track_id_a"] or not row["track_id_b"]:
            continue
        tid_a = int(row["track_id_a"])
        tid_b = int(row["track_id_b"])
        frame = int(row["frame"])
        pairing_counter[tid_a][tid_b] += 1
        frames_per_pair[(tid_a, tid_b)].append(frame)

    view_a_to_global = {}
    view_b_to_global = {}
    global_id_info = {}

    excluded = []
    global_counter = 1

    # Sort by first-appearance frame so GlobalIDs are assigned in a
    # sensible, reproducible order rather than arbitrary dict order.
    def first_frame_for(tid_a):
        all_frames = [f for (a, b), fs in frames_per_pair.items() if a == tid_a for f in fs]
        return min(all_frames) if all_frames else float("inf")

    for tid_a in sorted(pairing_counter.keys(), key=first_frame_for):
        b_counts = pairing_counter[tid_a]
        dominant_b, dominant_count = b_counts.most_common(1)[0]
        total = sum(b_counts.values())
        purity_pct = 100.0 * dominant_count / total

        fails_purity = purity_pct < args.min_purity
        fails_coverage = dominant_count < args.min_coverage

        if fails_purity or fails_coverage:
            reason = []
            if fails_purity:
                reason.append(f"purity {purity_pct:.1f}% < {args.min_purity}%")
            if fails_coverage:
                reason.append(f"coverage {dominant_count} < {args.min_coverage} frames")
            excluded.append((tid_a, dominant_b, purity_pct, total, ", ".join(reason)))
            continue

        global_id = f"P{global_counter}"
        global_counter += 1

        matched_frames = frames_per_pair[(tid_a, dominant_b)]

        view_a_to_global[str(tid_a)] = global_id
        view_b_to_global[str(dominant_b)] = global_id
        global_id_info[global_id] = {
            "track_id_a": tid_a,
            "track_id_b": dominant_b,
            "purity_pct": round(purity_pct, 1),
            "matched_frame_count": dominant_count,
            "first_frame": min(matched_frames),
            "last_frame": max(matched_frames),
        }

    output = {
        "view_a_to_global": view_a_to_global,
        "view_b_to_global": view_b_to_global,
        "global_id_info": global_id_info,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Assigned {len(global_id_info)} GlobalIDs.")
    for gid, info in global_id_info.items():
        print(f"  {gid}: track_id_a={info['track_id_a']} <-> track_id_b={info['track_id_b']} "
              f"(purity={info['purity_pct']}%, frames {info['first_frame']}-{info['last_frame']}, "
              f"{info['matched_frame_count']} matched frames)")

    if excluded:
        print(f"\n{len(excluded)} track_id_a value(s) excluded (failed purity "
              f"and/or coverage threshold):")
        for tid_a, dom_b, purity, total, reason in excluded:
            print(f"  track_id_a={tid_a}: best guess track_id_b={dom_b} "
                  f"({purity:.1f}% pure over {total} matched frames) -- {reason}")
        print("These will show NO GlobalID label in the demo video rather than "
              "risk showing a wrong/flickering one. Investigate separately if "
              "you want to recover them (likely a ByteTrack ID switch case).")

    print(f"\nSaved mapping to {args.out}")


if __name__ == "__main__":
    main()
