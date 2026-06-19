"""
Pre-demo diagnostic: checks two things that determine how GlobalID
assignment needs to work.

1. COVERAGE: for each (track_id_a) that appears in 'matched' rows, what
   fraction of its total frame appearances are actually matched (vs
   unmatched/rejected)? Low coverage with gaps means GlobalID will need
   to persist through gap frames, not just frames with a fresh match.

2. STABILITY: for each track_id_a, does it consistently pair with the
   SAME track_id_b across all its matched frames, or does the pairing
   drift (e.g. track 9 matches track 4 for 100 frames, then matches
   track 7 for the next 50)? Drift usually means a ByteTrack ID switch
   happened in one of the views (occlusion, re-entry) -- this is
   important to know about regardless of the GlobalID demo, since it
   affects how trustworthy any single "track_id" is as a stand-in for
   a real person across the WHOLE sequence.

Usage:
    python diagnose_pairing_stability.py --matched ../matched_identities.csv
"""

import argparse
import csv
from collections import defaultdict, Counter


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", required=True)
    args = parser.parse_args()

    rows = load_rows(args.matched)

    # All frames where track_id_a appears at all (matched OR rejected),
    # vs frames where it specifically got a clean 'matched' status.
    all_appearances_a = defaultdict(set)
    matched_appearances_a = defaultdict(set)
    pairing_counter = defaultdict(Counter)  # track_id_a -> Counter({track_id_b: count})

    for row in rows:
        if not row["track_id_a"]:
            continue
        tid_a = int(row["track_id_a"])
        frame = int(row["frame"])
        all_appearances_a[tid_a].add(frame)

        if row["status"] == "matched" and row["track_id_b"]:
            tid_b = int(row["track_id_b"])
            matched_appearances_a[tid_a].add(frame)
            pairing_counter[tid_a][tid_b] += 1

    print(f"Total distinct track_id_a values seen: {len(all_appearances_a)}\n")
    print(f"{'track_id_a':<12}{'total_frames':<15}{'matched_frames':<16}"
          f"{'coverage_%':<12}{'dominant_b':<12}{'b_purity_%':<12}{'distinct_b_ids'}")
    print("-" * 95)

    unstable_tracks = []
    low_coverage_tracks = []

    for tid_a in sorted(all_appearances_a.keys()):
        total = len(all_appearances_a[tid_a])
        matched = len(matched_appearances_a[tid_a])
        coverage_pct = 100.0 * matched / total if total else 0.0

        b_counts = pairing_counter[tid_a]
        if b_counts:
            dominant_b, dominant_count = b_counts.most_common(1)[0]
            total_b_matches = sum(b_counts.values())
            purity_pct = 100.0 * dominant_count / total_b_matches
            distinct_b = len(b_counts)
        else:
            dominant_b, purity_pct, distinct_b = "none", 0.0, 0

        print(f"{tid_a:<12}{total:<15}{matched:<16}{coverage_pct:<12.1f}"
              f"{str(dominant_b):<12}{purity_pct:<12.1f}{distinct_b}")

        if distinct_b > 1 and purity_pct < 90:
            unstable_tracks.append((tid_a, distinct_b, purity_pct))
        if coverage_pct < 50:
            low_coverage_tracks.append((tid_a, coverage_pct))

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    if unstable_tracks:
        print(f"\n{len(unstable_tracks)} track(s) have UNSTABLE pairing "
              f"(matched to more than one track_id_b, <90% purity):")
        for tid_a, distinct_b, purity in unstable_tracks:
            print(f"  track_id_a={tid_a}: matched {distinct_b} different "
                  f"track_id_b values, dominant pairing only {purity:.1f}% pure")
        print(
            "\nThis means a simple 'track_id_a N always equals track_id_b M' "
            "lookup table will be WRONG for these tracks. GlobalID assignment "
            "needs to either (a) pick the dominant pairing and accept some "
            "frames will show no/wrong match, or (b) investigate why the "
            "pairing drifts -- likely a ByteTrack ID switch in one view "
            "(person occluded then reappeared with a new ID) rather than a "
            "geometry/ReID problem."
        )
    else:
        print("\nAll matched track_id_a values pair with a SINGLE consistent "
              "track_id_b. A simple static lookup table is sufficient -- "
              "no need for frame-by-frame re-resolution of GlobalID.")

    if low_coverage_tracks:
        print(f"\n{len(low_coverage_tracks)} track(s) have LOW coverage "
              f"(matched in <50% of their total frame appearances):")
        for tid_a, cov in low_coverage_tracks:
            print(f"  track_id_a={tid_a}: only {cov:.1f}% of frames matched")
        print(
            "\nFor the demo video, this means GlobalID will need to PERSIST "
            "through gap frames (carry the last known match forward) rather "
            "than disappearing every time a frame lacks a fresh 'matched' row -- "
            "otherwise the ID label will flicker on/off distractingly."
        )
    else:
        print("\nAll tracks have reasonable match coverage across their "
              "frame appearances.")


if __name__ == "__main__":
    main()
