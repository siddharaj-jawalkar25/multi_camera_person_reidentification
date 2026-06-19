"""
Validation check: does OSNet similarity actually separate correct
cross-view matches from incorrect/ambiguous ones for THIS camera pair?

This is the sanity check to run BEFORE building any fused matching
pipeline. If 'matched' pairs reliably score higher similarity than
'rejected ambiguity' pairs, the appearance signal is real and worth
fusing in. If the two distributions overlap heavily or are inverted,
that's critical information now -- it means OSNet embeddings aren't
separable for your specific elevated-vs-ground-level viewpoint gap,
and the ReID tiebreaker plan needs reconsidering (e.g. trying a
different model) before writing fusion logic around a signal that
turns out to be noise.

Inputs:
    - matched_identities.csv (output of hungarian_match.py)
    - detections.csv
    - View_001 / View_005 frame folders

What it does:
    1. Pulls 5 'matched' rows (good geometry agreement) and 5
       'rejected_too_far' rows (ambiguous/failed cases) from
       matched_identities.csv.
    2. Selection is NOT random -- it deliberately picks matched rows
       with LOW cost (clean, confident matches) and rejected rows
       that were flagged as ambiguity-type in your manual inspection
       (moderate cost, not the absurd outliers), since those are the
       most informative comparison: "clean geometric match" vs
       "genuinely confusing case" rather than vs "obviously broken".
    3. Computes OSNet cosine similarity for all 10 pairs.
    4. Prints similarity stats for each group, and whether the two
       distributions are actually separable.

Usage:
    python validate_reid_signal.py \
        --matched ../matched_identities.csv \
        --detections ../detections.csv \
        --frames_dir_a ../View_001 --frames_dir_b ../View_005 \
        --view_a View_001 --view_b View_005 \
        --max_rejected_cost 400

    --max_rejected_cost excludes the most extreme rejected outliers
    (e.g. your earlier 4946px case) from the "ambiguity" sample, since
    those are likely missed detections or empty frames, not genuine
    appearance-distinguishable ambiguity cases. We want the comparison
    to be fair -- testing whether ReID helps on cases where geometry
    was CLOSE but wrong, not cases where geometry had nothing to work
    with at all.
"""

import argparse
import csv
import random

import numpy as np

from reid_embedder import ReIDEmbedder


def load_matched_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def select_matched_sample(rows, n=5, seed=42):
    """Picks the N lowest-cost 'matched' rows -- the cleanest, most
    confident geometric matches, used as the 'should be visually similar'
    reference group."""
    matched = [r for r in rows if r["status"] == "matched" and r["cost_px"]]
    matched_sorted = sorted(matched, key=lambda r: float(r["cost_px"]))
    return matched_sorted[:n]


def select_rejected_sample(rows, n=5, max_cost=400, seed=42):
    """
    Picks N 'rejected_too_far' rows under max_cost, sampled across the
    cost range rather than just the lowest, since the point is to
    capture genuine ambiguity cases (moderate cost) rather than
    accidentally re-selecting near-matches that would have passed
    under a slightly looser threshold.
    """
    rejected = [r for r in rows if r["status"] == "rejected_too_far"
                and r["cost_px"] and float(r["cost_px"]) <= max_cost]
    if not rejected:
        return []

    rejected_sorted = sorted(rejected, key=lambda r: float(r["cost_px"]))
    if len(rejected_sorted) <= n:
        return rejected_sorted

    # Spread the sample across the cost range rather than clustering
    rng = random.Random(seed)
    step = len(rejected_sorted) / n
    indices = [int(i * step) for i in range(n)]
    return [rejected_sorted[i] for i in indices]


def run_group(embedder, rows, label, detections_csv, frames_dir_a, frames_dir_b,
              view_a, view_b):
    print(f"\n--- {label} ({len(rows)} pairs) ---")
    similarities = []

    for row in rows:
        frame = int(row["frame"])
        tid_a = int(row["track_id_a"])
        tid_b = int(row["track_id_b"])
        cost_px = float(row["cost_px"])

        sim = embedder.compare(
            detections_csv=detections_csv,
            frames_dir_a=frames_dir_a, frames_dir_b=frames_dir_b,
            view_a=view_a, view_b=view_b,
            frame=frame, track_id_a=tid_a, track_id_b=tid_b,
        )

        if sim is not None:
            print(f"  frame {frame:4d} | id_a={tid_a:3d} id_b={tid_b:3d} | "
                  f"geometry_cost={cost_px:6.1f}px | cosine_sim={sim:.4f}")
            similarities.append(sim)
        else:
            print(f"  frame {frame:4d} | id_a={tid_a:3d} id_b={tid_b:3d} | "
                  f"geometry_cost={cost_px:6.1f}px | EMBEDDING FAILED, skipped")

    return similarities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--frames_dir_a", required=True)
    parser.add_argument("--frames_dir_b", required=True)
    parser.add_argument("--view_a", required=True)
    parser.add_argument("--view_b", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_rejected_cost", type=float, default=400.0)
    parser.add_argument("--model_name", default="osnet_x1_0")
    args = parser.parse_args()

    rows = load_matched_rows(args.matched)

    matched_sample = select_matched_sample(rows, n=args.n)
    rejected_sample = select_rejected_sample(rows, n=args.n, max_cost=args.max_rejected_cost)

    if len(matched_sample) < args.n:
        print(f"WARNING: only found {len(matched_sample)} matched rows, wanted {args.n}.")
    if len(rejected_sample) < args.n:
        print(f"WARNING: only found {len(rejected_sample)} rejected rows under "
              f"{args.max_rejected_cost}px, wanted {args.n}. Try raising --max_rejected_cost.")

    print("Loading OSNet (first run will download pretrained weights)...")
    embedder = ReIDEmbedder(model_name=args.model_name)

    matched_sims = run_group(embedder, matched_sample, "MATCHED (clean geometry, expect HIGH similarity)",
                              args.detections, args.frames_dir_a, args.frames_dir_b,
                              args.view_a, args.view_b)

    rejected_sims = run_group(embedder, rejected_sample, "REJECTED AMBIGUITY (expect LOWER or MIXED similarity)",
                               args.detections, args.frames_dir_a, args.frames_dir_b,
                               args.view_a, args.view_b)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if matched_sims:
        print(f"\nMatched group (n={len(matched_sims)}):")
        print(f"  Mean similarity:   {np.mean(matched_sims):.4f}")
        print(f"  Median similarity: {np.median(matched_sims):.4f}")
        print(f"  Min / Max:         {np.min(matched_sims):.4f} / {np.max(matched_sims):.4f}")
    else:
        print("\nMatched group: no valid similarities computed.")

    if rejected_sims:
        print(f"\nRejected-ambiguity group (n={len(rejected_sims)}):")
        print(f"  Mean similarity:   {np.mean(rejected_sims):.4f}")
        print(f"  Median similarity: {np.median(rejected_sims):.4f}")
        print(f"  Min / Max:         {np.min(rejected_sims):.4f} / {np.max(rejected_sims):.4f}")
    else:
        print("\nRejected-ambiguity group: no valid similarities computed.")

    print()
    if matched_sims and rejected_sims:
        gap = np.mean(matched_sims) - np.mean(rejected_sims)
        print(f"Mean gap (matched - rejected): {gap:+.4f}")

        if gap > 0.1:
            print(
                "\nINTERPRETATION: matched pairs score meaningfully higher than "
                "rejected-ambiguity pairs. This is a real, usable separation -- "
                "OSNet similarity is a reasonable signal to fuse in as a tiebreaker "
                "for this camera pair. Proceed to hungarian_match_with_reid.py."
            )
        elif gap > 0.0:
            print(
                "\nINTERPRETATION: matched pairs score slightly higher on average, "
                "but the gap is small. The signal exists but is weak -- worth "
                "looking at the per-pair printout above to see if a few outliers "
                "are driving the difference, or if it's a consistent small effect. "
                "Still usable as a low-weight tiebreaker, but don't expect it to "
                "resolve hard cases on its own."
            )
        else:
            print(
                "\nINTERPRETATION: no separation, or rejected pairs score HIGHER "
                "than matched pairs on average. This is a real result, not a bug "
                "to explain away -- it suggests OSNet embeddings are not reliably "
                "separating people for this elevated-vs-ground-level viewpoint gap. "
                "Before building fusion logic around this signal, consider: "
                "(a) checking crop quality (are elevated-view crops too small/blurry?), "
                "(b) trying a different model (TransReID, CLIP), or "
                "(c) accepting that geometry alone may need to carry this POC, "
                "with ReID deferred rather than forced in."
            )
    else:
        print("Cannot compare groups -- one or both groups produced no valid "
              "similarities. Check the per-pair output above for errors "
              "(missing detections, bad crops, etc).")


if __name__ == "__main__":
    main()
