"""
Verify that a converted video's frame count matches its original source
frame folder exactly. Run this BEFORE exporting tracks to CSV -- if frame
counts don't match, frame numbers in your CSV won't correspond to the
correct original frame_XXXX.jpg files, and your cross-view matching will
silently compare the wrong frames against each other.

Usage:
    python verify_frame_alignment.py \
        --video cam1.mp4 \
        --frames_dir path/to/View_001
"""

import argparse
import os
import cv2


def count_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return count, fps


def count_folder_frames(frames_dir):
    files = [f for f in os.listdir(frames_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return len(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames_dir", required=True)
    args = parser.parse_args()

    video_count, fps = count_video_frames(args.video)
    folder_count = count_folder_frames(args.frames_dir)

    print(f"Video:  {args.video}")
    print(f"  Frame count: {video_count}")
    print(f"  FPS:         {fps}")
    print(f"Folder: {args.frames_dir}")
    print(f"  Frame count: {folder_count}")
    print()

    if video_count == folder_count:
        print(f"MATCH: both have {video_count} frames. "
              "Frame index N in the video should correspond to the Nth "
              "frame file in the folder (still confirm the ORDERING matches "
              "-- see note below).")
    else:
        diff = folder_count - video_count
        print(f"MISMATCH: folder has {folder_count} frames, video has "
              f"{video_count} frames (difference of {diff}).")
        print("This means frame numbers will NOT line up correctly between "
              "your CSV (built from the video) and the original frame_XXXX.jpg "
              "files (needed by the homography/projection scripts).")
        print("\nLikely causes:")
        print("  - ffmpeg dropped/duplicated frames during conversion "
              "(common if you didn't pin an exact frame rate)")
        print("  - the conversion script used a different frame range "
              "than the full folder")
        print("\nRecommended fix: re-convert using a command that preserves "
              "exact 1:1 frame correspondence, e.g. with ffmpeg:")
        print(f'  ffmpeg -i "{args.frames_dir}/frame_%04d.jpg" -vsync 0 '
              f'-frame_pts 0 "{args.video}"')
        print("(-vsync 0 prevents ffmpeg from dropping or duplicating frames "
              "to hit a target frame rate)")

    print("\nIMPORTANT: matching counts alone doesn't guarantee correct "
          "ORDER. Spot-check by extracting frame 0 and frame -1 (last) "
          "from the video and visually comparing them against "
          "frame_0000.jpg and the last file in the folder.")


if __name__ == "__main__":
    main()
