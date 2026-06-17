"""
Shared configuration for the whole project.
Both Person A and Person B import from this file — don't hardcode
thresholds/paths in your own scripts, change them here so everyone stays in sync.
"""

# ---- Paths ----
VIDEO_DIR = "data/videos"          # raw input videos, e.g. cam1.mp4, cam2.mp4
CROPS_DIR = "data/crops"           # cropped person images per track, written by detection/
EMBEDDINGS_DIR = "data/embeddings" # cached embeddings if needed
ANNOTATED_DIR = "data/annotated"   # output videos with boxes + IDs drawn

# ---- Cameras ----
# List every camera/video you're using. cam_id must match what you pass
# to detect_and_track.py --cam_id
CAMERAS = [
    {"cam_id": "cam1", "source": f"{VIDEO_DIR}/cam1.mp4"},
    {"cam_id": "cam2", "source": f"{VIDEO_DIR}/cam2.mp4"},
]

# ---- Detection / Tracking ----
YOLO_MODEL = "yolov8n.pt"   # nano = fastest, fine for person-only detection
TRACKER_CONFIG = "bytetrack.yaml"
DETECT_CLASSES = [0]        # 0 = person in COCO

# ---- ReID ----
REID_MODEL_NAME = "osnet_x1_0"   # pretrained on Market-1501 via torchreid
EMBEDDING_DIM = 512
TEMPORAL_WINDOW = 15             # number of recent frames to average per track

# ---- Matching ----
SIMILARITY_THRESHOLD = 0.85      # cosine similarity above this = same person
                                  # NOTE: tune this empirically on your own footage on Saturday
AMBIGUOUS_LOW = 0.45              # below this = confidently different person
AMBIGUOUS_HIGH = 0.65              # between LOW and HIGH = uncertain, log for review
TTL_SECONDS = 10                  # global ID released if unseen for this long

# ---- Homography / common coordinates (stretch goal) ----
USE_HOMOGRAPHY = False            # flip to True once calibrated, see calibration/
MAX_WALK_SPEED_MPS = 2.0          # used for cross-camera plausibility checks
