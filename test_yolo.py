from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.track(
    source="cam2.mp4",
    show=True,
    save=True,
    persist=True,
    tracker="bytetrack.yaml"
)