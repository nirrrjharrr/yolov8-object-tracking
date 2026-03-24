import cv2
from ultralytics import YOLO
 
# Config 
 
VIDEO_PATH     = "videos/people_walking.mp4"
MODEL_PATH     = "models/yolov8n.pt"
WINDOW_NAME    = "Object Detection + Tracking"
CONF_THRESHOLD = 0.3
 
BOX_COLOR  = (0, 255,   0)
TEXT_COLOR = (0, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS  = 2
 
# Drawing 
 
def draw_detection(frame, box: list[int], label: str, track_id: int) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
    cv2.putText(
        frame,
        f"{label}  ID {track_id}",
        (max(0, x1), max(20, y1 - 10)),
        FONT, FONT_SCALE, TEXT_COLOR, THICKNESS,
    )
 
# Core loop 
 
def process_frame(frame, result, model: YOLO) -> None:
    """Draw all tracked detections for a single result."""
    if result.boxes.id is None:
        return
 
    boxes   = result.boxes.xyxy.cpu().numpy()
    ids     = result.boxes.id.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs   = result.boxes.conf.cpu().numpy()
 
    for box, track_id, cls, conf in zip(boxes, ids, classes, confs):
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        draw_detection(frame, [x1, y1, x2, y2], model.names[cls], track_id)
 
 
def run(model: YOLO, cap: cv2.VideoCapture) -> None:
    while True:
        success, frame = cap.read()
        if not success:
            print("End of video.")
            break
 
        results = model.track(frame, conf=CONF_THRESHOLD, persist=True, verbose=False)
        for result in results:
            process_frame(frame, result, model)
 
        cv2.imshow(WINDOW_NAME, frame)
 
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC — exiting.")
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed — exiting.")
            break
 
# Entry point 
 
def main() -> None:
    print("Loading model…")
    model = YOLO(MODEL_PATH)
    # model = YOLO(MODEL_PATH).to("cuda")   # ← uncomment for GPU
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {VIDEO_PATH}")
 
    try:
        run(model, cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")
 
 
if __name__ == "__main__":
    main()
