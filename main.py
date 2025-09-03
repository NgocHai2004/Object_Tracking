from ultralytics import YOLO
import cv2
import time

# Load YOLOv11 model
model = YOLO("yolo11n.pt")

video_path = "36510-411342239_tiny.mp4"
cap = cv2.VideoCapture(video_path)

# Biến đếm
counter = 0
line_position = 200   # toạ độ y của vạch kẻ ngang
offset = 10           # độ lệch cho phép khi vượt qua vạch

# Lưu trạng thái đã đếm của ID
counted_ids = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Dự đoán + tracking (persist=True để giữ ID)
    results = model.track(frame, classes=[0], persist=True, verbose=False)

    # Annotated frame
    annotated = results[0].plot()

    # Vẽ vạch kẻ ngang
    cv2.line(annotated, (0, line_position), (annotated.shape[1], line_position), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # toạ độ bbox
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)   # tâm bbox
            cy = int((y1 + y2) / 2)

            # Vẽ tâm
            cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)

            # Kiểm tra nếu tâm vượt qua vạch
            if (line_position - offset) < cy < (line_position + offset):
                if obj_id not in counted_ids:
                    counter += 1
                    counted_ids.add(obj_id)

    # Hiển thị số lượng đã đếm
    cv2.putText(annotated, f"Count: {counter}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 People Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
