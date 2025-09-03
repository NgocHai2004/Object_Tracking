import cv2
import numpy as np
from ultralytics import YOLO
import time

polygons = [
        np.array([[335, 162], [234, 273], [494, 267]]) 
]

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)   
cv2.resizeWindow("Video", 1280, 720)         
video_path = "6387-191695740_tiny.mp4"
model = YOLO("yolo11n.pt")   
cap = cv2.VideoCapture(video_path)


entry_count = 0                
unique_ids_seen = set()       
last_inside = dict()           
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0], persist=True, verbose=False) 
    annotated = results[0].plot()  

    poly = polygons[0].astype(np.int32)

    cv2.polylines(annotated, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

    try:
        boxes = results[0].boxes.xyxy.cpu().numpy() 
    except Exception:
        boxes = np.array([])

    try:
        ids = results[0].boxes.id.int().cpu().tolist() 
    except Exception:
        ids = list(range(len(boxes)))

    present_ids = set()    
    current_inside_ids = set() 

    for box, obj_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        present_ids.add(obj_id)

        cv2.circle(annotated, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(annotated, f"ID:{obj_id}", (x1, y1 - 8), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        inside = cv2.pointPolygonTest(poly, (cx, cy), False) >= 0  

        prev = last_inside.get(obj_id, False)
        if inside and not prev:
            entry_count += 1
            unique_ids_seen.add(obj_id)  

        last_inside[obj_id] = inside

        if inside:
            current_inside_ids.add(obj_id)

    for known_id in list(last_inside.keys()):
        if known_id not in present_ids:
            last_inside[known_id] = False

    current_inside_count = len(current_inside_ids)
    unique_seen_count = len(unique_ids_seen)

    cv2.putText(annotated, f"Entries (transitions): {entry_count}", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated, f"Unique IDs entered: {unique_seen_count}", (10, 60), font, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Currently inside: {current_inside_count}", (10, 90), font, 0.7, (0, 255, 0), 2)

    cv2.imshow("People Counter (Polygon ROI)", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
