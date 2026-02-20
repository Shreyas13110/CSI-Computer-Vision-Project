import cv2
import time
from detector import YOLODetector
from utils import draw_custom_boxes

detector = YOLODetector()

cap = cv2.VideoCapture("videos/detection_video.mp4")

if not cap.isOpened():
    print("Camera not working")
    exit()

prev_time = 0
allowed_classes = None  

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (640, 360))
    
    detections = detector.get_results(frame)

    
    frame = draw_custom_boxes(frame, detections, allowed_classes)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    print(f"FPS: {fps:.2f} | Detections: {detections}")

    

    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
