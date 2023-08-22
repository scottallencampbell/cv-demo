from ultralytics import YOLO
import cv2

#model = YOLO("weights/yolov8n.pt")
model = YOLO("weights/yolov8l.pt")
results = model("images/taste.jpg", show=True)

print(results)

cv2.waitKey()