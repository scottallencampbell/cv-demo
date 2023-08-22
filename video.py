import math
from ultralytics import YOLO
import cv2
import utils 

def main():
    cap = cv2.VideoCapture("images/bali-small-crop-4.mp4")
   
    model = YOLO("weights/yolov8n.pt")

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                conf = math.ceil(box.conf[0] * 100) / 100
                item = r.names[box.cls[0].item()]

                if conf > .5 and (item == "motorcycle" or item == "car" or item == "person"):
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (150, 0, 0), 3)                    
                    utils.putTextRect(img, f"{item} {conf}", (max(0, x1), max(0, y1 - 10)), 2, 2, (255, 255, 255), (150, 0, 0))
                        
                    
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
        
if __name__ == "__main__":
    main()
   