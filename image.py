from ultralytics import YOLO
import cv2
import numpy as np

def overlay(image, mask, color, alpha, resize=None): 
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined
            
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
  
model = YOLO("weights/yolov8l-seg.pt")
original = cv2.imread("images/breck.jpg")
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray = np.stack((gray,)*3, axis=-1)
h, w, _ = original.shape

results = model.predict(original, conf=.5,classes=0)

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

if masks is not None:
    data = masks.data.cpu()
    count = 0
    
    for seg, box in zip(data.data.cpu().numpy(), boxes):        
        if int(box.cls) == 0:
            seg = cv2.resize(seg, (w, h))
            gray = overlay(gray, seg, colors[count % len(colors)], 0.2)
            count = count + 1
   
for mask in masks:
    for coord in masks.xy:
        ctr = np.array(coord).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(gray, [ctr], -1, (255,255,255), 2)
    
cv2.imshow('img', original)
cv2.imshow('gray', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
