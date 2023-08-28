import math
from ultralytics import YOLO
import cv2
import numpy as np 

classes = [0,2,39,41,45]
colors = [(0,0,255), (255,0,0), (0,255,0), (255,255,0), (0,255,255), (255,0,255)]
    
def add_overlay(img, mask, color, alpha, resize=None): 
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(img, mask=colored_mask, fill_value=color)
    overlay = masked.filled()

    if resize is not None:
        img = cv2.resize(img.transpose(1, 2, 0), resize)
        overlay = cv2.resize(overlay.transpose(1, 2, 0), resize)

    combined = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    return combined

def draw_contours(img, masks):
    for coord in masks.xy:
        ctr = np.array(coord).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [ctr], -1, (255,255,255), 2)            
                
def get_center_of_mass(seg):
    M = cv2.moments(seg)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    return [cX, cY]

def add_centered_text(img, seg, text):
    center_of_mass = get_center_of_mass(seg)
    text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
    cv2.putText(img, text, (center_of_mass[0]-int(text_width/2),center_of_mass[1]-int(text_height/2)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
def decorate(img, results, names):
    
    for r in results:
        boxes = r.boxes
        masks = r.masks  
        probs = r.probs  

    if masks is None:
        return img
    
    data = masks.data.cpu()
    
    for seg, box in zip(data.data.cpu().numpy(), boxes):    
        h, w, _ = img.shape
        seg = cv2.resize(seg, (w, h))
        index = classes.index(int(box.cls))
        color = colors[index % len(classes)]
        
        img = add_overlay(img, seg, color, 0.2)
        add_centered_text(img, seg, f'{names[int(box.cls)]} {int(box.conf*100)}%')
        
    draw_contours(img, masks)
    
    return img
        
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    model = YOLO("weights/yolov8n-seg.pt")
    
    while True:
        success, img = cap.read()
    
        if success == False:
            exit()
            
        results = model(img, stream=True, conf=.6, classes=classes)
        decorated = decorate(img, results, model.names)
        
        cv2.imshow("Image", decorated)
        cv2.waitKey(1)
                
if __name__ == "__main__":
    main()
   