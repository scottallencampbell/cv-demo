from ultralytics import YOLO
import cv2
import numpy as np

def get_box_coordinates(box):
    b = box[0].xyxy.numpy()[0]
    return int(b[0]), int(b[1]), int(b[2]), int(b[3])

def get_iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2
        
    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box
    
    y11, x11, y21, x21 = get_box_coordinates(box1)
    y12, x12, y22, x22 = get_box_coordinates(box2)
    
    yi1 = max(y11, y12)
    xi1 = max(x11, x12)
    yi2 = min(y21, y22)
    xi2 = min(x21, x22)
    inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (x21 - x11) * (y21 - y11)
    box2_area = (x22 - x12) * (y22 - y12)
    union_area = box1_area + box2_area - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou

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
  
model = YOLO("weights/yolov8n-seg.pt")
original = cv2.imread("images/double.jpg")
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray = np.stack((gray,)*3, axis=-1)
h, w, _ = original.shape

results = model.predict(original, conf=.5,classes=0)

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

b1, b1a, b1b, b1c = get_box_coordinates(boxes[0])
b2, b2a, b2b, b2c = get_box_coordinates(boxes[1])

cv2.rectangle(original, (b1, b1a), (b1b, b1c), (255,255,255), 1)
cv2.rectangle(original, (b2, b2a), (b2b, b2c), (255,255,255), 1)

if masks is not None:
    data = masks.data.cpu()
    count = 0
    
    for seg, box in zip(data.data.cpu().numpy(), boxes):   
        print(get_box_coordinates(box))     
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
