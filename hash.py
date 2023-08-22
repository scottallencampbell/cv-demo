import math
import cv2
import numpy as np

image_path = 'images/hash-7c.png'

def pre_process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    thin = cv2.ximgproc.thinning(thresh)
    return [gray, thin]
    
def get_interior_box(thin, gray): 
    contours, _= cv2.findContours(thin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    c = 0

    for i in contours:
        area = cv2.contourArea(i)
        
        if area > 1000 and area > max_area:
            max_area = area
            best_cnt = i

    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)

    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]

    out_blur = cv2.GaussianBlur(out, (5,5), 0)
    out_thresh = cv2.adaptiveThreshold(out_blur, 255, 1, 1, 11, 2)

    contours, _ = cv2.findContours(out_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = np.intp(cv2.boxPoints(rect))
    return box

def get_end_points(img):
    end_points = []  
    points = cv2.findNonZero(img)
    points = np.squeeze(points)  
    
    for p in points:
        x = p[0]
        y = p[1]
        n = 0        
        n += img[y - 1, x]
        n += img[y - 1, x - 1]
        n += img[y - 1, x + 1]
        n += img[y, x - 1]    
        n += img[y, x + 1]    
        n += img[y + 1, x]    
        n += img[y + 1, x - 1]
        n += img[y + 1, x + 1]
        n /= 255        
        
        if n == 1:
            end_points.append(p)
            
    center_of_mass = get_center_of_mass(end_points)
    sorted_end_points = sort_end_points_by_angle(end_points, center_of_mass)    
    return sorted_end_points

def rotate_image(img, angle):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
  return result
                           
def get_bounding_box(img):    
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    rectangle = cv2.minAreaRect(contours[0])
    center, dim, rot = rectangle
    box = np.intp(cv2.boxPoints(rectangle))    
    return [box, rot] 

def get_center_of_mass(points):
    x = 0
    y = 0

    for p in points:
        x = x + p[0]
        y = y + p[1]

    center_x = int(x / len(points))
    center_y = int(y / len(points)) 
    return [center_x, center_y]

def get_angle(point1, point2):
    angle = math.atan2((point2[0] - point1[0]), (point2[1] - point1[1]))
    degrees = math.degrees(angle)
    degrees = 360 - (540 + degrees) % 360
    return degrees

def get_rotational_loss(end_points):    
    vert_line_0 = get_angle(end_points[4], end_points[7])
    vert_line_1 = get_angle(end_points[3], end_points[0])

    if vert_line_0 > 180: vert_line_0 = 360 - vert_line_0
    if vert_line_1 > 180: vert_line_1 = 360 - vert_line_1
    
    horiz_line_0 = get_angle(end_points[6], end_points[1])
    horiz_line_1 = get_angle(end_points[5], end_points[2])
    
    loss = (vert_line_0)**2 + (vert_line_1)**2 + (90 - horiz_line_0)**2 + (90 - horiz_line_1)**2
    return loss

def sort_end_points_by_angle(end_points, center_of_mass):
    angles = []
    
    for p in end_points:
        angle = get_angle(center_of_mass, p)
        angles.append(angle)
       
    arr1 = np.array(angles)
    arr2 = np.array(end_points)    
    indexes = arr1.argsort()
    sorted_end_points = arr2[indexes]
    
    return sorted_end_points

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def get_cells(end_points):
    inter_top_left = line_intersection((end_points[4], end_points[7]), (end_points[6], end_points[1]))
    inter_top_right = line_intersection((end_points[3], end_points[0]), (end_points[6], end_points[1]))
    inter_bottom_left = line_intersection((end_points[4], end_points[7]), (end_points[5], end_points[2]))
    inter_bottom_right = line_intersection((end_points[3], end_points[0]), (end_points[5], end_points[2]))

    corner_top_left = (min(end_points[5][0], end_points[6][0]), min(end_points[7][1], end_points[0][1]))
    corner_top_right = (max(end_points[1][0], end_points[2][0]), min(end_points[7][1], end_points[0][1]))
    corner_bottom_left = (min(end_points[5][0], end_points[6][0]), max(end_points[4][1], end_points[3][1]))
    corner_bottom_right = (max(end_points[1][0], end_points[2][0]), max(end_points[4][1], end_points[3][1]))

    cells = []
    cells.append(np.array([corner_top_left, end_points[7], inter_top_left, end_points[6]]));
    cells.append(np.array([end_points[7], end_points[0], inter_top_right, inter_top_left]));
    cells.append(np.array([end_points[0], corner_top_right, end_points[1], inter_top_right]));
    cells.append(np.array([end_points[6], inter_top_left, inter_bottom_left, end_points[5]]));
    cells.append(np.array([inter_top_left, inter_top_right, inter_bottom_right, inter_bottom_left]));
    cells.append(np.array([inter_top_right, end_points[1], end_points[2], inter_bottom_right]));
    cells.append(np.array([end_points[5], inter_bottom_left, end_points[4], corner_bottom_left]));
    cells.append(np.array([inter_bottom_left, inter_bottom_right, end_points[3], end_points[4]]));
    cells.append(np.array([inter_bottom_right, end_points[2], corner_bottom_right, end_points[3]]));
    return cells;

original = cv2.imread(image_path)
copy = original.copy()
gray, processed = pre_process(copy)
interior_box = get_interior_box(processed, gray)
interior_box_rotation = get_angle(interior_box[0], interior_box[1])
      
# cv2.drawContours(copy, [interior_box], 0, (255,0, 0), 3) 

rotated = rotate_image(copy, interior_box_rotation)
_, processed = pre_process(rotated)
end_points = get_end_points(processed)
cells = get_cells(end_points)

#rotational_loss = get_rotational_loss(end_points)
#bounding_box, rotation = get_bounding_box(processed)

#for i, p in enumerate(end_points):
#    cv2.circle(rotated, p, 5, (0, 0, 255), 2)
#    cv2.putText(rotated, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.line(rotated, end_points[4], end_points[7], (255, 0, 0), 3)
cv2.line(rotated, end_points[3], end_points[0], (255, 0, 0), 3)
cv2.line(rotated, end_points[6], end_points[1], (0, 255, 0), 3)
cv2.line(rotated, end_points[5], end_points[2], (0, 255, 0), 3)

for i, cell in enumerate(cells):
    cv2.polylines(rotated, [cell], True, (0, 0, 255), 2)
    center_of_mass = get_center_of_mass(cell)
    cv2.putText(rotated, str(i), center_of_mass, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
#for p in bounding_box:
#    cv2.circle(rotated, p, 10, (255, 0, 0), 2)

cv2.imshow(f'Final', rotated) 

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
    