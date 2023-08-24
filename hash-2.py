import math
import cv2
import numpy as np 


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
            end_points.append([int(p[0]), int(p[1])])
            
    return end_points

def get_center_of_mass(points):
    x = 0
    y = 0


    for p in points:
        x = x + p[0][0]
        y = y + p[0][1]

    center_x = int(x / len(points))
    center_y = int(y / len(points)) 
    return [center_x, center_y]

def get_distance(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


image_path = 'images/hash-3.png'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("threshold_image", threshold_image)

contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
selected_contour = max(contours, key=lambda x: cv2.contourArea(x))
approx = cv2.approxPolyDP(selected_contour, 0.0068 * cv2.arcLength(selected_contour, True), True)

row,col,channels = np.shape(image)
blank_image = np.zeros((row, col, 3), np.uint8)
cv2.fillPoly(blank_image, pts =[approx], color=(255,255,255))

print(len(approx))
com = get_center_of_mass(approx)

for p in approx:
    cv2.circle(blank_image, p[0], 10, (255,0,255), 1)
    print(get_distance(com, p[0]))
    
cv2.drawContours(blank_image, [approx], 0, (255, 255, 255), 5)
gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
thin = cv2.ximgproc.thinning(thresh)
    
end_points = get_end_points(thin)

for p in end_points:
    cv2.circle(gray, p, 10, (255,0,255), 1)
    
cv2.imshow(f'Final', blank_image) 

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
    
    