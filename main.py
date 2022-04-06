import cv2
import numpy as np
import cv2 as cv
import math


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def apply_roi(img, roi):
    # resize ROI to match the original image size
    roi = cv2.resize(src=roi, dsize=(img.shape[1], img.shape[0]))

    assert img.shape[:2] == roi.shape[:2]

    # scale ROI to [0, 1] => binary mask
    thresh, roi = cv2.threshold(roi, 150,255, type=cv2.THRESH_BINARY)

    # apply ROI on the original image
    new_img = img * roi
    return new_img


imgTem = cv.imread('Battery/Template.PNG')
img = cv.imread('Battery/BAT0001.PNG')
roi = cv2.imread('Battery/ROI.png')
imgCricle = img.copy()
imgBlank = np.zeros_like(img)
height, width = img.shape[:2]
center = (width/2, height/2)


img = cv.medianBlur(img,5)
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imBulr = cv2.GaussianBlur(img, (7, 7), 1)
imgCanny = cv.Canny(imBulr,150, 255)
circles = cv.HoughCircles(imgCanny,cv.HOUGH_GRADIENT, 1, 50, param1=50, param2=43,)
circles = np.round(circles[0, :]).astype("int")

for (x, y, r) in circles:
    cv.circle(imgCricle, (x, y), 2, (0, 0, 255), 3)
    cv2.circle(imgCricle, (x, y), r, (0, 255, 255), 3)
#
x_0 = int(circles[0][2]*math.cos(math.pi/4)+circles[0][0])
y_0 = int(circles[0][2]*math.sin(math.pi/4)+circles[0][1])
x_1 = int(circles[0][2]*math.cos(-3*math.pi/4)+circles[0][0])
y_1 = int(circles[0][2]*math.sin(-3*math.pi/4)+circles[0][1])
#
cv2.circle(imgCricle, (x_0, y_0), 2, (255, 0, 255), 3)
cv2.circle(imgCricle, (x_1, y_1), 2, (255, 0, 255), 3)
cv2.line(imgCricle, (x_0,y_0),(x_1,y_1),(255, 0, 255),2)
cv2.putText(imgCricle,'d='+str(circles[0][2]*2), (circles[0][0],circles[0][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0, 0, 255),1)
print("đường kính =",format( circles[0][2]*2))

a = circles[2][0] - circles[1][0]
b = circles[2][1] - circles[1][1]
goc = math.acos(abs(-b) / math.sqrt((b**2)+(a**2)))
c = int(math.degrees(goc))

if (circles[2][0] < width/2):
    if circles[1][0] > circles[2][0]:
        angle = c
    else:

        angle = -c
else:
    if circles[1][0] < circles[2][0]:
        angle = c+180
    else:
        angle = -c+180
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
rotated_image = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(width, height))
rotated_image1 = rotated_image.copy()

Roi = apply_roi(rotated_image, roi)
AND = cv2.bitwise_xor(rotated_image, Roi )
imBulr1 = cv2.GaussianBlur(AND, (7, 7), 1)
imgGray1 = cv2.cvtColor(imBulr1, cv2.COLOR_BGR2GRAY)
res, thresh1 = cv2.threshold(imgGray1,150,255,cv2.THRESH_BINARY)
res2, thresh2 = cv2.threshold(imgGray1,150,255,cv2.THRESH_BINARY_INV)
imgCanny1 = cv2.Canny(thresh2,150, 255)


contours, hierarchy = cv2.findContours(imgCanny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
point_list=[]
for (i, c) in enumerate(contours):
    area = cv2.contourArea(c)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(c)
        point_list.append([ x, y, w, h])
        M = cv2.moments(c)
        centroidX = int(M["m10"] / M["m00"])
        centroidY = int(M["m01"] / M["m00"])

        cv2.rectangle(rotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)



pt1 = (point_list[0][0]+point_list[0][2]),(point_list[0][1])
x1 = point_list[0][0]+point_list[0][2]
y1 = point_list[0][1]
pt2 = (point_list[1][0]+point_list[1][2]),(point_list[1][1]+(point_list[1][3]))
x2 = point_list[1][0]+point_list[1][2]
y2 = point_list[1][1]+point_list[1][3]
cv2.circle(rotated_image, pt1, 2, (0, 0, 255), 3)
cv2.circle(rotated_image, pt2, 2, (0, 0, 255), 3)
cv2.line(rotated_image, pt1,pt2,(255, 0, 255),2)
d=((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
print("Khoang cách d =",format(d))

cv2.putText(rotated_image,'D1='+str(d), pt1,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0, 0, 255),1)



arrayImg = stackImages(0.5, ([img, imgGray, imgCanny, imgCricle,rotated_image1],
                        [Roi, AND, imgGray1, imgCanny1,rotated_image]))
cv.imshow('Test Vision', arrayImg)
cv.waitKey(0)
cv.destroyAllWindows()