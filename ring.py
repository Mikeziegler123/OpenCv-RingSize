# Author: Michael Ziegler
# Purpose: testing layered functionalities to measure finger size
# in openCV using python for computer vision image detection
from scipy.spatial import distance
from imutils import perspective
from imutils import contours
import cv2
import numpy as np
import argparse
import imutils

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# CONSTRUCT: the image argument and associate the image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
#ap.add_argument("-w", "--width", type=float, required=True,
#	help="width of the left-most object in the image ")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
# GRAY: convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
# BLUR: remove high frequency edges we are not concerned with for cleaner segmentation
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
# THRESHOLD:               ( image |  thval  |      type      |): returns tuple       
(T, thresh) = cv2.threshold(blurred, 72, 255, cv2.THRESH_BINARY)

#HAND
# FIND: contours of the hand
(cnt, hier) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# FIND: largest contour by area -> The Hand
hand = max(cnt, key=cv2.contourArea)
# DRAW: bounding box around hand
x,y,w,h = cv2.boundingRect(hand)
#print("x: ", x, ", y: ", y, ", w: ", w, ", h: ", h )
cv2.rectangle(image,(x-5,y-5),(x+w+5,y+h-100),(255,0,0),3)
# CROP: area around the hand
#imageCrop = image[y-20:h, x-50:x+w+50]
threshCrop = thresh[y-5:h-200, x-5:x+w+5]
#print("~~~Coordinates: x-50:x+w+50: ", x-50, ":", x+w+50, "  y-20:y+h+20: ", y-20, ":", y+h+20)
# FIND: new contours of the cropped hand
(contours, hierarchy) = cv2.findContours(threshCrop.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#CENTROID
# calculate x,y coordinate of center
M = cv2.moments(thresh)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])
pX = cX
# Draw circle to blackout under concavity points from the center
print("\nCentroid")
print("cX: ", cX, "  ", "cY: ", cX)
r = int((cY + y)/3)
cv2.circle(thresh, (cX, cY), r, (0, 0, 0), -1)
#FINGERS
# FIND: contours of each finger
(fingers, h) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# FIND: convex hull
convexHull = cv2.convexHull(hand, clockwise=True, returnPoints=False)
# FIND: convexity defects
defects = cv2.convexityDefects(hand, convexHull)

# FIND: bounding box for fingers and draw
#for f in fingers:
    #xf,yf,wf,hf = cv2.boundingRect(f)
  #  cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

#DEFECTS
#Print defect ONLY if the distance between itself and another is greater than 40
xS = []
yS = []
n=0
b=1
for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(hand[s][0])
    next = tuple(hand[s+1][0])
    end = tuple(hand[e][0])
    far = tuple(hand[f][0])
    x1 = defects[i][0][0]
    y1 = defects[i][0][1]
    xS.append(far)
    yS.append(far)
    if i < defects.shape[0]-1:
        x2 = defects[i+1][0][0]
        y2 = defects[i+1][0][1]
        distance = ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)
    else:
        cv2.circle(image,far, 6,[208, 146, 8],-1)
    prev = start
    # DRAW: convex hull onto original image
    cv2.line(image,start,end,[103, 205, 143],1)
    if distance > 40:
        # DRAW: convexity defects onto original image
        if n == 4 or n == 0 or n == 7 or n == 2 or n == 9:
            cv2.putText(image, "{:d} ".format(b), (far), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            b = b + 1
        n = n + 1
        cv2.circle(image, far, 6,[208, 146, 8], -1)
    else:
        continue
        cv2.circle(image,far, 6,[0, 0, 255], 1)

#SORT
#Sort by Convexity defect X Points & List.       
xS.sort(key = lambda x: x[0])
xhigh = ((xS[(len(xS)-1)][0]))
xlow = (xS[0][0])
print("\nX-SORTED LIST:")
print("Sort X Low Value: ", xlow)
print("Sort X High Value: ", xhigh)
print("Center: ", ((xhigh+xlow)/2))
#for i in xS:
  #  print(i)

print("\n")
#Sort by Convexity defect Y Points & List.
yS.sort(key = lambda x: x[1])
yhigh = ((yS[(len(yS)-1)][0]))
ylow = (yS[0][0])
print("\nY-SORTED LIST:")
print("Sort Y Low Value: ", ylow)
print("Sort Y High Value: ", yhigh)
print("Center: ", ((yhigh+ylow)/2))
#for i in yS:
#    print(i)
# DRAW: contour lines of hand onto original image
#cv2.drawContours(image, hand, -1, (140, 230, 240), 1)
# DRAW: contour lines of each Finger onto original image
c0 = (0, 0, 0)
c1 = (148, 0, 211)
c2 = (0, 0, 255)
c3 = (0, 255, 0)
c4 = (255, 255, 0)
c5 = (255, 0, 0)
colours = [c1, c2, c3, c4, c5, c0]

print("\n\n------AREA------")
c = 0
for k in fingers:
    area = cv2.contourArea(k)
    print(c, ": Area: ", area)
    c = c + 1
count = 0;
for fin in fingers:
    if cv2.contourArea(fin) > 1500 and cv2.contourArea(fin) < 12000:
        cv2.drawContours(image, fin, -1, (colours[count]), 2)
        count = count + 1


#OUTPUT:
# CREATE: output window(s)
cv2.namedWindow('Contours', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Threshold', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Interest_area', cv2.WINDOW_AUTOSIZE)

#cv2.namedWindow('Erode', cv2.WINDOW_AUTOSIZE)
#cv2.namedWindow('Dilate', cv2.WINDOW_AUTOSIZE)

# SHOW: result image
cv2.imshow('Contours', image)
cv2.imshow('Interest_area', threshCrop)
cv2.imshow('Threshold', thresh)
#cv2.imshow('Erode', erode)
#cv2.imshow('Dilate', dilate)   
# Press "q" to quit
if cv2.waitKey(0) & 0xFF == ord("q"):
    cv2.destroyAllWindows()

