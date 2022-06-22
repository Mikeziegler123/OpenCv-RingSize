import cv2
import numpy as np

image = cv2.imread('rawhandcoin3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY )
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
(T, thresh) = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# FIND: largest contour by area -> The Hand
maxArea = -1
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > maxArea:
        maxArea = cv2.contourArea(contours[i])
        hand = contours[i]
# FIND: convex hull
hull = cv2.convexHull(hand, returnPoints = False)
# FIND: convexity defects
defects = cv2.convexityDefects(hand, hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(hand[s][0])
    end = tuple(hand[e][0])
    far = tuple(hand[f][0])
    cv2.line(image,start,end,[103, 205, 143],1)
    cv2.circle(image,far,7,[36, 31, 255],-1)

cv2.imshow('Convexity Defects',image)

# Press "q" to quit
if cv2.waitKey(10000) & 0xFF == ord("q"):
    cv2.destroyAllWindows()

