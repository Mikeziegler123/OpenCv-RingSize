from scipy.spatial import distance
from imutils import perspective
from imutils import contours
import cv2
import numpy as np
import argparse
import imutils
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
size = 0
size2 = 0
i = 1
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY )
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    (T, thresh) = cv2.threshold(blurred, 126, 255, cv2.THRESH_BINARY)
    # FIND: contours
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # FIND: largest contour by area -> The Hand
    maxArea = -1
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > maxArea:
            maxArea = cv2.contourArea(contours[i])
            hand = contours[i]
        
    # FIND: convex hull
    convexHull = cv2.convexHull(hand, clockwise=False, returnPoints=False)
    # FIND: convexity defects
    defects = cv2.convexityDefects(hand, convexHull)



    #Print defect ONLY if the distance between it and another defect is below 40
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(hand[s][0])
        next = tuple(hand[s+1][0])
        end = tuple(hand[e][0])
        far = tuple(hand[f][0])
        # DRAW: convex hull onto original image
        cv2.line(frame,start,end,[0, 0, 255],1)
        # DRAW: convexity defects onto original image
        #cv2.circle(frame,far, 3,[208, 146, 8],-1)
        cv2.circle(frame,far, 3,[208, 146, 8],-1)
    
    # DRAW: contour lines of hand onto original image
    cv2.drawContours(frame, hand, -1, (145, 230, 240), 1)
    #--------------

    M = cv2.moments(thresh)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"] + 130)
    cY = int(M["m01"] / M["m00"] +45)

    # put text and highlight the center
    cv2.circle(frame, (cX, cY), size, (0, 0, 255), -1)
    cv2.circle(frame, (cX, cY), size, (5, 116, 252), 6)



    if size <= 100:
        if i % 2 == 0:
            size = size + 3
        i = i + 1
    if size <= 20:
        size2 = size2 + 1

   # print("MaxArea: ", maxArea)
    if maxArea <= 15000:
        if size > 10:
            size = size - 5 
    
    cv2.putText(frame, "FireBall", (cX - 25, cY - 25),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2)
        #--------------

    frameS = cv2.resize(frame,(500,420),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    threshS = cv2.resize(thresh,(500,420),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    cv2.imshow('Webcam', frameS)
    cv2.imshow('Thresh', threshS)

    c = cv2.waitKey(1)
    if cv2.waitKey(60) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
