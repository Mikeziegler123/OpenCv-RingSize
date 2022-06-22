import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
from  PIL  import Image

# initialize video from the webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    #Capture Frame-by-frame
    
    ret, frame = video.read()
    if ret == True:
        img = frame
        img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
            cv2.destroyAllWindows()
        original = frame
        l = int(max(5, 6))
        u = int(min(6, 6))

        ed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.GaussianBlur(img, (21, 51), 3)
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(edges, l, u)

        _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

        data = mask.tolist()
        sys.setrecursionlimit(10**8)
        for i in  range(len(data)):
            for j in  range(len(data[i])):
                if data[i][j] !=  255:
                    data[i][j] =  -1
                else:
                    break
            for j in  range(len(data[i])-1, -1, -1):
                if data[i][j] !=  255:
                    data[i][j] =  -1
                else:
                    break
        image = np.array(data)
        image[image !=  -1] =  255
        image[image ==  -1] =  0

        mask = np.array(image, np.uint8)

        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask ==  0] =  255
        cv2.imwrite('bg.png', result)

        img = Image.open('bg.png')
        img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] ==  255  and item[1] ==  255  and item[2] ==  255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save("img.png", "PNG")
        
        # Use the q button to quit the operation
        if cv2.waitKey(60) & 0xff == ord('q'):
            break
        else:
            break
        cv2.destroyAllWindows()
        video.release()

