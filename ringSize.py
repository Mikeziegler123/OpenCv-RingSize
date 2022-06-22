# Author: Michael Ziegler
# Purpose: testing layered functionalities in openCV using python
# for computer vision image detection

image = "rawhandcoin.jpg"

cv2.IMREAD_COLOR(image)

cv2.IMREAD_GRAYSCALE(image)

cv2.IMREAD_UNCHANGED(image)

window_out = "Output"

cv2.imshow(window_out, image )

cv2.waitKey(0)
cv2.destroyAllWindows()
