import cv2 as cv
import numpy as np
import math
def Draw_the_lines(lines,img):
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(img, pt1, pt2, (255,255,255), 1, cv.LINE_AA)





img = cv.imread("test_img.bmp", cv.IMREAD_GRAYSCALE)
ret3, th3 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
ret,thresh1 = cv.threshold(img,52,255,cv.THRESH_BINARY)
kn=cv.Canny(img, 52, 255)
dst = cv.Canny(img, 50, 200, None, 3)


res = cv.findContours(kn, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, res[0], contourIdx=-1, color=(255),thickness=-1)

lines = cv.HoughLines(dst, 2, np.pi / 180, 150, None, 0, 0)

Draw_the_lines(lines,kn)
cv.imshow("blinchik_porog", thresh1)
cv.imshow("blinchik_otsu", th3)
cv.imshow("blinchik_kn", kn)
cv.imshow("blinchik", img)
cv.waitKey()

# detected_circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 20, param1=50,
#                                    param2=30, minRadius=1, maxRadius=40)
#
# # Draw circles that are detected.
# if detected_circles is not None:
#
#     # Convert the circle parameters a, b and r to integers.
#     detected_circles = np.uint16(np.around(detected_circles))
#
#     for pt in detected_circles[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
#
#         # Draw the circumference of the circle.
#         cv.circle(img, (a, b), r, (0, 255, 0), 2)
#
#         # Draw a small circle (of radius 1) to show the center.
#         cv.circle(img, (a, b), 1, (0, 0, 255), 3)
#         cv.imshow("Detected Circle", img)
#         cv.waitKey(0)
