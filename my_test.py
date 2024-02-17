import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import stats
import statistics
from SignLib import custom_sort, get_center, get_beg_point, get_polar_coordinates_list, polar_to_decart
import statistics
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from dataclasses import dataclass
import time

start = time.time()


@dataclass
class Section:
    id: int = None
    index: tuple[int] = None
    positive: bool = None
    perimeter: int = None
    distance: float = None
    curvature: float = None
    square: float = None
    deviation_mean: float = None
    defect: bool = None


def defect_detection(
        positive,
        deviation_mean,
        square,
):
    if positive:
        return False

    if abs(square) > 2 and abs(deviation_mean) > 6:
        return True
    return False


def signature_analysis(
        signature: list[float],
        contur
):
    len_signature = len(signature)
    signature_filtred = savgol_filter(signature, 15, 1)

    # avg_dist = sum(signature) / len(signature)
    # med = statistics.median(signature)
    trim_mean = stats.trim_mean(signature_filtred, 0.2)
    square_ideal = (trim_mean ** 2) * 3.14
    signature_filtred_norm = np.array(signature_filtred) - trim_mean

    signature_norm = list(map(lambda x: ((x - trim_mean) / trim_mean * 100), signature_filtred))

    ax = np.full((1, len(signature)), 0, dtype=int)

    idx = np.argwhere(np.diff(np.sign(ax - np.array(signature_filtred_norm))))[:, 1]
    idx_filtred = rem_dist(idx, (len_signature / 100) * 2)

    zones_list = []

    united = np.concatenate([np.array(signature_filtred_norm[idx_filtred[-1]:]),
                             np.array(signature_filtred_norm[0:idx_filtred[0]])])
    square = round((np.trapz(united) / square_ideal) * 100, 3)
    section_positive = 0 < square
    perimeter = len(united)
    p1 = contur[idx_filtred[-1]][0].tolist()
    p2 = contur[idx_filtred[0]][0].tolist()
    distance = round(distance_calculate(p1, p2), 3)
    curvature = round(distance / perimeter, 2)
    zones_list.append(Section(id=0, index=(idx_filtred[-1], idx_filtred[0]), positive=section_positive,
                              perimeter=perimeter, distance=distance, curvature=curvature, square=square))
    for i in range(len(idx_filtred) - 1):
        arr = signature_filtred_norm[idx_filtred[i]: idx_filtred[i + 1]]
        deviation_mean = round(stats.trim_mean(arr, 0.2) / trim_mean * 100, 3)
        square = round((np.trapz(arr) / square_ideal) * 100,
                       3)
        section_positive = 0 < square
        perimeter = idx_filtred[i + 1] - idx_filtred[i]
        p1 = contur[idx_filtred[i]][0].tolist()
        p2 = contur[idx_filtred[i + 1]][0].tolist()
        distance = round(distance_calculate(p1, p2), 3)
        curvature = round(distance / perimeter, 2)

        defect = defect_detection(section_positive,
                                  deviation_mean,
                                  square,
                                  )
        zones_list.append(
            Section(
                id=i + 1,
                index=(idx_filtred[i], idx_filtred[i + 1]),
                positive=section_positive,
                perimeter=perimeter,
                distance=distance,
                curvature=curvature,
                square=square,
                deviation_mean=deviation_mean,
                defect=defect
            )
        )

    return zones_list, idx_filtred, signature_norm, trim_mean


def distance_calculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return int(dis)


def rem_dist(arr, dist):
    new = [arr[0]]
    for n in arr[1:]:
        if n - new[-1] > dist:
            new.append(n)
    return np.array(new)


if __name__ == "__main__":
    img = cv2.imread("test_img.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = 20
    ret, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst = cv2.Canny(image=img, threshold1=200, threshold2=600, L2gradient=True, apertureSize=5)
    # get threshold image
    cv2.imshow("th3", th3)
    cv2.imshow("dst", dst)
    cv2.imshow("thresh_img", thresh_img)
    # cv2.imshow("th3", th3)

    # find contours without approx
    contours, _ = cv2.findContours(image=dst, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    contours = list(contours)
    contours.sort(key=custom_sort)

    sel_countour = contours[0]

    cv2.drawContours(img, [sel_countour], 0, (0, 255, 0), 1)
    M = cv2.moments(sel_countour)
    if M['m00'] != 0:
        cx1 = int(M['m10'] / M['m00'])
        cy1 = int(M['m01'] / M['m00'])

        cv2.circle(img, (cx1,cy1), radius=2, color=(0, 255, 255), thickness=-1)

    sign = []
    for xy in sel_countour:
        x = xy[0][0]
        y = xy[0][1]
        distance = distance_calculate((cx1, cy1), (x, y))
        sign.append(distance)

    zones_list, idx, signature_norm, trim_mean = signature_analysis(sign, sel_countour)

    # max_ind = argrelextrema(np.array(sign), np.greater)
    cv2.circle(img, (cx1, cy1), int(trim_mean), (0, 0, 130), 1)
    cv2.circle(img, sel_countour[idx[2]][0].tolist(), radius=0, color=(0, 255, 255), thickness=-1)
    cv2.circle(img, sel_countour[idx[3]][0].tolist(), radius=0, color=(0, 255, 255), thickness=-1)
    cv2.imshow("blinchik_porog", img)

    x = np.arange(0, len(sign))
    end = time.time()
    print(end - start)
    print(zones_list)

    plt.plot(signature_norm)

    # p1 = [0, len(sign)]
    # p2 = [int(trim_mean), int(trim_mean)]
    plt.plot(x[idx], np.array(signature_norm)[idx], 'ro')
    # plt.plot(p1, p2)
    plt.grid()

    plt.show()
