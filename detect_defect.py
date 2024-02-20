import math
import time
from dataclasses import dataclass
from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from typing import Literal

start = time.time()


@dataclass
class Section:
    id: int = None
    index: tuple[Any] = None
    positive: bool = None
    perimeter: float = None
    distance_shortest: float = None
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


def get_contur(
        img_path: str,
        threshold1: int = 200,
        threshold2: int = 600,
        apertureSize: int = 5,
        thresh: int = 20,
        method: Literal["CANNY", "THRESH_BINARY", "THRESH_OTSU"] = 'CANNY'
):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "CANNY":
        dst = cv2.Canny(image=img, threshold1=threshold1, threshold2=threshold2, L2gradient=True,
                        apertureSize=apertureSize)
    elif method == "THRESH_BINARY":
        _, dst = cv2.threshold(src=gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    elif method == "THRESH_OTSU":
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, dst = cv2.threshold(src=gray, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise NameError('Incorrect method name')

    # get threshold image
    # cv2.imshow("th3", th3)
    # cv2.imshow("dst", dst)
    # cv2.imshow("thresh_img", thresh_img)
    # cv2.imshow("th3", th3)

    # find contours without approx
    contours, _ = cv2.findContours(image=dst, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key=lambda cont: -cont.shape[0])
    sel_contour = contours[0]
    return sel_contour, img


def get_center_contur(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    else:
        return None, None
    return cx, cy


def get_signature(contour, cx, cy):
    signature = []
    for xy in contour:
        x = xy[0][0]
        y = xy[0][1]
        distance = distance_calculate((cx, cy), (x, y))
        signature.append(distance)
    return signature


def signature_analysis(
        signature: list[float],
        contur
):
    len_signature = len(signature)

    signature_filtered = savgol_filter(x=signature,
                                       window_length=int((len_signature / 100) * 0.6),
                                       polyorder=1,
                                       )
    # avg_dist = sum(signature) / len(signature)
    # med = statistics.median(signature)
    trim_mean = stats.trim_mean(signature_filtered, 0.2)
    square_ideal = (trim_mean ** 2) * math.pi
    signature_filtered_norm = np.array(signature_filtered) - trim_mean

    signature_norm = list(map(lambda x: ((x - trim_mean) / trim_mean * 100), signature_filtered))

    ax = np.full((1, len(signature)), 0, dtype=int)

    idx = np.argwhere(np.diff(np.sign(ax - np.array(signature_filtered_norm))))[:, 1]
    idx_filtered = rem_dist(idx, (len_signature / 100) * 2)

    zones_list = []

    united = np.concatenate([np.array(signature_filtered_norm[idx_filtered[-1]:]),
                             np.array(signature_filtered_norm[0:idx_filtered[0]])])

    deviation_mean = round(stats.trim_mean(united, 0.2) / trim_mean * 100, 3)
    square = round((np.trapz(united) / square_ideal) * 100, 3)
    section_positive = 0 < square
    perimeter = len(united)
    perimeter_norm = round(perimeter / len_signature * 100, 1)
    p1 = contur[idx_filtered[-1]][0].tolist()
    p2 = contur[idx_filtered[0]][0].tolist()
    distance = round(distance_calculate(p1, p2), 3)
    curvature = round(distance / perimeter, 2)

    defect = defect_detection(
        section_positive,
        deviation_mean,
        square,
    )
    zones_list.append(Section(
        id=0,
        index=(idx_filtered[-1], idx_filtered[0]),
        positive=section_positive,
        perimeter=perimeter_norm,
        distance_shortest=distance,
        curvature=curvature,
        square=square,
        deviation_mean=deviation_mean,
        defect=defect
    ))
    for i in range(len(idx_filtered) - 1):
        arr = signature_filtered_norm[idx_filtered[i]: idx_filtered[i + 1]]
        deviation_mean = round(stats.trim_mean(arr, 0.2) / trim_mean * 100, 3)
        square = round((np.trapz(arr) / square_ideal) * 100, 3)
        section_positive = 0 < square
        perimeter = (idx_filtered[i + 1] - idx_filtered[i])
        perimeter_norm = round(perimeter / len_signature * 100, 1)
        p1 = contur[idx_filtered[i]][0].tolist()
        p2 = contur[idx_filtered[i + 1]][0].tolist()
        distance = round(distance_calculate(p1, p2), 3)
        curvature = round(distance / perimeter, 2)

        defect = defect_detection(section_positive,
                                  deviation_mean,
                                  square,
                                  )
        zones_list.append(
            Section(
                id=i + 1,
                index=(idx_filtered[i], idx_filtered[i + 1]),
                positive=section_positive,
                perimeter=perimeter_norm,
                distance_shortest=distance,
                curvature=curvature,
                square=square,
                deviation_mean=deviation_mean,
                defect=defect
            )
        )

    return zones_list, idx_filtered, signature_norm, trim_mean


def get_result(zones: list[Section]):
    count = 0
    for zone in zones:
        if zone.defect:
            count += 1
            print(
                "Defect №{} "
                "\n-- normalized perimeter - {} % "
                "\n-- normalized area of the defect relative to the trend - {} % "
                "\n-- average deviation from the trend {} % "
                "\n-- the directness parameter {}".format(count,
                                                          zone.perimeter,
                                                          zone.square,
                                                          zone.deviation_mean,
                                                          zone.curvature)
            )
    if count == 0:
        print("No defects found")
    else:
        print("--- Total defects {} ---".format(count))


def draw_defects(
        img,
        contour,
        zones: list[Section]
):
    color_defect = (0, 255, 255)
    for zona in zones:
        if zona.defect:
            cv2.circle(img, contour[zona.index[0]][0].tolist(), radius=3, color=color_defect, thickness=-1)
            cv2.circle(img, contour[zona.index[1]][0].tolist(), radius=3, color=color_defect, thickness=-1)
            # cv2.line(img=img,
            #          pt1=(contour[zona.index[0]][0].tolist()),
            #          pt2=(contour[zona.index[1]][0].tolist()),
            #          color=color_defect,
            #          thickness=1)
            cv2.polylines(img=img, pts=[contour[zona.index[0]:zona.index[1]]], isClosed=False, color=color_defect,
                          thickness=1)


def detect_defect(
        img_path: str,
        contour_extraction_method: Literal["CANNY", "THRESH_BINARY", "THRESH_OTSU"],
        visualize: bool
):
    contour, img = get_contur(img_path=img_path, method=contour_extraction_method)
    cx, cy = get_center_contur(contour=contour)
    sign = get_signature(contour=contour, cx=cx, cy=cy)
    zones_list, idx, signature_norm, trim_mean = signature_analysis(sign, contour)
    get_result(zones_list)
    end = time.time()
    print("Processing time {} s".format(round(end - start, 3)))

    if visualize:
        # center of trend and trend
        cv2.circle(img, (cx, cy), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, (cx, cy), int(trim_mean), (0, 0, 130), 1)
        # contur
        cv2.drawContours(img, [contour], 0, (0, 255, 0), 1)
        # defect
        draw_defects(img=img, contour=contour, zones=zones_list)

        cv2.imshow("img", img)

        x = np.arange(0, len(sign))
        plt.plot(signature_norm)
        plt.plot(x[idx], np.array(signature_norm)[idx], "ro")
        plt.grid()
        plt.show()


def parse_opt():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ip", "--img_path",
        type=str,
        default='test_img.bmp',
        help="path to image"
    )
    parser.add_argument(
        "-cem", "--contour_extraction_method",
        choices=["CANNY", "THRESH_BINARY", "THRESH_OTSU"],
        type=str,
        default="CANNY",
        help="contour_extraction_method"
    )
    parser.add_argument(
        "-v", "--visualize",
        action='store_true',
        help="contour_extraction_method"
    )

    return parser.parse_args()


def main(
        opt
):
    detect_defect(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
