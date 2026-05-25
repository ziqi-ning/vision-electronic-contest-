import sys
import os

# Lazy mock pyzbar (not available on Windows dev machine)
try:
    import pyzbar.pyzbar as pyzbar
except Exception:
    import unittest.mock as _mock
    pyzbar = _mock.MagicMock()
    sys.modules['pyzbar'] = _mock.MagicMock()
    sys.modules['pyzbar.pyzbar'] = pyzbar

# numpy 2.x removed numpy.lib.utils.info and numpy.core.defchararray.{center,lower}
# str_format lives in numpy.core.arrayprint (still available in numpy 2.x)
try:
    from numpy.core.arrayprint import str_format
except Exception:
    str_format = str  # fallback for numpy 2.x compatibility

import numpy as np
import math as mh
import cv2 as cv
import time
import serial
import apriltag
from src.utils.logger import get_logger

logger = get_logger(__name__)


def decodeDisplay(_img):
    flag = 0
    image = cv.cvtColor(_img, cv.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(image)
    x = 0
    y = 0
    apriltag_id = 0
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv.rectangle(_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        text = "{} ({})".format(barcodeData, barcodeType) + str_format(barcode.rect)
        cv.putText(_img, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                   .5, (0, 0, 255), 1)
        logger.info("Found %s barcode: %s", barcodeType, barcodeData)
        x = int((2 * x + w) / 2)
        y = int((2 * y + h) / 2)
        apriltag_id = int(barcodeData)
        flag = 1
    return _img, x, y, apriltag_id, flag


def opencv_find_april_tag(imgsrc, cam_info, apriltag_detetor):
    gray = cv.cvtColor(imgsrc, cv.COLOR_BGR2GRAY)
    tags = apriltag_detetor.detect(gray, return_image=False)
    for tag in tags:
        (ptA, ptB, ptC, ptD) = tag.corners
        info_t = [cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy]
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        M, e1, e2 = apriltag_detetor.detection_pose(tag, info_t, cam_info.tag_size_m)
        R_Mat = np.array(M[0:3, 0:3])
        P_Mat = np.array(M[0:3, 3])
        yaw = 180 * mh.atan2(R_Mat[1][0], R_Mat[0][0]) / mh.pi
        pitch = 180 * mh.asin(R_Mat[2][0]) / mh.pi
        roll = 180 * mh.atan2(R_Mat[2][1], R_Mat[2][2]) / mh.pi
        px = format(P_Mat[0], '.4f')
        py = format(P_Mat[1], '.4f')
        pz = format(P_Mat[2], '.4f')
        len_ab = int(mh.sqrt(np.square(ptB[0] - ptA[0]) + np.square(ptB[1] - ptA[1])))
        len_bc = int(mh.sqrt(np.square(ptB[0] - ptC[0]) + np.square(ptB[1] - ptC[1])))
        len_cd = int(mh.sqrt(np.square(ptC[0] - ptD[0]) + np.square(ptC[1] - ptD[1])))
        len_da = int(mh.sqrt(np.square(ptD[0] - ptA[0]) + np.square(ptD[1] - ptA[1])))
        max_len = max(len_ab, len_bc, len_cd, len_da)
        side_len = max_len
        flag = 1
        (x, y) = tuple(tag.center.astype(int))
        return x, y, tag.tag_id, side_len, flag, int(P_Mat[0] * 1000), int(P_Mat[1] * 1000), int(P_Mat[2] * 1000)
    return 0, 0, 0, 0, 0, 0, 0, 0


def QR_detect(detector, img):
    flag = 0
    data = 0
    data, points, _ = detector.detectAndDecode(img)
    if points is not None:
        points = points[0].astype(int)
        if data:
            data = int(data)
            center = tuple(np.mean(points, axis=0).astype(int))
            if center[0] < 0 or center[1] < 0:
                return img, flag, 0, (0, 0), 0
            flag = 1
            left_length = int(np.linalg.norm(points[0] - points[3]))
            x = int(center[0])
            y = int(center[1])
            pixel = int(left_length * left_length)
            cv.circle(img, center, 5, (0, 255, 0), -1)
            cv.putText(img, f"Center: {center}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(img, f"Side Length: {left_length}px", (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.polylines(img, [points], True, (0, 0, 255), 2)
            logger.info("Decoded Data: %s", data)
            return img, flag, data, x, y, min(60000, pixel)
        return img, flag, 0, (0, 0), 0
    return img, flag, 0, (0, 0), 0
