import numpy as np
import cv2
import outsite
import colorblob
import allin
from datetime import datetime

class mode:
    def __init__(self):
        self.mode = 0
        self.color = 'none'



if __name__ == '__main__':

    ctl = mode()
    ctl.mode = 0
    color = "blue"
    score = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "M": 0, "X": 0}

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        score = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "M": 0, "X": 0}
        outimage = {"A": None, "B": None, "C": None, "D": None, "E": None, "M": None, "X": None}

        flag, outimage["A"], score["A"] = allin.template_matching(cv2.imread("E:\\templates\\A.jpg"), frame, color)


        flag, outimage["B"], score["B"] = allin.template_matching(cv2.imread("E:\\templates\\B.jpg"), frame, color)


        flag, outimage["C"], score["C"] = allin.template_matching(cv2.imread("E:\\templates\\C.jpg"), frame, color)


        flag, outimage["D"], score["D"] = allin.template_matching(cv2.imread("E:\\templates\\D.jpg"), frame, color)


        flag, outimage["E"], score["E"] = allin.template_matching(cv2.imread("E:\\templates\\E.jpg"), frame, color)


        flag, outimage["M"], score["M"] = allin.template_matching(cv2.imread("E:\\templates\\M.jpg"), frame, color)


        flag, outimage["X"], score["X"] = allin.template_matching(cv2.imread("E:\\templates\\X.jpg"), frame, color)


        max_key = min(score, key=score.get)
        max_score = score[max_key]
        # print(f"最高得分的字母是: {max_key}，得分: {max_score}")

        if max_key in outimage and outimage[max_key] is not None:
            cv2.imshow("Highest Score Image", outimage[max_key])
            if cv2.waitKey(1) == ord('q'):
                break