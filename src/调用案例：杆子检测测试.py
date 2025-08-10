import cv2
import numpy as np
import allin

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        frame = allin.color_bonous_for_line(frame, "red")
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break