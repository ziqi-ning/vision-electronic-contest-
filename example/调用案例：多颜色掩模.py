import cv2
import numpy as np
import allin
import colorblob

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        composite_img = allin.color_bonous_multi_color(frame, "red", "red")
        cv2.imshow("multi_color", composite_img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.release()
