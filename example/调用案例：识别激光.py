import numpy as np
import cv2
import colorblob
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
        composite_img = allin.color_bonous_laser_small_area(frame, "red_laser", bais=0, min_area=0)
        flag, result_img, center, radius = colorblob.detect_laser(composite_img, light_bais=25, min_area=0, max_area=5000)
        cv2.imshow("Frame", result_img)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
