import numpy as np
import cv2
import outsite
import colorblob
import allin
from datetime import datetime

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
    
    while True:

        ret, frame = cam.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        
        composite_img, type_list = allin.give_me_a_color_and_i_will_give_you_a_shape(frame, "red", bais=30)

        cv2.imshow("composite_img", composite_img)

        if cv2.waitKey(1) == ord('q'):
                break
        
    cam.release()
    cv2.destroyAllWindows()