import numpy as np
import cv2

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


if __name__ == '__main__':
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
            exit()
    except:
        print("Error: Cannot open webcam")
        exit()

    cv2.namedWindow("hsv", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("h->","hsv",0,255,lambda x:None)
    cv2.createTrackbar("s->","hsv",0,255,lambda x:None)
    cv2.createTrackbar("v->","hsv",0,255,lambda x:None)
    cv2.createTrackbar("<-h","hsv",0,180,lambda x:None)
    cv2.createTrackbar("<-s","hsv",0,255,lambda x:None)
    cv2.createTrackbar("<-v","hsv",0,255,lambda x:None)
    cv2.setTrackbarPos("<-h","hsv",180)
    cv2.setTrackbarPos("<-s","hsv",255)
    cv2.setTrackbarPos("<-v","hsv",255)
    cv2.namedWindow("CLAHE",cv2.WINDOW_NORMAL)
    cv2.createTrackbar("clipLimit","CLAHE",100,400,lambda x:None)
    cv2.setTrackbarPos("clipLimit","CLAHE",100)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam")
            break
        # 获取阈值
        h_low = cv2.getTrackbarPos("h->","hsv")
        s_low = cv2.getTrackbarPos("s->","hsv")
        v_low = cv2.getTrackbarPos("v->","hsv")
        h_high = cv2.getTrackbarPos("<-h","hsv")
        s_high = cv2.getTrackbarPos("<-s","hsv")
        v_high = cv2.getTrackbarPos("<-v","hsv")
        clipLimit = cv2.getTrackbarPos("clipLimit","CLAHE")/100.0
        # 创建阈值区间
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
        lower_hsv = np.array([h_low, s_low, v_low])
        upper_hsv = np.array([h_high, s_high, v_high])
        # 拆开成HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        # CLAHE均衡
        V_new = clahe.apply(V)
        hsv_new = cv2.merge((H, S, V_new))
        mask = cv2.inRange(hsv_new, lower_hsv, upper_hsv)
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        # 合成掩模
        result = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("hsv_new", hsv_new)
        cv2.imshow("user2", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


