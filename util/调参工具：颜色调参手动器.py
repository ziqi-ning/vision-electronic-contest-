import numpy as np
import cv2
import yaml
import os

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# config/scene.yaml 保存路径（相对于脚本所在目录的上一级）
CONFIG_YAML_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "scene.yaml")

# 当前 trackbar 位置缓存（避免导出时重新 get）
_current_lower = [0, 43, 46]   # [H, S, V] lower bound
_current_upper = [180, 255, 255]  # [H, S, V] upper bound
_current_clip_limit = 1.0


def _update_current_values():
    """读取所有 trackbar 当前值，存入缓存"""
    global _current_lower, _current_upper, _current_clip_limit
    _current_lower = [
        cv2.getTrackbarPos("h->", "hsv"),
        cv2.getTrackbarPos("s->", "hsv"),
        cv2.getTrackbarPos("v->", "hsv"),
    ]
    _current_upper = [
        cv2.getTrackbarPos("<-h", "hsv"),
        cv2.getTrackbarPos("<-s", "hsv"),
        cv2.getTrackbarPos("<-v", "hsv"),
    ]
    _current_clip_limit = cv2.getTrackbarPos("clipLimit", "CLAHE") / 100.0


def on_export(x):
    """导出当前参数到 config/scene.yaml"""
    _update_current_values()
    config_path = os.path.normpath(CONFIG_YAML_PATH)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    config = {
        "colors": {
            "red": {
                "lower": _current_lower,
                "upper": _current_upper,
            }
        },
        "morphology": {
            "erode_iter": 2,
            "dilate_iter": 2,
        },
        "clahe": {
            "clip_limit": _current_clip_limit,
        },
        "detection": {
            "min_area": 1200,
            "roi_bais": 20,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"[调参工具] 参数已导出到 {config_path}")
    print(f"  red.lower  = {_current_lower}")
    print(f"  red.upper  = {_current_upper}")
    print(f"  clipLimit = {_current_clip_limit}")


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
    cv2.createButton("导出参数到 scene.yaml", on_export, None, cv2.QT_PUSH_BUTTON, 0)
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


