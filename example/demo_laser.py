"""
demo_laser.py — 激光点检测演示
场景：极亮红色小点在暗背景中游走
API：LaserROIExtractor + colorblob.detect_laser

运行前先生成测试视频：
    python generate_test_video.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline.roi_extractor import LaserROIExtractor, composite_ROIs
import src.colorblob as colorblob

VIDEO = os.path.join(os.path.dirname(__file__), "videos", "test_laser.avi")

extractor = LaserROIExtractor(min_area=0)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"找不到视频文件：{VIDEO}")
    print("请先运行 python generate_test_video.py 生成测试视频")
    sys.exit(1)

print("激光点检测演示 | 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 第一步：提取激光颜色区域（min_area=0 允许极小色块）
    roi_list = extractor.extract(frame, "red_laser", bais=0, min_area=0)
    composite = composite_ROIs(frame, roi_list)

    # 第二步：在提取区域内定位激光亮点
    flag, result_img, center, radius = colorblob.detect_laser(
        composite, light_bais=25, min_area=0, max_area=5000
    )

    annotated = frame.copy()
    if flag == 1:
        cx, cy = center
        cv2.circle(annotated, (cx, cy), radius + 4, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)
        cv2.putText(annotated, f"laser ({cx},{cy}) r={radius}",
                    (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    display = cv2.hconcat([annotated, result_img])
    cv2.imshow("Laser Detection  [left: annotated | right: detect_laser result]", display)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
