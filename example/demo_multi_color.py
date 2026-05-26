"""
demo_multi_color.py — 多颜色同时检测演示
场景：红色和绿色色块同时出现在画面中
API：MultiColorROIExtractor

运行前先生成测试视频：
    python generate_test_video.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline.roi_extractor import MultiColorROIExtractor, composite_ROIs

VIDEO = os.path.join(os.path.dirname(__file__), "videos", "test_multi_color.avi")

extractor = MultiColorROIExtractor()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"找不到视频文件：{VIDEO}")
    print("请先运行 python generate_test_video.py 生成测试视频")
    sys.exit(1)

print("多颜色检测演示 | 同时检测红+绿 | 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # MultiColorROIExtractor 的 color 参数格式为 "color1+color2"
    roi_list = extractor.extract(frame, "red+green", bais=20)

    composite = composite_ROIs(frame, roi_list)

    annotated = frame.copy()
    for item in roi_list:
        cx, cy = item["center"]
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
        cv2.putText(annotated, f"({cx},{cy})", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    display = cv2.hconcat([annotated, composite])
    cv2.imshow("Multi-Color Detection  [left: annotated | right: ROI composite]", display)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
