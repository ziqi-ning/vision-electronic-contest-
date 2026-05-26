"""
demo_color.py — 颜色检测演示
场景：单色色块（红/绿/蓝）在画面中移动
API：CircleROIExtractor

运行前先生成测试视频：
    python generate_test_video.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline.roi_extractor import CircleROIExtractor, composite_ROIs

VIDEO = os.path.join(os.path.dirname(__file__), "videos", "test_color_single.avi")
COLOR = "red"   # 可改为 "green" / "blue"

extractor = CircleROIExtractor()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"找不到视频文件：{VIDEO}")
    print("请先运行 python generate_test_video.py 生成测试视频")
    sys.exit(1)

print(f"颜色检测演示 | 颜色={COLOR} | 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 提取指定颜色的所有 ROI 区域
    roi_list = extractor.extract(frame, COLOR, bais=20)

    # 合成结果图
    composite = composite_ROIs(frame, roi_list)

    # 在原图上标注检测到的色块中心
    annotated = frame.copy()
    for item in roi_list:
        cx, cy = item["center"]
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(annotated, f"{COLOR} ({cx},{cy})", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 左：原图+标注  右：ROI 合成图
    display = cv2.hconcat([annotated, composite])
    cv2.imshow("Color Detection  [left: annotated | right: ROI composite]", display)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
