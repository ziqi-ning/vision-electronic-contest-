"""
demo_pole.py — 杆子（平行竖线）检测演示
场景：两根平行红色竖线在画面中左右漂移
API：LineROIExtractor

运行前先生成测试视频：
    python generate_test_video.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline.roi_extractor import LineROIExtractor, composite_ROIs

VIDEO = os.path.join(os.path.dirname(__file__), "videos", "test_pole.avi")

extractor = LineROIExtractor()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"找不到视频文件：{VIDEO}")
    print("请先运行 python generate_test_video.py 生成测试视频")
    sys.exit(1)

print("杆子检测演示 | 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # LineROIExtractor 内部已完成直线检测，结果存在 pole_groups 字段
    roi_list = extractor.extract(frame, "red", bais=5)

    composite = composite_ROIs(frame, roi_list)

    # 标注杆子中心线
    for item in roi_list:
        for pole in item.get("pole_groups", []):
            for cx in pole.get("center", []):
                cv2.line(composite, (cx, 0), (cx, 480), (0, 255, 0), 2)
                cv2.putText(composite, f"pole x={cx}", (cx + 5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    display = cv2.hconcat([frame, composite])
    cv2.imshow("Pole Detection  [left: raw | right: result]", display)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
