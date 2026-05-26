"""
demo_multi_shape.py — 多区域形状同时识别演示
场景：梯形、三角形、圆同时出现在画面中，Pipeline 对每个色块区域独立识别
API：DetectionPipeline.run

运行前先生成测试视频：
    python generate_test_video.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline.orchestrator import DetectionPipeline

VIDEO = os.path.join(os.path.dirname(__file__), "videos", "test_multi_shape.avi")
SHAPE_NAMES = {0: "ellipse", 1: "trapezoid", 2: "triangle", 3: "pole"}

pipeline = DetectionPipeline()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"找不到视频文件：{VIDEO}")
    print("请先运行 python generate_test_video.py 生成测试视频")
    sys.exit(1)

print("多区域形状识别演示 | 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    type_list, composite = pipeline.run(frame, "red", bais=20)

    # 标注每个识别到的形状
    for item in type_list:
        shape_name = SHAPE_NAMES.get(item["type"], "unknown")
        center = item["center"]
        if center and center != (0, 0):
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(composite, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(composite, shape_name,
                        (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 左上角显示检测到的形状列表
    summary = "  ".join(SHAPE_NAMES.get(i["type"], "?") for i in type_list) or "none"
    cv2.putText(composite, f"detected: {summary}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    display = cv2.hconcat([frame, composite])
    cv2.imshow("Multi-Region Shape Detection  [left: raw | right: result]", display)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
