"""
demo_shape.py — 形状识别演示（梯形 / 三角形 / 椭圆）
场景：单个红色形状在画面中移动，Pipeline 自动识别形状类型
API：DetectionPipeline.run

支持的视频：
    test_trapezoid.avi  — 梯形
    test_triangle.avi   — 三角形
    test_ellipse.avi    — 圆/椭圆

运行前先生成测试视频：
    python generate_test_video.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from src.pipeline.orchestrator import DetectionPipeline

SHAPE_NAMES = {0: "ellipse", 1: "trapezoid", 2: "triangle", 3: "pole"}

# 切换这里来测试不同形状
VIDEO_NAME = "test_trapezoid.avi"   # 可改为 test_triangle.avi / test_ellipse.avi
VIDEO = os.path.join(os.path.dirname(__file__), "videos", VIDEO_NAME)

pipeline = DetectionPipeline()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"找不到视频文件：{VIDEO}")
    print("请先运行 python generate_test_video.py 生成测试视频")
    sys.exit(1)

print(f"形状识别演示 | {VIDEO_NAME} | 按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 一次调用完成：颜色检测 → ROI 提取 → 形状分类
    type_list, composite = pipeline.run(frame, "red", bais=20)

    # 在合成图上标注识别结果
    for item in type_list:
        shape_name = SHAPE_NAMES.get(item["type"], "unknown")
        center = item["center"]
        if center and center != (0, 0):
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(composite, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(composite, f"{shape_name} ({cx},{cy})",
                        (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 左：原图  右：检测结果
    display = cv2.hconcat([frame, composite])
    cv2.imshow(f"Shape Detection: {VIDEO_NAME}  [left: raw | right: result]", display)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
