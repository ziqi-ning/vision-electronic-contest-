"""
测试视频生成脚本
所有场景目标均缓慢移动（沿椭圆/正弦路径漂移），模拟真实拍摄。
场景列表：
  场景1  颜色检测 - 红、绿、蓝、黄色块依次缓慢移动
  场景2  多颜色同时检测 - 红+绿同框各自移动
  场景3  形状识别 - 梯形（红色）缓慢移动
  场景4  形状识别 - 三角形（红色）缓慢移动
  场景5  形状识别 - 圆形→椭圆（红色）缓慢移动
  场景6  多区域形状同时识别 - 梯形+三角形+圆各自移动
  场景7  杆子检测 - 两根竖直平行红色细杆整体缓慢左右漂移
  场景8  激光点检测 - 极亮小红点缓慢游走
"""

import cv2
import numpy as np
import math
import os

W, H = 640, 480
FPS = 25
OUTPUT = "test_video.avi"

# 各场景独立视频输出目录
VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example", "videos")


def hsv_to_bgr(h, s, v):
    px = np.uint8([[[h, s, v]]])
    return tuple(int(x) for x in cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0])


RED_BGR    = hsv_to_bgr(165, 220, 220)
GREEN_BGR  = hsv_to_bgr(56,  200, 200)
BLUE_BGR   = hsv_to_bgr(112, 200, 200)
YELLOW_BGR = hsv_to_bgr(30,  220, 220)


def blank():
    return np.full((H, W, 3), 240, dtype=np.uint8)


def label(frame, text, pos=(20, 40)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (50, 50, 50), 2, cv2.LINE_AA)


def drift(i, n, amp_x=80, amp_y=60, phase=0):
    """返回当前帧的 (dx, dy) 漂移量，走一个缓慢的椭圆轨迹"""
    t = 2 * math.pi * i / n + phase
    dx = int(amp_x * math.sin(t))
    dy = int(amp_y * math.sin(t * 0.7 + 1.0))
    return dx, dy


def clamp_pt(x, y, margin=20):
    return max(margin, min(W - margin, x)), max(margin, min(H - margin, y))


# ─────────────────────────────────────────────
# 场景生成函数
# ─────────────────────────────────────────────

def scene_color_single(n_frames=300):
    """场景1：红绿蓝黄色块，每种75帧，色块缓慢漂移"""
    frames = []
    colors = [
        ("Red",    RED_BGR),
        ("Green",  GREEN_BGR),
        ("Blue",   BLUE_BGR),
        ("Yellow", YELLOW_BGR),
    ]
    per = n_frames // len(colors)
    for name, bgr in colors:
        for i in range(per):
            f = blank()
            dx, dy = drift(i, per, amp_x=100, amp_y=70)
            cx, cy = 320 + dx, 240 + dy
            x1, y1 = cx - 90, cy - 80
            x2, y2 = cx + 90, cy + 80
            cv2.rectangle(f, (x1, y1), (x2, y2), bgr, -1)
            label(f, f"Color Detection: {name}")
            frames.append(f)
    return frames


def scene_multi_color(n_frames=200):
    """场景2：红+绿同框，各自独立漂移"""
    frames = []
    for i in range(n_frames):
        f = blank()
        # 红色块
        dx1, dy1 = drift(i, n_frames, amp_x=60, amp_y=50, phase=0)
        cx1, cy1 = 160 + dx1, 240 + dy1
        cv2.rectangle(f, (cx1-70, cy1-80), (cx1+70, cy1+80), RED_BGR, -1)
        # 绿色块
        dx2, dy2 = drift(i, n_frames, amp_x=60, amp_y=50, phase=math.pi)
        cx2, cy2 = 480 + dx2, 240 + dy2
        cv2.rectangle(f, (cx2-70, cy2-80), (cx2+70, cy2+80), GREEN_BGR, -1)
        label(f, "Multi-Color: Red + Green")
        frames.append(f)
    return frames


def scene_trapezoid(n_frames=200):
    """场景3：红色梯形缓慢漂移"""
    frames = []
    # 梯形相对中心的偏移点
    base_pts = np.array([[-100, -80], [100, -80], [140, 80], [-140, 80]], np.int32)
    for i in range(n_frames):
        f = blank()
        dx, dy = drift(i, n_frames, amp_x=110, amp_y=80)
        cx, cy = 320 + dx, 240 + dy
        pts = base_pts + np.array([cx, cy])
        # 防止超出边界
        pts[:, 0] = np.clip(pts[:, 0], 5, W - 5)
        pts[:, 1] = np.clip(pts[:, 1], 5, H - 5)
        cv2.fillPoly(f, [pts], RED_BGR)
        label(f, "Shape: Trapezoid")
        frames.append(f)
    return frames


def scene_triangle(n_frames=200):
    """场景4：红色三角形缓慢漂移"""
    frames = []
    base_pts = np.array([[0, -110], [-130, 110], [130, 110]], np.int32)
    for i in range(n_frames):
        f = blank()
        dx, dy = drift(i, n_frames, amp_x=110, amp_y=70)
        cx, cy = 320 + dx, 240 + dy
        pts = base_pts + np.array([cx, cy])
        pts[:, 0] = np.clip(pts[:, 0], 5, W - 5)
        pts[:, 1] = np.clip(pts[:, 1], 5, H - 5)
        cv2.fillPoly(f, [pts], RED_BGR)
        label(f, "Shape: Triangle")
        frames.append(f)
    return frames


def scene_ellipse(n_frames=250):
    """场景5：前半段圆、后半段椭圆，均缓慢漂移"""
    frames = []
    half = n_frames // 2
    for i in range(half):
        f = blank()
        dx, dy = drift(i, half, amp_x=110, amp_y=80)
        cx, cy = 320 + dx, 240 + dy
        cx, cy = clamp_pt(cx, cy, margin=140)
        cv2.ellipse(f, (cx, cy), (120, 120), 0, 0, 360, RED_BGR, -1)
        label(f, "Shape: Circle")
        frames.append(f)
    for i in range(half):
        f = blank()
        dx, dy = drift(i, half, amp_x=110, amp_y=80, phase=math.pi)
        cx, cy = 320 + dx, 240 + dy
        cx, cy = clamp_pt(cx, cy, margin=170)
        cv2.ellipse(f, (cx, cy), (150, 95), 0, 0, 360, RED_BGR, -1)
        label(f, "Shape: Ellipse")
        frames.append(f)
    return frames


def scene_multi_shape(n_frames=250):
    """场景6：梯形+三角形+圆同框，三个目标各自独立漂移"""
    frames = []
    trap_base = np.array([[-70, -60], [70, -60], [90, 60], [-90, 60]], np.int32)
    tri_base  = np.array([[0, -70], [-70, 70], [70, 70]], np.int32)
    for i in range(n_frames):
        f = blank()

        # 梯形 - 左侧区域漂移
        dx1, dy1 = drift(i, n_frames, amp_x=50, amp_y=40, phase=0)
        c1x, c1y = 130 + dx1, 160 + dy1
        pts1 = trap_base + np.array([c1x, c1y])
        pts1[:, 0] = np.clip(pts1[:, 0], 5, W - 5)
        pts1[:, 1] = np.clip(pts1[:, 1], 5, H - 5)
        cv2.fillPoly(f, [pts1], RED_BGR)

        # 三角形 - 右侧区域漂移
        dx2, dy2 = drift(i, n_frames, amp_x=50, amp_y=40, phase=2.1)
        c2x, c2y = 510 + dx2, 160 + dy2
        pts2 = tri_base + np.array([c2x, c2y])
        pts2[:, 0] = np.clip(pts2[:, 0], 5, W - 5)
        pts2[:, 1] = np.clip(pts2[:, 1], 5, H - 5)
        cv2.fillPoly(f, [pts2], RED_BGR)

        # 圆 - 下方区域漂移
        dx3, dy3 = drift(i, n_frames, amp_x=60, amp_y=40, phase=4.2)
        c3x, c3y = 320 + dx3, 370 + dy3
        c3x, c3y = clamp_pt(c3x, c3y, margin=90)
        cv2.ellipse(f, (c3x, c3y), (80, 80), 0, 0, 360, RED_BGR, -1)

        label(f, "Multi-Region: Trap + Tri + Circle")
        frames.append(f)
    return frames


def scene_pole(n_frames=200):
    """场景7：两根竖直平行红色细杆，整体左右+上下缓慢漂移"""
    frames = []
    gap = 12        # 两杆间距，满足 5 < avg_distance < 70
    pole_h = 320    # 杆长，远大于 min_line_length=100
    for i in range(n_frames):
        f = blank()
        dx, dy = drift(i, n_frames, amp_x=120, amp_y=60)
        cx = 320 + dx
        y_top = max(10,  (H - pole_h) // 2 + dy)
        y_bot = min(H-10, y_top + pole_h)
        x1 = cx - gap // 2
        x2 = cx + gap // 2
        x1 = max(5, min(W-5, x1))
        x2 = max(5, min(W-5, x2))
        cv2.line(f, (x1, y_top), (x1, y_bot), RED_BGR, 3)
        cv2.line(f, (x2, y_top), (x2, y_bot), RED_BGR, 3)
        label(f, "Shape: Pole (Parallel Lines)")
        frames.append(f)
    return frames


def scene_laser(n_frames=250):
    """
    场景8：极亮红色小点缓慢游走
    背景暗，激光点是画面中 V 通道最亮的像素，触发 detect_laser 的阈值逻辑。
    """
    frames = []
    bg_color = (60, 60, 60)
    laser_bgr = (0, 0, 255)   # 纯红，V=255 最亮

    for i in range(n_frames):
        f = np.full((H, W, 3), bg_color[0], dtype=np.uint8)
        f[:, :] = bg_color

        # 用两个正弦叠加，走出不规则但平滑的路径
        lx = int(320 + 150 * math.sin(2 * math.pi * i / n_frames)
                      +  50 * math.sin(2 * math.pi * i / n_frames * 3))
        ly = int(240 + 100 * math.cos(2 * math.pi * i / n_frames * 0.7)
                      +  40 * math.cos(2 * math.pi * i / n_frames * 2.3))
        lx = max(10, min(W-10, lx))
        ly = max(10, min(H-10, ly))

        cv2.circle(f, (lx, ly), 4, laser_bgr, -1)
        cv2.circle(f, (lx, ly), 1, (255, 255, 255), -1)  # 中心纯白，V最大

        label(f, "Laser Detection", pos=(20, 40))
        frames.append(f)
    return frames


# ─────────────────────────────────────────────
# 写入视频
# ─────────────────────────────────────────────

def write_single_video(frames, output_path):
    """将帧列表写入单个视频文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_video(scenes, output=OUTPUT):
    """写入合并视频（保留向后兼容）"""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output, fourcc, FPS, (W, H))
    total = sum(len(s) for s in scenes)
    written = 0
    for scene_frames in scenes:
        for frame in scene_frames:
            writer.write(frame)
            written += 1
            if written % 200 == 0:
                print(f"  进度: {written}/{total} 帧  ({written/FPS:.1f}s)")
    writer.release()
    print(f"\n✅ 合并视频已生成: {output}")
    print(f"   总帧数: {total}  时长: {total/FPS:.1f}s  分辨率: {W}x{H}  FPS: {FPS}")


if __name__ == "__main__":
    print("开始生成测试视频（所有目标持续移动）...")

    scene_map = [
        ("color_single", scene_color_single()),
        ("multi_color",  scene_multi_color()),
        ("trapezoid",    scene_trapezoid()),
        ("triangle",     scene_triangle()),
        ("ellipse",      scene_ellipse()),
        ("multi_shape",  scene_multi_shape()),
        ("pole",         scene_pole()),
        ("laser",        scene_laser()),
    ]

    # 各场景独立视频
    for name, frames in scene_map:
        path = os.path.join(VIDEO_DIR, f"test_{name}.avi")
        write_single_video(frames, path)
        print(f"  ✅ test_{name}.avi  ({len(frames)} 帧, {len(frames)/FPS:.1f}s)")

    # 同时生成合并视频（方便整体预览）
    print("\n生成合并视频...")
    write_video([frames for _, frames in scene_map])
    print("\n全部完成。")
