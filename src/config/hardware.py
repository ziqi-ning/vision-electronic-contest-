"""
硬件参数配置
从 main.py 提取的相机内参、雷达外参及 AprilTag 参数
"""

# ========== 相机配置 ==========
class CameraConfig:
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    FPS = 50

    FX = (3.6 / 3.6736) * IMAGE_WIDTH   # 水平焦距（像素）
    FY = (3.6 / 2.7384) * IMAGE_HEIGHT  # 垂直焦距（像素）

    CX = IMAGE_WIDTH / 2   # 光心 x
    CY = IMAGE_HEIGHT / 2   # 光心 y


# ========== 雷达外参配置 ==========
class RadarConfig:
    DELTA_X = 0       # X 方向偏移（右侧为正，单位：米）
    DELTA_Y = 0.09    # Y 方向偏移（前方为正），雷达在相机后方 0.09m
    DELTA_Z = -0.12   # Z 方向偏移（上方为正），雷达在相机上方 0.12m（实际飞机 -0.18m，便于测试）
    CAMERA_PITCH_DEG = 0  # 相机俯仰角度（度）
    CAMERA_HEIGHT = 0.03  # 相机高度（米）
    ANGLE_TOLERANCE_RAD = 0.5  # ±2.86° 容差（约 3°）
    SCAN_TOPIC = "/scan"


# ========== AprilTag 配置 ==========
class AprilTagConfig:
    TAG_SIZE_M = 0.15  # Apriltag 边长（米）
    TAG_FAMILY = "tag36h11"


# ========== 兼容层：保留原 main.py 中的数据结构 ==========
IMAGE_WIDTH = CameraConfig.IMAGE_WIDTH
IMAGE_HEIGHT = CameraConfig.IMAGE_HEIGHT

fx = CameraConfig.FX
fy = CameraConfig.FY
cx = CameraConfig.CX
cy = CameraConfig.CY

delta_x = RadarConfig.DELTA_X
delta_y = RadarConfig.DELTA_Y
delta_z = RadarConfig.DELTA_Z
camera_pitch_deg = RadarConfig.CAMERA_PITCH_DEG
camera_height = RadarConfig.CAMERA_HEIGHT
angle_tolerance_rad = RadarConfig.ANGLE_TOLERANCE_RAD

cam_info_tag_size_m = AprilTagConfig.TAG_SIZE_M
cam_info_fx = (3.6 / 3.6736) * IMAGE_WIDTH
cam_info_fy = (3.6 / 2.7384) * IMAGE_HEIGHT
cam_info_cx = IMAGE_WIDTH / 2
cam_info_cy = IMAGE_HEIGHT / 2

camera_params = (
    fx, fy, cx, cy,
    delta_x, delta_y, delta_z,
    camera_pitch_deg, angle_tolerance_rad,
    camera_height
)


class StaticCameraParams:
    fx = 0
    fy = 0
    cx = 0
    cy = 0
    tag_size_m = 0


cam_info = StaticCameraParams()
cam_info.tag_size_m = cam_info_tag_size_m
cam_info.fx = cam_info_fx
cam_info.fy = cam_info_fy
cam_info.cx = cam_info_cx
cam_info.cy = cam_info_cy
