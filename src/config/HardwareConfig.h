#pragma once

#include "Types.h"
#include "Version.h"

#include <string>

namespace fvs {

// ——————————————————————————————————
// 相机内参配置（与 Python 版 hardware.py CameraConfig 完全一致）
// 焦距计算：fx = (sensor_width_mm / pixel_width) * image_width
//          fy = (sensor_height_mm / pixel_height) * image_height
// ——————————————————————————————————

struct CameraConfig {
    static constexpr int   kImageWidth  = 640;
    static constexpr int   kImageHeight = 480;
    static constexpr int   kFPS        = 50;

    // 水平焦距 fx = (3.6 / 3.6736) * 640 ≈ 628.7 px
    static constexpr float kFX()  { return (3.6f / 3.6736f) * static_cast<float>(kImageWidth); }
    // 垂直焦距 fy = (3.6 / 2.7384) * 480 ≈ 631.0 px
    static constexpr float kFY()  { return (3.6f / 2.7384f) * static_cast<float>(kImageHeight); }
    static constexpr float kCX()  { return static_cast<float>(kImageWidth)  / 2.0f; }
    static constexpr float kCY()  { return static_cast<float>(kImageHeight) / 2.0f; }
};

// ——————————————————————————————————
// 雷达外参配置（与 Python 版 hardware.py RadarConfig 完全一致）
// 坐标系：X=右  Y=前  Z=上
// ——————————————————————————————————

struct RadarConfig {
    static constexpr float kDeltaX             =  0.00f;  // m，右侧为正
    static constexpr float kDeltaY             =  0.09f;  // m，前方为正（雷达在相机后方 9cm）
    static constexpr float kDeltaZ             = -0.12f;  // m，上方为正（实际飞机 -0.18m，便于测试）
    static constexpr float kCameraPitchDeg     =  0.0f;   // deg，相机俯仰角
    static constexpr float kCameraHeight       =  0.03f;  // m，相机高度
    static constexpr float kAngleToleranceRad   =  0.5f;   // rad，±28.6° 容差
    static constexpr float kScanTopic           =  0.0f;   // ROS scan topic placeholder
};

// ——————————————————————————————————
// AprilTag 配置（与 Python 版 hardware.py AprilTagConfig 完全一致）
// ——————————————————————————————————

struct AprilTagConfig {
    static constexpr float kTagSizeM   = 0.15f;   // m，Tag 边长
    static constexpr char* kTagFamily  = "tag36h11";
};

// ——————————————————————————————————
// CameraParams 工厂：从硬件配置生成完整内参
// 等价于 Python 版的 camera_params tuple
// ——————————————————————————————————

inline CameraParams make_camera_params() {
    CameraParams p;
    p.fx               = CameraConfig::kFX();
    p.fy               = CameraConfig::kFY();
    p.cx               = CameraConfig::kCX();
    p.cy               = CameraConfig::kCY();
    p.delta_x          = RadarConfig::kDeltaX;
    p.delta_y          = RadarConfig::kDeltaY;
    p.delta_z          = RadarConfig::kDeltaZ;
    p.camera_pitch     = RadarConfig::kCameraPitchDeg * CV_PI / 180.0f;
    p.angle_tolerance  = RadarConfig::kAngleToleranceRad;
    p.camera_height    = RadarConfig::kCameraHeight;
    return p;
}

} // namespace fvs
