#pragma once

#include <cstdint>

namespace fvs {

// ——————————————————————————————————
// 工作模式枚举（与 Python 版 modes.py 完全一致）
// ——————————————————————————————————

enum class WorkMode : uint8_t {
    IDLE        = 0x00,  // 红色检测 + 雷达测距
    CIRCLE      = 0x01,  // 白色色块内椭圆检测
    SOUND       = 0x02,  // 纯雷达，无视觉
    IDLE2       = 0x03,  // 绿色色块 + 雷达测距
    APRILTAG    = 0x04,  // AprilTag 位姿估计
    COLOR_BLOB  = 0x05,  // 蓝色色块检测
    BARCODE     = 0x06,  // 条码识别
    QR_2024     = 0x07   // 二维码识别（24年真题）
};

// ——————————————————————————————————
// 形状类型枚举
// ——————————————————————————————————

enum class ShapeType : int {
    UNKNOWN   = -1,
    ELLIPSE   =  0,
    TRAPEZOID =  1,
    TRIANGLE  =  2,
    POLE      =  3
};

constexpr const char* shape_type_to_string(ShapeType s) {
    switch (s) {
        case ShapeType::ELLIPSE:   return "ellipse";
        case ShapeType::TRAPEZOID: return "trapezoid";
        case ShapeType::TRIANGLE:  return "triangle";
        case ShapeType::POLE:       return "pole";
        default:                   return "unknown";
    }
}

// ——————————————————————————————————
// 颜色名称常量
// ——————————————————————————————————

namespace ColorNames {
    constexpr const char* RED       = "red";
    constexpr const char* GREEN     = "green";
    constexpr const char* BLUE      = "blue";
    constexpr const char* YELLOW    = "yellow";
    constexpr const char* BLACK     = "black";
    constexpr const char* WHITE     = "white";
    constexpr const char* CYAN      = "cyan";
    constexpr const char* MAGENTA   = "magenta";
    constexpr const char* ORANGE    = "orange";
    constexpr const char* PURPLE    = "purple";
    constexpr const char* RED_LASER = "red_laser";
    constexpr const char* ALL       = "all";
}

// ——————————————————————————————————
// 协议常量
// ——————————————————————————————————

namespace Protocol {
    constexpr uint8_t  kFrameHead1      = 0xFF;
    constexpr uint8_t  kFrameHead2      = 0xFC;
    constexpr uint8_t  kCommandBase     = 0xA0;
    constexpr uint8_t  kPackageHead1    = 0xFF;
    constexpr uint8_t  kPackageHead2    = 0xFE;
    constexpr uint8_t  kMaxDataLen      = 50;
    constexpr size_t   kPacketSize      = 36;

    // 雷达融合默认值（无数据时返回）
    constexpr int kDefaultDistance = 3000;  // 30m（单位 mm）
    constexpr int kDefaultAngle     = 40000; // 400°（单位 0.01°）

    // 超声波传感器数量
    constexpr size_t kRangeSensorCount = 4;
}

// ——————————————————————————————————
// 相机默认值
// ——————————————————————————————————

namespace Camera {
    constexpr int   kWidth          = 640;
    constexpr int   kHeight         = 480;
    constexpr int   kFPS            = 50;
    constexpr float kFX             = 628.7f;
    constexpr float kFY             = 631.0f;
    constexpr float kCX             = 320.0f;
    constexpr float kCY             = 240.0f;
    constexpr float kTagSizeM       = 0.15f;
}

// ——————————————————————————————————
// 雷达默认值
// ——————————————————————————————————

namespace Radar {
    constexpr float kDeltaX             =  0.00f; // m
    constexpr float kDeltaY             =  0.09f;  // m（雷达在相机后方9cm）
    constexpr float kDeltaZ             = -0.12f;  // m
    constexpr float kCameraPitchDeg     =  0.0f;   // deg
    constexpr float kCameraHeight       =  0.03f;  // m
    constexpr float kAngleToleranceRad  =  0.5f;   // rad
    constexpr float kObstacleClusterThreshold = 0.03f; // 连续3点距离差≤3cm归为一簇
    constexpr const char* kScanTopic = "/scan";
}

// ——————————————————————————————————
// 检测参数默认值
// ——————————————————————————————————

namespace Detection {
    constexpr int   kDefaultBais        = 20;    // HSV 阈值偏移量
    constexpr int   kLaserBais         = 25;    // 激光点检测阈值偏移
    constexpr int   kMinLaserArea      = 0;     // 激光点最小面积
    constexpr int   kMaxLaserArea      = 5000;  // 激光点最大面积
    constexpr int   kErodeIter         = 2;     // 腐蚀迭代次数
    constexpr int   kDilateIter        = 2;     // 膨胀迭代次数
    constexpr float kPoleAngleBucketDeg = 5.0f;  // 杆线霍夫角度分桶间隔（度）
    constexpr int   kPoleMinSpacing    = 5;     // 平行杆线最小间距（px）
    constexpr int   kPoleMaxSpacing    = 70;    // 平行杆线最大间距（px）
    constexpr int   kEllipseMaxOneRadiusRatio = 2; // 第二大椭圆与最大椭圆半径比阈值
}

// ——————————————————————————————————
// 串口默认值
// ——————————————————————————————————

namespace Serial {
    constexpr int      kBaudRate      = 256000;
    constexpr int      kMaxReconnect  = 5;
    constexpr uint64_t kReconnectIntervalMs = 1000;
    constexpr uint64_t kSendPeriodMs  = 50;     // ≤50ms，对应50Hz刷新率
}

// ——————————————————————————————————
// 日志级别
// ——————————————————————————————————

enum class LogLevel : int {
    DEBUG   = 0,
    INFO    = 1,
    WARNING = 2,
    ERROR   = 3
};

} // namespace fvs
