#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fvs {

// ——————————————————————————————————
// 几何基础
// ——————————————————————————————————

struct BoundingBox {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    BoundingBox() = default;
    BoundingBox(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}

    cv::Point2i center() const { return {x + width / 2, y + height / 2}; }
    int area() const { return width * height; }
};

struct RadarPoint {
    float distance_m = 0.0f;
    float angle_rad = 0.0f;
};

// ——————————————————————————————————
// 相机与雷达参数
// ——————————————————————————————————

struct CameraParams {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float delta_x = 0.0f;       // 雷达相对相机 X 偏移（m）
    float delta_y = 0.0f;        // 雷达相对相机 Y 偏移（m）
    float delta_z = 0.0f;        // 雷达相对相机 Z 偏移（m）
    float camera_pitch = 0.0f;   // 相机俯仰角（rad）
    float angle_tolerance = 0.5f; // 角度容差（rad）
    float camera_height = 0.03f; // 相机高度（m）

    // 兼容 Python 版 get_camera_params() 返回的 10 元组顺序
    std::array<float, 10> to_array() const {
        return {fx, fy, cx, cy, delta_x, delta_y, delta_z, camera_pitch, angle_tolerance, camera_height};
    }

    static CameraParams from_hardware_config();
};

// ——————————————————————————————————
// 雷达数据
// ——————————————————————————————————

struct RadarScan {
    double timestamp = 0.0;
    std::vector<RadarPoint> points;
    std::vector<RadarPoint> obstacles;
};

// ——————————————————————————————————
// 检测结果（统一返回类型）
// ——————————————————————————————————

struct DetectionResult {
    std::string type;                     // 检测类型，如 "ellipse"、"trapezoid"
    float confidence = 0.0f;
    std::optional<BoundingBox> bbox;
    std::optional<cv::Point2i> center;   // 像素坐标 (col, row)
    std::unordered_map<std::string, cv::Variant> extra; // 扩展字段
};

// ——————————————————————————————————
// 颜色检测 ROI 结果
// ——————————————————————————————————

struct ColorROIResult {
    std::string color;                    // 颜色名称
    cv::Point2i center{0, 0};             // 色块中心像素坐标
    int area = 0;                          // 面积（像素数）
    BoundingBox bbox;                      // 包围盒
    cv::Mat mask;                          // 对应掩模
};

// ——————————————————————————————————
// 形状检测结果
// ——————————————————————————————————

struct ShapeResult {
    std::string type;                      // "ellipse" / "trapezoid" / "triangle" / "pole"
    cv::Point2i center{0, 0};
    float confidence = 0.0f;
    int area = 0;
    BoundingBox bbox;
    cv::RotatedRect rotated_rect;          // 椭圆拟合的旋转矩形（用于 ellipse）
    std::vector<cv::Point2i> contour;      // 原始轮廓点
    std::vector<float> side_lengths;       // 梯形/三角形的边长列表
    std::vector<cv::Vec4i> pole_lines;     // 杆线的霍夫线段对 (x1,y1,x2,y2)
};

// ——————————————————————————————————
// 雷达融合结果
// ——————————————————————————————————

struct FusionResult {
    cv::Point2i pixel{0, 0};              // 输入的像素坐标
    std::optional<float> distance_m;        // 融合后的距离（m）
    std::optional<float> angle_rad;         // 雷达角度（rad）
    bool obstacle_detected = false;
};

// ——————————————————————————————————
// AprilTag 检测结果
// ——————————————————————————————————

struct AprilTagResult {
    int tag_id = -1;
    cv::Point2i center{0, 0};
    cv::Vec3d tvec{0, 0, 0};  // 位移向量 (x, y, z) 单位 m
    cv::Vec3d rvec{0, 0, 0};  // 旋转向量
    float distance_m = 0.0f;
    float angle_rad = 0.0f;
};

// ——————————————————————————————————
// 二维码/条码检测结果
// ——————————————————————————————————

struct MarkerResult {
    int flag = 0;                         // 0=未检测到，1=检测到
    int x = 0;
    int y = 0;
    int pixel = 0;                        // 面积
    std::string data;                      // 编码内容
    cv::Mat debug_frame;                  // 调试用叠加帧
};

// ——————————————————————————————————
// UART 目标数据结构（36字节协议）
// ——————————————————————————————————

#pragma pack(push, 1)
struct TargetData {
    uint8_t  frame_head1 = 0xFF;
    uint8_t  frame_head2 = 0xFC;
    uint8_t  func_code   = 0;   // 0xA0 + mode
    uint8_t  length       = 0;
    int16_t  x           = 0;
    int16_t  y           = 0;
    int16_t  pixel       = 0;
    uint8_t  flag        = 0;
    uint8_t  state       = 0;
    int16_t  angle       = 0;
    int16_t  distance    = 0;
    int16_t  apriltag_id = 0;
    int16_t  img_width   = 640;
    int16_t  img_height  = 480;
    uint8_t  fps         = 0;
    uint8_t  reserved1   = 0;
    uint8_t  reserved2   = 0;
    uint8_t  reserved3   = 0;
    int16_t  range_s1    = 0;   // 超声波1 前
    int16_t  range_s2    = 0;   // 超声波2 左
    int16_t  range_s3    = 0;   // 超声波3 右
    int16_t  range_s4    = 0;   // 超声波4 后
    uint8_t  camera_id   = 0x02;
    uint8_t  checksum   = 0;

    TargetData() = default;

    static constexpr size_t kSize = 36;
    static constexpr uint8_t kFrameHead1 = 0xFF;
    static constexpr uint8_t kFrameHead2 = 0xFC;
    static constexpr uint8_t kCommandBase = 0xA0;
    static constexpr uint8_t kPackageHead1 = 0xFF;
    static constexpr uint8_t kPackageHead2 = 0xFE;

    void set_func_code(uint8_t mode) { func_code = static_cast<uint8_t>(kCommandBase + mode); }
    uint8_t get_mode() const { return func_code > kCommandBase ? func_code - kCommandBase : 0; }

    uint8_t compute_checksum() const;
    void fill_and_seal();
    std::vector<uint8_t> to_bytes() const;
};
#pragma pack(pop)

// ——————————————————————————————————
// HSV 颜色阈值表（从 YAML 加载）
// ——————————————————————————————————

struct HSVThreshold {
    cv::Scalar lower;
    cv::Scalar upper;
};

using HSVThresholdMap = std::unordered_map<std::string, HSVThreshold>;

HSVThresholdMap default_hsv_thresholds();

} // namespace fvs
