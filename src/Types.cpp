#include "Types.h"
#include "Version.h"

#include <algorithm>
#include <numeric>

namespace fvs {

CameraParams CameraParams::from_hardware_config() {
    CameraParams p;
    p.fx              = Camera::kFX;
    p.fy              = Camera::kFY;
    p.cx              = Camera::kCX;
    p.cy              = Camera::kCY;
    p.delta_x         = Radar::kDeltaX;
    p.delta_y         = Radar::kDeltaY;
    p.delta_z         = Radar::kDeltaZ;
    p.camera_pitch     = Radar::kCameraPitchDeg * CV_PI / 180.0f;
    p.angle_tolerance  = Radar::kAngleToleranceRad;
    p.camera_height    = Radar::kCameraHeight;
    return p;
}

uint8_t TargetData::compute_checksum() const {
    std::array<uint8_t, kSize> arr;
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(this);
    std::copy(ptr, ptr + kSize, arr.begin());
    // checksum 字段自身不参与累加（即最后一字节不加入）
    uint8_t sum = 0;
    for (size_t i = 0; i < kSize - 1; ++i) {
        sum += arr[i];
    }
    return sum & 0xFF;
}

void TargetData::fill_and_seal() {
    length = static_cast<uint8_t>(kSize - 5);
    checksum = compute_checksum();
}

std::vector<uint8_t> TargetData::to_bytes() const {
    return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(this),
                                reinterpret_cast<const uint8_t*>(this) + kSize);
}

// ——————————————————————————————————
// 默认 HSV 阈值表（对应 Python 版 scene.yaml）
// ——————————————————————————————————

HSVThresholdMap default_hsv_thresholds() {
    using namespace cv;
    HSVThresholdMap thresholds;
    thresholds["red"]        = {{0, 100, 100},   {10, 255, 255}};
    thresholds["red_laser"] = {{0, 0, 200},      {180, 30, 255}};
    thresholds["green"]      = {{40, 50, 50},     {80, 255, 255}};
    thresholds["blue"]       = {{90, 80, 50},     {130, 255, 255}};
    thresholds["yellow"]     = {{20, 100, 100},   {30, 255, 255}};
    thresholds["black"]      = {{0, 0, 0},        {180, 255, 50}};
    thresholds["white"]      = {{0, 0, 150},      {180, 30, 255}};
    thresholds["cyan"]       = {{80, 80, 50},     {100, 255, 255}};
    thresholds["magenta"]    = {{140, 80, 50},    {160, 255, 255}};
    thresholds["orange"]     = {{10, 100, 100},   {25, 255, 255}};
    thresholds["purple"]     = {{120, 80, 50},    {150, 255, 255}};
    thresholds["all"]        = {{0, 0, 0},         {180, 255, 255}};
    return thresholds;
}

} // namespace fvs
