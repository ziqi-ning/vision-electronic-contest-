#pragma once

#include <cstddef>
#include <cstdint>

namespace fvs {

// ——————————————————————————————————
// UART 36 字节协议字段定义
// 与 Python 版 protocol.py 完全一致
// ——————————————————————————————————

// 帧头
constexpr uint8_t  kProtocolFrameHead1 = 0xFF;  // 帧头字节1
constexpr uint8_t  kProtocolFrameHead2 = 0xFC;  // 帧头字节2
constexpr uint8_t  kProtocolPackageHead1 = 0xFF; // 包头（状态机解析用）
constexpr uint8_t  kProtocolPackageHead2 = 0xFE; // 包头（状态机解析用）
constexpr uint8_t  kProtocolFuncBase = 0xA0;     // 功能字基础值

// ——————————————————————————————————
// 协议字段偏移量（从数据包起始地址计算）
// 对应 Python 版的 IDX_* 常量
// ——————————————————————————————————

namespace ProtocolFieldIdx {
    constexpr size_t kHeader     = 0;  // 0xFF
    constexpr size_t kHeader2    = 1;  // 0xFC
    constexpr size_t kFunc       = 2;  // 0xA0 + mode
    constexpr size_t kLength     = 3;  // 数据长度字段
    constexpr size_t kXHigh      = 4;  // x 高字节
    constexpr size_t kXLow       = 5;  // x 低字节
    constexpr size_t kYHigh      = 6;  // y 高字节
    constexpr size_t kYLow       = 7;  // y 低字节
    constexpr size_t kPixelHigh  = 8;  // pixel 高字节
    constexpr size_t kPixelLow   = 9;  // pixel 低字节
    constexpr size_t kFlag       = 10; // 标志位
    constexpr size_t kState      = 11; // 状态
    constexpr size_t kAngleHigh  = 12; // 角度高字节
    constexpr size_t kAngleLow   = 13; // 角度低字节
    constexpr size_t kDistHigh   = 14; // 距离高字节
    constexpr size_t kDistLow    = 15; // 距离低字节
    constexpr size_t kApriltagHigh = 16; // Apriltag ID 高字节
    constexpr size_t kApriltagLow  = 17; // Apriltag ID 低字节
    constexpr size_t kImgWidthHigh = 18; // 图像宽度高字节
    constexpr size_t kImgWidthLow  = 19; // 图像宽度低字节
    constexpr size_t kImgHeightHigh = 20; // 图像高度高字节
    constexpr size_t kImgHeightLow  = 21; // 图像高度低字节
    constexpr size_t kFPS         = 22; // 帧率
    constexpr size_t kReserved1  = 23; // reserved
    constexpr size_t kReserved2  = 24; // reserved
    constexpr size_t kReserved3  = 25; // reserved
    constexpr size_t kRangeS1High = 26; // 超声波1 前
    constexpr size_t kRangeS1Low  = 27;
    constexpr size_t kRangeS2High = 28; // 超声波2 左
    constexpr size_t kRangeS2Low  = 29;
    constexpr size_t kRangeS3High = 30; // 超声波3 右
    constexpr size_t kRangeS3Low  = 31;
    constexpr size_t kRangeS4High = 32; // 超声波4 后
    constexpr size_t kRangeS4Low  = 33;
    constexpr size_t kCameraId    = 34; // 相机 ID
    constexpr size_t kChecksum   = 35; // 校验和（最后一字节）

    // 协议包总长
    constexpr size_t kPacketSize = 36;
}

// ——————————————————————————————————
// 协议包字段宽度（字节数）
// ——————————————————————————————————

namespace ProtocolFieldWidth {
    constexpr size_t kX         = 2;
    constexpr size_t kY         = 2;
    constexpr size_t kPixel     = 2;
    constexpr size_t kAngle     = 2;
    constexpr size_t kDistance  = 2;
    constexpr size_t kApriltagId = 2;
    constexpr size_t kImgWidth  = 2;
    constexpr size_t kImgHeight = 2;
    constexpr size_t kRangeS    = 2;   // 每个超声波
}

// ——————————————————————————————————
// 辅助函数：从高/低字节组装 16 位有符号整数（与 Python struct.unpack('>h') 等价）
// Big-endian: 高字节在前，低字节在后
// ——————————————————————————————————

inline int16_t unpack_int16_le(uint8_t lo, uint8_t hi) {
    return static_cast<int16_t>(static_cast<uint16_t>(lo) |
                                (static_cast<uint16_t>(hi) << 8));
}

inline int16_t unpack_int16_be(uint8_t hi, uint8_t lo) {
    return unpack_int16_le(lo, hi);
}

// ——————————————————————————————————
// 辅助函数：将 16 位有符号整数拆分为高/低字节
// Big-endian: 高字节在前，低字节在后
// ——————————————————————————————————

inline void pack_int16_be(int16_t value, uint8_t& hi, uint8_t& lo) {
    uint16_t u = static_cast<uint16_t>(value);
    hi = static_cast<uint8_t>((u >> 8) & 0xFF);
    lo = static_cast<uint8_t>(u & 0xFF);
}

// ——————————————————————————————————
// 校验和计算：累加所有字节（不含 checksum 字段本身），取低 8 位
// 与 Python 版 compute_checksum() 完全等价
// ——————————————————————————————————

inline uint8_t compute_protocol_checksum(const uint8_t* data, size_t len) {
    uint8_t sum = 0;
    for (size_t i = 0; i < len - 1; ++i) {  // 不含最后一个 checksum 字节
        sum += data[i];
    }
    return sum & 0xFF;
}

} // namespace fvs
