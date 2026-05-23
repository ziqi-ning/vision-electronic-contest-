"""
UART 串口通信协议配置
从 uartuse.py 提取的协议字段定义
"""

# ========== 帧头与包头 ==========
FRAME_HEAD = bytes([0xFF, 0xFC])   # 数据帧头
PACKAGE_HEAD = bytes([0xFF, 0xFE])  # 包头（解析状态机用）

# ========== 协议功能字 ==========
FUNC_BASE = 0xA0  # 功能字基础值

# ========== 字段偏移与宽度（字节数）==========
FIELD_OFFSETS = {
    "x": 0,
    "y": 2,
    "pixel": 4,
}
FIELD_WIDTHS = {
    "x": 2,
    "y": 2,
    "pixel": 2,
}

# ========== 协议命令字偏移量（协议数组索引）==========
# 对应 package_blobs_data 中各字段在 data 数组中的位置（从 HEADER[0] 开始计）
IDX_HEADER = 0          # 0xFF
IDX_HEADER2 = 1         # 0xFC
IDX_FUNC = 2             # 0xA0 + mode
IDX_LEN = 3              # 数据长度字段
IDX_X_HIGH = 4           # x 高字节
IDX_X_LOW = 5            # x 低字节
IDX_Y_HIGH = 6           # y 高字节
IDX_Y_LOW = 7            # y 低字节
IDX_PIXEL_HIGH = 8       # pixel 高字节
IDX_PIXEL_LOW = 9         # pixel 低字节
IDX_FLAG = 10            # 标志位
IDX_STATE = 11           # 状态
IDX_ANGLE_HIGH = 12      # 角度高字节
IDX_ANGLE_LOW = 13       # 角度低字节
IDX_DIST_HIGH = 14        # 距离高字节
IDX_DIST_LOW = 15         # 距离低字节
IDX_APRILTAG_HIGH = 16   # Apriltag ID 高字节
IDX_APRILTAG_LOW = 17     # Apriltag ID 低字节
IDX_IMG_WIDTH_HIGH = 18   # 图像宽度高字节
IDX_IMG_WIDTH_LOW = 19    # 图像宽度低字节
IDX_IMG_HEIGHT_HIGH = 20  # 图像高度高字节
IDX_IMG_HEIGHT_LOW = 21    # 图像高度低字节
IDX_FPS = 22              # 帧率
IDX_RANGE_S1_HIGH = 26    # 超声波1 前
IDX_RANGE_S1_LOW = 27
IDX_RANGE_S2_HIGH = 28    # 超声波2 左
IDX_RANGE_S2_LOW = 29
IDX_RANGE_S3_HIGH = 30    # 超声波3 右
IDX_RANGE_S3_LOW = 31
IDX_RANGE_S4_HIGH = 32    # 超声波4 后
IDX_RANGE_S4_LOW = 33
IDX_CAMERA_ID = 34        # 相机 ID
IDX_CHECKSUM = -1         # 校验和（最后一字节）
