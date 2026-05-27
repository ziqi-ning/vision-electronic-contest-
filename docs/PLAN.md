# FVS-Cpp 施工计划

> **文档类型**：执行方案\
> **基于**：SPEC.md 需求规格\
> **总工期估算**：Phase 0-5，约 7~12 周\
> **分支策略**：`cpp-restruct`，从 `main` 分出

---

## 一、分 Phase 总览

```
Phase 0 ─── 项目骨架（1周，5个子任务）
  └─ T0.1 ~ T0.5

Phase 1 ─── 通信层（1周，3个子任务）
  └─ T1.1 ~ T1.3

Phase 2 ─── 检测层（2~3周，4个子任务）【最高风险】
  └─ T2.1 ~ T2.4

Phase 3 ─── 融合层（1~2周，3个子任务）
  └─ T3.1 ~ T3.3

Phase 4 ─── 编排层 + 模式层（1~2周，4个子任务）
  └─ T4.1 ~ T4.4

Phase 5 ─── 测试 + 优化（1~2周，4个子任务）
  └─ T5.1 ~ T5.4
```

**关键路径**：T0.1 → T1.1 → T2.1 → T3.1 → T4.3 → T5.2

---

## 二、Phase 0：项目骨架

**目标**：建立 C++ 项目基础结构，不改任何核心逻辑。

### T0.1 建立 CMake 项目结构

```
FVS-Cpp/
├── CMakeLists.txt              # 顶层构建（Debug/Release + sanitizer）
├── conanfile.txt               # 依赖管理（OpenCV 4.8+ / yaml-cpp / spdlog）
├── src/
│   ├── main.cpp
│   ├── config/                 # 相机/雷达/协议配置
│   ├── detection/              # 检测层
│   ├── fusion/                 # 融合层
│   ├── comm/                   # 通信层
│   ├── modes/                  # 模式层
│   ├── pipeline/              # 编排层
│   ├── hardware/              # 外设层
│   └── utils/                 # 工具层
├── include/                    # 公共头文件
├── tests/
│   ├── unit/
│   └── integration/
├── config/                     # 运行时配置（YAML）
└── docs/
```

CMakeLists.txt 关键配置：
- C++17 标准
- Debug: `-fsanitize=address,undefined`；Release: LTO 优化
- Unity Build 加速编译
- clang-tidy 集成

### T0.2 建立统一类型系统

直接从 Python 版的 `src/core/types.py` 翻译为 C++ 头文件 `include/Types.h`：

```cpp
// 核心数据结构
struct BoundingBox { int x, y, width, height; };
struct DetectionResult { string type; float confidence; cv::Point2i center; ... };
struct RadarPoint { float distance_m; float angle_rad; };
struct FusionResult { cv::Point2i pixel; std::optional<float> distance_m; ... };
struct TargetData { int x, y, pixel, flag, angle, distance, ...; };
struct CameraParams { float fx, fy, cx, cy, delta_x, ...; };
```

同时建立枚举：
```cpp
enum class WorkMode : uint8_t { IDLE=0, CIRCLE=1, SOUND=2, IDLE2=3,
                                 APRILTAG=4, COLOR_BLOB=5, BARCODE=6, QR_2024=7 };
enum class ShapeType : int { ELLIPSE=0, TRAPEZOID=1, TRIANGLE=2, POLE=3, UNKNOWN=-1 };
```

### T0.3 建立配置加载层

翻译 Python 版 `src/config/` 四个配置模块为 C++：

| Python 文件 | C++ 文件 | 职责 |
|------------|---------|------|
| `hardware.py` | `src/config/HardwareConfig.h` | 相机内参（fx/fy/cx/cy）、雷达外参（delta_x/y/z） |
| `scene.py` | `src/config/SceneConfig.h/.cpp` | HSV阈值表（12种颜色）、形态学参数，从YAML加载 |
| `protocol.py` | `src/config/Protocol.h` | 帧头（0xFF 0xFC）、字段偏移/宽度、包长（36字节） |
| `modes.py` | `include/Version.h` | 8种工作模式枚举 |

配置结构体使用 YAML-CPP 加载，与 Python 版的 `scene.yaml` 完全兼容。

### T0.4 建立日志系统

使用 spdlog 封装，建立 `src/utils/Logger.h`：

```cpp
class Logger {
public:
    static Logger& instance();
    void debug(const char* fmt, ...);
    void info(const char* fmt, ...);
    void warn(const char* fmt, ...);
    void error(const char* fmt, ...);
};
```

日志级别语义与 Python 版完全一致：
- `DEBUG`：每帧 FPS、检测数量、中间变量
- `INFO`：模式切换、串口收发、心跳
- `WARNING`：降级操作（雷达断线切换SimRadarSource）
- `ERROR`：异常堆栈

### T0.5 建立工具层

| 文件 | 职责 |
|------|------|
| `src/utils/RingBuffer.h` | 无锁环形队列（单生产者/单消费者），用于帧缓冲 |
| `src/utils/MathUtils.h` | NumPy 等效函数：`atan2`、`diff`、`mean`、`median`、`norm`、`deg2rad` |
| `src/utils/Profiler.h` | 轻量性能分析器（计时器，日志输出各模块耗时） |

**Phase 0 验收标准**：
- `cmake .. && make` 编译通过，无警告
- `make test` 运行所有 GoogleTest 测试（目前仅有占位测试）
- `spdlog` 日志输出正常

---

## 三、Phase 1：通信层

**目标**：实现 UART 串口通信，与下位机固件联调通过。

### T1.1 SerialPort C++ 实现

基于 Boost.Asio 实现异步串口，协议严格照搬 Python 版：

```cpp
class SerialPort {
public:
    bool open(const string& port, int baudrate = 256000);
    void close();
    bool send(const TargetData& data);     // 组36字节包发送
    int  readMode();                        // 读模式码（0x00~0x07）
    int  getMode() const;

private:
    void startReadHeader();                 // 状态机：等0xFF 0xFE帧头
    void startReadPayload(size_t len);      // 读后续字节
    void computeChecksum(const uint8_t* data, size_t len);
};
```

36字节数据包结构（`#pragma pack(push,1)`）：

| 偏移 | 字段 | 字节数 |
|------|------|--------|
| 0 | 0xFF | 1 |
| 1 | 0xFC | 1 |
| 2 | 0xA0 + mode | 1 |
| 3 | length | 1 |
| 4-5 | x | 2 |
| 6-7 | y | 2 |
| 8-9 | pixel | 2 |
| 10 | flag | 1 |
| 11 | state | 1 |
| 12-13 | angle | 2 |
| 14-15 | distance | 2 |
| 16-17 | apriltag_id | 2 |
| 18-19 | img_width | 2 |
| 20-21 | img_height | 2 |
| 22 | fps | 1 |
| 23-25 | reserved | 3 |
| 26-27 | range_s1（超声波前） | 2 |
| 28-29 | range_s2（左） | 2 |
| 30-31 | range_s3（右） | 2 |
| 32-33 | range_s4（后） | 2 |
| 34 | camera_id | 1 |
| 35 | checksum（最后一字节累加和低8位） | 1 |

### T1.2 LEDController C++ 实现

基于 pigpio / WiringPi 实现 GPIO 控制：

```cpp
class LEDController {
public:
    enum class Color { INDIGO, PURPLE, YELLOW, BLUE, GREEN, RED, EMPTY, WHITE };
    void setColor(Color c);   // RGB三色GPIO组合，对应7种颜色
    void clear();             // 全灭
};
```

GPIO 引脚编码与 Python 版 `facility2.py` 完全一致。

### T1.3 串口联调验证

- 在开发机（Windows）上用 USB转TTL 模块连接测试
- 或在 Linux 上用 `/dev/pts/N` 模拟下位机
- 验证：Python版发送 ←→ C++版接收，逐字节完全一致

**Phase 1 验收标准**：
- 串口发送的数据包与 Python 版 `uartuse.py` 逐字节一致
- 模式码读取正确（0x00~0x07）
- 校验和计算正确

---

## 四、Phase 2：检测层（最高风险）

**目标**：完成所有视觉检测算法的 C++ 实现，与 Python 版逐帧对比结果一致。

### T2.1 ColorDetector C++ 实现

翻译 `src/colorblob.py` → `src/detection/ColorDetector.h/.cpp`：

```cpp
class ColorDetector {
public:
    // 入口函数，等价于 colorblob.detect_color()
    std::vector<ColorROIResult> detect(const cv::Mat& frame,
                                       const std::string& color,
                                       int bais = 20);

    // 辅助函数
    cv::Mat  createMask(const cv::Mat& hsv, const std::string& color);
    cv::Mat  erodeDilate(const cv::Mat& mask, int erode_iter=2, int dilate_iter=2);
    cv::Mat  compositeROI(const cv::Mat& frame, const std::vector<ColorROIResult>& results);
    cv::Mat  drawResults(const cv::Mat& frame, const std::vector<ColorROIResult>& results);

    // 激光点检测，等价于 colorblob.detect_laser()
    std::tuple<int, cv::Mat, cv::Point2i, int>
        detectLaser(const cv::Mat& roi, int light_bais=25,
                    int min_area=0, int max_area=5000);

    // 多颜色ROI，等价于 allin.color_bonous_multi_color()
    cv::Mat compositeMultiColor(const cv::Mat& frame,
                                const std::string& color1,
                                const std::string& color2);
};
```

**关键实现细节**：
- HSV阈值：`std::unordered_map<std::string, std::pair<cv::Scalar, cv::Scalar>>` 查找表
- 形态学：`cv::morphologyEx(mask, MORPH_OPEN, kernel)` 一行替代 erode+dilate
- 轮廓迭代：严格按 Python 顺序（面积排序→取最大→计算矩→圆拟合→排序）
- `detect_laser`：`cv::inRange(hsv, lower_white, upper_white)` 后取最亮像素位置

### T2.2 ShapeRecognizer C++ 实现

翻译 `src/outsite.py` → `src/detection/ShapeRecognizer.h/.cpp`：

```cpp
class ShapeRecognizer {
public:
    // 等价于 outsite.detect_ellipses()（max_one=False）
    std::vector<ShapeResult>
        detectEllipses(const cv::Mat& roi, const cv::Point2i& offset = {0,0});

    // 等价于 outsite.detect_ellipses()（max_one=True）
    std::vector<ShapeResult>
        detectEllipsesMaxOne(const cv::Mat& roi, const cv::Point2i& offset = {0,0});

    // 等价于 outsite.detect_trapezoids()
    std::vector<ShapeResult>
        detectTrapezoids(const cv::Mat& roi, const cv::Point2i& offset = {0,0});

    // 等价于 outsite.detect_triangle()
    std::vector<ShapeResult>
        detectTriangle(const cv::Mat& roi, const cv::Point2i& offset = {0,0});

    // 等价于 outsite.find_longest_straight_line()
    std::vector<ShapeResult>
        findPoleLines(const cv::Mat& roi, const cv::Point2i& offset = {0,0});
};
```

**高风险实现点**（需对照 Python 版仔细验证）：

1. **`detect_ellipses_max_one`**：第二大椭圆筛选（半径差异过大则丢弃）
2. **`detect_trapezoids`**：
   - 四边形角度余弦计算（numpy.angle 等效实现）
   - 梯形面积排序（`std::sort` vs Python `sorted(list, key=lambda x: x[1])`）
   - 第二、三、四长边长度排序
3. **`detect_triangle`**：
   - 三边长度计算（`sqrt(dx²+dy²)`）
   - 旋转不变性处理逻辑（`rotate_times` 计数）
4. **`find_pole_lines`**：
   - 霍夫变换后角度分桶（5度间隔）
   - 每桶内质量排序
   - 平行线间距计算（`abs(x1-x2)`）
   - pole_groups 嵌套结构映射

### T2.3 MarkerDetector C++ 实现

翻译 `src/other.py` → `src/detection/MarkerDetector.h/.cpp`：

```cpp
class MarkerDetector {
public:
    // 等价于 other.QR_detect()
    std::tuple<int, cv::Mat, int/*flag*/, int/*x*/, int/*y*/, int/*pixel*/, int/*data*/>
        detectQR(const cv::Mat& frame);

    // 等价于 other.opencv_find_april_tag()
    std::tuple<int, int, int, float, int, float, float, float>
        detectAprilTag(const cv::Mat& frame,
                       const CameraParams& cam_info,
                       apriltag::Detector& detector);

    // 等价于 other.decodeDisplay()
    std::tuple<cv::Mat, int, int, std::string, int/*flag*/>
        detectBarcode(const cv::Mat& frame);
};
```

注意：AprilTag C++ SDK 的位姿估计 API 与 Python 版 `apriltag.Detector` 不同，需单独适配。

### T2.4 检测层逐帧对比验证

使用合成视频（与 Python 版相同场景）：

```
测试脚本：
  1. Python版: pipeline.run(frame) → type_list_python
  2. C++版:   pipeline.run(frame) → type_list_cpp
  3. 对比:    逐帧比较 type_list 长度、中心坐标（±3px）、形状类型（完全一致）
```

通过标准：100帧视频中，检测结果偏差超过阈值的不超过3帧。

**Phase 2 验收标准**：
- 所有检测函数返回值与 Python 版像素级一致
- `detect_ellipses()` 第二大椭圆筛选行为一致
- `find_pole_lines()` 平行线分组逻辑一致

---

## 五、Phase 3：融合层

**目标**：雷达数据获取和相机-雷达融合算法 C++ 实现。

### T3.1 RadarSource 抽象基类

翻译 `src/radar/base.py` + `ros_source.py` + `sim_source.py`：

```cpp
// 抽象基类，对应 Python 版的 RadarSource
class RadarSource {
public:
    virtual ~RadarSource() = default;
    virtual bool init() = 0;                          // 初始化
    virtual RadarScanResult getScan() = 0;             // 获取一帧雷达数据
    virtual bool isHealthy() const = 0;                 // 健康状态检测
};

// ROS数据源，对应 Python 版的 ROSRadarSource
class ROSRadarSource : public RadarSource {
    // 使用 roscpp 订阅 /scan 话题
};

// 模拟数据源，对应 Python 版的 SimRadarSource
class SimRadarSource : public RadarSource {
    // 从文件加载或程序生成模拟雷达数据
};

// 自动检测，对应 Python 版的 _auto_detect_source()
std::unique_ptr<RadarSource> createRadarSource();
```

ROS 集成注意：`find_package(roscpp REQUIRED)`，ROS 不可用时自动回退到 SimRadarSource。

### T3.2 RadarFusion 融合核心

翻译 `src/radar/fusion.py` → `src/fusion/RadarFusion.h/.cpp`：

```cpp
class RadarFusion {
public:
    explicit RadarFusion(std::unique_ptr<RadarSource> source = nullptr);

    // 等价于 Python 版的 angle_to_distance()
    std::pair<int, int> angleToDistance(float start_deg, float end_deg);

    // 等价于 Python 版的 site_to_distance()
    std::pair<int, int> siteToDistance(float u, float v, const CameraParams& params);

    // 等价于 Python 版的 get_obstacle()
    std::vector<std::pair<int, int>> getObstacles();

private:
    // 对应 Python 版的 _detect_obstacles()
    std::vector<RadarPoint> detectObstacleClusters(const std::vector<RadarPoint>& points);

    std::unique_ptr<RadarSource> source_;
};
```

**关键算法点**：
- `siteToDistance` 中的相机坐标系转换（射线与地面交点计算）
- `_detect_obstacles` 中的点群聚类（连续3点距离差≤0.03m归为一簇）
- 无数据时返回 `(3000, 40000)`（30m / 400°）

### T3.3 融合结果对比验证

固定像素坐标输入（320, 240），对比 Python 和 C++ 版的距离输出，偏差 ≤ 5cm。

**Phase 3 验收标准**：
- 自动数据源切换正确（ROS可用/不可用）
- `siteToDistance()` 返回值与 Python 版误差 ≤ 5cm
- 点群聚类数量一致

---

## 六、Phase 4：编排层 + 模式层

**目标**：Pipeline 编排和8种工作模式的 C++ 实现，主程序完成。

### T4.1 ModeHandler 抽象基类

翻译 `src/modes/base.py` 的虚基类设计：

```cpp
class ModeHandler {
public:
    virtual ~ModeHandler() = default;
    virtual WorkMode mode() const = 0;
    virtual void process(const cv::Mat& frame) = 0;
    virtual void setOutputCallback(OutputCallback cb) = 0;

protected:
    void sendResult(const TargetData& data);
    std::optional<FusionResult> fuseWithRadar(const cv::Point2i& pixel);

    SerialPort* serial_ = nullptr;
    RadarFusion* radar_ = nullptr;
};
```

### T4.2 各 ModeHandler 实现

翻译 `src/modes/idle_mode.py`、`qr_mode.py`、`stub_modes.py`：

```cpp
class IdleMode      : public ModeHandler { WorkMode mode() const override { return WorkMode::IDLE; } };
class CircleMode    : public ModeHandler { WorkMode mode() const override { return WorkMode::CIRCLE; } };
class SoundMode     : public ModeHandler { WorkMode mode() const override { return WorkMode::SOUND; } };
class Idle2Mode     : public ModeHandler { WorkMode mode() const override { return WorkMode::IDLE2; } };
class AprilTagMode  : public ModeHandler { WorkMode mode() const override { return WorkMode::APRILTAG; } };
class ColorBlobMode : public ModeHandler { WorkMode mode() const override { return WorkMode::COLOR_BLOB; } };
class BarcodeMode   : public ModeHandler { WorkMode mode() const override { return WorkMode::BARCODE; } };
class QRMode        : public ModeHandler { WorkMode mode() const override { return WorkMode::QR_2024; } };
```

### T4.3 DetectionPipeline 编排器

翻译 `src/pipeline/orchestrator.py` + `roi_extractor.py` + `shape_classifier.py`：

```cpp
class ROIExtractor {  // 抽象基类
public:
    virtual ~ROIExtractor() = default;
    virtual std::vector<cv::Mat> extract(const cv::Mat& frame,
                                          const std::vector<ColorROIResult>& color_results) = 0;
};

class CircleROIExtractor : public ROIExtractor { ... };
class RectROIExtractor  : public ROIExtractor { ... };
class ORBROIExtractor  : public ROIExtractor { ... };

class ShapeClassifier {
public:
    std::vector<ShapeResult> classify(const cv::Mat& roi, const cv::Point2i& offset);
};

class DetectionPipeline {
public:
    explicit DetectionPipeline(std::unique_ptr<ROIExtractor> roi_extractor = nullptr);

    // 等价于 Python 版的 pipeline.run()
    std::pair<std::vector<ShapeResult>, cv::Mat>
        run(const cv::Mat& frame, const std::string& color, int bais = 20);

private:
    std::unique_ptr<ROIExtractor> roi_extractor_;
    std::unique_ptr<ShapeClassifier> shape_classifier_;
    std::unique_ptr<ColorDetector> color_detector_;
};
```

### T4.4 主程序 main.cpp

对应 Python 版的 `main.py`，asyncio 双协程 → C++ 多线程：

```cpp
int main() {
    // 初始化
    auto serial = std::make_unique<SerialPort>();
    serial->open("/dev/ttyAMA4", 256000);

    auto radar_source = createRadarSource();   // 自动检测 ROS/Sim
    auto radar = std::make_unique<RadarFusion>(std::move(radar_source));

    auto pipeline = std::make_unique<DetectionPipeline>();

    std::map<WorkMode, std::unique_ptr<ModeHandler>> handlers;
    // ... 注册8种ModeHandler

    // 线程模型
    RingBuffer<2> frame_queue;  // 最多2帧缓冲
    std::atomic<bool> running{true};

    // 取帧线程
    std::thread grabber([&]() {
        cv::VideoCapture cap(0);
        cap.set(CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cv::Mat frame;
        while (running) {
            cap >> frame;
            frame_queue.push(std::move(frame));
        }
    });

    // 处理线程
    while (running) {
        cv::Mat frame;
        if (frame_queue.pop(frame)) {
            int mode = serial->readMode();
            auto it = handlers.find(static_cast<WorkMode>(mode));
            if (it != handlers.end()) {
                it->second->process(frame);
            }
        }
    }

    grabber.join();
    return 0;
}
```

**Phase 4 验收标准**：
- 8种模式均能正确处理（idle/qr 优先验证）
- 主程序帧率 ≥ 50 FPS（开发机x86基准）
- 串口数据与 Python 版一致

---

## 七、Phase 5：测试 + 优化

### T5.1 单元测试

GoogleTest 覆盖各模块，关键测试用例：

```cpp
// test_color_detector.cpp
TEST(ColorDetector, DetectRedBlock) { ... }
TEST(ColorDetector, EmptyFrameReturnsEmpty) { ... }
TEST(ColorDetector, LaserDetection) { ... }

// test_shape_recognizer.cpp
TEST(ShapeRecognizer, EllipseDetection) { ... }
TEST(ShapeRecognizer, TrapezoidDetection) { ... }
TEST(ShapeRecognizer, TriangleDetection) { ... }
TEST(ShapeRecognizer, PoleLineDetection) { ... }

// test_fusion.cpp
TEST(RadarFusion, SiteToDistance) { ... }
TEST(RadarFusion, AngleToDistance) { ... }

// test_protocol.cpp
TEST(Protocol, PacketChecksum) { ... }
TEST(Protocol, PacketFieldsAlignment) { ... }
```

### T5.2 集成测试

基于合成视频逐帧对比 Python vs C++ 输出：

```cpp
// test_integration.cpp
TEST_F(IntegrationTest, ColorDetectionOnSyntheticVideo) { ... }
TEST_F(IntegrationTest, ShapeRecognitionOnSyntheticVideo) { ... }
TEST_F(IntegrationTest, FullPipelineOnAllScenes) { ... }
```

### T5.3 性能优化

- 使用 `perf` / `gprof` 分析热点
- SIMD 优化 HSV 掩模生成（用 `cv::Vec3b` 逐像素遍历替换 `cv::inRange` 的热点路径）
- 内存池减少 `cv::Mat` 分配开销

### T5.4 嵌入式交叉编译验证

```bash
# 在 Linux 开发机上交叉编译 ARM 版
cmake .. -DCMAKE_TOOLCHAIN_FILE=toolchain/arm-linux-gnueabihf.cmake
make -j$(nproc)
scp fvs_main pi@raspberrypi.local:~/
```

**Phase 5 验收标准**：
- 所有单元测试通过
- 集成测试在8个合成视频场景上通过率 ≥ 95%
- ARM 平台（树莓派4）帧率 ≥ 40 FPS

---

## 八、执行顺序

```
T0.1 (CMake骨架)
  ↓ T0.2 (类型系统)
  ↓ T0.3 (配置层)
  ↓ T0.4 (日志)
  ↓ T0.5 (工具层)
  ↓ T1.1 (串口) ←── 高优先级，协议固定，无变化风险
  ↓ T1.2 (LED)
  ↓ T1.3 (串口联调)
  ↓ T2.1 (颜色检测) ←── 较高优先级
  ↓ T2.2 (形状识别) ←── 最高风险，最后验证
  ↓ T2.3 (特殊标记)
  ↓ T2.4 (逐帧对比验证)
  ↓ T3.1 (RadarSource抽象)
  ↓ T3.2 (RadarFusion核心)
  ↓ T3.3 (融合对比验证)
  ↓ T4.1 (ModeHandler基类)
  ↓ T4.2 (各ModeHandler)
  ↓ T4.3 (Pipeline编排)
  ↓ T4.4 (main.cpp)
  ↓ T5.1 (单元测试)
  ↓ T5.2 (集成测试)
  ↓ T5.3 (性能优化)
  ↓ T5.4 (交叉编译)
```

**每个 Phase 完成后打一个 git commit，保留可回滚的工作版本。**

---

## 九、风险评估

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| T2.2 形状识别数值精度不一致 | 高 | 高 | 建立逐像素对比测试套件 |
| asyncio → 多线程引入竞态条件 | 高 | 高 | RingBuffer 无锁设计，充分测试并发场景 |
| AprilTag C++ SDK API 差异 | 中 | 中 | Phase 2 T2.3 单独测试 API 兼容性 |
| 嵌入式平台交叉编译工具链复杂 | 中 | 中 | Phase 0 同步建立 Docker 交叉编译镜像 |
| NumPy → C++ 数学函数重写出错 | 中 | 中 | 参照 Python 版逐函数验证 |
| 工期估算偏差 | 中 | 中 | 按1.5倍估算（7~12周 → 10~18周） |
