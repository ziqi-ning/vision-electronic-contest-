#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <string>
#include <unordered_map>
#include <optional>

namespace fvs {

// ——————————————————————————————————
// 运行时配置加载器：从 YAML 文件加载场景参数
// 与 Python 版 scene.py 的 load_scene_config() 完全等价
// ——————————————————————————————————

struct SceneConfigLoader {
    // 加载 config/scene.yaml，返回配置对象；如果文件不存在或解析失败返回 std::nullopt
    static std::optional<SceneConfigLoader> load(const std::string& config_path);
    static std::optional<SceneConfigLoader> load_default();

    // —— HSV 阈值表（从 YAML 的 colors 字段读取）——
    HSVThresholdMap hsv_thresholds;

    // —— 形态学参数（从 YAML 的 morphology 字段读取）——
    int morphology_erode_iter  = 2;
    int morphology_dilate_iter = 2;

    // —— CLAHE 参数（从 YAML 的 clahe 字段读取）——
    double clahe_clip_limit = 1.0;

    // —— 检测参数（从 YAML 的 detection 字段读取）——
    int detection_min_area = 1200;
    int detection_roi_bais = 20;

    // 便捷访问接口（与 Python 版 SCENE 字典等价）
    const HSVThreshold& get_hsv_threshold(const std::string& color) const;
    bool has_hsv_threshold(const std::string& color) const;
};

// ——————————————————————————————————
// 全局默认配置（静态单例，程序启动时自动初始化）
// 等价于 Python 版的 `SCENE = load_scene_config()`
// ——————————————————————————————————

class GlobalSceneConfig {
public:
    static GlobalSceneConfig& instance();

    // 从 YAML 文件加载；如果加载失败使用内置默认值
    bool load(const std::string& config_path);
    bool reload();

    // 获取运行时配置
    const SceneConfigLoader& config() const { return config_; }
    const HSVThresholdMap& hsv_thresholds() const { return config_.hsv_thresholds; }
    int erode_iter()   const { return config_.morphology_erode_iter; }
    int dilate_iter()  const { return config_.morphology_dilate_iter; }
    int min_area()     const { return config_.detection_min_area; }
    int roi_bais()     const { return config_.detection_roi_bais; }

    const HSVThreshold& get_hsv_threshold(const std::string& color) const {
        return config_.get_hsv_threshold(color);
    }

    bool is_loaded_from_file() const { return loaded_from_file_; }
    const std::string& last_path() const { return last_path_; }

private:
    GlobalSceneConfig() = default;
    SceneConfigLoader config_;
    std::string last_path_;
    bool loaded_from_file_ = false;
};

} // namespace fvs
