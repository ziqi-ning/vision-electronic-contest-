#include "SceneConfig.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <sstream>

namespace fvs {

// ——————————————————————————————————
// SceneConfigLoader 成员方法实现
// ——————————————————————————————————

const HSVThreshold& SceneConfigLoader::get_hsv_threshold(const std::string& color) const {
    static const HSVThreshold kEmpty{{0,0,0}, {0,0,0}};
    auto it = hsv_thresholds.find(color);
    return (it != hsv_thresholds.end()) ? it->second : kEmpty;
}

bool SceneConfigLoader::has_hsv_threshold(const std::string& color) const {
    return hsv_thresholds.find(color) != hsv_thresholds.end();
}

std::optional<SceneConfigLoader> SceneConfigLoader::load(const std::string& config_path) {
    try {
        YAML::Node root = YAML::LoadFile(config_path);
        SceneConfigLoader loader;

        // —— 解析 colors 字段 ——
        if (root["colors"] && root["colors"].IsMap()) {
            for (const auto& kv : root["colors"]) {
                std::string color_name = kv.first.as<std::string>();
                YAML::Node color_node = kv.second;

                cv::Scalar lower, upper;
                if (color_node["lower"] && color_node["lower"].IsSequence()) {
                    auto lower_seq = color_node["lower"].as<std::vector<int>>();
                    if (lower_seq.size() >= 3) {
                        lower = {static_cast<double>(lower_seq[0]),
                                 static_cast<double>(lower_seq[1]),
                                 static_cast<double>(lower_seq[2])};
                    }
                }
                if (color_node["upper"] && color_node["upper"].IsSequence()) {
                    auto upper_seq = color_node["upper"].as<std::vector<int>>();
                    if (upper_seq.size() >= 3) {
                        upper = {static_cast<double>(upper_seq[0]),
                                 static_cast<double>(upper_seq[1]),
                                 static_cast<double>(upper_seq[2])};
                    }
                }
                loader.hsv_thresholds[color_name] = {lower, upper};
            }
        }

        // —— 解析 morphology 字段 ——
        if (root["morphology"] && root["morphology"].IsMap()) {
            loader.morphology_erode_iter  = root["morphology"]["erode_iter"]
                                               .as<int>(2);
            loader.morphology_dilate_iter = root["morphology"]["dilate_iter"]
                                               .as<int>(2);
        }

        // —— 解析 clahe 字段 ——
        if (root["clahe"] && root["clahe"].IsMap()) {
            loader.clahe_clip_limit = root["clahe"]["clip_limit"]
                                         .as<double>(1.0);
        }

        // —— 解析 detection 字段 ——
        if (root["detection"] && root["detection"].IsMap()) {
            loader.detection_min_area = root["detection"]["min_area"]
                                           .as<int>(1200);
            loader.detection_roi_bais = root["detection"]["roi_bais"]
                                           .as<int>(20);
        }

        return loader;

    } catch (const std::exception& e) {
        std::cerr << "[SceneConfig] Failed to load " << config_path
                  << ": " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::optional<SceneConfigLoader> SceneConfigLoader::load_default() {
    SceneConfigLoader loader;
    loader.hsv_thresholds = default_hsv_thresholds();
    loader.morphology_erode_iter  = 2;
    loader.morphology_dilate_iter = 2;
    loader.clahe_clip_limit       = 1.0;
    loader.detection_min_area     = 1200;
    loader.detection_roi_bais     = 20;
    return loader;
}

// ——————————————————————————————————
// GlobalSceneConfig 单例实现
// ——————————————————————————————————

GlobalSceneConfig& GlobalSceneConfig::instance() {
    static GlobalSceneConfig instance_;
    return instance_;
}

bool GlobalSceneConfig::load(const std::string& config_path) {
    last_path_ = config_path;
    auto loaded = SceneConfigLoader::load(config_path);
    if (loaded.has_value()) {
        config_ = std::move(loaded.value());
        loaded_from_file_ = true;
        return true;
    }
    // YAML 加载失败时使用内置默认值
    auto defaults = SceneConfigLoader::load_default();
    if (defaults.has_value()) {
        config_ = std::move(defaults.value());
        loaded_from_file_ = false;
        return false;
    }
    loaded_from_file_ = false;
    return false;
}

bool GlobalSceneConfig::reload() {
    return load(last_path_);
}

} // namespace fvs
