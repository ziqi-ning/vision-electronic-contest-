#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <string>

class Logger {
public:
    static Logger& instance();

    void init(const std::string& log_dir = "logs", const std::string& filename = "fvs.log");

    void debug(const char* fmt, ...);
    void info(const char* fmt, ...);
    void warn(const char* fmt, ...);
    void error(const char* fmt, ...);

    spdlog::logger& logger() { return *logger_; }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger() = default;
    ~Logger() = default;

    std::shared_ptr<spdlog::logger> logger_;
};

#define LOGD(fmt, ...) Logger::instance().debug(fmt, ##__VA_ARGS__)
#define LOGI(fmt, ...) Logger::instance().info(fmt, ##__VA_ARGS__)
#define LOGW(fmt, ...) Logger::instance().warn(fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) Logger::instance().error(fmt, ##__VA_ARGS__)
