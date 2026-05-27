#include "Logger.h"
#include <cstdarg>
#include <cstdio>
#include <vector>

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

void Logger::init(const std::string& log_dir, const std::string& filename) {
    if (logger_) return;

    std::vector<spdlog::sink_ptr> sinks;
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
    sinks.push_back(console_sink);

    if (!log_dir.empty()) {
        try {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_dir + "/" + filename, 1024 * 1024 * 10, 3);
            file_sink->set_level(spdlog::level::debug);
            sinks.push_back(file_sink);
        } catch (const spdlog::spdlog_ex& ex) {
            console_sink->warn("Failed to create file sink: {}", ex.what());
        }
    }

    logger_ = std::make_shared<spdlog::logger>("FVS", begin(sinks), end(sinks));
    logger_->set_level(spdlog::level::trace);
    spdlog::set_default_logger(logger_);
}

void Logger::debug(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger_->debug("{}", buf);
}

void Logger::info(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger_->info("{}", buf);
}

void Logger::warn(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger_->warn("{}", buf);
}

void Logger::error(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    logger_->error("{}", buf);
}
