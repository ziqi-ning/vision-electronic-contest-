/**
 * @file main.cpp
 * @brief FVS-Cpp 主程序入口
 *
 * 功能：无人机多传感器融合 C++ 版主程序
 * 平台：ARM嵌入式（树莓派 / Jetson）
 */

#include <csignal>
#include <atomic>
#include <memory>
#include <iostream>

#include "spdlog/spdlog.h"

int main(int, char*[]) {
    // 初始化日志
    spdlog::set_level(spdlog::level::info);
    spdlog::info("FVS-Cpp starting...");

    // SIGINT handler：使用静态变量避免 lambda 捕获问题（C++ 不能把捕获的 lambda 转函数指针）
    static std::atomic<bool> sigint_received{false};
    std::signal(SIGINT, +[](int) {
        sigint_received.store(true, std::memory_order_release);
        spdlog::info("Received SIGINT, shutting down...");
    });

    // TODO Phase 4: 初始化串口、雷达、Pipeline
    // TODO Phase 4: 初始化各 ModeHandler
    // TODO Phase 4: 启动取帧线程 + 处理线程

    while (!sigint_received.load(std::memory_order_acquire)) {
        // 占位：实际帧处理循环在 T4.4 实现
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    spdlog::info("FVS-Cpp stopped.");
    return 0;
}
