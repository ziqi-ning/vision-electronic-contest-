#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

class Profiler {
public:
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::microseconds;

    void start(const std::string& name);
    void stop(const std::string& name);

    void report() const;

    struct Record {
        std::string name;
        int64_t total_us;
        int count;
        int64_t min_us;
        int64_t max_us;
    };
    std::vector<Record> records() const;

    Profiler() = default;
    ~Profiler() = default;

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

private:
    struct Entry {
        Clock::time_point start;
        int64_t total_us = 0;
        int count = 0;
        int64_t min_us = INT64_MAX;
        int64_t max_us = 0;
    };

    std::map<std::string, Entry> entries_;
    std::map<std::string, Clock::time_point> running_;
};

void Profiler::start(const std::string& name) {
    running_[name] = Clock::now();
}

void Profiler::stop(const std::string& name) {
    auto it = running_.find(name);
    if (it == running_.end()) return;

    auto elapsed = std::chrono::duration_cast<Duration>(Clock::now() - it->second).count();
    running_.erase(it);

    auto& e = entries_[name];
    e.total_us += elapsed;
    e.count += 1;
    e.min_us = std::min(e.min_us, elapsed);
    e.max_us = std::max(e.max_us, elapsed);
}

void Profiler::report() const {
    for (const auto& kv : entries_) {
        const auto& e = kv.second;
        double avg = e.count > 0 ? static_cast<double>(e.total_us) / e.count / 1000.0 : 0.0;
        double total = static_cast<double>(e.total_us) / 1000.0;
        printf("[PROFILER] %-20s  count=%4d  avg=%8.2fms  total=%9.2fms  min=%7.2fms  max=%7.2fms\n",
               kv.first.c_str(), e.count, avg, total,
               static_cast<double>(e.min_us) / 1000.0,
               static_cast<double>(e.max_us) / 1000.0);
    }
}

std::vector<Profiler::Record> Profiler::records() const {
    std::vector<Record> result;
    for (const auto& kv : entries_) {
        const auto& e = kv.second;
        Record r;
        r.name = kv.first;
        r.total_us = e.total_us;
        r.count = e.count;
        r.min_us = e.min_us;
        r.max_us = e.max_us;
        result.push_back(r);
    }
    return result;
}
