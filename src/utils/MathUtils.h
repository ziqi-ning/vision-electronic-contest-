#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

namespace MathUtils {

inline double atan2(double y, double x) {
    return std::atan2(y, x);
}

template <typename T>
std::vector<double> diff(const std::vector<T>& arr) {
    if (arr.size() < 2) return {};
    std::vector<double> result(arr.size() - 1);
    for (size_t i = 1; i < arr.size(); ++i) {
        result[i - 1] = static_cast<double>(arr[i]) - static_cast<double>(arr[i - 1]);
    }
    return result;
}

template <typename T>
double mean(const std::vector<T>& arr) {
    if (arr.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& v : arr) sum += static_cast<double>(v);
    return sum / arr.size();
}

template <typename T>
double median(std::vector<T> arr) {
    if (arr.empty()) return 0.0;
    size_t n = arr.size();
    if (n % 2 == 0) {
        std::nth_element(arr.begin(), arr.begin() + n / 2 - 1, arr.end());
        std::nth_element(arr.begin(), arr.begin() + n / 2, arr.end());
        return (static_cast<double>(arr[n / 2 - 1]) + static_cast<double>(arr[n / 2])) * 0.5;
    } else {
        std::nth_element(arr.begin(), arr.begin() + n / 2, arr.end());
        return static_cast<double>(arr[n / 2]);
    }
}

template <typename T>
double norm(const std::vector<T>& a, const std::vector<T>& b) {
    double sum_sq = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq);
}

inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

inline double rad2deg(double rad) {
    return rad * 180.0 / M_PI;
}

inline double angleDiff(double a, double b) {
    double d = std::fmod(a - b + M_PI, 2.0 * M_PI) - M_PI;
    return (d < -M_PI) ? d + 2.0 * M_PI : d;
}

inline double euclideanDist(const cv::Point2i& p1, const cv::Point2i& p2) {
    double dx = static_cast<double>(p1.x - p2.x);
    double dy = static_cast<double>(p1.y - p2.y);
    return std::sqrt(dx * dx + dy * dy);
}

}
