#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity);

    bool push(T item);
    std::optional<T> pop();
    bool empty() const;
    bool full() const;
    size_t size() const;
    size_t capacity() const { return capacity_; }

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

private:
    const size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<size_t> count_{0};
};

template <typename T>
RingBuffer<T>::RingBuffer(size_t capacity)
    : capacity_(capacity), buffer_(capacity) {}

template <typename T>
bool RingBuffer<T>::push(T item) {
    if (full()) return false;
    buffer_[tail_.load(std::memory_order_relaxed)] = std::move(item);
    tail_.store((tail_.load(std::memory_order_relaxed) + 1) % capacity_,
                std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_release);
    return true;
}

template <typename T>
std::optional<T> RingBuffer<T>::pop() {
    if (empty()) return std::nullopt;
    T item = std::move(buffer_[head_.load(std::memory_order_relaxed)]);
    head_.store((head_.load(std::memory_order_relaxed) + 1) % capacity_,
                std::memory_order_relaxed);
    count_.fetch_sub(1, std::memory_order_release);
    return item;
}

template <typename T>
bool RingBuffer<T>::empty() const {
    return count_.load(std::memory_order_acquire) == 0;
}

template <typename T>
bool RingBuffer<T>::full() const {
    return count_.load(std::memory_order_acquire) == capacity_;
}

template <typename T>
size_t RingBuffer<T>::size() const {
    return count_.load(std::memory_order_acquire);
}
