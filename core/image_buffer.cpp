#include "image_buffer.hpp"

#include <algorithm>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace mfad {

ImageBuffer::ImageBuffer(int width, int height)
    : width_(width), height_(height), pixels_(width * height, Eigen::Vector3f::Zero()) {}

void ImageBuffer::set_pixel(int x, int y, const Eigen::Vector3f& color) {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        pixels_[y * width_ + x] = color;
    }
}

Eigen::Vector3f ImageBuffer::get_pixel(int x, int y) const {
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
        return pixels_[y * width_ + x];
    }
    return Eigen::Vector3f::Zero();
}

bool ImageBuffer::write_png(const std::string& filepath, float gamma, bool apply_aces) const {
    std::vector<unsigned char> bytes(width_ * height_ * 3);
    const float inv_gamma = 1.0f / gamma;

    // ACES filmic tone mapping
    auto aces_tonemap = [](float x) -> float {
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Eigen::Vector3f& c = pixels_[y * width_ + x];

            float rc = std::max(0.0f, c.x());
            float gc = std::max(0.0f, c.y());
            float bc = std::max(0.0f, c.z());

            if (apply_aces) {
                rc = aces_tonemap(rc);
                gc = aces_tonemap(gc);
                bc = aces_tonemap(bc);
            }

            float r = std::pow(std::clamp(rc, 0.0f, 1.0f), inv_gamma);
            float g = std::pow(std::clamp(gc, 0.0f, 1.0f), inv_gamma);
            float b_ch = std::pow(std::clamp(bc, 0.0f, 1.0f), inv_gamma);

            int idx = (y * width_ + x) * 3;
            bytes[idx + 0] = static_cast<unsigned char>(r * 255.0f + 0.5f);
            bytes[idx + 1] = static_cast<unsigned char>(g * 255.0f + 0.5f);
            bytes[idx + 2] = static_cast<unsigned char>(b_ch * 255.0f + 0.5f);
        }
    }

    return stbi_write_png(filepath.c_str(), width_, height_, 3, bytes.data(), width_ * 3) != 0;
}

bool ImageBuffer::read_png(const std::string& filepath) {
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char* data = stbi_load(filepath.c_str(), &w, &h, &channels, 3);
    if (!data) {
        return false;
    }
    width_ = w;
    height_ = h;
    pixels_.resize(w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            float r = static_cast<float>(data[idx + 0]) / 255.0f;
            float g = static_cast<float>(data[idx + 1]) / 255.0f;
            float b = static_cast<float>(data[idx + 2]) / 255.0f;
            pixels_[y * w + x] = Eigen::Vector3f(r, g, b);
        }
    }
    stbi_image_free(data);
    return true;
}

bool ImageBuffer::write_hdr(const std::string& filepath) const {
    std::vector<float> float_data(width_ * height_ * 3);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Eigen::Vector3f& c = pixels_[y * width_ + x];
            int idx = (y * width_ + x) * 3;
            float_data[idx + 0] = c.x();
            float_data[idx + 1] = c.y();
            float_data[idx + 2] = c.z();
        }
    }
    return stbi_write_hdr(filepath.c_str(), width_, height_, 3, float_data.data()) != 0;
}

bool ImageBuffer::read_hdr(const std::string& filepath) {
    int w = 0;
    int h = 0;
    int channels = 0;
    float* data = stbi_loadf(filepath.c_str(), &w, &h, &channels, 3);
    if (!data) {
        return false;
    }
    width_ = w;
    height_ = h;
    pixels_.resize(static_cast<size_t>(w * h));
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            pixels_[static_cast<size_t>(y * w + x)] =
                Eigen::Vector3f(data[idx + 0], data[idx + 1], data[idx + 2]);
        }
    }
    stbi_image_free(data);
    return true;
}

}  // namespace mfad
