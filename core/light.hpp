#pragma once

#include <Eigen/Dense>

namespace mfad {

/**
 * @brief Point light source emitting isotropic radiant intensity.
 */
struct PointLight {
    Eigen::Vector3f position{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f intensity{1.0f, 1.0f, 1.0f};
    float radius{0.0f};
    // When true, used only by Stage 0/1 direct lighting (stand-in for an area light).
    // Path tracing samples the real area light instead and must skip this to avoid double-counting.
    bool direct_only{false};

    PointLight() = default;
    PointLight(const Eigen::Vector3f& pos, const Eigen::Vector3f& inten, float rad = 0.0f,
               bool direct_only_flag = false)
        : position(pos), intensity(inten), radius(rad), direct_only(direct_only_flag) {}
};

}  // namespace mfad
