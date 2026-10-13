#pragma once

#include "camera.hpp"
#include "ellipsoid.hpp"
#include "hittable.hpp"
#include "light.hpp"
#include "material.hpp"
#include "sphere.hpp"
#include "triangle.hpp"

#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mfad {

struct CameraData {
    Eigen::Vector3d eye{0.0, 0.0, 2.0};
    Eigen::Vector3d look_at{0.0, 0.0, 0.0};
    Eigen::Vector3d up{0.0, 1.0, 0.0};
    double vfov{60.0};
    double aspect_ratio{16.0 / 9.0};

    Eigen::Matrix4d view_matrix{Eigen::Matrix4d::Identity()};
    Eigen::Matrix4d projection_matrix{Eigen::Matrix4d::Identity()};
    std::unique_ptr<Camera> camera;
};

struct MaterialData {
    std::string name;
    std::string type;  // "diffuse", "glass", "metal", "checker", "diffuse_light"
    Eigen::Vector3d albedo{1.0, 1.0, 1.0};
    double ior{1.5};
    double fuzz{0.0};
    std::shared_ptr<Material> material;
};

struct MeshData {
    std::string name;
    std::string file;
    Eigen::Matrix4d transform{Eigen::Matrix4d::Identity()};
    Eigen::MatrixXd vertices;  // 3 x V
    Eigen::MatrixXi faces;     // F x 3
    std::string material_name;
    std::shared_ptr<Material> material;
    std::vector<std::shared_ptr<Triangle>> triangles;
};

struct QuadricData {
    std::string name;
    std::string type;  // "sphere", "ellipsoid"
    Eigen::Vector3d center{0.0, 0.0, 0.0};
    Eigen::Vector3d radii{1.0, 1.0, 1.0};
    Eigen::Vector3d rotation_deg{0.0, 0.0, 0.0};
    Eigen::Matrix3d rotation_matrix{Eigen::Matrix3d::Identity()};
    Eigen::Matrix4d quadric_matrix{Eigen::Matrix4d::Identity()};
    std::string material_name;
    std::shared_ptr<Material> material;
    std::shared_ptr<Hittable> hittable;
};

/**
 * @brief Representation of an MFAD 3D Scene with linear algebra matrices and hittable geometry.
 */
class Scene {
public:
    CameraData camera_data;
    std::map<std::string, MaterialData> materials;
    std::vector<MeshData> meshes;
    std::vector<QuadricData> quadrics;
    std::vector<PointLight> point_lights;
    std::vector<std::shared_ptr<Hittable>> area_lights;
    HittableList hittables;

    // Optional environment (parsed from scene JSON "environment" block)
    bool use_sky_gradient = false;
    Eigen::Vector3f background_color{0.015f, 0.015f, 0.02f};
    Eigen::Vector3f ambient_color{0.02f, 0.02f, 0.025f};

    /**
     * @brief Number of high-level geometric objects in the scene (meshes + quadrics).
     */
    size_t object_count() const { return meshes.size() + quadrics.size(); }

    /**
     * @brief Total number of ray-traceable primitives (individual triangles + quadrics).
     */
    size_t total_primitive_count() const {
        size_t tri_count = 0;
        for (const auto& m : meshes) {
            tri_count += m.triangles.size();
        }
        return tri_count + quadrics.size();
    }

    /**
     * @brief Prints a structured summary of loaded scene elements and object count.
     */
    void print_summary(std::ostream& os = std::cout) const {
        os << "========================================\n";
        os << "  Scene Summary\n";
        os << "========================================\n";
        os << "Camera: eye=[" << camera_data.eye.transpose() << "], look_at=["
           << camera_data.look_at.transpose() << "], vfov=" << camera_data.vfov << "\n";
        os << "Materials (" << materials.size() << "):\n";
        for (const auto& [name, mat] : materials) {
            os << "  - " << name << " (type: " << mat.type << ")\n";
        }
        os << "Meshes (" << meshes.size() << "):\n";
        for (const auto& m : meshes) {
            os << "  - " << m.name << " (" << m.vertices.cols() << " vertices, " << m.faces.rows()
               << " faces, " << m.triangles.size() << " triangles)\n";
        }
        os << "Quadrics (" << quadrics.size() << "):\n";
        for (const auto& q : quadrics) {
            os << "  - " << q.name << " (type: " << q.type << ", center=[" << q.center.transpose()
               << "], radii=[" << q.radii.transpose() << "])\n";
        }
        os << "Point Lights (" << point_lights.size() << "):\n";
        for (const auto& l : point_lights) {
            os << "  - pos=[" << l.position.transpose() << "], intensity=["
               << l.intensity.transpose() << "]\n";
        }
        os << "Total Object Count: " << object_count() << "\n";
        os << "Total Primitive Count: " << total_primitive_count() << "\n";
        os << "========================================\n";
    }
};

}  // namespace mfad
