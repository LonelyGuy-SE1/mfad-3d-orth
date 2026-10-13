#include "scene_loader.hpp"

#include "plane.hpp"
#include "quad.hpp"
#include "stages/stage_projection.hpp"
#include "stages/stage_transforms.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace mfad {

namespace {

// Helper: Euler rotation in degrees to 3x3 rotation matrix matching prep/stage_eigen.py (Rz * Ry *
// Rx)
Eigen::Matrix3d euler_to_rotation_matrix(const Eigen::Vector3d& angles_deg) {
    const double deg_to_rad = 3.14159265358979323846 / 180.0;
    double rx = angles_deg.x() * deg_to_rad;
    double ry = angles_deg.y() * deg_to_rad;
    double rz = angles_deg.z() * deg_to_rad;

    Eigen::Matrix3d mx = Eigen::Matrix3d::Identity();
    mx(1, 1) = std::cos(rx);
    mx(1, 2) = -std::sin(rx);
    mx(2, 1) = std::sin(rx);
    mx(2, 2) = std::cos(rx);

    Eigen::Matrix3d my = Eigen::Matrix3d::Identity();
    my(0, 0) = std::cos(ry);
    my(0, 2) = std::sin(ry);
    my(2, 0) = -std::sin(ry);
    my(2, 2) = std::cos(ry);

    Eigen::Matrix3d mz = Eigen::Matrix3d::Identity();
    mz(0, 0) = std::cos(rz);
    mz(0, 1) = -std::sin(rz);
    mz(1, 0) = std::sin(rz);
    mz(1, 1) = std::cos(rz);

    return mz * my * mx;
}

// Helper: Simple and robust OBJ file loader
bool load_obj_file(const std::filesystem::path& path, Eigen::MatrixXd& vertices_out,
                   Eigen::MatrixXi& faces_out) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[SceneLoader] Failed to open OBJ file: " << path << std::endl;
        return false;
    }

    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> faces;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            double x = 0.0, y = 0.0, z = 0.0;
            iss >> x >> y >> z;
            verts.emplace_back(x, y, z);
        } else if (token == "f") {
            std::vector<int> poly_indices;
            std::string face_token;
            while (iss >> face_token) {
                // Parse vertex index before first '/'
                size_t slash = face_token.find('/');
                std::string idx_str =
                    (slash != std::string::npos) ? face_token.substr(0, slash) : face_token;
                if (!idx_str.empty()) {
                    int idx = std::stoi(idx_str);
                    if (idx > 0) {
                        poly_indices.push_back(idx - 1);
                    } else if (idx < 0) {
                        poly_indices.push_back(static_cast<int>(verts.size()) + idx);
                    }
                }
            }

            // Triangulate polygons (triangle fan)
            for (size_t i = 1; i + 1 < poly_indices.size(); ++i) {
                faces.emplace_back(poly_indices[0], poly_indices[i], poly_indices[i + 1]);
            }
        }
    }

    // Convert into Eigen matrices (3 x V for vertices, F x 3 for faces)
    vertices_out.resize(3, verts.size());
    for (size_t i = 0; i < verts.size(); ++i) {
        vertices_out.col(static_cast<Eigen::Index>(i)) = verts[i];
    }

    faces_out.resize(faces.size(), 3);
    for (size_t i = 0; i < faces.size(); ++i) {
        faces_out.row(static_cast<Eigen::Index>(i)) = faces[i];
    }

    return true;
}

}  // namespace

Scene SceneLoader::load_from_json(const std::string& filepath, double aspect_ratio) {
    std::filesystem::path scene_path(filepath);
    std::filesystem::path scene_dir = scene_path.parent_path();

    std::ifstream file(scene_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open scene file: " + filepath);
    }

    nlohmann::json j;
    file >> j;

    Scene scene;

    // 0. Optional environment (sky / background / ambient fill)
    if (j.contains("environment") && j["environment"].is_object()) {
        const auto& env = j["environment"];
        scene.use_sky_gradient = env.value("sky_gradient", scene.use_sky_gradient);
        if (env.contains("background") && env["background"].is_array() &&
            env["background"].size() >= 3) {
            scene.background_color =
                Eigen::Vector3f(env["background"][0].get<float>(), env["background"][1].get<float>(),
                                env["background"][2].get<float>());
        }
        if (env.contains("ambient") && env["ambient"].is_array() && env["ambient"].size() >= 3) {
            scene.ambient_color = Eigen::Vector3f(env["ambient"][0].get<float>(),
                                                  env["ambient"][1].get<float>(),
                                                  env["ambient"][2].get<float>());
        }
    }

    // 1. Parse Camera
    if (j.contains("camera")) {
        const auto& cam_j = j["camera"];
        if (cam_j.contains("eye")) {
            scene.camera_data.eye =
                Eigen::Vector3d(cam_j["eye"][0].get<double>(), cam_j["eye"][1].get<double>(),
                                cam_j["eye"][2].get<double>());
        }
        if (cam_j.contains("look_at")) {
            scene.camera_data.look_at = Eigen::Vector3d(cam_j["look_at"][0].get<double>(),
                                                        cam_j["look_at"][1].get<double>(),
                                                        cam_j["look_at"][2].get<double>());
        }
        if (cam_j.contains("up")) {
            scene.camera_data.up =
                Eigen::Vector3d(cam_j["up"][0].get<double>(), cam_j["up"][1].get<double>(),
                                cam_j["up"][2].get<double>());
        }
        if (cam_j.contains("vfov")) {
            scene.camera_data.vfov = cam_j["vfov"].get<double>();
        }
        scene.camera_data.aspect_ratio = aspect_ratio;

        // Construct 4x4 view matrix
        Eigen::Vector3d w = (scene.camera_data.eye - scene.camera_data.look_at).normalized();
        Eigen::Vector3d u = scene.camera_data.up.cross(w).normalized();
        Eigen::Vector3d v = w.cross(u);

        Eigen::Matrix4d view = Eigen::Matrix4d::Identity();
        view.block<1, 3>(0, 0) = u.transpose();
        view.block<1, 3>(1, 0) = v.transpose();
        view.block<1, 3>(2, 0) = w.transpose();
        view(0, 3) = -u.dot(scene.camera_data.eye);
        view(1, 3) = -v.dot(scene.camera_data.eye);
        view(2, 3) = -w.dot(scene.camera_data.eye);
        scene.camera_data.view_matrix = view;

        // Construct 4x4 projection matrix using stage_project_image_plane
        auto proj_stage =
            stage_project_image_plane(Eigen::Vector3d(0.0, 0.0, -1.0), scene.camera_data.vfov,
                                      scene.camera_data.aspect_ratio, 100, 100, 0.1, 1000.0);
        scene.camera_data.projection_matrix = proj_stage.value.projection_matrix;

        // Instantiate Camera object
        float aperture = 0.0f;
        float focus_dist_val = -1.0f;
        if (cam_j.contains("aperture")) {
            aperture = static_cast<float>(cam_j["aperture"].get<double>());
        }
        if (cam_j.contains("focus_dist")) {
            focus_dist_val = static_cast<float>(cam_j["focus_dist"].get<double>());
        }
        scene.camera_data.camera = std::make_unique<Camera>(
            scene.camera_data.eye.cast<float>(), scene.camera_data.look_at.cast<float>(),
            scene.camera_data.up.cast<float>(), static_cast<float>(scene.camera_data.vfov),
            static_cast<float>(scene.camera_data.aspect_ratio), aperture, focus_dist_val);
    }

    // 2. Parse Materials
    if (j.contains("materials") && j["materials"].is_object()) {
        for (auto& [name, mat_j] : j["materials"].items()) {
            MaterialData mdata;
            mdata.name = name;
            mdata.type = mat_j.value("type", "diffuse");

            if (mat_j.contains("albedo") && mat_j["albedo"].is_array()) {
                mdata.albedo = Eigen::Vector3d(mat_j["albedo"][0].get<double>(),
                                               mat_j["albedo"][1].get<double>(),
                                               mat_j["albedo"][2].get<double>());
            }
            if (mat_j.contains("ior")) {
                mdata.ior = mat_j["ior"].get<double>();
            }
            if (mat_j.contains("fuzz")) {
                mdata.fuzz = mat_j["fuzz"].get<double>();
            }

            // Create concrete C++ Material instance
            if (mdata.type == "checker") {
                Eigen::Vector3f color2(0.2f, 0.2f, 0.2f);
                float scale = 2.0f;
                if (mat_j.contains("color2") && mat_j["color2"].is_array()) {
                    color2 = Eigen::Vector3f(mat_j["color2"][0].get<float>(),
                                             mat_j["color2"][1].get<float>(),
                                             mat_j["color2"][2].get<float>());
                }
                if (mat_j.contains("scale"))
                    scale = mat_j["scale"].get<float>();
                mdata.material =
                    std::make_shared<CheckerMaterial>(mdata.albedo.cast<float>(), color2, scale);
            } else if (mdata.type == "diffuse" || mdata.type == "lambertian") {
                mdata.material = std::make_shared<Lambertian>(mdata.albedo.cast<float>());
            } else if (mdata.type == "glass" || mdata.type == "dielectric") {
                mdata.material = std::make_shared<Dielectric>(static_cast<float>(mdata.ior));
            } else if (mdata.type == "metal" || mdata.type == "mirror") {
                mdata.material = std::make_shared<Metal>(mdata.albedo.cast<float>(),
                                                         static_cast<float>(mdata.fuzz));
            } else if (mdata.type == "diffuse_light" || mdata.type == "light") {
                mdata.material = std::make_shared<DiffuseLight>(mdata.albedo.cast<float>(),
                                                                mat_j.value("two_sided", false));
            } else {
                mdata.material = std::make_shared<Lambertian>(mdata.albedo.cast<float>());
            }

            scene.materials[name] = mdata;
        }
    }

    // 3. Parse Meshes
    if (j.contains("meshes") && j["meshes"].is_array()) {
        for (const auto& mesh_j : j["meshes"]) {
            MeshData mdata;
            mdata.name = mesh_j.value("name", "mesh");
            mdata.file = mesh_j.value("file", "");
            mdata.material_name = mesh_j.value("material", "");

            if (mesh_j.contains("transform") && mesh_j["transform"].is_array()) {
                const auto& t_arr = mesh_j["transform"];
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        mdata.transform(r, c) = t_arr[r][c].get<double>();
                    }
                }
            }

            // Resolve material
            if (!mdata.material_name.empty() && scene.materials.count(mdata.material_name)) {
                mdata.material = scene.materials[mdata.material_name].material;
            } else {
                mdata.material = std::make_shared<Lambertian>(Eigen::Vector3f(0.8f, 0.8f, 0.8f));
            }

            // Load geometry
            if (!mdata.file.empty()) {
                std::filesystem::path obj_path = scene_dir / mdata.file;
                if (!load_obj_file(obj_path, mdata.vertices, mdata.faces)) {
                    std::cerr << "[SceneLoader] Warning: could not load OBJ: " << obj_path
                              << std::endl;
                }
            } else if (mesh_j.contains("vertices") && mesh_j.contains("faces")) {
                // Inline vertices & faces (e.g. from prepared json)
                const auto& v_arr = mesh_j["vertices"];
                const auto& f_arr = mesh_j["faces"];

                if (v_arr.is_array() && !v_arr.empty()) {
                    if (v_arr[0].is_array() && v_arr.size() == 3) {
                        // 3 x V format
                        size_t num_v = v_arr[0].size();
                        mdata.vertices.resize(3, num_v);
                        for (int r = 0; r < 3; ++r) {
                            for (size_t c = 0; c < num_v; ++c) {
                                mdata.vertices(r, c) = v_arr[r][c].get<double>();
                            }
                        }
                    } else if (v_arr[0].is_array()) {
                        // V x 3 format
                        size_t num_v = v_arr.size();
                        mdata.vertices.resize(3, num_v);
                        for (size_t r = 0; r < num_v; ++r) {
                            for (int c = 0; c < 3; ++c) {
                                mdata.vertices(c, r) = v_arr[r][c].get<double>();
                            }
                        }
                    }
                }

                if (f_arr.is_array() && !f_arr.empty()) {
                    size_t num_f = f_arr.size();
                    mdata.faces.resize(num_f, 3);
                    for (size_t r = 0; r < num_f; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            mdata.faces(r, c) = f_arr[r][c].get<int>();
                        }
                    }
                }
            }

            // Apply 4x4 affine transform to vertices if non-identity
            if (mdata.vertices.cols() > 0) {
                Eigen::Matrix4d T = mdata.transform;
                for (Eigen::Index col = 0; col < mdata.vertices.cols(); ++col) {
                    Eigen::Vector4d p_homo(mdata.vertices(0, col), mdata.vertices(1, col),
                                           mdata.vertices(2, col), 1.0);
                    Eigen::Vector4d p_trans = T * p_homo;
                    if (std::abs(p_trans.w()) > 1e-12) {
                        mdata.vertices.col(col) = p_trans.head<3>() / p_trans.w();
                    }
                }
            }

            // Construct Triangle hittables
            for (Eigen::Index fi = 0; fi < mdata.faces.rows(); ++fi) {
                int i0 = mdata.faces(fi, 0);
                int i1 = mdata.faces(fi, 1);
                int i2 = mdata.faces(fi, 2);

                if (i0 < mdata.vertices.cols() && i1 < mdata.vertices.cols() &&
                    i2 < mdata.vertices.cols()) {
                    Eigen::Vector3f v0 = mdata.vertices.col(i0).cast<float>();
                    Eigen::Vector3f v1 = mdata.vertices.col(i1).cast<float>();
                    Eigen::Vector3f v2 = mdata.vertices.col(i2).cast<float>();

                    auto tri = std::make_shared<Triangle>(v0, v1, v2, Eigen::Vector3f::Ones(),
                                                          mdata.material);
                    mdata.triangles.push_back(tri);
                    scene.hittables.add(tri);
                }
            }

            scene.meshes.push_back(std::move(mdata));
        }
    }

    // 4. Parse Quadrics
    if (j.contains("quadrics") && j["quadrics"].is_array()) {
        for (const auto& quad_j : j["quadrics"]) {
            QuadricData qdata;
            qdata.name = quad_j.value("name", "quadric");
            qdata.type = quad_j.value("type", "sphere");
            qdata.material_name = quad_j.value("material", "");

            if (quad_j.contains("center") && quad_j["center"].is_array()) {
                qdata.center = Eigen::Vector3d(quad_j["center"][0].get<double>(),
                                               quad_j["center"][1].get<double>(),
                                               quad_j["center"][2].get<double>());
            }

            if (qdata.type == "sphere") {
                double rad = quad_j.value("radius", 1.0);
                qdata.radii = Eigen::Vector3d(rad, rad, rad);
            } else if (quad_j.contains("radii") && quad_j["radii"].is_array()) {
                qdata.radii = Eigen::Vector3d(quad_j["radii"][0].get<double>(),
                                              quad_j["radii"][1].get<double>(),
                                              quad_j["radii"][2].get<double>());
            }

            if (quad_j.contains("rotation_deg") && quad_j["rotation_deg"].is_array()) {
                qdata.rotation_deg = Eigen::Vector3d(quad_j["rotation_deg"][0].get<double>(),
                                                     quad_j["rotation_deg"][1].get<double>(),
                                                     quad_j["rotation_deg"][2].get<double>());
                qdata.rotation_matrix = euler_to_rotation_matrix(qdata.rotation_deg);
            } else if (quad_j.contains("axes") && quad_j["axes"].is_array()) {
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        qdata.rotation_matrix(r, c) = quad_j["axes"][r][c].get<double>();
                    }
                }
            }

            // Compute algebraic quadric matrix Q = [A, b; b^T, d]
            Eigen::Matrix3d inv_sq = Eigen::Matrix3d::Zero();
            inv_sq(0, 0) = 1.0 / (qdata.radii.x() * qdata.radii.x());
            inv_sq(1, 1) = 1.0 / (qdata.radii.y() * qdata.radii.y());
            inv_sq(2, 2) = 1.0 / (qdata.radii.z() * qdata.radii.z());
            Eigen::Matrix3d A = qdata.rotation_matrix * inv_sq * qdata.rotation_matrix.transpose();
            Eigen::Vector3d b = -A * qdata.center;
            double d = qdata.center.dot(A * qdata.center) - 1.0;

            qdata.quadric_matrix = Eigen::Matrix4d::Zero();
            qdata.quadric_matrix.block<3, 3>(0, 0) = A;
            qdata.quadric_matrix.block<3, 1>(0, 3) = b;
            qdata.quadric_matrix.block<1, 3>(3, 0) = b.transpose();
            qdata.quadric_matrix(3, 3) = d;

            if (quad_j.contains("matrix") && quad_j["matrix"].is_array()) {
                const auto& m_arr = quad_j["matrix"];
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        qdata.quadric_matrix(r, c) = m_arr[r][c].get<double>();
                    }
                }
            }

            // Resolve material
            if (!qdata.material_name.empty() && scene.materials.count(qdata.material_name)) {
                qdata.material = scene.materials[qdata.material_name].material;
            } else {
                qdata.material = std::make_shared<Lambertian>(Eigen::Vector3f(0.8f, 0.8f, 0.8f));
            }

            // Instantiate hittable
            if (qdata.type == "sphere" && std::abs(qdata.radii.x() - qdata.radii.y()) < 1e-6 &&
                std::abs(qdata.radii.x() - qdata.radii.z()) < 1e-6) {
                auto sph = std::make_shared<Sphere>(qdata.center.cast<float>(),
                                                    static_cast<float>(qdata.radii.x()),
                                                    Eigen::Vector3f::Ones(), qdata.material);
                qdata.hittable = sph;
                scene.hittables.add(sph);
            } else {
                auto ell = std::make_shared<Ellipsoid>(
                    qdata.center.cast<float>(), qdata.radii.cast<float>(),
                    qdata.rotation_matrix.cast<float>(), Eigen::Vector3f::Ones(), qdata.material);
                qdata.hittable = ell;
                scene.hittables.add(ell);
            }

            scene.quadrics.push_back(std::move(qdata));
        }
    }

    // Parse Planes
    if (j.contains("planes") && j["planes"].is_array()) {
        for (const auto& p_j : j["planes"]) {
            Eigen::Vector3f point(0, 0, 0);
            Eigen::Vector3f normal(0, 1, 0);
            if (p_j.contains("point"))
                point = Eigen::Vector3f(p_j["point"][0], p_j["point"][1], p_j["point"][2]);
            if (p_j.contains("normal"))
                normal = Eigen::Vector3f(p_j["normal"][0], p_j["normal"][1], p_j["normal"][2]);
            std::string mat_name = p_j.value("material", "default");
            std::shared_ptr<Material> mat =
                scene.materials.count(mat_name) ? scene.materials[mat_name].material : nullptr;
            scene.hittables.add(
                std::make_shared<Plane>(point, normal, Eigen::Vector3f::Ones(), mat));
        }
    }

    // Parse Quads
    if (j.contains("quads") && j["quads"].is_array()) {
        for (const auto& q_j : j["quads"]) {
            Eigen::Vector3f Q(0, 0, 0), u(1, 0, 0), v(0, 1, 0);
            if (q_j.contains("Q"))
                Q = Eigen::Vector3f(q_j["Q"][0], q_j["Q"][1], q_j["Q"][2]);
            if (q_j.contains("u"))
                u = Eigen::Vector3f(q_j["u"][0], q_j["u"][1], q_j["u"][2]);
            if (q_j.contains("v"))
                v = Eigen::Vector3f(q_j["v"][0], q_j["v"][1], q_j["v"][2]);
            std::string mat_name = q_j.value("material", "default");
            std::shared_ptr<Material> mat =
                scene.materials.count(mat_name) ? scene.materials[mat_name].material : nullptr;
            auto quad = std::make_shared<Quad>(Q, u, v, Eigen::Vector3f::Ones(), mat);
            scene.hittables.add(quad);
            if (mat && mat_name.find("light") != std::string::npos) {
                scene.area_lights.push_back(quad);
            }

            // If it's a light, add it to Area Lights? We'll let main.cpp extract Area Lights from
            // hittables!
        }
    }

    // 5. Parse Lights
    if (j.contains("lights") && j["lights"].is_array()) {
        for (const auto& light_j : j["lights"]) {
            Eigen::Vector3f pos = Eigen::Vector3f::Zero();
            Eigen::Vector3f intensity = Eigen::Vector3f::Ones();

            if (light_j.contains("position") && light_j["position"].is_array()) {
                pos = Eigen::Vector3f(light_j["position"][0].get<float>(),
                                      light_j["position"][1].get<float>(),
                                      light_j["position"][2].get<float>());
            }
            if (light_j.contains("intensity") && light_j["intensity"].is_array()) {
                intensity = Eigen::Vector3f(light_j["intensity"][0].get<float>(),
                                            light_j["intensity"][1].get<float>(),
                                            light_j["intensity"][2].get<float>());
            } else if (light_j.contains("color") && light_j["color"].is_array()) {
                intensity = Eigen::Vector3f(light_j["color"][0].get<float>(),
                                            light_j["color"][1].get<float>(),
                                            light_j["color"][2].get<float>());
            }
            float radius = light_j.value("radius", 0.0f);
            bool direct_only = light_j.value("direct_only", false);
            scene.point_lights.emplace_back(pos, intensity, radius, direct_only);
        }
    }

    // Mandatory Issue #20 requirement: print object count
    std::cout << "[SceneLoader] Loaded scene from '" << filepath << "': " << scene.meshes.size()
              << " meshes, " << scene.quadrics.size() << " quadrics, " << scene.point_lights.size()
              << " lights. Total object count: " << scene.object_count() << "." << std::endl;

    return scene;
}

}  // namespace mfad
