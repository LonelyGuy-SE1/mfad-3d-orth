#include "camera.hpp"
#include "direct_lighting.hpp"
#include "hittable.hpp"
#include "image_buffer.hpp"
#include "material.hpp"
#include "path_tracer.hpp"
#include "plane.hpp"
#include "quad.hpp"
#include "ray.hpp"
#include "scene_loader.hpp"
#include "sphere.hpp"
#include "stage_svd_denoise.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <string>
#include <vector>

// ============================================================
//  Single-source pipeline configuration (config.json)
//  Every knob of the render pipeline comes from ONE file.
// ============================================================
struct PipelineConfig {
    std::vector<std::string> scenes = {"scenes/final_demo.json"};
    std::string output_dir = "output";
    int width = 1920;
    int height = 1080;
    bool stage_raw = true;
    bool stage_direct = true;
    bool stage_path_traced = true;
    bool stage_denoise = true;
    int direct_spp = 32;
    int path_spp = 128;
    int max_bounces = 16;
    int min_rr_bounces = 3;
    int svd_rank = 120;
    float exposure = 1.0f;  // scale linear HDR before tone-map (<1 darkens)
};

PipelineConfig load_config(const std::string& path) {
    PipelineConfig cfg;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Config] Could not open " << path << ", using defaults" << std::endl;
        return cfg;
    }
    nlohmann::json j;
    file >> j;

    if (j.contains("scenes") && j["scenes"].is_array()) {
        cfg.scenes.clear();
        for (const auto& s : j["scenes"]) {
            cfg.scenes.push_back(s.get<std::string>());
        }
    } else {
        cfg.scenes = {j.value("scene", cfg.scenes.front())};
    }
    cfg.output_dir = j.value("output_dir", cfg.output_dir);
    if (j.contains("image")) {
        cfg.width = j["image"].value("width", cfg.width);
        cfg.height = j["image"].value("height", cfg.height);
    }
    if (j.contains("stages")) {
        cfg.stage_raw = j["stages"].value("raw", cfg.stage_raw);
        cfg.stage_direct = j["stages"].value("direct", cfg.stage_direct);
        cfg.stage_path_traced = j["stages"].value("path_traced", cfg.stage_path_traced);
        cfg.stage_denoise = j["stages"].value("denoise", cfg.stage_denoise);
    }
    cfg.direct_spp = j.value("direct_spp", cfg.direct_spp);
    cfg.path_spp = j.value("path_spp", cfg.path_spp);
    cfg.max_bounces = j.value("max_bounces", cfg.max_bounces);
    cfg.min_rr_bounces = j.value("min_rr_bounces", cfg.min_rr_bounces);
    cfg.svd_rank = j.value("svd_rank", cfg.svd_rank);
    cfg.exposure = j.value("exposure", cfg.exposure);
    return cfg;
}

void apply_exposure(mfad::ImageBuffer& image, float exposure) {
    if (std::abs(exposure - 1.0f) < 1e-6f) {
        return;
    }
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            image.set_pixel(x, y, image.get_pixel(x, y) * exposure);
        }
    }
}

// Stage 0: raw canvas — geometry and materials only, no shadows
void render_raw_frame(const std::string& out_path, const mfad::Scene& scene, int width, int height,
                      int spp, float exposure) {
    std::cout << "\n[Stage 0] Raw canvas (no shadows) -> " << out_path << std::endl;
    mfad::ImageBuffer raw_image(width, height);
    mfad::DirectLightingOptions dl_opts;
    dl_opts.use_sky_gradient = scene.use_sky_gradient;
    dl_opts.background_color = scene.background_color;
    dl_opts.ambient_color = scene.ambient_color;
    dl_opts.enable_shadows = false;

    mfad::render_direct_lighting(*scene.camera_data.camera, scene, raw_image, dl_opts, spp);
    apply_exposure(raw_image, exposure);
    raw_image.write_png(out_path);
}

// Stage 3: truncated SVD denoising of the path-traced image.
// apply_aces=true when input is linear HDR; false when input was already display-referred.
void denoise_and_save(mfad::ImageBuffer pt_image, const std::string& denoised_out, int svd_rank,
                      float exposure, bool apply_aces = true) {
    apply_exposure(pt_image, exposure);
    int width = pt_image.width();
    int height = pt_image.height();
    std::cout << "\n[Stage 3] SVD low-rank denoising (rank " << svd_rank << ") -> " << denoised_out
              << std::endl;
    // Split image into 3 height x width channel matrices (R, G, B)
    std::vector<Eigen::MatrixXd> channels(3, Eigen::MatrixXd(height, width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f c = pt_image.get_pixel(x, y);
            channels[0](y, x) = static_cast<double>(c.x());
            channels[1](y, x) = static_cast<double>(c.y());
            channels[2](y, x) = static_cast<double>(c.z());
        }
    }

    auto denoise_res = mfad::stage_svd_denoise_image(channels, svd_rank);
    mfad::ImageBuffer denoised_image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r =
                static_cast<float>(std::max(0.0, denoise_res.value.denoised_channels[0](y, x)));
            float g =
                static_cast<float>(std::max(0.0, denoise_res.value.denoised_channels[1](y, x)));
            float b =
                static_cast<float>(std::max(0.0, denoise_res.value.denoised_channels[2](y, x)));
            denoised_image.set_pixel(x, y, Eigen::Vector3f(r, g, b));
        }
    }
    denoised_image.write_png(denoised_out, 2.2f, apply_aces);
}

// Run all enabled pipeline stages for ONE scene.
// Every stage renders the same scene with the same camera and resolution;
// outputs land in <output_dir>/<scene_stem>/NN_*.png
bool run_scene_pipeline(const PipelineConfig& cfg, const std::string& scene_file) {
    const std::string stem = std::filesystem::path(scene_file).stem().string();
    const std::string out_dir = cfg.output_dir + "/" + stem;
    std::filesystem::create_directories(out_dir);
    const std::string raw_out = out_dir + "/01_raw.png";
    const std::string direct_out = out_dir + "/02_direct.png";
    const std::string pathtraced_out = out_dir + "/03_pathtraced.png";
    const std::string denoised_out = out_dir + "/04_denoised.png";

    std::cout << "\n[Scene] Loading " << scene_file << std::endl;
    mfad::Scene scene =
        mfad::SceneLoader::load_from_json(scene_file, static_cast<double>(cfg.width) / cfg.height);
    scene.print_summary();
    std::cout << "[Scene] Building BVH acceleration structure..." << std::endl;
    scene.hittables.build_bvh();

    const int width = cfg.width;
    const int height = cfg.height;

    // ---------------------------------------------------------
    // Stage 0: raw canvas (unshadowed geometry + materials)
    // ---------------------------------------------------------
    if (cfg.stage_raw) {
        render_raw_frame(raw_out, scene, width, height, cfg.direct_spp, cfg.exposure);
    }

    // ---------------------------------------------------------
    // Stage 1: direct lighting (Whitted reflection/refraction
    //          + shadow rays + Lambertian point lights)
    // ---------------------------------------------------------
    mfad::ImageBuffer direct_image(width, height);
    if (cfg.stage_direct) {
        std::cout << "\n[Stage 1] Direct lighting -> " << direct_out << std::endl;
        mfad::DirectLightingOptions dl_opts;
        dl_opts.use_sky_gradient = scene.use_sky_gradient;
        dl_opts.background_color = scene.background_color;
        dl_opts.ambient_color = scene.ambient_color;

        mfad::render_direct_lighting(*scene.camera_data.camera, scene, direct_image, dl_opts,
                                     cfg.direct_spp);
        apply_exposure(direct_image, cfg.exposure);
        direct_image.write_png(direct_out);
    }

    // ---------------------------------------------------------
    // Stage 2: Monte Carlo path tracing (global illumination,
    //          next-event estimation, Russian roulette)
    // ---------------------------------------------------------
    mfad::ImageBuffer pt_image(width, height);
    bool have_pathtraced = false;
    if (cfg.stage_path_traced) {
        std::cout << "\n[Stage 2] Path tracing (" << cfg.path_spp << " spp, max " << cfg.max_bounces
                  << " bounces) -> " << pathtraced_out << std::endl;
        mfad::PathTracerOptions pt_opts;
        pt_opts.use_sky_gradient = scene.use_sky_gradient;
        pt_opts.background_color = scene.background_color;
        pt_opts.max_bounces = cfg.max_bounces;
        pt_opts.min_rr_bounces = cfg.min_rr_bounces;
        pt_opts.sample_lights = true;
        pt_opts.firefly_clamp = 5.0f;
        for (auto& quad : scene.area_lights) {
            pt_opts.area_lights.push_back(std::dynamic_pointer_cast<mfad::Quad>(quad));
        }
        // Soft fill / rim point lights are additive with area lights. Skip
        // direct_only stand-ins that duplicate an area panel for Stage 0/1.
        for (const auto& pl : scene.point_lights) {
            if (!pl.direct_only) {
                pt_opts.point_lights.push_back(pl);
            }
        }
        std::cout << "[Stage 2] Lights: " << pt_opts.area_lights.size() << " area, "
                  << pt_opts.point_lights.size() << " point (NEE on)" << std::endl;
        if (pt_opts.area_lights.empty() && pt_opts.point_lights.empty()) {
            std::cerr << "[Stage 2] WARNING: no sampleable lights — image will be dark/wrong\n";
        }

        mfad::PathTracer tracer(pt_opts);
        tracer.render(*scene.camera_data.camera, scene.hittables, pt_image, cfg.path_spp);
        // Keep unexposed linear HDR for Stage 3 re-runs
        pt_image.write_hdr(out_dir + "/03_pathtraced.hdr");
        apply_exposure(pt_image, cfg.exposure);
        pt_image.write_png(pathtraced_out);
        have_pathtraced = true;
    }

    // ---------------------------------------------------------
    // Stage 3: SVD low-rank denoising (final LA application)
    // ---------------------------------------------------------
    if (cfg.stage_denoise) {
        bool input_is_linear = have_pathtraced;
        if (!have_pathtraced) {
            const std::string hdr_path = out_dir + "/03_pathtraced.hdr";
            if (std::filesystem::exists(hdr_path) && pt_image.read_hdr(hdr_path)) {
                std::cout << "[Stage 3] Loaded linear HDR from " << hdr_path << std::endl;
                input_is_linear = true;
            } else if (pt_image.read_png(pathtraced_out)) {
                std::cerr << "[Stage 3] Warning: denoise from display PNG (no HDR). "
                             "Prefer re-running with path_traced enabled.\n";
                for (int y = 0; y < pt_image.height(); ++y) {
                    for (int x = 0; x < pt_image.width(); ++x) {
                        Eigen::Vector3f c = pt_image.get_pixel(x, y);
                        c = c.array().pow(2.2f).matrix();
                        pt_image.set_pixel(x, y, c);
                    }
                }
                input_is_linear = false;
            } else {
                std::cerr << "[Stage 3] Skipped: no path-traced image at " << pathtraced_out
                          << std::endl;
                return false;
            }
        } else {
            // pt_image already had exposure applied for the PNG; reload HDR for clean denoise
            pt_image.read_hdr(out_dir + "/03_pathtraced.hdr");
        }
        denoise_and_save(pt_image, denoised_out, cfg.svd_rank, cfg.exposure, input_is_linear);
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "================================================" << std::endl;
    std::cout << "  MFAD Linear Algebra Path Tracer" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    std::cout << "OpenMP maximum threads: " << omp_get_max_threads() << std::endl;

    // Single config source: --config <file> (default: config.json)
    std::string config_path = "config.json";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }
    PipelineConfig cfg = load_config(config_path);
    std::cout << "[Config] " << config_path << ":" << std::endl;
    for (const auto& s : cfg.scenes) {
        std::cout << "  scene       : " << s << std::endl;
    }
    std::cout << "  output_dir  : " << cfg.output_dir << std::endl;
    std::cout << "  image       : " << cfg.width << "x" << cfg.height << std::endl;
    std::cout << "  direct_spp  : " << cfg.direct_spp << std::endl;
    std::cout << "  path_spp    : " << cfg.path_spp << std::endl;
    std::cout << "  svd_rank    : " << cfg.svd_rank << std::endl;
    std::cout << "  exposure    : " << cfg.exposure << std::endl;

    std::filesystem::create_directories(cfg.output_dir);

    bool all_ok = true;
    for (const auto& scene_file : cfg.scenes) {
        std::cout << "\n================================================" << std::endl;
        std::cout << "  Scene: " << scene_file << std::endl;
        std::cout << "================================================" << std::endl;
        if (!run_scene_pipeline(cfg, scene_file)) {
            std::cerr << "[ERROR] Pipeline failed for " << scene_file << std::endl;
            all_ok = false;
        }
    }

    std::cout << "\n[DONE] All outputs in " << cfg.output_dir << "/<scene>/" << std::endl;
    return all_ok ? 0 : 1;
}
