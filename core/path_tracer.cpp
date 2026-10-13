#include "path_tracer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>

namespace mfad {

PathTracer::PathTracer(const PathTracerOptions& options) : options_(options) {
    update_cached_lights();
}

void PathTracer::set_options(const PathTracerOptions& options) {
    options_ = options;
    update_cached_lights();
}

void PathTracer::update_cached_lights() {
    cached_area_lights_.clear();
    for (const auto& light_quad : options_.area_lights) {
        if (!light_quad || !light_quad->material()) {
            continue;
        }
        auto diff_light = dynamic_cast<const DiffuseLight*>(light_quad->material().get());
        if (!diff_light) {
            continue;
        }
        CachedAreaLight cal;
        cal.quad = light_quad;
        cal.normal = light_quad->normal();
        cal.area = light_quad->area();
        cal.emit = diff_light->emit();
        cal.two_sided = diff_light->two_sided();
        cached_area_lights_.push_back(cal);
    }
}

Eigen::Vector3f PathTracer::trace_ray(const Ray& r, const Hittable& scene) const {
    Eigen::Vector3f L = Eigen::Vector3f::Zero();
    Eigen::Vector3f throughput = Eigen::Vector3f::Ones();
    Ray cur_ray = r;
    bool prev_was_specular = true;

    for (int bounce = 0; bounce < options_.max_bounces; ++bounce) {
        HitRecord rec;
        if (!scene.hit(cur_ray, 0.001f, 1e8f, rec)) {
            if (options_.use_sky_gradient) {
                // Muted fill sky so area lights stay dominant
                float dir_y = cur_ray.direction.y();
                const Eigen::Vector3f sky_zenith(0.08f, 0.12f, 0.22f);
                const Eigen::Vector3f sky_horizon(0.18f, 0.20f, 0.24f);
                const Eigen::Vector3f ground_nadir(0.06f, 0.06f, 0.07f);

                Eigen::Vector3f sky;
                if (dir_y >= 0.0f) {
                    float t = dir_y;
                    sky = (1.0f - t) * sky_horizon + t * sky_zenith;
                } else {
                    float t = -dir_y;
                    sky = (1.0f - t) * sky_horizon + t * ground_nadir;
                }
                L += throughput.cwiseProduct(sky);
            } else {
                L += throughput.cwiseProduct(options_.background_color);
            }
            break;
        }

        // Add emission from surface
        if (rec.material != nullptr) {
            Eigen::Vector3f emit = rec.material->emitted(cur_ray, rec);
            if (!emit.isZero()) {
                if (!options_.sample_lights || prev_was_specular) {
                    L += throughput.cwiseProduct(emit);
                }
            }
        }

        // Scatter ray
        ScatterRecord srec;
        if (rec.material == nullptr || !rec.material->scatter(cur_ray, rec, srec)) {
            break;
        }

        // Check if current material is specular (dielectric or metal)
        bool is_specular = (rec.material && rec.material->is_specular());

        // Direct light sampling (Next Event Estimation) on diffuse surfaces
        if (options_.sample_lights && !is_specular) {
            constexpr float pi = 3.14159265358979323846f;
            const Eigen::Vector3f albedo = srec.attenuation;

            // Sample Area Lights (Quads)
            for (const auto& light : cached_area_lights_) {
                // Sample random point on quad light
                Eigen::Vector3f light_pt = light.quad->sample_point(random_float(), random_float());
                Eigen::Vector3f d = light_pt - rec.point;
                float dist_sq = d.squaredNorm();
                float dist = std::sqrt(dist_sq);
                if (dist < 1e-4f) {
                    continue;
                }

                Eigen::Vector3f dir = d / dist;
                float cos_theta_s = rec.normal.dot(dir);
                if (cos_theta_s <= 0.0f) {
                    continue;
                }

                float cos_theta_l = -light.normal.dot(dir);
                if (light.two_sided) {
                    cos_theta_l = std::abs(cos_theta_l);
                }
                if (cos_theta_l <= 0.0f) {
                    continue;
                }

                // Cast shadow ray to check occlusion
                Ray shadow_ray(rec.point + 1e-4f * rec.normal, dir);
                HitRecord occluder;
                if (!scene.hit(shadow_ray, 0.001f, dist - 0.001f, occluder)) {
                    float G = (cos_theta_s * cos_theta_l) / dist_sq;
                    L += (G * light.area / pi) *
                         throughput.cwiseProduct(albedo).cwiseProduct(light.emit);
                }
            }

            // Sample Point Lights
            for (const auto& pl : options_.point_lights) {
                Eigen::Vector3f d = pl.position - rec.point;
                float dist_sq = d.squaredNorm();
                float dist = std::sqrt(dist_sq);
                if (dist < 1e-4f) {
                    continue;
                }

                Eigen::Vector3f dir = d / dist;
                float cos_theta_s = rec.normal.dot(dir);
                if (cos_theta_s <= 0.0f) {
                    continue;
                }

                Ray shadow_ray(rec.point + 1e-4f * rec.normal, dir);
                HitRecord occluder;
                float t_max = dist - pl.radius - 0.001f;
                bool occluded = false;
                if (t_max > 0.001f && scene.hit(shadow_ray, 0.001f, t_max, occluder)) {
                    if (!occluder.material ||
                        occluder.material->emitted(shadow_ray, occluder).squaredNorm() <= 1e-4f) {
                        occluded = true;
                    }
                }
                if (!occluded) {
                    float atten = 1.0f / dist_sq;
                    L += (cos_theta_s * atten / pi) *
                         throughput.cwiseProduct(albedo).cwiseProduct(pl.intensity);
                }
            }
        }

        prev_was_specular = is_specular;
        throughput = throughput.cwiseProduct(srec.attenuation);
        cur_ray = srec.scattered;

        // Russian Roulette termination (unbiased)
        if (bounce >= options_.min_rr_bounces) {
            // Guard against NaN in throughput (can occur from degenerate geometry)
            if (!throughput.allFinite()) {
                break;
            }
            // Photometric luminance: Y = 0.2126 R + 0.7152 G + 0.0722 B
            float p_survive =
                0.2126f * throughput.x() + 0.7152f * throughput.y() + 0.0722f * throughput.z();
            p_survive = std::clamp(p_survive, options_.rr_survival_clamp_min,
                                   options_.rr_survival_clamp_max);

            if (random_float() > p_survive) {
                break;  // Path terminated
            }
            throughput /= p_survive;  // Boost surviving throughput to maintain unbiased estimator
        }
    }

    if (options_.firefly_clamp > 0.0f) {
        return L.cwiseMin(Eigen::Vector3f::Constant(options_.firefly_clamp));
    }
    return L;
}

void PathTracer::render(const Camera& camera, const Hittable& scene, ImageBuffer& buffer,
                        int samples_per_pixel) const {
    const int width = buffer.width();
    const int height = buffer.height();

    int rows_done = 0;
    auto t_start = std::chrono::steady_clock::now();
    int last_reported_pct = -1;
#pragma omp parallel for schedule(dynamic, 8)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Eigen::Vector3f pixel_col = Eigen::Vector3f::Zero();
            for (int s = 0; s < samples_per_pixel; ++s) {
                float u = (static_cast<float>(x) + random_float()) / static_cast<float>(width - 1);
                // Scanline flip: bottom-left is (0, 0)
                float v = (static_cast<float>(height - 1 - y) + random_float()) /
                          static_cast<float>(height - 1);

                Ray ray = camera.generate_ray(u, v);
                pixel_col += trace_ray(ray, scene);
            }
            pixel_col /= static_cast<float>(samples_per_pixel);
            buffer.set_pixel(x, y, pixel_col);
        }
#pragma omp critical
        {
            ++rows_done;
            int pct = 100 * rows_done / height;
            if (pct != last_reported_pct && (pct % 5 == 0 || rows_done == height)) {
                last_reported_pct = pct;
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t_start).count();
                double total_est = (pct > 0) ? (elapsed / pct * 100.0) : 0.0;
                double remaining = std::max(0.0, total_est - elapsed);
                std::cerr << "\r[PathTracer] Progress: " << pct << "% (" << rows_done << "/"
                          << height << " rows) - Elapsed: " << std::fixed << std::setprecision(1)
                          << elapsed << "s, ETA: " << std::fixed << std::setprecision(1)
                          << remaining << "s    " << std::flush;
            }
        }
    }
    if (height > 0)
        std::cerr << std::endl;
}

}  // namespace mfad
