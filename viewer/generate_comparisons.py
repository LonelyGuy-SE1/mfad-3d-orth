#!/usr/bin/env python3
"""Generate labeled side-by-side comparison images for presentations and reports."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_paths = [
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidate_paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def generate_all_comparisons(root_dir: Path = Path(".")) -> None:
    raw_path = root_dir / "final_demo_raw.png"
    direct_path = root_dir / "final_demo_direct.png"
    pt_path = root_dir / "final_demo_pathtraced.png"
    denoised_path = root_dir / "final_demo_denoised.png"
    spp32_path = root_dir / "pathtraced_32spp.png"
    spp512_path = root_dir / "pathtraced_512spp.png"
    spp32k_path = root_dir / "pathtraced_32768spp.png"

    # Create intuitive alias copies in root
    if raw_path.exists():
        shutil.copyfile(raw_path, root_dir / "initial_canvas.png")
    if direct_path.exists():
        shutil.copyfile(direct_path, root_dir / "initial_direct.png")
    if pt_path.exists():
        shutil.copyfile(pt_path, root_dir / "final_pathtraced.png")
    if denoised_path.exists():
        shutil.copyfile(denoised_path, root_dir / "final_denoised.png")

    title_font = get_font(22, bold=True)
    header_font = get_font(16, bold=True)
    sub_font = get_font(12, bold=False)

    pad = 20
    top_banner = 70
    bottom_caption = 52

    # 1. comparison_initial_vs_final.png
    if raw_path.exists() and pt_path.exists():
        img_init = Image.open(raw_path)
        img_final = Image.open(pt_path)
        w, h = img_init.size

        canvas_w = pad * 3 + w * 2
        canvas_h = top_banner + h + bottom_caption + pad

        comp1 = Image.new("RGB", (canvas_w, canvas_h), color=(18, 20, 24))
        draw1 = ImageDraw.Draw(comp1)
        draw1.text(
            (canvas_w // 2, 35),
            "MFAD Path Tracer: Initial Canvas vs. Final Path-Traced Render",
            fill=(240, 240, 245),
            font=title_font,
            anchor="mm",
        )

        x1, y1 = pad, top_banner
        x2, y2 = pad * 2 + w, top_banner

        comp1.paste(img_init, (x1, y1))
        comp1.paste(img_final, (x2, y2))
        draw1.rectangle([x1 - 1, y1 - 1, x1 + w, y1 + h], outline=(60, 68, 80), width=1)
        draw1.rectangle([x2 - 1, y2 - 1, x2 + w, y2 + h], outline=(60, 68, 80), width=1)

        draw1.text(
            (x1 + w // 2, y1 + h + 18),
            "INITIAL VERSION: Raw Canvas (Pre-GI)",
            fill=(100, 180, 255),
            font=header_font,
            anchor="mm",
        )
        draw1.text(
            (x1 + w // 2, y1 + h + 38),
            "Unshadowed materials, dielectric glass refraction, chrome reflection",
            fill=(160, 165, 175),
            font=sub_font,
            anchor="mm",
        )

        draw1.text(
            (x2 + w // 2, y2 + h + 18),
            "FINAL VERSION: Monte Carlo Path Tracing (32,768 SPP)",
            fill=(80, 220, 120),
            font=header_font,
            anchor="mm",
        )
        draw1.text(
            (x2 + w // 2, y2 + h + 38),
            "16 bounces, soft shadows, diffuse color bleeding, caustics (21.9 min)",
            fill=(160, 165, 175),
            font=sub_font,
            anchor="mm",
        )

        out1 = root_dir / "comparison_initial_vs_final.png"
        comp1.save(out1)
        print(f"[SUCCESS] Saved {out1}")

    # 2. comparison_stages.png (2x2 grid)
    if all(p.exists() for p in [raw_path, direct_path, pt_path, denoised_path]):
        img0 = Image.open(raw_path)
        img1 = Image.open(direct_path)
        img2 = Image.open(pt_path)
        img3 = Image.open(denoised_path)
        w, h = img0.size

        grid_w = pad * 3 + w * 2
        grid_h = top_banner + (h + bottom_caption) * 2 + pad

        comp2 = Image.new("RGB", (grid_w, grid_h), color=(18, 20, 24))
        draw2 = ImageDraw.Draw(comp2)
        draw2.text(
            (grid_w // 2, 35),
            "MFAD Progressive 4-Stage Rendering Pipeline (scenes/final_demo.json)",
            fill=(240, 240, 245),
            font=title_font,
            anchor="mm",
        )

        stages = [
            (
                img0,
                pad,
                top_banner,
                "STAGE 0: Raw Initial Canvas (Pre-GI)",
                "Authentic materials, unshadowed geometry, direct ambient",
                (100, 180, 255),
            ),
            (
                img1,
                pad * 2 + w,
                top_banner,
                "STAGE 1: Direct Lighting (Whitted)",
                "Raytraced shadow rays, sharp directional shadows, specular bounce",
                (255, 195, 75),
            ),
            (
                img2,
                pad,
                top_banner + h + bottom_caption,
                "STAGE 2: Path Traced GI (32,768 SPP)",
                "Full Monte Carlo global illumination, soft penumbras, caustics",
                (80, 220, 120),
            ),
            (
                img3,
                pad * 2 + w,
                top_banner + h + bottom_caption,
                "STAGE 3: SVD Low-Rank Denoising",
                "Truncated SVD (Rank 120 matrix approximation) across RGB",
                (210, 120, 255),
            ),
        ]

        for img, x, y, title, subtitle, color in stages:
            comp2.paste(img, (x, y))
            draw2.rectangle([x - 1, y - 1, x + w, y + h], outline=(60, 68, 80), width=1)
            draw2.text((x + w // 2, y + h + 18), title, fill=color, font=header_font, anchor="mm")
            draw2.text(
                (x + w // 2, y + h + 38), subtitle, fill=(160, 165, 175), font=sub_font, anchor="mm"
            )

        out2 = root_dir / "comparison_stages.png"
        comp2.save(out2)
        print(f"[SUCCESS] Saved {out2}")

    # 3. comparison_noise_convergence.png (3-panel)
    if all(p.exists() for p in [spp32_path, spp512_path, spp32k_path]):
        img_32 = Image.open(spp32_path)
        img_512 = Image.open(spp512_path)
        img_32k = Image.open(spp32k_path)
        w, h = img_32.size

        strip_w = pad * 4 + w * 3
        strip_h = top_banner + h + bottom_caption + pad

        comp3 = Image.new("RGB", (strip_w, strip_h), color=(18, 20, 24))
        draw3 = ImageDraw.Draw(comp3)
        draw3.text(
            (strip_w // 2, 35),
            "Monte Carlo Sampling Convergence: Noise & Firefly Dissolution",
            fill=(240, 240, 245),
            font=title_font,
            anchor="mm",
        )

        spp_panels = [
            (
                img_32,
                pad,
                "INITIAL: 32 SPP (1.20 s)",
                "Heavy grain, visible firefly spikes in caustics pool",
                (255, 100, 100),
            ),
            (
                img_512,
                pad * 2 + w,
                "INTERMEDIATE: 512 SPP (19.62 s)",
                "Grain reduced by 4x, caustics starting to resolve",
                (255, 195, 75),
            ),
            (
                img_32k,
                pad * 3 + w * 2,
                "CONVERGED: 32,768 SPP (21.91 min)",
                "Noise-free reference, pristine caustics & reflections",
                (80, 220, 120),
            ),
        ]

        for img, x, title, subtitle, color in spp_panels:
            y = top_banner
            comp3.paste(img, (x, y))
            draw3.rectangle([x - 1, y - 1, x + w, y + h], outline=(60, 68, 80), width=1)
            draw3.text((x + w // 2, y + h + 18), title, fill=color, font=header_font, anchor="mm")
            draw3.text(
                (x + w // 2, y + h + 38), subtitle, fill=(160, 165, 175), font=sub_font, anchor="mm"
            )

        out3 = root_dir / "comparison_noise_convergence.png"
        comp3.save(out3)
        print(f"[SUCCESS] Saved {out3}")


if __name__ == "__main__":
    generate_all_comparisons()
