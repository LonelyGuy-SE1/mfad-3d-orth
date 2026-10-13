#!/bin/bash
# ============================================================
#  MFAD Path Tracer — single entry point
#
#  ALL settings (scene, resolution, spp, stages, svd rank,
#  output folder) live in config.json. Edit that one file.
#
#  Usage:
#    ./run.sh                     # full pipeline, config.json defaults
#    ./run.sh my_config.json      # same pipeline, different config
# ============================================================
set -e
cd "$(dirname "$0")"

CONFIG=${1:-config.json}

# Build if the binary is missing or any source is newer
if [ ! -f build/path_tracer ] || \
   [ -n "$(find core stages \( -name '*.cpp' -o -name '*.hpp' \) -newer build/path_tracer -print -quit)" ]; then
    echo "[build] Compiling..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build build -j"$(nproc)" > /dev/null
    echo "[build] Done."
fi

./build/path_tracer --config "$CONFIG"
