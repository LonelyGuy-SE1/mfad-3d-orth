# Plan: live single-pixel path trace (optional)

Matrix traces already exist for the linear-algebra stages. Path radiance sampling
is Monte Carlo and is not logged per pixel by default.

## Available now

1. `docs/SINGLE_PIXEL_TRACE.json` — pixel walkthrough with concrete \(Q\), \(A\), SVD numbers.
2. `docs/prep_stage_trace.json` — machine trace from `python -m prep` (`mesh_checks`, `eigen`).
3. PNG pipeline outputs under `output/`.

## Ideal logged pixel (future)

For pixel \((x,y)\), one sample:

1. Camera basis \(Q_{\mathrm{cam}}\) + \(Q^\top Q\), \(\det\)
2. Primary ray \(o,d\)
3. Each bounce: hit, normal, local Duff \(Q\), NEE contribution, throughput
4. Accumulated radiance
5. Write into channel matrices (full image still needed for SVD)

## Implementation sketch

1. Add `PathTracerOptions::debug_pixel = {x,y}`.
2. Log via existing `mfad::Trace` (`trace/trace.hpp`, schema in `docs/TRACE_SPEC.md`).
3. Run a low-res config and dump `output/pixel_trace.json`.

```bash
python -m prep scenes/prep_demo.json -o /tmp/scene_prepared.json --trace docs/prep_stage_trace.json
./run.sh
```
