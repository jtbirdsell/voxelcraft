# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch Minecraft-style voxel sandbox in Rust + wgpu (no game engine). Everything is generated in code — textures, fonts, and worldgen; the only assets are the WGSL shaders in `assets/shaders/`. Developed for a Windows RTX 4090 box (DX12 + hardware ray tracing + DLSS), and also builds and runs on macOS / Apple Silicon (Metal backend → software DDA lighting; hardware RT + DLSS stay Windows/RTX-only). The DLSS deps are gated to `cfg(windows)`, so plain `cargo run --release` works on both OSes; `--no-default-features` additionally builds without any NVIDIA SDK on Windows.

## Commands

```sh
cargo run --release                      # day-to-day build (debug builds run worldgen/mesher too slow to play)
cargo test --release                     # all tests (worldgen determinism, physics, inventory, save round-trips)
cargo test --release <test_name>        # single test
cargo build --no-default-features        # no DLSS — compiles without NVIDIA SDKs / libclang / Streamline
cargo build --profile dist               # fat-LTO shipping/benchmark build (release profile has LTO off for fast relinks)
```

Tests are inline `#[cfg(test)]` unit tests in the module they cover (heaviest in `gameplay/`, `world/worldgen.rs`, `persistence.rs`, `game.rs`).

### Headless verification

`VOXELCRAFT_SHOT=path.png cargo run --release` renders one frame offscreen to a PNG and exits — the standard way to verify a change without a human. Companion knobs: `VOXELCRAFT_CAM="x,y,z,yaw,pitch"`, `VOXELCRAFT_TIME=secs`, `VOXELCRAFT_PLACE="x,y,z,id;..."`, `VOXELCRAFT_SCREEN=inv|craft|furnace`, `VOXELCRAFT_ROOM`, `VOXELCRAFT_SURVIVAL=1`, `VOXELCRAFT_CRACK="x,y,z,progress"`. `VOXELCRAFT_PERSIST_TEST=1` round-trips save/load state with no window. Rendering knobs: `VOXELCRAFT_BACKEND=dx12|vulkan|gl|metal`, `VOXELCRAFT_TRACER=dda|hwrt`, `VOXELCRAFT_RTX=0|1|2` (force lighting off/shadows/shadows+GI — cost isolation), `VOXELCRAFT_GI=fragment|compute` (fragment is the bit-for-bit parity oracle for the deferred compute path), `VOXELCRAFT_GI_RAYS=N`, `VOXELCRAFT_RENDER_SCALE=0.25..1.0` (Metal-only sub-native swapchain, physical fraction; defaults to 0.57× logical on macOS Retina — set `1.0` for native-pixel shots / cross-OS comparisons), quality-tier overrides (`VOXELCRAFT_SUN_DIST`, `VOXELCRAFT_GI_DIST`, `VOXELCRAFT_GI_SUN_DIST`, `VOXELCRAFT_GI_ACCUM=1|0`, `VOXELCRAFT_WSMOOTH`, `VOXELCRAFT_WREFL`, `VOXELCRAFT_WDEPTH`, `VOXELCRAFT_VOLUME_CHUNKS`, `VOXELCRAFT_RENDER_DISTANCE`, `VOXELCRAFT_UPLOAD_BUDGET` — see README table; Windows resolves to the legacy values), `VOXELCRAFT_DLSS=off|rr`, `VOXELCRAFT_SS=N`, `VOXELCRAFT_FG=1`, `VOXELCRAFT_GPU_WATCHDOG=secs` (windowed wedge fail-safe — save + exit when submitted frames stop completing; default 5, `0` off).

**Headless benchmark (P21):** `VOXELCRAFT_BENCH=N cargo run --release` renders N timed frames offscreen (no swapchain — safe on this Mac) and prints p50/p95 `gpu_frame` + per-pass GPU timings (stage-boundary timestamp queries; per-pass render timings OVERLAP on Apple TBDR — trust `gpu_frame` absolutely, per-pass relatively; `gi_compute` is reliable). Knobs: `VOXELCRAFT_BENCH_WARMUP`, `VOXELCRAFT_BENCH_LABEL`, `VOXELCRAFT_BENCH_CSV` (append rows for sweeps), `VOXELCRAFT_BENCH_GPU=0`. Bench defaults GI rays to the tier's interactive value (not the SHOT path's 64). Canonical scene: default cam + `VOXELCRAFT_TIME=0.30`. M3 reference: tier defaults ≈ 13.7 ms p50; pre-tier ≈ 147 ms. The GPU target is ~13–14 ms, NOT 16.6: the windowed loop's CPU work serializes behind the GPU (Metal `desired_maximum_frame_latency: 1`) and ProMotion quantizes any cycle over 16.67 ms straight down to 40 FPS — headroom is the difference between 60 and 40 in practice.

**macOS GPU-hang hazard:** interactive (windowed) runs on Apple Silicon have wedged the AGX GPU driver irrecoverably (2026-06-03: machine froze, power-button reset; the `gpuEvent`/WindowServer-watchdog telemetry attributed it to voxelcraft's present path — headless was unaffected and every shader loop is statically bounded). Verify with headless `VOXELCRAFT_SHOT` runs, never a windowed launch; if a windowed test is unavoidable, warn the user first, keep it short, and rely on the GPU watchdog + surface backoff (both added after the incident) to fail safe.

## Critical constraints

- **wgpu is pinned exact (`=29.0.3`) and patched** to a fork (`jtbirdsell/wgpu`) that adds raw DX12 accessors for DLSS. The experimental hardware-RT API breaks every wgpu release — bumping wgpu is a deliberate mini-project, not a routine update. The patch rev must match `dlss_wgpu_dx12`'s own wgpu pin.
- **DLSS is feature-gated AND Windows-gated** (`dlss`, `frame-generation`; both default-on, but the real modules compile only under `all(feature, target_os = "windows")` — see `src/gfx.rs`). When off *or off-Windows*, `gfx/dlss_stub.rs` / `gfx/frame_gen_stub.rs` compile in place of the real modules via `#[path]` aliases (uninhabited types, constructors return `None`), so engine plumbing like `Option<DlssRender>` stays valid. Keep both feature combinations compiling — and note macOS builds only type-check the stubs, so changes to the DLSS surface (renderer/app/capture call sites) still require a Windows build before merge.
- **Backend default is platform-aware** (`device.rs::backend_order`): DX12 first on Windows/Linux, Metal first on macOS (no hardware RT there → software DDA tracer; `rt_probe`/`rt_spike` degrade to no-op diagnostics). Keep the non-macOS order unchanged when editing. On macOS the swapchain defaults to `quality::METAL_LOGICAL_SCALE` (0.57) × logical resolution (`Gpu::render_scale`, Metal drawable-stretch) — the Retina ray-budget lever.
- **Platform quality tier (P21, `gfx/quality.rs`)**: every GPU-cost knob resolves once at startup — `Quality::maxed()` on non-Metal backends **must stay equal to the pre-P21 hardcoded defaults** (unit tests anchor each field; the values also flow into the shader `Volume.paramsg` uniform so the shared HW-RT/DDA WGSL is bit-identical on Windows). Tune the Mac in `Quality::metal()` + `METAL_LOGICAL_SCALE`, never by editing shared shader literals. Tier changes need a `VOXELCRAFT_BENCH` before/after on the Mac and (eventually) a defaults-PNG byte-compare on the Windows box.
- **Worldgen is seed-deterministic** — only player-edited chunks are persisted (LZ4 to `saves/world/chunks.bin`); everything else regenerates from the seed. Worldgen changes that alter output break existing saves and the determinism tests; refactors are expected to be byte-identical (see commit `P20`).
- **Saves are forward-compatible**: older `level.bin`/`chunks.bin` must keep loading. Extend persistence formats additively.

## Architecture

Module layout uses a `foo.rs + foo/` facade pattern (M33-G1 grouping); submodules are re-exported at the crate root so `crate::camera`-style paths still resolve.

- `main.rs` — module tree + winit event-loop entry. `app.rs` — `App`/`State`, input routing, per-frame update + render, headless screenshot path.
- `game.rs` — the streaming manager: keeps chunks generated/meshed around the camera, frustum-culls, applies edits, ticks fluids/furnaces, unloads distant chunks.
- `world/` — `chunk` (32³ blocks + light), `block`, `worldgen` (noise terrain/biomes/caves/ores/trees, `Arc`-shared across workers), `light` (skylight + block-light flood, baked per-vertex at mesh time), `worker` (crossbeam pool).
- `mesh/mesher.rs` — binary greedy mesher splitting geometry into opaque/water/glass buckets, plus cross-billboards and slab/stair/door/fence per-cell boxes. Block orientation/state lives in a per-block **block-state byte**, baked into geometry at mesh time so the GPU path never sees it.
- `gfx/` — wgpu device/surface (`device`), pipelines + HDR frame recording (`renderer`), render-graph targets (`graph`: HDR scene buffer + G-buffer of normal/motion/position/albedo), hardware RT acceleration structures (`rt`: per-chunk BLAS + per-frame TLAS), GPU voxel volume for the software DDA fallback tracer (`voxel_volume`), procedural texture atlas (`texture`), DLSS Ray Reconstruction (`dlss`) and Frame Generation (`frame_gen`).
- `gameplay/` — swept-AABB player physics (`player`), mobs/items/XP entities + AI state machine (`entity`), item/tool registry + `Inventory` (`item`), data-driven recipes (`crafting`, `smelting`), DDA block targeting (`raycast`), difficulty rules (`rules`), generic `Container` shared by chests/furnaces.
- `ui/` — HUD/hotbar/inventory screens (`overlay`), embedded 8×8 bitmap font (`font`).
- `persistence.rs` — LZ4 chunk save/load, level header, inventory/survival state.
- `build.rs` — stages `dxcompiler.dll`/`dxil.dll` (DXC, required for DX12 hardware RT) and the Streamline/DLSS-G DLLs next to the exe; all non-fatal if missing.

**Threading model:** workers generate chunks and build meshes from immutable snapshots; the main thread is the *only* mutator of the world map and does budgeted per-frame GPU uploads. Keep it that way — don't add world mutation off the main thread.

**Lighting paths:** two parallel implementations exist by design — hardware `rayQuery` (BLAS/TLAS) and a software DDA march over a toroidal voxel volume — switchable via `VOXELCRAFT_TRACER`. Same for GI: deferred compute pass vs. in-shader fragment gather (`VOXELCRAFT_GI`), kept as a parity oracle. Lighting changes generally need to be made (or at least verified) in both.

## Conventions

- Commit subjects use a milestone prefix (`P20: ...`) or conventional type (`fix(visuals): ...`); code comments reference milestone tags like `(M33-G8)`.
- The README's Features and Architecture sections are the spec of record — update them when behavior or module layout changes.
