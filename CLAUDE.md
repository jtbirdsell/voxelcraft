# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A from-scratch Minecraft-style voxel sandbox in Rust + wgpu (no game engine). Everything is generated in code — textures, fonts, and worldgen; the only assets are the WGSL shaders in `assets/shaders/`. Developed for a Windows RTX 4090 box (DX12 + hardware ray tracing + DLSS), but builds without any NVIDIA SDK via `--no-default-features`.

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

`VOXELCRAFT_SHOT=path.png cargo run --release` renders one frame offscreen to a PNG and exits — the standard way to verify a change without a human. Companion knobs: `VOXELCRAFT_CAM="x,y,z,yaw,pitch"`, `VOXELCRAFT_TIME=secs`, `VOXELCRAFT_PLACE="x,y,z,id;..."`, `VOXELCRAFT_SCREEN=inv|craft|furnace`, `VOXELCRAFT_ROOM`, `VOXELCRAFT_SURVIVAL=1`, `VOXELCRAFT_CRACK="x,y,z,progress"`. `VOXELCRAFT_PERSIST_TEST=1` round-trips save/load state with no window. Rendering knobs: `VOXELCRAFT_BACKEND=dx12|vulkan|gl`, `VOXELCRAFT_TRACER=dda|hwrt`, `VOXELCRAFT_GI=fragment|compute` (fragment is the bit-for-bit parity oracle for the deferred compute path), `VOXELCRAFT_GI_RAYS=N`, `VOXELCRAFT_DLSS=off|rr`, `VOXELCRAFT_SS=N`, `VOXELCRAFT_FG=1`.

## Critical constraints

- **wgpu is pinned exact (`=29.0.3`) and patched** to a fork (`jtbirdsell/wgpu`) that adds raw DX12 accessors for DLSS. The experimental hardware-RT API breaks every wgpu release — bumping wgpu is a deliberate mini-project, not a routine update. The patch rev must match `dlss_wgpu_dx12`'s own wgpu pin.
- **DLSS is feature-gated** (`dlss`, `frame-generation`; both default-on). When off, `gfx/dlss_stub.rs` / `gfx/frame_gen_stub.rs` compile in place of the real modules via `#[path]` aliases (uninhabited types, constructors return `None`), so engine plumbing like `Option<DlssRender>` stays valid. Keep both feature combinations compiling.
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
