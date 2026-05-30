# Voxelcraft

A Minecraft-equivalent voxel sandbox written from scratch in **Rust + wgpu**, tuned for an
RTX 4090 / i9-14900K. Infinite procedurally-generated world, multithreaded chunk streaming,
break/place building, day/night, transparent water, world save/load, and **ray-traced sun
shadows** computed against the actual voxel geometry on the GPU.

## Run

The Rust toolchain (MSVC) is already installed. From the project directory:

```powershell
cargo run --release
```

First release build takes ~50s (fat LTO); after that it's instant. A debug build (`cargo run`)
compiles faster but the worldgen/meshing run unoptimized, so use `--release` to play.

`cargo test --release` runs the unit tests (worldgen determinism, physics, persistence).

## Controls

| Input | Action |
|---|---|
| **WASD** | Move |
| **Mouse** | Look |
| **Space** | Jump (walk) / ascend (fly) |
| **Left-Shift** | Descend (fly) |
| **Left-Ctrl** | Sprint / fly boost |
| **F** | Toggle fly / walk |
| **Left-click** | Break block |
| **Right-click** | Place block |
| **1–9 / scroll** | Select hotbar block |
| **R** | Toggle ray-traced shadows |
| **P** | Save world |
| **Esc** | Quit |

The world saves automatically on quit to `saves/world/`.

## What's implemented

- **M1** — winit 0.30 + wgpu 29 on the **DX12** backend, depth buffer, fly camera.
- **M2** — infinite world streamed as 32³ chunks, generated + **binary greedy meshed** across a
  worker pool (cores − 2 threads), CPU frustum culling.
- **M3** — multi-octave OpenSimplex2 terrain, parameter-space **biomes**, 3D-noise **caves**,
  ores, sea-level water, deterministic **trees** (no cross-chunk writes). Seed-deterministic.
- **M4** — swept-AABB **player physics** (walk/fly, gravity, jump), 3D-DDA **block targeting**,
  **break/place** with incremental re-mesh, block highlight, hotbar + crosshair HUD.
- **M5** — **day/night** cycle, dynamic sky + distance **fog**, translucent **water**, world
  **save/load** (LZ4-packed edited chunks; unedited chunks regenerate from the seed).
- **M6** — **ray-traced sun shadows**: a GPU-resident 256³ toroidal voxel occupancy volume that
  follows the player; the fragment shader DDA-marches a shadow ray toward the sun.

Performance: **~144 fps (vsync-capped)** at render distance 12 with shadows on — the GPU has
large headroom.

## Architecture

```
src/
  gpu.rs            wgpu device/surface/depth; DX12-preferred adapter selection
  camera.rs         fly camera + globals UBO (view-proj, sun, sky/fog)
  environment.rs    day/night: sun direction, sky/fog color, ambient/intensity
  world.rs          Chunk (32³), World store, neighbor view for meshing
  worldgen.rs       noise terrain, biomes, caves, ores, trees (Arc-shared across workers)
  mesher.rs         binary greedy mesher → opaque + translucent geometry
  worker.rs         crossbeam worker pool (generate + mesh off the main thread)
  game.rs           streaming manager: gen/mesh budgets, frustum cull, edits, saves
  voxel_volume.rs   GPU voxel occupancy volume for ray-traced shadows
  raycast.rs        Amanatides–Woo voxel DDA (block targeting)
  player.rs         AABB collision, gravity/jump/fly, input
  frustum.rs        Gribb–Hartmann frustum culling
  renderer.rs       pipelines (opaque/water/highlight/HUD), frame recording
  overlay.rs        block highlight + crosshair/hotbar geometry
  persistence.rs    LZ4 chunk save/load + level header
  capture.rs        offscreen screenshot (headless verification)
assets/shaders/     chunk / water / line / ui WGSL
```

**Hardware-driven choices:** worker threads keep all cores busy on generation/meshing while the
render thread stays light; greedy meshing + per-chunk draws keep geometry cheap; the over-powered
GPU is spent on ray-traced shadows rather than sitting idle.

## Possible next steps

Flowing fluids, survival (health/hunger, mobs, crafting), texture atlas, GPU-driven indirect
rendering for much larger render distances, and extending the voxel ray tracer from shadows to
full GI / reflections.
