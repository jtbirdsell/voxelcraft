# Voxelcraft

A Minecraft-equivalent voxel sandbox written from scratch in **Rust + wgpu**, tuned for an
RTX 4090 / i9-14900K. Infinite procedurally-generated world, multithreaded chunk streaming,
break/place building, day/night, transparent water, **flowing fluids**, **survival**
(health/hunger/fall damage), **mobs + item drops**, world save/load, and **ray-traced lighting** —
sun shadows, ambient occlusion, one-bounce colored global illumination, water reflections, and
**emissive lava** that lights the scene — computed against the actual voxel geometry on the GPU.

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
| **R** | Cycle ray-traced lighting: off → shadows → shadows + GI |
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
- **M6** — **ray-traced sun shadows**: a GPU-resident 256³ toroidal voxel volume that follows the
  player; the fragment shader DDA-marches a shadow ray toward the sun.
- **M7** — **ray-traced ambient occlusion + one-bounce global illumination**: the volume now stores
  block ids (not just occupancy), so cosine-weighted hemisphere rays gather sky radiance on a miss
  and sun-lit *material color* on a hit — soft contact AO and colored light bleeding between blocks.
  `R` cycles off → shadows → shadows + GI; interactive traces a few rays per pixel, the headless
  screenshot path traces 64 for a clean image.
- **M8** — **ray-traced water reflections**: the water surface marches a mirror ray through the
  same voxel volume (lit material on a hit, sky + a sun glint on a miss) and blends it in by a
  Schlick-Fresnel term, so water mirrors the shoreline and sky at grazing angles and shows its own
  tint head-on. The DDA tracer now lives in one shared `rtx_common.wgsl` used by both shaders.
- **M9** — **flowing fluids + emissive lava**: placed water/lava are simulated by a cellular tick
  (a bounded frontier flood that falls, then spreads with diminishing reach, and cascades over
  ledges). Lava is **emissive** — the per-vertex color carries an emission channel, and GI /
  reflection rays treat a lava hit as a light source, so a lava lake glows and washes nearby blocks
  in orange indirect light (and reflects in water).
- **M10** — **survival basics**: in walk mode the player has health + hunger; a hard fall deals
  damage past a safe distance, hunger drains (faster sprinting), regenerates health when full and
  starves when empty, and death respawns at spawn. Red/orange pip bars render above the hotbar
  (flying is treated as creative — invulnerable, bars hidden).
- **M11** — **mobs + item drops**: an entity system with the same swept-AABB voxel collision as the
  player. Mobs wander with a small random AI; breaking a block drops a small item cube that falls,
  rests, bobs/spins, and is collected when walked over. Entities are drawn as boxes through the
  chunk pipeline, so they pick up the same ray-traced shadows / AO / GI as the world.

Performance: **~144 fps (vsync-capped)** at render distance 12 with shadows on — the GPU has
large headroom, which GI and reflections spend on per-pixel ray tracing.

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
  game.rs           streaming manager: gen/mesh budgets, frustum cull, edits, fluid tick, saves
  voxel_volume.rs   GPU voxel material volume (block ids) for ray-traced shadows + AO/GI
  raycast.rs        Amanatides–Woo voxel DDA (block targeting)
  player.rs         AABB collision, gravity/jump/fly, input, survival (health/hunger/fall damage)
  entity.rs         mobs + dropped items: AABB physics, wander AI, box geometry (GI-lit)
  frustum.rs        Gribb–Hartmann frustum culling
  renderer.rs       pipelines (opaque/water/highlight/HUD), frame recording
  overlay.rs        block highlight + crosshair/hotbar geometry
  persistence.rs    LZ4 chunk save/load + level header
  capture.rs        offscreen screenshot (headless verification)
assets/shaders/     rtx_common (shared bindings + voxel DDA tracer) + chunk / water / line / ui WGSL
```

**Hardware-driven choices:** worker threads keep all cores busy on generation/meshing while the
render thread stays light; greedy meshing + per-chunk draws keep geometry cheap; the over-powered
GPU is spent on ray-traced shadows, ambient occlusion and global illumination rather than idling.

## Possible next steps

An inventory + crafting grid; mob health/combat and more creature types; a texture atlas;
GPU-driven indirect rendering for much larger render distances; and a temporal/spatial denoiser so
interactive GI can use fewer rays per pixel without noise.
