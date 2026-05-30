# Voxelcraft

A Minecraft-style voxel sandbox written from scratch in **Rust + [wgpu](https://wgpu.rs)** (DirectX 12),
tuned for a high-end PC (developed on an RTX 4090 / i9-14900K). It features an infinite,
procedurally-generated world streamed across all CPU cores, a procedural texture atlas, real
block-light + skylight (dark caves, glowing torches), a full inventory, and **ray-traced lighting** —
sun shadows, ambient occlusion, one-bounce colored global illumination, water reflections, and
emissive blocks — all computed against the actual voxel geometry on the GPU.

> **Status:** the engine and world systems are complete and the survival loop is in active
> development (see [Roadmap](#roadmap)). Everything here is built from scratch — no game engine, and
> the textures, fonts, and worldgen are all generated in code.

## Run

Requires the Rust toolchain (stable, MSVC on Windows). From the project directory:

```sh
cargo run --release
```

The first release build takes ~50s (fat LTO); after that it's instant. A debug build (`cargo run`)
compiles faster but runs the worldgen/mesher unoptimized, so use `--release` to play.

```sh
cargo test --release   # worldgen determinism, physics, inventory + save round-trips
```

The world auto-saves to `saves/world/` on quit.

## Controls

| Input | Action |
|---|---|
| **WASD** | Move |
| **Mouse** | Look |
| **Space** | Jump (walk) / ascend (fly) |
| **Left-Shift** | Descend (fly) |
| **Left-Ctrl** | Sprint / fly boost |
| **F** | Toggle fly / walk |
| **Left-click (hold)** | Mine the targeted block (progressive, hardness-timed) |
| **Right-click** | Place block, or open a crafting table / furnace |
| **Q** | Drop one of the selected item |
| **1–9 / scroll** | Select hotbar slot |
| **E** | Open / close inventory (with a 2×2 craft grid) |
| **R** | Cycle ray-traced lighting: off → shadows → shadows + GI |
| **F3** | Toggle debug overlay (fps, position, biome, facing) |
| **P** | Save world |
| **Esc** | Close menu, or quit |

Mining yields drops (stone → cobblestone, ores need the right pickaxe tier) that fall into your
inventory; the hotbar shows stack counts, durability bars, and the selected item's name.

## Features

**World & rendering**
- Infinite world streamed as 32³ chunks, generated and **binary greedy-meshed** across a worker pool
  (cores − 2 threads), with CPU frustum culling.
- Multi-octave OpenSimplex terrain with parameter-space **biomes**, 3D-noise **caves**, depth-banded
  **ores**, a jagged **bedrock** floor, **deepslate** at depth, and deterministic **trees** and
  surface **decoration** (flowers, tall grass, cactus). Fully seed-deterministic.
- A **procedural texture atlas** painted in code at startup (stone, ores, planks, bricks, foliage,
  …), with cross-billboard plants drawn via **alpha cutout**.
- **Day/night** cycle with a dynamic sky, distance fog, translucent **water**, and **flowing fluids**
  (water/lava cellular simulation). Animated water ripples and lava.
- **Block-light + skylight** flood (0–15) baked per-vertex: caves are genuinely dark, and torches /
  glowstone / lava illuminate their surroundings.

**Ray-traced lighting** (a GPU-resident 768×256 toroidal voxel volume the fragment shaders DDA-march)
- **Sun shadows** against the real voxel geometry.
- **Ambient occlusion + one-bounce colored global illumination** (cosine-weighted hemisphere rays
  gather sky on a miss and sun-lit material color on a hit).
- **Water reflections** via a Schlick-Fresnel mirror ray, with depth-based clarity (Beer–Lambert).
- **Emissive blocks** (lava, glowstone, torches) that cast colored light into the scene and reflect.

**Gameplay & UI**
- Swept-AABB **player physics** (walk/fly, gravity, jump, sprint), 3D-DDA block targeting, break/place
  with incremental re-meshing, and a block highlight.
- **Progressive mining**: hold to break, timed by per-block hardness, with a crack overlay; bedrock is
  unbreakable. **Tools** (5 tiers × pickaxe/axe/shovel/sword/hoe) speed up mining, gate ore drops by
  harvest level, and wear down with **durability**.
- **Crafting**: a data-driven shaped + shapeless recipe registry; a 2×2 grid in the inventory and a
  3×3 grid at a crafting table, with a live result preview (planks, sticks, tools, furnace, chest, …).
- **Smelting**: right-click a furnace for an input/fuel/output screen with a live burning-fuel flame
  gauge and smelt-progress arrow; a `step_furnaces` tick consumes fuel (planks/logs/sticks) to smelt
  ores into **iron / gold ingots** (and cobblestone back to stone). Breaking a lit furnace spills it.
- A real **inventory** (9 hotbar + 27 main + armor + cursor), stack merging/splitting, an inventory
  screen with drag/drop and tooltips, item drops (blocks **and** tools) as world entities, and
  persistence. Survival basics: health + hunger, fall damage, regen/starvation, death drops + respawn.
- **Mobs + item drops**: AABB entities that wander and drop collectible items, lit by the same
  ray-traced pipeline as the world.
- A from-scratch **bitmap-font** text renderer and an **F3 debug overlay**.

Performance: comfortably **vsync-capped** at render distance 12 with full ray-traced GI on an RTX 4090;
the GPU has large headroom, which the lighting spends on per-pixel ray tracing.

## Architecture

```
src/
  main.rs           module tree + winit event-loop entry point
  app.rs            App/State, input routing, per-frame update + render, headless screenshot path
  gpu.rs            wgpu device/surface/depth; DX12-preferred adapter selection
  camera.rs         fly camera + globals UBO (view-proj, sun, sky/fog, time)
  environment.rs    day/night: sun direction, sky/fog color, ambient/intensity
  world.rs          Chunk (32³ blocks + light), World store, neighborhood view for meshing
  worldgen.rs       noise terrain, biomes, caves, ores, trees, decoration (Arc-shared across workers)
  light.rs          skylight + block-light flood (baked per-vertex during meshing)
  mesher.rs         binary greedy mesher (opaque/water) + cross-billboard plants
  texture.rs        procedural block texture atlas (painted in code)
  font.rs           embedded 8×8 bitmap font, baked to an atlas
  worker.rs         crossbeam worker pool (generate + mesh off the main thread)
  game.rs           streaming manager: gen/mesh budgets, frustum cull, edits, fluids, saves
  voxel_volume.rs   GPU voxel material volume (block ids) for ray-traced shadows + AO/GI
  raycast.rs        Amanatides–Woo voxel DDA (block targeting)
  player.rs         AABB collision, gravity/jump/fly, input, survival
  entity.rs         mobs + dropped items: AABB physics, wander AI, GI-lit box geometry
  item.rs           item + tool registry, ItemStack (durability), Inventory, slot-click logic
  crafting.rs       data-driven shaped/shapeless recipe registry + grid matching
  smelting.rs       furnace smelt-recipe + fuel tables (drives the furnace tick in game.rs)
  overlay.rs        HUD, hotbar, inventory/crafting screens, block + crack highlight (UI geometry)
  frustum.rs        Gribb–Hartmann frustum culling
  renderer.rs       pipelines (chunk/water/highlight/UI) + atlas/font bind groups, frame recording
  persistence.rs    LZ4 chunk save/load, level header, inventory save_state
  capture.rs        offscreen screenshot (headless verification)
assets/shaders/     rtx_common (shared bindings + voxel DDA tracer) + chunk / water / line / ui WGSL
```

**Hardware-driven choices:** worker threads keep all cores busy on generation/meshing/lighting while
the render thread stays light; greedy meshing keeps geometry cheap; the over-powered GPU is spent on
ray-traced shadows, AO, and global illumination rather than idling.

### Headless verification

Setting `VOXELCRAFT_SHOT=path.png` renders a single frame offscreen to a PNG and exits — used to
verify each change without a human in the loop. Companion debug knobs: `VOXELCRAFT_CAM="x,y,z,yaw,pitch"`,
`VOXELCRAFT_TIME=secs`, `VOXELCRAFT_PLACE="x,y,z,id;..."`, `VOXELCRAFT_SCREEN=inv|craft|furnace`,
`VOXELCRAFT_CRACK="x,y,z,progress"`, `VOXELCRAFT_ROOM`.

## Roadmap

Done: the full engine, world generation, lighting, rendering, the block/item library, inventory,
**progressive mining, tools + durability, crafting, and furnace smelting**. Next up: food & deeper
survival, armor; then typed mobs + combat, structures (dungeons/villages), an RTX temporal denoiser,
particles + audio, redstone, and additional dimensions.

## License

[MIT](LICENSE) © Jordan Birdsell. Built with the help of Claude Code.
