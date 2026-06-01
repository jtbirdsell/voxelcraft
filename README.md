# Voxelcraft

A Minecraft-style voxel sandbox written from scratch in **Rust + [wgpu](https://wgpu.rs)** (Vulkan,
with a DirectX 12 fallback), tuned for a high-end PC (developed on an RTX 4090 / i9-14900K). It features an infinite,
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

## GPU backend

Defaults to the **DX12** backend with hardware ray tracing via the **DXC** shader compiler (the
default FXC cannot compile ray-tracing shaders). The build script stages `dxcompiler.dll` +
`dxil.dll` next to the executable — from a vendored `dll/` directory if present, else your installed
Windows SDK; if neither is found, DX12 falls back to FXC and the software DDA tracer. (`dxil.dll` is a
Microsoft redistributable and is **not** committed to this repo — it is copied from your local
Windows SDK at build time.)

Override with `VOXELCRAFT_BACKEND=dx12|vulkan|gl`. **Vulkan** is the fallback: it also has hardware RT
and is the only backend wgpu exposes **DLSS** on, but DX12 renders better here so it's the default.
GL has no hardware RT (software DDA tracer only).

## Controls

| Input | Action |
|---|---|
| **WASD** | Move |
| **Mouse** | Look |
| **Space** | Jump (walk) / swim up / ascend (fly) |
| **Left-Shift** | Sneak (walk — won't walk off ledges) / descend (fly) |
| **Left-Ctrl** | Sprint (widens FOV) / fly boost |
| **F** | Toggle fly / walk |
| **Left-click (hold)** | Mine the targeted block (progressive, hardness-timed) |
| **Right-click** | Place block; open a crafting table / furnace / chest; **open/close a door or trapdoor**; **hold to eat** the selected food |
| **Q** | Drop one of the selected item |
| **1–9 / scroll** | Select hotbar slot |
| **E** | Open / close inventory (with a 2×2 craft grid) |
| **R** | Cycle ray-traced lighting: off → shadows → shadows + GI |
| **G** | Cycle difficulty: Peaceful → Easy → Normal → Hard |
| **F3** | Toggle debug overlay (fps, position, biome, facing, difficulty) |
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
  biome-gated surface **decoration** — flowers, tall grass, **ferns**, **red/brown mushrooms**,
  **sugar cane** along water edges, **pumpkins**, cactus, and **ice** on frozen lakes in snowy
  biomes. Fully seed-deterministic.
- A **procedural texture atlas** painted in code at startup (stone, ores, planks, bricks, foliage,
  …), with cross-billboard plants drawn via **alpha cutout**.
- **Translucent glass** (its own alpha-blended, depth-writing render pass — see-through and tinted,
  non-occluding to light) plus **slabs and stairs** as half/partial blocks emitted through the
  mesher's per-cell path; the greedy mesher now splits geometry into opaque / water / glass buckets.
  **Slabs** place as a **bottom or top half** (chosen by the clicked face / where on a side you click);
  adding a matching slab into the empty half forms a **double slab** (a full block that drops two).
  **Logs** orient to the axis of the face you place them against (upright / east-west / north-south),
  with the end-grain on the two perpendicular faces. All carried in a per-block **block-state byte**
  (the same mechanism that orients stairs), baked into geometry at mesh time so the GPU path is untouched.
- **Doors & trapdoors**: a **2-tall wooden door** (facing + hinge + open) and a **wooden trapdoor**
  (facing + half + open), both **right-click to open/close** — a closed door blocks the doorway, an
  open one swings aside so you can walk through; breaking either door half removes both. Thin oriented
  panels that don't occlude light or cast full-cube shadows.
- **Fences, walls & glass panes**: **connection-aware** blocks — a central post grows a thin arm
  toward each adjacent fence/wall/pane or solid block, so a row reads as a continuous rail/wall/window
  (a lone pane is a flat sheet). The shape is derived from neighbors at mesh time; fences are 1.5 blocks
  tall so they can't be jumped.
- **Torches, levers & buttons**: small **attached fixtures** — a thin 3D torch stick that stands on the
  floor or angles off a wall (the attach face comes from the surface you click), and a lever / button
  mounted the same way. All walk-through and non-occluding, but still breakable; the torch keeps its
  block-light glow. **Right-click flips a lever / presses a button** (the on/pressed state drives redstone
  later). The attach face + on bit live in the block-state byte.
- **Day/night** cycle with a dynamic sky, distance fog, translucent **water**, and **flowing fluids**
  (water/lava cellular simulation). Animated water ripples and lava.
- **Block-light + skylight** flood (0–15) baked per-vertex: caves are genuinely dark, and torches /
  glowstone / lava illuminate their surroundings.

**Ray-traced lighting** — traced on the RTX **hardware ray-tracing cores** (a per-chunk BLAS + a
per-frame TLAS over the greedy-meshed geometry, inline `rayQuery`), with a software DDA march over a
768×256 toroidal voxel volume as a switchable fallback (`VOXELCRAFT_TRACER=dda`). The world renders
through an **HDR (Rgba16Float) deferred pipeline** — a G-buffer (normal, motion vectors, world
position, albedo + skylight) and an ACES tonemap.
- **Sun shadows** against the real voxel geometry.
- **Ambient occlusion + one-bounce colored global illumination** (cosine-weighted hemisphere rays
  gather sky on a miss and sun-lit material color on a hit), gathered in a **deferred compute pass**
  that writes a noisy demodulated irradiance buffer and composites it back. `VOXELCRAFT_GI=fragment`
  restores the in-shader gather as a bit-for-bit parity oracle.
- **DLSS Ray Reconstruction** (`VOXELCRAFT_DLSS=rr`, NVIDIA RTX): the scene renders at a reduced
  resolution and DLSS denoises that noisy GI **and** upscales to the output resolution on the Tensor
  cores. Quality presets via `VOXELCRAFT_DLSS_QUALITY=dlaa|quality|balanced|performance`; degrades
  gracefully to native resolution when unavailable.
- **DLSS Frame Generation** (`VOXELCRAFT_FG=1`, NVIDIA RTX 40+): stacks on Ray Reconstruction — DLSS-G
  (via NVIDIA **Streamline**) interpolates an extra frame between each rendered one (~2× FPS) on the
  Optical-Flow hardware. The render-resolution depth/motion guides are resized to output res, and the
  HUD is tagged as a separate UI layer so DLSS-G recomposites it crisply instead of interpolating it.
  Needs the Streamline DLLs staged beside the exe (`STREAMLINE_SDK` set), a composited/focused window,
  and a non-vsync present mode; degrades gracefully to no frame generation.
- **Supersampling / DLDSR-style** (`VOXELCRAFT_SS=1.5`, NVIDIA RTX): render the scene *above* the
  window (×SS) with DLSS in DLAA mode, then a sharp Catmull-Rom pass downscales to the window — the
  documented best practice for spare GPU headroom (renders the noisy GI at super-res too, so it
  doubles as a GI denoiser). Clamped 1.0–4.0; 1.0 = off. Also `VOXELCRAFT_GI_RAYS=N` (hemisphere GI
  samples, default 8) and `VOXELCRAFT_GI_ACCUM=1` (opt-in motion-reprojected GI temporal accumulation).
- **Water reflections** via a Schlick-Fresnel mirror ray, with depth-based clarity (Beer–Lambert).
- **Emissive blocks** (lava, glowstone, torches) that cast colored light into the scene and reflect.

**Gameplay & UI**
- Swept-AABB **player physics** (walk/fly, gravity, jump, sprint), 3D-DDA block targeting, break/place
  with incremental re-meshing, and a block highlight.
- **Progressive mining**: hold to break, timed by per-block hardness, with a crack overlay; bedrock is
  unbreakable. **Tools** (5 tiers × pickaxe/axe/shovel/sword/hoe) speed up mining, gate ore drops by
  harvest level, and wear down with **durability**.
- **Crafting**: a data-driven shaped + shapeless recipe registry built from family generators (so it
  scales); a 2×2 grid in the inventory and a 3×3 grid at a crafting table, with a live result preview.
  Covers the **full tool + armor progression** (all five tool classes and four armor pieces across
  wood/stone/iron/gold/diamond), torches, slabs/stairs, table/furnace/chest, planks and sticks.
- **Smelting**: right-click a furnace for an input/fuel/output screen with a live burning-fuel flame
  gauge and smelt-progress arrow; a `step_furnaces` tick burns fuel (coal/charcoal/logs/planks/sticks)
  to smelt **raw iron/gold → ingots**, **sand → glass**, **logs → charcoal**, cobblestone → stone, and
  **raw meat → cooked food**. Mined ores drop their material (coal/diamond/redstone/lapis directly;
  iron/gold as raw ore). Breaking a lit furnace spills it.
- A real **inventory** (9 hotbar + 27 main + armor + cursor), stack merging/splitting, an inventory
  screen with drag/drop and tooltips, item drops (blocks **and** tools) as world entities. Block items
  render as **textured icons** (sampled from the block atlas) in every slot, not flat color swatches.
- **Chests**: right-click a chest for a 27-slot storage screen (drag/drop against your inventory);
  breaking a chest spills its contents. A generic `Container` so future storage blocks reuse it.
- **Persistence**: edited chunks (LZ4 blocks **+ block-state**), the level header, the full inventory +
  armor + tool durability, **furnace + chest contents**, and **player survival**
  (health/hunger/air/saturation/XP/level) all round-trip across save/reload; older saves load
  forward-compatibly.
- **Survival depth**: health + hunger with saturation-fueled regen and starvation; fall damage;
  **swimming** (buoyancy + paddle up) with an **air/drowning** bubble meter; **lava contact damage**;
  **sneak** (Shift won't let you walk off ledges) and a sprint **FOV** kick; death drops + respawn.
- **Eating**: hold right-click on a food to eat it (~1.6 s), restoring hunger + saturation. Raw mob
  drops (beef/pork/chicken/mutton) are edible but weak; **cook them in a furnace** (→ steak, cooked
  porkchop/chicken/mutton) for far more, plus bread and apples.
- **Difficulty** (cycle with **G**, saved per world): **Peaceful** (no hostile spawns, existing
  hostiles vanish, hunger never depletes and health regenerates passively), **Easy** / **Normal** /
  **Hard** scale incoming hostile damage (0.5× / 1× / 1.5×) and how far starvation can hurt you (Easy
  floors at 10 HP, Normal at 1 HP, Hard can kill). Environmental damage (fall/lava/drowning) is never
  scaled. Shown on the F3 overlay.
- **Experience**: mining ores (coal/redstone/lapis/diamond) drops glowing **XP orbs** that home in and
  grant points; an XP bar + level counter on the HUD.
- **Armor**: 4 equip slots (helmet/chestplate/leggings/boots) × leather/iron/gold/diamond, drag-to-equip
  in the inventory (each slot only accepts its piece). Equipped defense points reduce incoming damage
  (~4%/point, capped at 80%) for fall/lava/mob hits; a steel armor bar shows on the HUD.
- **Typed mobs + AI**: eight species (cow, pig, sheep, chicken, zombie, skeleton, creeper, spider),
  each with its own size, health, and a distinct **multi-box model** (quadrupeds, humanoids, a tall
  creeper, a wide-low spider), lit by the same ray-traced pipeline as the world. An **Idle / Wander /
  Flee / Chase / Attack** state machine drives them: passive species flee when you crowd them, hostile
  species **chase you on sight** (raycast line-of-sight) and close to attack range. Movement is real
  **local navigation** — mobs **step up** one-block ledges, **avoid walking off cliffs** (passives) and
  **into lava**, and **deflect around walls** rather than stuttering into them (a chaser pursues
  relentlessly, dropping off ledges to follow you but still routing around lava). They **spawn
  naturally** by Minecraft-style rules: hostiles appear only in the **dark** (block-light 0) at night,
  on any solid surface a short distance out; passive animals appear in **packs** on **sunlit grass**,
  drawn from a **biome-appropriate** pool. Hostiles and animals have **separate population caps** and
  despawn once you roam far away. Dropped items + XP orbs share the same entity physics.
- **Combat (modern 1.9-style)**: left-click melees the nearest mob in your reach (ray-vs-AABB beats
  mining when a mob is closer than the block). Each weapon has its own **attack-speed cooldown** shown
  as a recharge bar under the crosshair — swing before it refills and the hit is weak (damage scales
  `0.2 + 0.8·charge²`), so timed strikes beat spam. A fully-charged **fall attack** lands a **critical**
  (+50%); a charged **sword** swing on the ground does a **sweep** (light AoE to nearby mobs); a
  **sprint** attack trades the crit for extra knockback. Hits apply **knockback** + a red **hurt-flash**
  and drop XP on kill; hostile mobs in range hit **back** for contact damage, reduced by your armor.
- **Ranged + explosions**: **skeletons** loose arrows at you when they have a clear shot (gravity-arced
  projectiles that stick in blocks or deal damage on a hit); **creepers** prime a fuse in your face and
  **detonate** — a spherical crater blown out of the world plus radial blast damage that falls off with
  distance (and is reduced by armor).
- **Bow**: craft a bow (string + sticks) and arrows (flint/stick/feather); **hold right-click to draw**
  (a charge bar fills above the crosshair) and release to loose a **gravity-arced arrow** whose speed and
  damage scale with the draw — a full draw fires a faster, harder **critical** arrow. Arrows are owner-
  tagged, so yours strike mobs (never yourself); drawing spends one arrow (free in creative).
- **Shield**: craft a shield (planks + iron); **hold right-click to raise** it (a cue lights up beside the
  crosshair once it's ready, after a brief delay). A raised shield **fully blocks** melee, arrow, and
  explosion damage arriving from the **front** (a facing-vs-source test) — turn into the danger to soak it.
  Environmental damage (falling, lava, drowning, starvation) is never blocked.
- A from-scratch **bitmap-font** text renderer and an **F3 debug overlay**.

Performance: comfortably **vsync-capped** at render distance 12 with full ray-traced GI on an RTX 4090;
the GPU has large headroom, which the lighting spends on per-pixel ray tracing.

## Architecture

```
src/
  main.rs           module tree + winit event-loop entry point
  app.rs            App/State, input routing, per-frame update + render, headless screenshot path
  gpu.rs            wgpu device/surface/depth; Vulkan-default adapter selection (+DX12/GL), RT-capable device
  camera.rs         fly camera + globals UBO (view-proj, sun, sky/fog, time)
  environment.rs    day/night: sun direction, sky/fog color, ambient/intensity
  world.rs          Chunk (32³ blocks + light), World store, neighborhood view for meshing
  worldgen.rs       noise terrain, biomes, caves, ores, trees, decoration (Arc-shared across workers)
  light.rs          skylight + block-light flood (baked per-vertex during meshing)
  mesher.rs         binary greedy mesher (opaque/water/glass) + cross-billboards + slab/stair boxes
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
  renderer.rs       pipelines (chunk/water/glass/GI-compute/composite/tonemap/highlight/UI), HDR frame recording
  graph.rs          render-graph targets: HDR scene buffer + G-buffer (normal/motion/position/albedo)
  rt.rs             hardware ray-tracing acceleration structures (per-chunk BLAS + per-frame TLAS)
  persistence.rs    LZ4 chunk save/load, level header, inventory save_state
  capture.rs        offscreen screenshot (headless verification)
assets/shaders/     rtx_common (shared bindings + DDA tracer + GI) + atlas + sun_vis_hw (hardware rayQuery)
                    + gi_compute / gi_composite + chunk / water / glass / tonemap / line / ui WGSL
```

**Hardware-driven choices:** worker threads keep all cores busy on generation/meshing/lighting while
the render thread stays light; greedy meshing keeps geometry cheap; the over-powered GPU is spent on
ray-traced shadows, AO, and global illumination rather than idling.

### Headless verification

Setting `VOXELCRAFT_SHOT=path.png` renders a single frame offscreen to a PNG and exits — used to
verify each change without a human in the loop. Companion debug knobs: `VOXELCRAFT_CAM="x,y,z,yaw,pitch"`,
`VOXELCRAFT_TIME=secs`, `VOXELCRAFT_PLACE="x,y,z,id;..."`, `VOXELCRAFT_SCREEN=inv|craft|furnace`,
`VOXELCRAFT_CRACK="x,y,z,progress"`, `VOXELCRAFT_ROOM`, `VOXELCRAFT_SURVIVAL=1` (HUD with air + XP +
armor bars). Rendering knobs: `VOXELCRAFT_BACKEND=vulkan|dx12|gl`, `VOXELCRAFT_TRACER=dda|hwrt`
(software DDA vs hardware ray query), `VOXELCRAFT_GI=fragment|compute` (in-shader vs deferred GI),
`VOXELCRAFT_GI_RAW=1` (dump the raw GI irradiance buffer), `VOXELCRAFT_GI_RAYS=N` (GI samples/pixel,
default 8), `VOXELCRAFT_GI_ACCUM=1` (opt-in GI temporal accumulation), `VOXELCRAFT_DLSS=off|rr` +
`VOXELCRAFT_DLSS_QUALITY=dlaa|quality|balanced|performance` (DLSS Ray Reconstruction),
`VOXELCRAFT_SS=1.5` (supersample ×SS then downscale — render above native; clamped 1–4),
`VOXELCRAFT_FG=1` (DLSS Frame Generation on top of RR; needs `STREAMLINE_SDK` set + a focused window).
DLSS needs `DLSS_SDK` + `LIBCLANG_PATH` set at build time. **Build features:** DLSS is gated behind the
default `dlss` (Ray Reconstruction + supersampling) and `frame-generation` (DLSS-G; implies `dlss`)
cargo features — `cargo build --no-default-features` compiles and runs the engine **natively with no
NVIDIA SDK / libclang / Streamline** (no RR/FG), and `--features dlss` builds RR without Frame
Generation. Frame Generation can be smoke-tested headlessly with `VOXELCRAFT_FG_TEST=N` (auto-pans the
camera for N frames, logs the max `num_frames_actually_presented` — 2 means DLSS-G is generating — then
exits). `VOXELCRAFT_PERSIST_TEST=1` round-trips a sample inventory/armor/furnace/survival state through
the real save/load (no window).

## Roadmap

Done: the full engine, world generation, lighting, rendering, the block/item library, inventory,
**progressive mining, tools + durability, crafting, furnace smelting, survival depth** (swimming/air,
lava damage, sneak, XP & levels), **armor, and full save/reload persistence**. Next up: billboard
decoration + world content, typed mobs + combat, structures (dungeons/villages), an RTX temporal
denoiser, particles + audio, redstone, and additional dimensions.

## License

[MIT](LICENSE) © Jordan Birdsell. Built with the help of Claude Code.
