# Voxelcraft

A Minecraft-style voxel sandbox written from scratch in **Rust + [wgpu](https://wgpu.rs)** (DX12 /
Vulkan / Metal), tuned for a high-end PC (developed on an RTX 4090 / i9-14900K) but it also builds
and runs on **macOS / Apple Silicon** via the Metal backend (software-traced lighting — hardware RT
and DLSS remain Windows/RTX-only). It features an infinite, procedurally-generated world streamed
across all CPU cores, a procedural texture atlas, real block-light + skylight (dark caves, glowing
torches), a full inventory, and **ray-traced lighting** — sun shadows, ambient occlusion, one-bounce
colored global illumination, water reflections, and emissive blocks — all computed against the
actual voxel geometry on the GPU.

> **Status:** the engine and world systems are complete and the survival loop is in active
> development (see [Roadmap](#roadmap)). Everything here is built from scratch — no game engine, and
> the textures, fonts, and worldgen are all generated in code.

## Run

Requires the Rust toolchain (stable; MSVC on Windows, Xcode Command Line Tools on macOS). The same
plain command works on both — no NVIDIA SDK is needed off-Windows (the DLSS deps are gated to
`cfg(windows)`). From the project directory:

```sh
cargo run --release
```

The first release build takes ~50s (fat LTO); after that it's instant. A debug build (`cargo run`)
compiles faster but runs the worldgen/mesher unoptimized, so use `--release` to play.

```sh
cargo test --release   # worldgen determinism, physics, inventory + save round-trips
```

Launching drops you at the **main menu** — *Singleplayer*, *Settings*, *Quit*. *Singleplayer* opens the
**world-select** screen listing every saved world (name + seed, most-recently-played first); click a
world to play it, the row's **X** to delete it (a second click on the red **Delete?** confirms — no
one-misclick world loss), or **Create New World** to make one. The create screen takes a **name**, an
optional **seed** (a number is used as-is; any other text is hashed; blank = random), and a **game
mode** — **Survival** (the default: no flying, real damage and hunger, finite items; spawns on dry
land) or **Creative** (the infinite block palette, F-toggled flight, no damage). Each world lives in
its own `saves/worlds/<name>/` directory and saves there on quit / *Save & Quit*. A pre-N4 single world
at `saves/world/` is migrated into the list automatically on first launch (legacy worlds stay creative).

In-game, **Esc** opens the **pause menu** (Resume / Settings / Save & Quit to Menu) and frees the cursor.
`VOXELCRAFT_SKIPMENU=1` boots straight into the most-recently-played world (the pre-menu behaviour, for
scripts/screenshots).

### Settings

A live **General** tab (FOV, mouse sensitivity, render distance, view-bob, difficulty) and a
**Graphics** tab (DLSS mode/quality, supersampling, GI mode/rays, tracer, backend, frame generation —
each tagged *(restart)* since they're baked at device/pipeline creation). Persisted to
`saves/settings.cfg` as plain `key=value` text (a sibling of the per-world dirs, world-independent); the
`VOXELCRAFT_*` env vars still override at startup. *(The General-tab live-apply and the Graphics-tab
restart hooks land in N5.)*

## GPU backend

The default is platform-aware. On Windows/Linux it's the **DX12** backend with hardware ray tracing
via the **DXC** shader compiler (the default FXC cannot compile ray-tracing shaders). The build
script stages `dxcompiler.dll` + `dxil.dll` next to the executable — from a vendored `dll/`
directory if present, else your installed Windows SDK; if neither is found, DX12 falls back to FXC
and the software DDA tracer. (`dxil.dll` is a Microsoft redistributable and is **not** committed to
this repo — it is copied from your local Windows SDK at build time.)

On **macOS** the default is **Metal**, the only native backend there. Metal exposes no hardware ray
tracing through wgpu (and no DLSS), so the lighting runs the **software DDA tracer** + the deferred
compute GI — the same fully-featured fallback path GL uses, with a tuned-down **platform quality
tier** (below) sized for 60 FPS on Apple-silicon GPUs. The swapchain renders at 0.57× of the
window's *logical* resolution by default (the compositor stretches the drawable); override with
`VOXELCRAFT_RENDER_SCALE=0.25..1.0`, a fraction of *physical* pixels (Metal-only; `1.0` = native
pixels, e.g. for screenshots).

### Platform quality tier

A `Quality` tier resolves once at startup (`gfx/quality.rs`): every non-Metal backend gets the
**maxed** tier — exactly the pre-tier hardcoded defaults, unit-test-anchored so the Windows/RTX
experience can never drift — while **Metal** gets values benchmarked for 60 FPS on a base M3.
Each knob is only a default; the env vars override either tier:

| Knob (env var) | Maxed (Windows) | Metal tier | Controls |
|---|---|---|---|
| `VOXELCRAFT_GI_RAYS` | 8 | 2 | GI hemisphere rays/pixel |
| `VOXELCRAFT_GI_ACCUM` | off | **on** | GI temporal accumulation (denoises the low ray count) |
| `VOXELCRAFT_GI_DIST` | 22 | 12 | GI bounce-ray range (blocks) |
| `VOXELCRAFT_GI_SUN_DIST` | 22 | 12 | secondary sun ray from GI hits |
| `VOXELCRAFT_SUN_DIST` | 96 | 18 | primary sun-shadow ray range |
| `VOXELCRAFT_WSMOOTH` | 3 | 1 | water depth-clarity smoothing radius |
| `VOXELCRAFT_WREFL` | 80 | 24 | water reflection-ray range |
| `VOXELCRAFT_WDEPTH` | 24 | 12 | water depth-march cap |
| `VOXELCRAFT_VOLUME_CHUNKS` | 24 | 20 | tracer voxel-volume extent (chunks; ~302 → ~210 MB) |
| `VOXELCRAFT_RENDER_DISTANCE` | 12 | 8 | chunk render distance (fog scales with it) |
| `VOXELCRAFT_UPLOAD_BUDGET` | 48 | 32 | volume chunk uploads/frame |

The tier values flow through the `Volume` uniform, so the shared HW-RT/DDA shader code is
bit-identical on Windows by construction, and the `VOXELCRAFT_GI=fragment` parity oracle holds on
both paths.

Override with `VOXELCRAFT_BACKEND=dx12|vulkan|gl|metal` (a backend your OS lacks just fails adapter
selection). **Vulkan** is the Windows fallback: it also has hardware RT, but DX12 renders better
here so it's the default. GL has no hardware RT (software DDA tracer only).

## Controls

| Input | Action |
|---|---|
| **WASD** | Move |
| **Mouse** | Look |
| **Space** | Jump (walk) / swim up / ascend (fly) |
| **Left-Shift** | Sneak (walk — won't walk off ledges) / descend (fly) |
| **Left-Ctrl** | Sprint (widens FOV) / fly boost |
| **F** | Toggle fly / walk (**Creative worlds only**) |
| **Left-click (hold)** | Mine the targeted block (progressive, hardness-timed) |
| **Right-click** | Place block; open a crafting table / furnace / chest; **open/close a door or trapdoor**; **hold to eat** the selected food |
| **Q** | Drop one of the selected item |
| **1–9 / scroll** | Select hotbar slot |
| **E** | Open / close inventory (with a 2×2 craft grid) |
| **R** | Cycle ray-traced lighting: off → shadows → shadows + GI |
| **G** | Cycle difficulty: Peaceful → Easy → Normal → Hard |
| **F3** | Toggle debug overlay (fps, position, biome, facing, difficulty) |
| **P** | Save world |
| **Esc** | Close GUI screen, else open the **pause menu** |

Mining yields drops (stone → cobblestone, ores need the right pickaxe tier) that fall into your
inventory; the hotbar shows stack counts, durability bars, and the selected item's name.

## Features

**World & rendering**
- Infinite world streamed as 32³ chunks, generated and **binary greedy-meshed** across a worker pool
  (cores − 2 threads), with CPU frustum culling.
- Multi-octave OpenSimplex terrain (deepened: sea level y96, ~90 blocks of underground) with
  parameter-space **biomes**, a jagged **bedrock** floor, a noise-perturbed **deepslate** boundary,
  and deterministic **trees** and biome-gated surface **decoration** — flowers, tall grass, **ferns**,
  **red/brown mushrooms**, **sugar cane** along water edges, **pumpkins**, cactus, and **ice** on
  frozen lakes in snowy biomes. Fully seed-deterministic.
- **Full underground overhaul** (a cross-chunk-safe region-cell feature engine): **every overworld ore**
  — coal, iron, **copper**, gold, redstone, lapis, diamond, **emerald** — each with a **deepslate
  variant**, generated as connected **ore veins/blobs** on **1.18 triangular altitude bands**, plus rare
  **large copper/iron veins**. Stone-variant blobs (tuff, granite, diorite, andesite) + dirt/gravel/clay,
  and the building family (storage blocks, polished/brick deepslate, cut copper). **Rich caves** — cheese
  caverns, spaghetti tunnels, thin noodle worms, **ravines**, underground **aquifers** + **lava lakes**.
  **Cave biomes**: amethyst **geodes** (calcite/budding amethyst that grow clusters), **dripstone**,
  **lush** caves (moss, cave vines + glow berries, azalea), and the **Deep Dark** — the **sculk** family
  (sculk + veins + sensors + shriekers + a catalyst that spreads sculk when mobs die) and the **Warden**.
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
  glowstone / lava illuminate their surroundings. A small day/night-scaled **ambient floor** keeps
  fully sun-shadowed, sky-occluded surfaces (a pit bottom, an overhang, a deep concave corner) from
  collapsing to pure black — they read as a deep shadow rather than a black hole, without lifting the
  open-sky terrain (the floor is weighted by `1 − skylight`).
- **Smooth lighting + ambient occlusion** (Minecraft-style): the greedy mesher samples skylight and
  block-light **per vertex** — averaging the four cells touching each corner — and bakes a geometric
  **AO** darkening into concave corners, so light gradients across a face are smooth (Gouraud-
  interpolated) instead of flat-shaded with hard per-face steps. Uniform-lit runs still greedy-merge
  into big quads; only light gradients (shadow edges, AO corners, a torch's falloff) split into the
  per-cell quads that carry the gradient. Tracer- and GI-path-independent (baked into the vertex).

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
- Substepped swept-AABB **player physics** (walk/fly, gravity, jump, sprint) — tunnel-proof at any
  frame rate (terminal-velocity falls land on 1-block floors; entities are speed-capped the same way)
  with frame-rate-independent jump/fall integration (fixed ~120 Hz substeps), 3D-DDA block targeting,
  break/place with incremental re-meshing, and a block highlight. Physics pauses until the chunks
  around the player stream in, so an unloaded chunk never collides as air; a save made underground
  (a cave home) resumes in place — only a position genuinely inside solid blocks lifts to the surface.
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
  In creative the inventory shows a **paged palette** of every block, tool, material, and armor piece
  (click a slot to grab it; scroll to page) so the full ~110-block library is reachable directly.
- A **first-person view-model**: the held item is drawn in front of the camera — a textured 3D cube
  for blocks, a painted sprite for tools/weapons/food, a fist for the empty hand. It **swings** when
  you mine/attack/use and animates per action (raise food to eat, draw a bow, raise a shield), with
  walk-bob, look-sway, and an equip lower→raise on hotbar swaps. It's lit by the local block light
  (dims in caves) and renders in its own LDR pass after the DLSS resolve, isolated from the denoiser.
  A subtle camera head-bob accompanies it (`VOXELCRAFT_VIEWBOB=0` to disable).
- **Chests**: right-click a chest for a 27-slot storage screen (drag/drop against your inventory);
  breaking a chest spills its contents. A generic `Container` so future storage blocks reuse it.
- **Persistence**: edited chunks (LZ4 blocks **+ block-state**), the level header, the full inventory +
  armor + tool durability, **furnace + chest contents**, and **player survival**
  (health/hunger/air/saturation/XP/level) all round-trip across save/reload; older saves load
  forward-compatibly.
- **Game modes** (chosen at world creation, persisted per world): **Survival** — no flying, full
  damage/hunger/air simulation, finite items, death drops your inventory at the death site; **Creative**
  — the paged infinite palette, F-toggled flight, and total invulnerability (no hunger, fall, lava,
  drowning, or mob damage), on foot or airborne.
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
- **Typed mobs + AI**: thirteen species — cow, pig, sheep, chicken, zombie, skeleton, creeper, spider,
  plus a **wolf** (neutral; turns on you when struck), a tall **enderman**, a **slime** (bounces, and
  **splits into smaller slimes** when killed until the smallest pops), a passive **villager**, and the
  **Warden** — a blind, **vibration-guided** Deep-Dark boss (500 HP) summoned by a **sculk shrieker**:
  it emerges from the ground, crushes in melee, fires a ranged **sonic boom**, and digs away when it
  loses your trail (so **sneak** to evade it). Each has its own size, health, and a distinct
  **multi-box model**, lit by the same ray-traced pipeline as the world. An **Idle / Wander /
  Flee / Chase / Attack** state machine drives them: passive species flee when you crowd them, hostile
  species **chase you on sight** (raycast line-of-sight) and close to attack range. Movement is real
  **local navigation** — mobs **step up** one-block ledges, **avoid walking off cliffs** (passives) and
  **into lava**, and **deflect around walls** rather than stuttering into them (a chaser pursues
  relentlessly, dropping off ledges to follow you but still routing around lava). They **spawn
  naturally** by Minecraft-style rules: hostiles appear only in the **dark** (block-light 0) at night,
  on any solid surface a short distance out; passive animals appear in **packs** on **sunlit grass**,
  drawn from a **biome-appropriate** pool. Hostiles and animals have **separate population caps** and
  despawn once you roam far away. Dropped items + XP orbs share the same entity physics.
- **Life cycle**: **feed** an animal its food (wheat → cows/sheep, seeds → chickens, carrots → pigs) to
  put it in love-mode; two in-love animals nearby breed a **baby** (a half-size juvenile that grows up
  after a while). **Zombies and skeletons burn** in direct daylight when they're out in the open sun.
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
the GPU has large headroom, which the lighting spends on per-pixel ray tracing. On Apple Silicon
(software DDA path) the per-pixel ray cost dominates — the Metal **quality tier** (see *GPU
backend*) lands 60 FPS on a base M3 (13.7 ms GPU p50 on the benchmark vista, from 147 ms before
the tier) by shortening ray ranges, quartering GI rays into the temporal accumulator, and rendering
at 0.57× logical resolution; the headroom under the 16.67 ms vblank absorbs the windowed loop's
serialized CPU work (Metal runs one frame in flight), keeping ProMotion from quantizing busy views
down to 40 FPS. `VOXELCRAFT_BENCH=N` measures any knob combination headlessly.

## Architecture

Modules are grouped into subsystems with a `foo.rs + foo/` facade pattern; submodules are re-exported
at the crate root so flat `crate::camera`-style paths keep resolving.

```
src/
  main.rs             module tree + winit event-loop entry point
  app.rs              App/State, input routing, per-frame update + render, headless screenshot path
  game.rs             streaming manager: gen/mesh budgets, frustum cull, edits, fluids, furnace tick, saves
  persistence.rs      LZ4 chunk save/load, level header, inventory/container/survival save_state
  world/              world data + generation
    chunk.rs          Chunk (32³ blocks + light), World store, neighborhood view for meshing
    block.rs          block registry: u16 ids + property tables
    worldgen.rs       noise terrain, climate-model biomes, caves, ores, trees, decoration (Arc-shared across workers)
    light.rs          skylight + block-light flood (baked per-vertex during meshing)
    worker.rs         crossbeam worker pool (generate + mesh off the main thread)
  mesh/
    mesher.rs         binary greedy mesher (opaque/water/glass) + cross-billboards + slab/stair/door/fence boxes
  gfx/                GPU / rendering
    device.rs         wgpu instance/surface/device/queue; platform-aware adapter selection — DX12 (Win/Linux) / Metal (macOS), +Vulkan/GL; RT-capable device
    camera.rs         fly camera + globals UBO (view-proj, sun, sky/fog, time)
    environment.rs    day/night: sun direction, sky/fog color, ambient/intensity
    renderer.rs       pipelines (chunk/water/glass/GI-compute/composite/tonemap/highlight/UI), HDR frame recording
    graph.rs          render-graph targets: HDR scene buffer + G-buffer (normal/motion/position/albedo)
    rt.rs             hardware ray-tracing acceleration structures (per-chunk BLAS + per-frame TLAS)
    voxel_volume.rs   GPU voxel material volume (block ids) for the software DDA shadows + AO/GI
    dlss.rs           DLSS Ray Reconstruction / Super Resolution (feature `dlss`; else dlss_stub.rs)
    frame_gen.rs      DLSS Frame Generation via Streamline (feature `frame-generation`; else frame_gen_stub.rs)
    texture.rs        procedural block texture atlas (painted in code)
    quality.rs        platform quality tier: maxed (Windows, == legacy defaults) vs Metal (60 FPS on M3)
    frustum.rs        Gribb–Hartmann frustum culling
    capture.rs        offscreen screenshot (headless verification)
    bench.rs          headless benchmark (VOXELCRAFT_BENCH=N): per-pass GPU timestamps + frame stats
    rt_probe.rs       hardware-RT capability probe (VOXELCRAFT_RT_PROBE=1)
    rt_spike.rs       self-contained hardware-RT proof, isolated from the main render path
  gameplay/
    player.rs         AABB collision, gravity/jump/fly, input, survival
    entity.rs         mobs + dropped items + XP orbs: AABB physics, AI state machine, GI-lit box models
    item.rs           item + tool registry, ItemStack (durability), Inventory, slot-click logic
    crafting.rs       data-driven shaped/shapeless recipe registry + grid matching
    smelting.rs       furnace smelt-recipe + fuel tables (drives the furnace tick in game.rs)
    food.rs           hunger + saturation values for edible items
    container.rs      generic block-entity container (chests; future barrels/hoppers)
    rules.rs          difficulty (Peaceful/Easy/Normal/Hard), persisted per world
    raycast.rs        Amanatides–Woo voxel DDA (block targeting)
  ui/
    overlay.rs        HUD, hotbar, inventory/crafting screens, block + crack highlight (UI geometry)
    font.rs           embedded 8×8 bitmap font, baked to an atlas
assets/shaders/       rtx_common (shared bindings + DDA tracer + GI) + atlas + sun_vis_hw (hardware rayQuery)
                      + gi_compute / gi_composite / gi_temporal + gbuf_downscale / gbuf_upscale / ss_downscale
                      (DLSS + supersampling resizes) + chunk / water / glass / tonemap / line / ui WGSL
```

**Hardware-driven choices:** worker threads keep all cores busy on generation/meshing/lighting while
the render thread stays light; greedy meshing keeps geometry cheap; the over-powered GPU is spent on
ray-traced shadows, AO, and global illumination rather than idling.

### Headless verification

Setting `VOXELCRAFT_SHOT=path.png` renders a single frame offscreen to a PNG and exits — used to
verify each change without a human in the loop. Companion debug knobs: `VOXELCRAFT_CAM="x,y,z,yaw,pitch"`,
`VOXELCRAFT_TIME=secs`, `VOXELCRAFT_PLACE="x,y,z,id;..."`, `VOXELCRAFT_SCREEN=inv|craft|furnace`,
`VOXELCRAFT_CRACK="x,y,z,progress"`, `VOXELCRAFT_ROOM`, `VOXELCRAFT_SURVIVAL=1` (HUD with air + XP +
armor bars), `VOXELCRAFT_DARK=1` (the Warden's pulsing **Darkness** screen dim),
`VOXELCRAFT_HELD=<item_id>` + `VOXELCRAFT_VM_POSE=swing|eat|draw|shield|equip|idle` +
`VOXELCRAFT_VM_T=<0..1>` (force the first-person held item + an animation pose),
`VOXELCRAFT_MENU=main|worlds|create|settings|pause` (render one menu screen to a PNG, no world).
Rendering knobs:
`VOXELCRAFT_BACKEND=vulkan|dx12|gl|metal`, `VOXELCRAFT_TRACER=dda|hwrt`
(software DDA vs hardware ray query), `VOXELCRAFT_RENDER_SCALE=0.25..1.0` (Metal-only sub-native
swapchain; defaults to 0.60× logical resolution on macOS — set `1.0` for native-pixel screenshots /
cross-OS comparisons), `VOXELCRAFT_RTX=0|1|2` (force lighting off / shadows / shadows+GI — cost
isolation), `VOXELCRAFT_GI=fragment|compute` (in-shader vs deferred GI),
`VOXELCRAFT_GI_RAW=1` (dump the raw GI irradiance buffer), `VOXELCRAFT_GI_RAYS=N` (GI samples/pixel,
tier default), `VOXELCRAFT_GI_ACCUM=1|0` (GI temporal accumulation; tier default),
`VOXELCRAFT_AMBIENT_FLOOR=<frac>` (ambient lift on sky-occluded shadowed surfaces; default 0.25, `0`
restores the old pure-black behaviour), the quality-tier
knobs above (`VOXELCRAFT_SUN_DIST`, `VOXELCRAFT_GI_DIST`, `VOXELCRAFT_GI_SUN_DIST`,
`VOXELCRAFT_WREFL`, `VOXELCRAFT_WDEPTH`, `VOXELCRAFT_VOLUME_CHUNKS`, `VOXELCRAFT_RENDER_DISTANCE`,
`VOXELCRAFT_UPLOAD_BUDGET`), `VOXELCRAFT_DLSS=off|rr` +
`VOXELCRAFT_DLSS_QUALITY=dlaa|quality|balanced|performance` (DLSS Ray Reconstruction),
`VOXELCRAFT_SS=1.5` (supersample ×SS then downscale — render above native; clamped 1–4),
`VOXELCRAFT_FG=1` (DLSS Frame Generation on top of RR; needs `STREAMLINE_SDK` set + a focused window),
`VOXELCRAFT_GPU_WATCHDOG=secs` (windowed GPU-wedge fail-safe: if submitted frames stop completing
for this long, save the world and exit instead of feeding a dead driver queue; default 5, `0` off).
DLSS needs `DLSS_SDK` + `LIBCLANG_PATH` set at build time. **Build features:** DLSS is gated behind the
default `dlss` (Ray Reconstruction + supersampling) and `frame-generation` (DLSS-G; implies `dlss`)
cargo features — `cargo build --no-default-features` compiles and runs the engine **natively with no
NVIDIA SDK / libclang / Streamline** (no RR/FG), and `--features dlss` builds RR without Frame
Generation. Frame Generation can be smoke-tested headlessly with `VOXELCRAFT_FG_TEST=N` (auto-pans the
camera for N frames, logs the max `num_frames_actually_presented` — 2 means DLSS-G is generating — then
exits). `VOXELCRAFT_PERSIST_TEST=1` round-trips a sample inventory/armor/furnace/survival state through
the real save/load (no window).

**Headless benchmark:** `VOXELCRAFT_BENCH=N` renders N timed frames offscreen (never touching the
swapchain) and prints p50/p95 frame + per-pass GPU timings from stage-boundary timestamp queries
(the only flavor Apple GPUs support; whole-frame wall time elsewhere if unavailable). Defaults to
the tier's interactive GI rays so numbers reflect gameplay cost. Knobs: `VOXELCRAFT_BENCH_WARMUP=W`
(discarded frames, default max(3, N/10)), `VOXELCRAFT_BENCH_LABEL=tag`, `VOXELCRAFT_BENCH_CSV=path`
(append one row per run for A/B sweeps), `VOXELCRAFT_BENCH_GPU=0` (wall-time only). Canonical bench
scene: default camera + `VOXELCRAFT_TIME=0.30`, e.g.
`VOXELCRAFT_BENCH=120 VOXELCRAFT_TIME=0.30 cargo run --release`. On Apple-silicon TBDR the per-pass
render timings overlap (passes execute concurrently) — treat them as relative weights and
`gpu_frame` as the absolute verdict; the compute-pass (`gi_compute`) timing is reliable.

## Roadmap

Done: the full engine, world generation, lighting (ray-traced GI with DLSS Ray Reconstruction
denoising + Frame Generation), rendering, the block/item library and surface decoration, inventory,
**progressive mining, tools + durability, crafting, furnace smelting, survival depth** (swimming/air,
lava damage, sneak, XP & levels), **armor, full save/reload persistence**, and **typed mobs + combat**
— twelve species with AI, natural spawning, breeding + babies, difficulty, melee (1.9-style timing,
crits, sweeps), skeleton arrows + creeper explosions, and the bow + shield. Next up: structures
(dungeons/villages), particles + audio, redstone, and additional dimensions.

## License

[MIT](LICENSE) © Jordan Birdsell. Built with the help of Claude Code.
