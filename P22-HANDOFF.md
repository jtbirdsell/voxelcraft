# P22 handoff — Metal hardware RT (session state, 2026-06-04)

Branch-scoped working notes for resuming P22 from another machine. **Delete this file when P22 merges.**
Written on the M3 MacBook mid-milestone; next session may be the Windows RTX box (see "Windows tasks")
or back on the Mac (see "Mac tasks").

## Mission

Enable hardware ray tracing on macOS/Metal. The wgpu Metal AS-sync bug (gfx-rs/wgpu#9215) that was
the prime suspect for voxelcraft's Metal hwrt first-frame hang is now fixed in our fork. Decisions
already made with Jordan: Metal default flips to hwrt only if bench p50 ≤ DDA's ~13.7 ms; tracer-aware
quality-tier retune is in scope this milestone; one supervised windowed test at the very end.

## State of every repo (all pushed)

| Repo | Branch | Rev | What it is |
|---|---|---|---|
| `jtbirdsell/wgpu` | `voxelcraft/metal-as-sync` | `9335871` | v29.0.3 + dx12 DLSS patches (`d81d755`) + Metal AS fence fix. **This is the pinned rev.** |
| `jtbirdsell/wgpu` | `fix/metal-as-sync` | `a9778b0` | Same fix rebased on upstream trunk → open upstream PR [gfx-rs/wgpu#9645](https://github.com/gfx-rs/wgpu/pull/9645) (awaiting maintainers; nothing to do). |
| `jtbirdsell/dlss_wgpu_dx12` | `wgpu-bump-933587` | `901c6b4` | = consumed `3e79f06` + ONLY the wgpu pin bump to `9335871` (both deps + dev-deps). Deliberately NOT main (main is 50 commits ahead with API refactors — separate project). |
| `jtbirdsell/voxelcraft` | `p22-metal-hwrt` (this branch) | — | main + the two P22 commits below + this file. |

Local checkouts on the Mac: `~/code/wgpu` (+ worktree `~/code/wgpu-voxelcraft`), `~/code/dlss_wgpu_dx12`.

## Done so far (commits on this branch)

1. **`889aa8a` P22: unify wgpu pin at 933587 across dlss_wgpu_dx12** — Cargo.lock had split into TWO
   wgpu graphs (voxelcraft patched to `9335871`, dlss still pinning `d81d755`), which would break the
   Windows DLSS build. Now unified: zero `d81d755` refs in the lock;
   `cargo tree --target x86_64-pc-windows-msvc -d` shows a single wgpu/naga graph; both feature combos
   build on macOS; `cargo test --release` 98/98.
2. **`05203e5` P22: bound headless GPU waits** — SHOT/AS_STATS/RT_SPIKE all waited with
   `PollType::wait_indefinitely()` (the exact documented spin-forever). New shared helpers
   `gpu::headless_wait_idle` / `gpu::headless_wait_map` (`HEADLESS_DEADLINE` = 30 s → log + `exit(70)`,
   never touching the GPU again) at every headless wait site, plus a bounded completion check after
   SHOT warm-up frame 0. Verified: DDA SHOT still renders normally.

## ★ Key finding of the day (rung 1 of the verification ladder)

`VOXELCRAFT_RT_SPIKE=/tmp/spike.png cargo run --release` on Metal/M3:

```
RT_SPIKE backend: Metal
RT_SPIKE: scene has 76 vertices, 38 triangles
RT_SPIKE: acceleration structures built          ← fence fix WORKS (this waits for completion)
ERROR RT_SPIKE readback: GPU readback did not complete within 30s — appears wedged; aborting (exit 70)
```

**The AS build now completes on Metal (the fence fix did its job), but the `rayQuery` trace dispatch
itself wedges.** That is a second, distinct bug the fence fix never claimed to cover. The machine
survives fine (no AGX restart events, no diagnostic reports — process kill reclaims the submission;
the bounded waits turn what used to be a frozen process into a clean exit-70).

### Prime suspect

PR #9645's testing showed the upstream RT examples **pass on current trunk on this exact M3** — but our
fork base is v29.0.3. Diffing pin→trunk shows upstream **rewrote the naga MSL ray-query lowering**
after v29.0.3: there is a new `naga/src/back/msl/ray.rs` module (intersection handling rebuilt around
`metal::raytracing::intersection_query` with a generated `ray_query_get_intersection` function, etc.).
Our pin has the old lowering; trunk has the new one. The old lowering producing a wedging trace on
AGX/M3 is the leading hypothesis. (Also noted: gfx-rs/wgpu#9100's example "failures" turned out to be
the MTL *shader-validation* harness, fixed upstream by `disable_mtl_shader_validation()` in tests
(`daced7ac`) — i.e. NOT evidence of a trace hang; our hang is something the examples-at-trunk don't hit
because trunk has the new lowering.)

### Where the investigation stopped (interrupted mid-diff)

Last command run, in `~/code/wgpu`:
```sh
git diff 933587187 upstream/trunk -- naga/src/back/msl/ | grep -iE "intersect|ray"
```
showed the `mod ray;` rewrite. Next concrete steps are in "Mac tasks" below.

## Verification ladder status (headless only; every run wrapped in a timeout)

macOS has no GNU `timeout`; use `perl -e 'alarm shift; exec @ARGV or die' 90 cargo run --release`.
The in-process 30 s `HEADLESS_DEADLINE` is the primary bound. **Commit before every GPU run.**

| Rung | Test | Status |
|---|---|---|
| 0 | builds + cargo test + DDA SHOT baseline | ✅ pass |
| 1 | `VOXELCRAFT_RT_SPIKE` on Metal (same-submission AS build + trace) | ⚠️ AS build passes; **trace wedges** (exit 70) |
| 2 | `VOXELCRAFT_TRACER=hwrt VOXELCRAFT_AS_STATS=1 VOXELCRAFT_SHOT=…` (real-world AS builds) | ⏸ blocked on rung 1 trace fix |
| 3 | `VOXELCRAFT_TRACER=hwrt VOXELCRAFT_BENCH=30 VOXELCRAFT_RTX=1` then `=2` | ⏸ |
| 4 | full SHOT hwrt | ⏸ |
| 5 | parity PNGs dda vs hwrt (`VOXELCRAFT_GI_RAYS=64`) | ⏸ |

Remaining phases after the ladder: bench comparison → Metal default flip decision →
`Quality::metal_hwrt()` tracer-aware tier (constraint: `Quality::maxed()` + its anchor tests stay
byte-identical) → README/CLAUDE.md updates → supervised windowed test (user present — windowed Metal
is the path class that wedged the AGX driver on 2026-06-03; never launch one unattended).

## Windows tasks (can be done independently of the Metal trace bug)

This is the **pending gate** flagged in `889aa8a` — the dependency graph Windows consumes changed:

1. `git fetch && git checkout p22-metal-hwrt` (or cherry-pick `889aa8a` + `05203e5` onto main).
2. `cargo build --release` (default features; needs DLSS_SDK + LIBCLANG_PATH + STREAMLINE_SDK as
   usual). This proves the unified pin compiles against real `dlss_wgpu_dx12` — the Mac only
   type-checks the stubs.
3. Quick hwrt sanity (the fence commit is Metal-only code, DX12 should be bit-identical):
   `VOXELCRAFT_SHOT=p22_win.png cargo run --release` → compare against a pre-P22 shot
   (also closes P21's pending defaults-PNG byte-compare if you take it at tier defaults).
4. Run the game briefly; confirm DLSS RR + FG still initialize (watch the log lines).
5. If green: this branch is safe to merge to main from the Windows side, and
   `jtbirdsell/dlss_wgpu_dx12` `wgpu-bump-933587` can merge to its main (or stay a pinned branch —
   voxelcraft pins the rev either way).

## Mac tasks (the Metal trace bug)

1. **Confirm the codegen hypothesis cheaply (no GPU):** generate MSL for `assets/shaders/rt_spike.wgsl`
   with the pin's naga vs trunk's naga (`cargo run -p naga-cli -- <in.wgsl> out.metal` from each
   checkout — input must be the fully concatenated source rt_spike feeds wgpu) and eyeball the
   intersection loop (`while rayQueryProceed`) lowering for a non-terminating pattern.
2. **Find the lowering-rewrite commit(s):** in `~/code/wgpu`,
   `git log --oneline 4cbe6232b..upstream/trunk -- naga/src/back/msl/ray.rs` (file is new on trunk;
   find the PR that added it and any follow-up fixes). Assess cherry-pickability onto `9335871`
   (naga-internal → no engine ABI impact expected, but it may drag prerequisite naga refactors).
3. **If cherry-pickable:** new fork rev on `voxelcraft/metal-as-sync` → bump BOTH repos' pins again
   (voxelcraft `[patch]` + `dlss_wgpu_dx12` branch — keep them identical, that's the invariant) →
   re-run rung 1. If rt_spike passes, walk rungs 2–5.
4. **If not cherry-pickable:** options are (a) backport a minimal fix by hand, (b) test rt_spike
   against a trunk-based wgpu in a scratch crate to confirm trunk really fixes it before investing,
   (c) park Metal hwrt behind the existing opt-in until the next planned wgpu bump (the
   "deliberate mini-project").
5. Diagnostics if needed: `MTL_DEBUG_LAYER=1` on rt_spike; Metal frame capture via `MTLCaptureManager`
   (headless-safe); `VOXELCRAFT_GI=fragment` / `VOXELCRAFT_RTX=1|2` bisects which shader wedges in the
   full engine (rungs 3+ only).

## Cheat sheet

```sh
# bounded GPU run (no GNU timeout on macOS)
perl -e 'alarm shift; exec @ARGV or die' 90 cargo run --release

# rung 1 (the failing one)
VOXELCRAFT_RT_SPIKE=/tmp/spike.png RUST_LOG=info cargo run --release

# canonical bench scene (Phase 4)
VOXELCRAFT_BENCH=200 VOXELCRAFT_TIME=0.30 VOXELCRAFT_TRACER=hwrt cargo run --release
# M3 reference: DDA tier defaults ≈ 13.7 ms p50; target ≤ ~13–14 ms
```

Plan file (Mac-local): `~/.claude/plans/alright-i-think-we-ve-smooth-wadler.md`.
wgpu fix-effort context (Mac-local, git-excluded): `~/code/wgpu/CLAUDE.md`.
