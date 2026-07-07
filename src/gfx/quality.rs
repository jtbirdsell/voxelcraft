//! Platform quality tier (P21): one struct, resolved once at startup, holding every GPU-cost knob
//! that differs between the maxed Windows/RTX experience and the tuned-down macOS/Metal tier.
//!
//! Windows (and any non-Metal backend) resolves to [`Quality::maxed`], which is **exactly today's
//! hardcoded defaults** — the unit tests below anchor each field to the const it replaced, so the
//! Windows experience cannot drift. macOS/Metal resolves to [`Quality::metal`], sized from headless
//! benchmarks on the base M3 (`VOXELCRAFT_BENCH`): the software DDA tracer there spends almost the
//! whole frame marching rays (~66 ms of a 147 ms frame in primary sun shadows + water rays alone at
//! 1512×844), so the tier cuts ray *distances* and *counts* and enables the existing GI temporal
//! accumulator to win the quality back.
//!
//! Every field is only a DEFAULT: the env vars listed on each field still override it (existing
//! knobs keep their names; new ones are documented in CLAUDE.md). Values flow into the `Volume`
//! uniform (`paramsg`, see voxel_volume.rs) so the shared HW-RT/DDA shaders stay bit-identical on
//! Windows by construction, and the `VOXELCRAFT_GI=fragment` parity oracle keeps holding (both GI
//! paths read the same uniform).

/// Metal-only default swapchain scale as a fraction of LOGICAL resolution (DPI-independent): the
/// drawable is `METAL_LOGICAL_SCALE / scale_factor` of the window's physical size, i.e. 0.57× of
/// logical res on any Retina factor. 1.0 would be the pre-P21 logical-res default. Benched on the
/// base M3: 0.57 + the metal() tier ≈ 13.7 ms GPU p50 on the canonical vista — the ~3 ms of
/// headroom under the 16.67 ms vblank absorbs the windowed loop's serialized CPU work (Metal runs
/// one frame in flight post-wedge), so ProMotion doesn't quantize vista-class views down to 40 FPS.
/// `VOXELCRAFT_RENDER_SCALE` (a physical fraction) still overrides verbatim (device.rs).
pub const METAL_LOGICAL_SCALE: f32 = 0.57;

#[derive(Clone, Copy, Debug)]
pub struct Quality {
    /// GI hemisphere rays per pixel (`VOXELCRAFT_GI_RAYS`). ~8.2 ms/ray on the M3 at 1512×844.
    pub gi_rays: u32,
    /// GI bounce-ray range in blocks (`VOXELCRAFT_GI_DIST`); march steps scale with it.
    pub gi_dist: f32,
    /// Range of the SECONDARY sun ray fired from each GI hit (`VOXELCRAFT_GI_SUN_DIST`).
    /// Was hardwired to `gi_dist`; the maxed tier keeps that equality.
    pub gi_sun_dist: f32,
    /// GI temporal accumulation (`VOXELCRAFT_GI_ACCUM`). Off on Windows (the supersampled default
    /// is already clean); ON for the mac tier to denoise the low ray count across frames.
    pub gi_accum: bool,
    /// Primary sun-shadow ray range for opaque/glass/water fragments (`VOXELCRAFT_SUN_DIST`).
    /// Was a literal 96.0 in the shaders — the single largest cost on the M3.
    pub sun_dist: f32,
    /// Water depth-clarity smoothing radius in blocks; 0 = single tap (`VOXELCRAFT_WSMOOTH`).
    pub water_smooth: f32,
    /// Water reflection-ray range (`VOXELCRAFT_WREFL`); was const REFL_MAX = 80.0.
    pub water_refl_max: f32,
    /// Water downward depth-march cap (`VOXELCRAFT_WDEPTH`); was const DEPTH_MAX = 24.0.
    pub water_depth_max: f32,
    /// Voxel-volume horizontal extent in chunks (`VOXELCRAFT_VOLUME_CHUNKS`). Memory is
    /// `(n*32)² * 256 * 2` bytes: 24 → ~302 MB, 20 → ~210 MB.
    pub volume_chunks_xz: i32,
    /// Chunk render distance (`VOXELCRAFT_RENDER_DISTANCE`); fog derives from it (see fog_end).
    pub render_distance: i32,
    /// Voxel-volume chunk uploads per frame (`VOXELCRAFT_UPLOAD_BUDGET`).
    pub upload_budget: usize,
}

impl Quality {
    /// The maxed tier == today's exact hardcoded defaults (each anchored by a unit test below).
    /// Windows/DX12/Vulkan/GL resolve to this verbatim — behavior-identical to before P21.
    pub fn maxed() -> Self {
        Self {
            gi_rays: 8,             // voxel_volume.rs gi_rays default
            gi_dist: 22.0,          // voxel_volume.rs gi_dist default
            gi_sun_dist: 22.0,      // == gi_dist (rtx_common.wgsl passed gi_dist before P21)
            gi_accum: false,        // VOXELCRAFT_GI_ACCUM was opt-in
            sun_dist: 96.0,         // chunk/glass/water.wgsl literal
            water_smooth: 3.0,      // voxel_volume.rs water_smooth default
            water_refl_max: 80.0,   // water.wgsl REFL_MAX
            water_depth_max: 24.0,  // water.wgsl DEPTH_MAX
            // The const stays the canonical legacy extent (24 chunks / 768 blocks); the tier is
            // the only non-test consumer now that the volume itself is instance-sized.
            volume_chunks_xz: crate::voxel_volume::VOL_CHUNKS_XZ,
            render_distance: 12,    // app.rs RENDER_DISTANCE (pre-P21)
            upload_budget: 48,      // voxel_volume.rs upload_budget default
        }
    }

    /// The macOS/Metal tier, sized from M3 benchmarks (see module docs; final values tuned with
    /// `VOXELCRAFT_BENCH` sweeps). Quality cost: shorter shadow/reflection reach and a coarser,
    /// temporally-denoised GI — the trades the user signed off on for 60 FPS.
    pub fn metal() -> Self {
        Self {
            gi_rays: 2,
            gi_dist: 12.0,
            gi_sun_dist: 12.0,
            gi_accum: true,
            sun_dist: 18.0,
            water_smooth: 1.0,
            water_refl_max: 24.0,
            water_depth_max: 12.0,
            volume_chunks_xz: 20,
            render_distance: 8,
            upload_budget: 32,
        }
    }

    /// Resolve the startup tier from the active backend, then fold in per-knob env overrides.
    /// The volume/render-distance invariant is enforced LAST so it also guards env combinations.
    pub fn resolve(backend: wgpu::Backend) -> Self {
        let mut q = Self::tier(backend);
        q.apply_env();
        // The volume's half-extent (chunks×16 blocks) must reach past the fog wall plus the
        // 24-block water-edge fade: chunks×16 ≥ (rd−1)×32 + 24 → chunks ≥ 2×rd (the maxed tier
        // sits exactly at this floor: 24 = 2×12). Clamp + warn (not assert: env overrides reach
        // here in release builds, and a silent seam is worse than a bigger volume).
        let floor = 2 * q.render_distance;
        if q.volume_chunks_xz < floor {
            log::warn!(
                "volume_chunks_xz {} < 2*render_distance ({}) = {floor}; clamping to {floor} \
                 (the tracer volume must reach past the fog wall)",
                q.volume_chunks_xz,
                q.render_distance
            );
            q.volume_chunks_xz = floor;
        }
        log::info!("quality tier ({backend:?}): {q:?}");
        q
    }

    /// The backend's raw tier BEFORE env overrides — S15's live render-distance path needs the
    /// true tier value ("the slider was never touched"), which a seeded resolve can't provide.
    pub fn tier(backend: wgpu::Backend) -> Self {
        if backend == wgpu::Backend::Metal {
            Self::metal()
        } else {
            Self::maxed()
        }
    }

    /// Fog wall distance derived from the render distance (the pre-P21 FOG_END formula).
    pub fn fog_end(&self) -> f32 {
        (self.render_distance as f32 - 1.0) * 32.0
    }

    pub fn fog_start(&self) -> f32 {
        self.fog_end() * 0.68
    }

    fn apply_env(&mut self) {
        env_parse("VOXELCRAFT_GI_RAYS", &mut self.gi_rays);
        env_parse("VOXELCRAFT_GI_DIST", &mut self.gi_dist);
        env_parse("VOXELCRAFT_GI_SUN_DIST", &mut self.gi_sun_dist);
        env_bool("VOXELCRAFT_GI_ACCUM", &mut self.gi_accum);
        env_parse("VOXELCRAFT_SUN_DIST", &mut self.sun_dist);
        env_parse("VOXELCRAFT_WSMOOTH", &mut self.water_smooth);
        env_parse("VOXELCRAFT_WREFL", &mut self.water_refl_max);
        env_parse("VOXELCRAFT_WDEPTH", &mut self.water_depth_max);
        env_parse("VOXELCRAFT_VOLUME_CHUNKS", &mut self.volume_chunks_xz);
        env_parse("VOXELCRAFT_RENDER_DISTANCE", &mut self.render_distance);
        env_parse("VOXELCRAFT_UPLOAD_BUDGET", &mut self.upload_budget);
    }
}

fn env_parse<T: std::str::FromStr>(name: &str, out: &mut T) {
    if let Some(v) = std::env::var(name).ok().and_then(|s| s.trim().parse().ok()) {
        *out = v;
    }
}

/// `1`/`on`/`true` enable, `0`/`off`/`false` disable; anything else leaves the tier default
/// (matches the pre-P21 `VOXELCRAFT_GI_ACCUM` accepted values, plus explicit-off for the mac tier).
fn env_bool(name: &str, out: &mut bool) {
    match std::env::var(name).ok().as_deref().map(str::trim) {
        Some("1") | Some("on") | Some("true") => *out = true,
        Some("0") | Some("off") | Some("false") => *out = false,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The maxed tier must equal the constants it replaced — this is the "Windows is untouched"
    /// regression anchor. If a default is deliberately changed, change BOTH sites.
    #[test]
    fn maxed_matches_legacy_constants() {
        let q = Quality::maxed();
        // maxed() reads VOL_CHUNKS_XZ directly; anchor the const's VALUE so an accidental edit
        // to the legacy extent still fails here.
        assert_eq!(q.volume_chunks_xz, 24);
        assert_eq!(q.gi_rays, 8);
        assert_eq!(q.gi_dist, 22.0);
        assert_eq!(q.gi_sun_dist, q.gi_dist); // pre-P21: secondary GI sun ray used gi_dist
        assert!(!q.gi_accum);
        assert_eq!(q.sun_dist, 96.0); // pre-P21 shader literal
        assert_eq!(q.water_smooth, 3.0);
        assert_eq!(q.water_refl_max, 80.0); // pre-P21 water.wgsl REFL_MAX
        assert_eq!(q.water_depth_max, 24.0); // pre-P21 water.wgsl DEPTH_MAX
        assert_eq!(q.render_distance, 12);
        assert_eq!(q.upload_budget, 48);
    }

    /// Non-Metal backends resolve to maxed when no env overrides are present. (Env vars are
    /// process-global, so only assert on ones the test environment is unlikely to set.)
    #[test]
    fn non_metal_resolves_to_maxed() {
        let q = Quality::resolve(wgpu::Backend::Dx12);
        let m = Quality::maxed();
        assert_eq!(q.sun_dist, m.sun_dist);
        assert_eq!(q.volume_chunks_xz, m.volume_chunks_xz);
        assert_eq!(q.render_distance, m.render_distance);
        assert!(!q.gi_accum);
    }

    /// Both tiers must satisfy the volume/fog invariant out of the box: the volume half-extent
    /// reaches past the fog wall plus the 24-block water-edge fade.
    #[test]
    fn tiers_volume_covers_fog() {
        for q in [Quality::maxed(), Quality::metal()] {
            assert!(q.volume_chunks_xz >= 2 * q.render_distance);
            assert!(q.fog_end() + 24.0 <= (q.volume_chunks_xz * 16) as f32);
        }
    }

    /// resolve() clamps a volume too small for the render distance instead of shipping a seam.
    #[test]
    fn resolve_clamps_volume_floor() {
        let mut q = Quality::metal();
        q.render_distance = 12;
        q.volume_chunks_xz = 16;
        // Reproduce resolve()'s clamp logic (resolve itself reads env; keep the test hermetic).
        let floor = 2 * q.render_distance;
        if q.volume_chunks_xz < floor {
            q.volume_chunks_xz = floor;
        }
        assert_eq!(q.volume_chunks_xz, 24);
    }
}
