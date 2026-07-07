//! Global, world-independent player settings (M35), persisted to `saves/settings.cfg` as a tiny
//! hand-written `key=value` text format (the repo has no serde; this matches the persistence style).
//!
//! Resolution order at startup is **env override > persisted setting > built-in default**: the engine
//! reads `Settings` as the source of truth, and the `VOXELCRAFT_*` env vars remain a dev override (so
//! existing scripts/screenshots behave identically). The graphics fields that are baked at
//! device/pipeline creation are "apply on restart"; the rest apply live (N5 wires the live hooks).
#![allow(dead_code)] // Settings + persistence — loaded/applied by the scene flow in N3–N5.

use std::path::PathBuf;

/// Where the global settings file lives (sibling of the per-world save dirs).
pub fn settings_path() -> PathBuf {
    PathBuf::from("saves").join("settings.cfg")
}

macro_rules! str_enum {
    ($(#[$m:meta])* $name:ident { $($variant:ident => $s:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub fn as_str(self) -> &'static str { match self { $(Self::$variant => $s),+ } }
            pub fn from_str(s: &str) -> Option<Self> {
                match s { $($s => Some(Self::$variant),)+ _ => None }
            }
            /// All variants, in declaration order (for the settings cycler).
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            /// The next variant (wraps) — used by the Graphics-tab "< value >" cyclers.
            pub fn next(self) -> Self {
                let all = Self::ALL;
                let i = all.iter().position(|&v| v == self).unwrap_or(0);
                all[(i + 1) % all.len()]
            }
        }
    };
}

str_enum!(DlssMode { Off => "off", SuperResolution => "sr", RayReconstruction => "rr" });
str_enum!(DlssQuality { Dlaa => "dlaa", Quality => "quality", Balanced => "balanced", Performance => "performance", UltraPerformance => "ultra" });
str_enum!(GiMode { Deferred => "compute", Fragment => "fragment" });
str_enum!(Tracer { Hardware => "hwrt", Software => "dda" });
str_enum!(Backend { Auto => "auto", Dx12 => "dx12", Vulkan => "vulkan", Metal => "metal", Gl => "gl" });

/// Persisted player settings. Defaults mirror the previous hardcoded constants / env defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    // --- applied live (N5) ---
    pub fov: f32,             // base vertical FOV, degrees
    pub sensitivity: f32,     // mouse-look multiplier
    pub render_distance: i32, // chunks
    pub view_bob: bool,       // camera + item head-bob
    // --- applied on restart (device/pipeline-baked) ---
    pub dlss_mode: DlssMode,
    pub dlss_quality: DlssQuality,
    pub supersample: f32, // 1.0..4.0 (DLDSR-style); 1.0 = off
    pub gi_mode: GiMode,
    pub gi_rays: u32, // hemisphere samples/pixel
    pub tracer: Tracer,
    pub backend: Backend,
    pub frame_gen: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            fov: 70.0,
            sensitivity: 0.0022,
            render_distance: 12,
            view_bob: true,
            dlss_mode: DlssMode::RayReconstruction,
            dlss_quality: DlssQuality::Quality,
            supersample: 1.0,
            gi_mode: GiMode::Deferred,
            gi_rays: 8,
            tracer: Tracer::Hardware,
            backend: Backend::Auto,
            // Mirrors frame_gen::requested(): FG is opt-in (env default OFF). The old `true` was
            // a placebo — the toggle displayed on while the engine never enabled it.
            frame_gen: false,
        }
    }
}

impl Settings {
    /// Load from `saves/settings.cfg`, defaulting any missing/unknown key (forward/back compatible).
    pub fn load() -> Settings {
        let mut s = Settings::default();
        if let Ok(text) = std::fs::read_to_string(settings_path()) {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    s.apply_kv(k.trim(), v.trim());
                }
            }
        }
        s
    }

    /// Apply one `key=value` line; unknown keys / unparseable values are ignored (keep the default).
    fn apply_kv(&mut self, k: &str, v: &str) {
        match k {
            // Numeric fields clamp to their slider ranges: a hand-edited cfg feeds the env
            // seeding (S15) and an oversized render_distance would drive the voxel volume past
            // wgpu's 3D-texture limit — a persistent crash on world entry.
            "fov" => {
                if let Ok(x) = v.parse::<f32>() {
                    self.fov = x.clamp(50.0, 110.0);
                }
            }
            "sensitivity" => {
                if let Ok(x) = v.parse::<f32>() {
                    self.sensitivity = x.clamp(0.0005, 0.006);
                }
            }
            "render_distance" => {
                if let Ok(x) = v.parse::<i32>() {
                    self.render_distance = x.clamp(4, 32);
                }
            }
            "view_bob" => self.view_bob = v == "1" || v == "true" || v == "on",
            "dlss_mode" => {
                if let Some(x) = DlssMode::from_str(v) {
                    self.dlss_mode = x;
                }
            }
            "dlss_quality" => {
                if let Some(x) = DlssQuality::from_str(v) {
                    self.dlss_quality = x;
                }
            }
            "supersample" => {
                if let Ok(x) = v.parse::<f32>() {
                    self.supersample = x.clamp(1.0, 4.0);
                }
            }
            "gi_mode" => {
                if let Some(x) = GiMode::from_str(v) {
                    self.gi_mode = x;
                }
            }
            "gi_rays" => {
                if let Ok(x) = v.parse::<u32>() {
                    self.gi_rays = x.clamp(1, 16);
                }
            }
            "tracer" => {
                if let Some(x) = Tracer::from_str(v) {
                    self.tracer = x;
                }
            }
            "backend" => {
                if let Some(x) = Backend::from_str(v) {
                    self.backend = x;
                }
            }
            "frame_gen" => self.frame_gen = v == "1" || v == "true" || v == "on",
            _ => {}
        }
    }

    /// Serialize to the `key=value` text body.
    pub fn to_text(&self) -> String {
        let b = |x: bool| if x { "1" } else { "0" };
        format!(
            "# Voxelcraft settings — edit in-game (Settings menu) or here.\n\
             fov={}\n\
             sensitivity={}\n\
             render_distance={}\n\
             view_bob={}\n\
             dlss_mode={}\n\
             dlss_quality={}\n\
             supersample={}\n\
             gi_mode={}\n\
             gi_rays={}\n\
             tracer={}\n\
             backend={}\n\
             frame_gen={}\n",
            self.fov,
            self.sensitivity,
            self.render_distance,
            b(self.view_bob),
            self.dlss_mode.as_str(),
            self.dlss_quality.as_str(),
            self.supersample,
            self.gi_mode.as_str(),
            self.gi_rays,
            self.tracer.as_str(),
            self.backend.as_str(),
            b(self.frame_gen),
        )
    }

    /// S15/N5: the `VOXELCRAFT_*` seeds this settings object implies — every field that differs
    /// from the built-in default, as (env key, value). The caller sets each var ONLY when it is
    /// absent, preserving the documented resolution order (env override > persisted > default).
    pub fn env_seed_pairs(&self) -> Vec<(&'static str, String)> {
        let d = Settings::default();
        let mut out: Vec<(&'static str, String)> = Vec::new();
        if self.dlss_mode != d.dlss_mode {
            out.push(("VOXELCRAFT_DLSS", self.dlss_mode.as_str().into()));
        }
        if self.dlss_quality != d.dlss_quality {
            out.push(("VOXELCRAFT_DLSS_QUALITY", self.dlss_quality.as_str().into()));
        }
        if (self.supersample - d.supersample).abs() > 1e-3 {
            out.push(("VOXELCRAFT_SS", self.supersample.to_string()));
        }
        if self.gi_mode != d.gi_mode {
            out.push(("VOXELCRAFT_GI", self.gi_mode.as_str().into()));
        }
        if self.gi_rays != d.gi_rays {
            out.push(("VOXELCRAFT_GI_RAYS", self.gi_rays.to_string()));
        }
        if self.tracer != d.tracer {
            out.push(("VOXELCRAFT_TRACER", self.tracer.as_str().into()));
        }
        if self.backend != d.backend && self.backend != Backend::Auto {
            out.push(("VOXELCRAFT_BACKEND", self.backend.as_str().into()));
        }
        if self.frame_gen != d.frame_gen {
            out.push(("VOXELCRAFT_FG", if self.frame_gen { "1" } else { "0" }.into()));
        }
        if self.render_distance != d.render_distance {
            out.push(("VOXELCRAFT_RENDER_DISTANCE", self.render_distance.to_string()));
        }
        out
    }

    /// Persist to `saves/settings.cfg` (best-effort; logs on failure). Tmp+rename so a crash
    /// mid-write can't truncate the file (S3 — same pattern as the world saves).
    pub fn save(&self) {
        let _ = std::fs::create_dir_all("saves");
        let path = settings_path();
        let tmp = path.with_extension("cfg.tmp");
        let write = std::fs::write(&tmp, self.to_text()).and_then(|()| std::fs::rename(&tmp, &path));
        if let Err(e) = write {
            log::error!("failed to save settings: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_through_text() {
        let mut s = Settings::default();
        s.fov = 95.0;
        s.sensitivity = 0.004;
        s.render_distance = 20;
        s.view_bob = false;
        s.dlss_mode = DlssMode::Off;
        s.dlss_quality = DlssQuality::Performance;
        s.supersample = 1.5;
        s.gi_mode = GiMode::Fragment;
        s.gi_rays = 4;
        s.tracer = Tracer::Software;
        s.backend = Backend::Vulkan;
        s.frame_gen = false;
        // Re-parse the serialized body into a fresh defaulted Settings.
        let mut back = Settings::default();
        for line in s.to_text().lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line.split_once('=').unwrap();
            back.apply_kv(k.trim(), v.trim());
        }
        assert_eq!(s, back);
    }

    #[test]
    fn out_of_range_values_clamp_to_slider_bounds() {
        let mut s = Settings::default();
        s.apply_kv("render_distance", "9999");
        s.apply_kv("gi_rays", "0");
        s.apply_kv("fov", "500");
        s.apply_kv("supersample", "40");
        assert_eq!(s.render_distance, 32, "a hand-edited cfg cannot blow the voxel volume");
        assert_eq!(s.gi_rays, 1);
        assert_eq!(s.fov, 110.0);
        assert_eq!(s.supersample, 4.0);
    }

    #[test]
    fn unknown_keys_and_bad_values_keep_defaults() {
        let mut s = Settings::default();
        s.apply_kv("nonsense", "123");
        s.apply_kv("fov", "not-a-number");
        s.apply_kv("dlss_mode", "bogus");
        assert_eq!(s, Settings::default());
    }

    #[test]
    fn s15_env_seed_pairs_only_non_defaults() {
        assert!(Settings::default().env_seed_pairs().is_empty(), "defaults seed nothing");
        let mut s = Settings::default();
        s.tracer = Tracer::Software;
        s.render_distance = 16;
        s.frame_gen = true;
        s.backend = Backend::Auto; // Auto never seeds a backend
        let pairs = s.env_seed_pairs();
        assert_eq!(pairs.len(), 3, "exactly the changed fields seed: {pairs:?}");
        assert!(pairs.contains(&("VOXELCRAFT_TRACER", "dda".into())));
        assert!(pairs.contains(&("VOXELCRAFT_RENDER_DISTANCE", "16".into())));
        assert!(pairs.contains(&("VOXELCRAFT_FG", "1".into())));
    }

    #[test]
    fn str_enum_cycles_and_parses() {
        assert_eq!(DlssMode::Off.next(), DlssMode::SuperResolution);
        assert_eq!(DlssMode::RayReconstruction.next(), DlssMode::Off); // wraps
        assert_eq!(GiMode::from_str("fragment"), Some(GiMode::Fragment));
        assert_eq!(Tracer::from_str("nope"), None);
    }
}
