//! Audio engine (S9): a rodio output stream + the fully code-synthesized sound bank
//! (`audio/synth.rs`). Short SFX are fire-and-forget sources with distance attenuation and
//! constant-power stereo pan BAKED per play (we synthesize mono buffers, so panning is just a
//! two-gain interleave); ambient music runs on a persistent `Sink`, crossfaded between the day and
//! night pad palettes by a per-frame volume lerp.
//!
//! `Audio::new` returns `None` (silent game, every call site no-ops) when the device is missing,
//! under `VOXELCRAFT_MUTE=1`, or during the windowed FG self-test. The headless SHOT/BENCH/MENU/
//! PERSIST paths never construct `State`, so they never touch audio at all.

pub mod synth;

use glam::Vec3;
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, OutputStreamHandle, Sink, Source};
use rustc_hash::FxHashMap;
use std::cell::Cell;

pub use synth::Sfx;

const MUSIC_VOL: f32 = 0.30; // ambient pads sit far under the foley
const MUSIC_FADE: f32 = 0.12; // volume units per second (slow crossfade)
const SFX_RANGE: f32 = 24.0; // world-sound audible radius (blocks)
const EXPLOSION_RANGE: f32 = 64.0;

pub struct Audio {
    /// Must outlive all playback: dropping the stream silences everything.
    _stream: OutputStream,
    handle: OutputStreamHandle,
    bank: FxHashMap<Sfx, Vec<f32>>,
    music: Sink,
    music_bright: bool,
    music_vol: f32,
    seg_ix: usize,
    day: [Vec<f32>; 2],
    night: [Vec<f32>; 2],
    /// Pitch-jitter rng; `Cell` so `play` stays `&self` at every call site.
    rng: Cell<u64>,
}

impl Audio {
    pub fn new() -> Option<Audio> {
        if std::env::var("VOXELCRAFT_MUTE").is_ok_and(|v| v != "0")
            || std::env::var("VOXELCRAFT_FG_TEST").is_ok()
        {
            return None;
        }
        let (stream, handle) = match OutputStream::try_default() {
            Ok(s) => s,
            Err(e) => {
                log::warn!("no audio device — playing silent: {e}");
                return None;
            }
        };
        let bank = Sfx::ALL.iter().map(|&s| (s, synth::synthesize(s))).collect();
        let music = Sink::try_new(&handle).ok()?;
        music.set_volume(0.0);
        Some(Audio {
            _stream: stream,
            handle,
            bank,
            music,
            music_bright: true,
            music_vol: 0.0,
            seg_ix: 0,
            day: [synth::music_segment(true, 0), synth::music_segment(true, 1)],
            night: [synth::music_segment(false, 0), synth::music_segment(false, 1)],
            rng: Cell::new(0x5EED_5EED),
        })
    }

    /// 0.92..1.08 pitch jitter so repeated sounds don't machine-gun.
    fn jitter(&self) -> f32 {
        let mut x = self.rng.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.set(x);
        0.92 + ((x >> 40) as f32 / (1u64 << 24) as f32) * 0.16
    }

    /// Non-positional sound (UI, the player's own actions).
    pub fn play(&self, sfx: Sfx, vol: f32) {
        self.play_panned(sfx, vol, 0.0);
    }

    /// World-positioned sound: quadratic distance falloff + camera-relative constant-power pan.
    pub fn play_at(&self, sfx: Sfx, vol: f32, pos: Vec3, cam_pos: Vec3, cam_fwd: Vec3) {
        let to = pos - cam_pos;
        let dist = to.length();
        let range = if sfx == Sfx::Explosion { EXPLOSION_RANGE } else { SFX_RANGE };
        if dist >= range {
            return;
        }
        let att = (1.0 - dist / range).powi(2);
        let right = cam_fwd.cross(Vec3::Y).normalize_or_zero();
        let pan = if dist > 1e-3 { (to / dist).dot(right).clamp(-1.0, 1.0) * 0.8 } else { 0.0 };
        self.play_panned(sfx, vol * att, pan);
    }

    fn play_panned(&self, sfx: Sfx, vol: f32, pan: f32) {
        if vol <= 0.01 {
            return;
        }
        let Some(mono) = self.bank.get(&sfx) else { return };
        // Constant-power pan baked into an interleaved stereo buffer (mono src, two gains).
        let theta = (pan.clamp(-1.0, 1.0) + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
        let (gl, gr) = (theta.cos(), theta.sin());
        let mut stereo = Vec::with_capacity(mono.len() * 2);
        for &v in mono {
            stereo.push(v * gl);
            stereo.push(v * gr);
        }
        let src = SamplesBuffer::new(2, synth::SAMPLE_RATE, stereo)
            .speed(self.jitter())
            .amplify(vol.min(1.5));
        let _ = self.handle.play_raw(src.convert_samples());
    }

    /// Per-frame: crossfade toward the day/night palette and keep one pad segment queued. The
    /// palette only flips once faded to silence, so a segment never hard-cuts mid-chord.
    pub fn tick_music(&mut self, day_factor: f32, dt: f32) {
        let want_bright = day_factor > 0.5;
        let goal = if want_bright == self.music_bright { MUSIC_VOL } else { 0.0 };
        let step = (goal - self.music_vol).clamp(-MUSIC_FADE * dt, MUSIC_FADE * dt);
        self.music_vol = (self.music_vol + step).clamp(0.0, MUSIC_VOL);
        self.music.set_volume(self.music_vol);
        if want_bright != self.music_bright && self.music_vol <= 0.001 {
            self.music_bright = want_bright;
            // rodio 0.20 gotcha: Sink::stop() only FLAGS the queue — until the audio callback's
            // next periodic pass drains it, sound_count stays > 0 and a same-frame append() walks
            // into sleep_until_end(), a blocking recv() on the MAIN thread (frame hitch; a
            // permanent freeze if the device stalls). A fresh sink never has that state.
            if let Ok(fresh) = rodio::Sink::try_new(&self.handle) {
                fresh.set_volume(self.music_vol);
                self.music = fresh;
            }
        }
        if self.music.len() <= 1 {
            self.seg_ix = (self.seg_ix + 1) % 2;
            let set = if self.music_bright { &self.day } else { &self.night };
            self.music
                .append(SamplesBuffer::new(1, synth::SAMPLE_RATE, set[self.seg_ix].clone()));
        }
    }
}

/// Block → (footstep, break) sound family, keyed off the existing registry predicates.
pub fn block_sfx(id: crate::block::BlockId) -> (Sfx, Sfx) {
    use crate::block;
    if id == block::GLASS || id == block::GLASS_PANE {
        return (Sfx::StepStone, Sfx::BreakGlass);
    }
    if id == block::LEAVES || id == block::AZALEA_LEAVES || block::is_plant(id) {
        return (Sfx::StepLeaves, Sfx::BreakLeaves);
    }
    if id == block::SAND || id == block::GRAVEL {
        return (Sfx::StepSand, Sfx::BreakSand);
    }
    if id == block::GRASS {
        return (Sfx::StepGrass, Sfx::BreakDirt);
    }
    match block::tool_class(id) {
        crate::block::ToolClass::Axe => (Sfx::StepWood, Sfx::BreakWood),
        crate::block::ToolClass::Shovel => (Sfx::StepDirt, Sfx::BreakDirt),
        crate::block::ToolClass::Pickaxe => (Sfx::StepStone, Sfx::BreakStone),
        _ => (Sfx::StepStone, Sfx::BreakStone),
    }
}

/// `VOXELCRAFT_AUDIO_TEST=1`: render the entire bank + a head of each music palette into one WAV
/// (hand-rolled header, 16-bit PCM) for human-free verification — segments are separated by 0.3 s
/// of silence and logged with their offsets so an RMS sweep can check each one. No device needed.
pub fn audio_selftest() {
    let mut pcm: Vec<f32> = Vec::new();
    let gap = vec![0.0f32; (0.3 * synth::SAMPLE_RATE as f32) as usize];
    for &sfx in synth::Sfx::ALL.iter() {
        let s = synth::synthesize(sfx);
        log::info!(
            "AUDIO_TEST {:?}: offset {:.2}s len {:.2}s",
            sfx,
            pcm.len() as f32 / synth::SAMPLE_RATE as f32,
            s.len() as f32 / synth::SAMPLE_RATE as f32
        );
        pcm.extend_from_slice(&s);
        pcm.extend_from_slice(&gap);
    }
    for bright in [true, false] {
        let m = synth::music_segment(bright, 0);
        log::info!(
            "AUDIO_TEST music(bright={bright}): offset {:.2}s",
            pcm.len() as f32 / synth::SAMPLE_RATE as f32
        );
        pcm.extend_from_slice(&m[..(6.0 * synth::SAMPLE_RATE as f32) as usize]);
        pcm.extend_from_slice(&gap);
    }
    let path = std::env::var("VOXELCRAFT_AUDIO_TEST_PATH").unwrap_or_else(|_| "audio_test.wav".into());
    let bytes = wav_bytes(&pcm, synth::SAMPLE_RATE);
    match std::fs::write(&path, bytes) {
        Ok(()) => log::info!("AUDIO_TEST: wrote {path} ({:.1}s)", pcm.len() as f32 / synth::SAMPLE_RATE as f32),
        Err(e) => log::error!("AUDIO_TEST: write failed: {e}"),
    }
}

fn wav_bytes(pcm: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (pcm.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + pcm.len() * 2);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    b.extend_from_slice(&2u16.to_le_bytes()); // block align
    b.extend_from_slice(&16u16.to_le_bytes()); // bits
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &v in pcm {
        b.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_and_families() {
        let w = wav_bytes(&[0.0, 0.5, -0.5], 44_100);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(w.len(), 44 + 6);
        // Family mapping sanity.
        assert_eq!(block_sfx(crate::block::STONE).1, Sfx::BreakStone);
        assert_eq!(block_sfx(crate::block::GLASS).1, Sfx::BreakGlass);
        assert_eq!(block_sfx(crate::block::WOOD).0, Sfx::StepWood);
        assert_eq!(block_sfx(crate::block::GRASS).0, Sfx::StepGrass);
        assert_eq!(block_sfx(crate::block::WHEAT_CROP).1, Sfx::BreakLeaves);
    }
}
