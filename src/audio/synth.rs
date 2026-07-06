//! Procedural sound synthesis (S9). Every SFX and music segment in the game is generated here in
//! code at startup — no audio asset files, matching the procedural texture/font ethos. Buffers are
//! mono f32 at 44.1 kHz; `audio.rs` bakes stereo pan + distance attenuation at play time.
//!
//! The palette is deliberately small: white noise, sine/triangle/saw oscillators, linear pitch
//! sweeps, attack/decay envelopes, and a one-pole low-pass — enough for readable Minecraft-style
//! foley without a sample library.

pub const SAMPLE_RATE: u32 = 44_100;
const SR: f32 = SAMPLE_RATE as f32;

/// Deterministic per-buffer RNG (xorshift) so the bank is identical every launch.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next(&mut self) -> f32 {
        // 0..1
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1 << 24) as f32
    }
    fn bi(&mut self) -> f32 {
        self.next() * 2.0 - 1.0 // -1..1
    }
}

/// A build context: sample buffer + helpers that add (mix) into it.
struct Buf {
    s: Vec<f32>,
}

impl Buf {
    fn new(secs: f32) -> Self {
        Buf { s: vec![0.0; (secs * SR) as usize] }
    }

    /// Mix a noise burst: `lp` one-pole low-pass coefficient 0..1 (1 = unfiltered), attack/decay
    /// envelope in seconds, amplitude `amp`, starting at `at` seconds.
    fn noise(&mut self, rng: &mut Rng, at: f32, attack: f32, decay: f32, lp: f32, amp: f32) {
        let start = (at * SR) as usize;
        let n = ((attack + decay) * SR) as usize;
        let mut y = 0.0f32;
        for i in 0..n {
            let idx = start + i;
            if idx >= self.s.len() {
                break;
            }
            let t = i as f32 / SR;
            let env = if t < attack { t / attack.max(1e-6) } else { 1.0 - (t - attack) / decay.max(1e-6) };
            let x = rng.bi();
            y += lp * (x - y); // one-pole low-pass
            self.s[idx] += y * env.max(0.0) * amp;
        }
    }

    /// Mix a tone with a linear pitch sweep `f0 -> f1` Hz over its length; `shape` 0 = sine,
    /// 1 = triangle, 2 = saw (harsher). Attack/decay envelope, amplitude, start offset.
    fn tone(&mut self, at: f32, attack: f32, decay: f32, f0: f32, f1: f32, shape: u8, amp: f32) {
        let start = (at * SR) as usize;
        let n = ((attack + decay) * SR) as usize;
        let mut phase = 0.0f32;
        for i in 0..n {
            let idx = start + i;
            if idx >= self.s.len() {
                break;
            }
            let t = i as f32 / SR;
            let frac = i as f32 / n.max(1) as f32;
            let f = f0 + (f1 - f0) * frac;
            phase = (phase + f / SR).fract();
            let env = if t < attack { t / attack.max(1e-6) } else { 1.0 - (t - attack) / decay.max(1e-6) };
            let v = match shape {
                0 => (phase * std::f32::consts::TAU).sin(),
                1 => 4.0 * (phase - 0.5).abs() - 1.0, // triangle
                _ => phase * 2.0 - 1.0,               // saw
            };
            self.s[idx] += v * env.max(0.0) * amp;
        }
    }

    /// Amplitude-modulate the whole buffer (tremolo/formant wobble for animal voices).
    fn wobble(&mut self, hz: f32, depth: f32) {
        for (i, v) in self.s.iter_mut().enumerate() {
            let m = 1.0 - depth * 0.5 * (1.0 + (i as f32 / SR * hz * std::f32::consts::TAU).sin());
            *v *= m;
        }
    }

    fn finish(mut self) -> Vec<f32> {
        // Soft-clip to [-1, 1] (tanh keeps stacked mixes from cracking) + a 2 ms fade-out tail.
        for v in self.s.iter_mut() {
            *v = v.tanh();
        }
        let n = self.s.len();
        let fade = (0.002 * SR) as usize;
        for i in 0..fade.min(n) {
            let k = i as f32 / fade as f32;
            self.s[n - 1 - i] *= k;
        }
        self.s
    }
}

/// Every sound in the game. Bank indices — keep in sync with `synthesize`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sfx {
    StepStone,
    StepDirt,
    StepGrass,
    StepSand,
    StepWood,
    StepLeaves,
    BreakStone,
    BreakDirt,
    BreakSand,
    BreakWood,
    BreakLeaves,
    BreakGlass,
    Place,
    DoorOpen,
    DoorClose,
    LeverClick,
    Click,
    PlayerHurt,
    MobHit,
    MobDeath,
    Land,
    Splash,
    Eat,
    Burp,
    BowShoot,
    ArrowHit,
    CreeperHiss,
    Explosion,
    ZombieGroan,
    SkeletonRattle,
    CowMoo,
    PigOink,
    SheepBaa,
    ChickenCluck,
    WolfBark,
    ItemPop,
    XpDing,
}

impl Sfx {
    pub const ALL: [Sfx; 37] = [
        Sfx::StepStone,
        Sfx::StepDirt,
        Sfx::StepGrass,
        Sfx::StepSand,
        Sfx::StepWood,
        Sfx::StepLeaves,
        Sfx::BreakStone,
        Sfx::BreakDirt,
        Sfx::BreakSand,
        Sfx::BreakWood,
        Sfx::BreakLeaves,
        Sfx::BreakGlass,
        Sfx::Place,
        Sfx::DoorOpen,
        Sfx::DoorClose,
        Sfx::LeverClick,
        Sfx::Click,
        Sfx::PlayerHurt,
        Sfx::MobHit,
        Sfx::MobDeath,
        Sfx::Land,
        Sfx::Splash,
        Sfx::Eat,
        Sfx::Burp,
        Sfx::BowShoot,
        Sfx::ArrowHit,
        Sfx::CreeperHiss,
        Sfx::Explosion,
        Sfx::ZombieGroan,
        Sfx::SkeletonRattle,
        Sfx::CowMoo,
        Sfx::PigOink,
        Sfx::SheepBaa,
        Sfx::ChickenCluck,
        Sfx::WolfBark,
        Sfx::ItemPop,
        Sfx::XpDing,
    ];
}

/// Synthesize one SFX. Deterministic per `Sfx` (seeded by its discriminant).
pub fn synthesize(sfx: Sfx) -> Vec<f32> {
    let mut r = Rng::new(sfx as u64 * 0x9E37_79B9 + 7);
    match sfx {
        // ── footsteps: short filtered taps, material = filter + length ──────────────────────────
        Sfx::StepStone => {
            let mut b = Buf::new(0.09);
            b.noise(&mut r, 0.0, 0.004, 0.06, 0.35, 0.55);
            b.finish()
        }
        Sfx::StepDirt | Sfx::StepGrass => {
            let mut b = Buf::new(0.1);
            b.noise(&mut r, 0.0, 0.006, 0.07, 0.18, 0.5);
            if sfx == Sfx::StepGrass {
                b.noise(&mut r, 0.01, 0.004, 0.05, 0.6, 0.12); // grass swish on top
            }
            b.finish()
        }
        Sfx::StepSand => {
            let mut b = Buf::new(0.12);
            b.noise(&mut r, 0.0, 0.01, 0.09, 0.5, 0.4);
            b.finish()
        }
        Sfx::StepWood => {
            let mut b = Buf::new(0.1);
            b.tone(0.0, 0.002, 0.07, 190.0, 150.0, 0, 0.4); // knock body
            b.noise(&mut r, 0.0, 0.003, 0.04, 0.3, 0.3);
            b.finish()
        }
        Sfx::StepLeaves => {
            let mut b = Buf::new(0.12);
            b.noise(&mut r, 0.0, 0.01, 0.09, 0.75, 0.3);
            b.finish()
        }
        // ── breaks: bigger crunches of the same materials ────────────────────────────────────────
        Sfx::BreakStone => {
            let mut b = Buf::new(0.25);
            b.noise(&mut r, 0.0, 0.005, 0.18, 0.3, 0.8);
            b.tone(0.0, 0.002, 0.1, 220.0, 90.0, 0, 0.25); // rubble body
            b.finish()
        }
        Sfx::BreakDirt => {
            let mut b = Buf::new(0.22);
            b.noise(&mut r, 0.0, 0.008, 0.16, 0.15, 0.7);
            b.finish()
        }
        Sfx::BreakSand => {
            let mut b = Buf::new(0.25);
            b.noise(&mut r, 0.0, 0.01, 0.2, 0.45, 0.6);
            b.finish()
        }
        Sfx::BreakWood => {
            let mut b = Buf::new(0.25);
            b.tone(0.0, 0.002, 0.12, 160.0, 90.0, 1, 0.5); // crack
            b.noise(&mut r, 0.02, 0.005, 0.15, 0.35, 0.5); // splinter
            b.finish()
        }
        Sfx::BreakLeaves => {
            let mut b = Buf::new(0.2);
            b.noise(&mut r, 0.0, 0.01, 0.16, 0.8, 0.45);
            b.finish()
        }
        Sfx::BreakGlass => {
            let mut b = Buf::new(0.3);
            b.noise(&mut r, 0.0, 0.002, 0.22, 0.95, 0.6); // bright shatter hiss
            for k in 0..4 {
                // glinting partials
                let f = 1800.0 + k as f32 * 700.0 + r.next() * 300.0;
                b.tone(0.004 * k as f32, 0.001, 0.12, f, f * 0.85, 0, 0.15);
            }
            b.finish()
        }
        Sfx::Place => {
            let mut b = Buf::new(0.1);
            b.tone(0.0, 0.002, 0.06, 140.0, 110.0, 0, 0.45); // soft thud
            b.noise(&mut r, 0.0, 0.003, 0.03, 0.25, 0.25);
            b.finish()
        }
        Sfx::DoorOpen | Sfx::DoorClose => {
            let mut b = Buf::new(0.35);
            let (f0, f1) = if sfx == Sfx::DoorOpen { (140.0, 230.0) } else { (230.0, 130.0) };
            b.tone(0.0, 0.05, 0.2, f0, f1, 2, 0.12); // creak sweep
            b.noise(&mut r, 0.22, 0.004, 0.06, 0.35, 0.4); // latch
            b.finish()
        }
        Sfx::LeverClick => {
            let mut b = Buf::new(0.09);
            b.noise(&mut r, 0.0, 0.001, 0.02, 0.5, 0.5);
            b.noise(&mut r, 0.045, 0.001, 0.03, 0.4, 0.4);
            b.finish()
        }
        Sfx::Click => {
            let mut b = Buf::new(0.05);
            b.tone(0.0, 0.001, 0.03, 900.0, 700.0, 0, 0.3);
            b.finish()
        }
        // ── damage / player ──────────────────────────────────────────────────────────────────────
        Sfx::PlayerHurt => {
            let mut b = Buf::new(0.22);
            b.tone(0.0, 0.01, 0.16, 160.0, 80.0, 2, 0.4); // low "oof"
            b.wobble(30.0, 0.4);
            b.finish()
        }
        Sfx::MobHit => {
            let mut b = Buf::new(0.14);
            b.tone(0.0, 0.002, 0.09, 130.0, 70.0, 0, 0.5); // punchy thud
            b.noise(&mut r, 0.0, 0.002, 0.05, 0.25, 0.3);
            b.finish()
        }
        Sfx::MobDeath => {
            let mut b = Buf::new(0.45);
            b.tone(0.0, 0.02, 0.4, 240.0, 60.0, 1, 0.4); // long descending cry
            b.wobble(12.0, 0.5);
            b.finish()
        }
        Sfx::Land => {
            let mut b = Buf::new(0.16);
            b.tone(0.0, 0.002, 0.1, 100.0, 55.0, 0, 0.55);
            b.noise(&mut r, 0.0, 0.004, 0.07, 0.2, 0.35);
            b.finish()
        }
        Sfx::Splash => {
            let mut b = Buf::new(0.4);
            b.noise(&mut r, 0.0, 0.01, 0.12, 0.55, 0.6); // impact spray
            b.noise(&mut r, 0.12, 0.05, 0.2, 0.35, 0.3); // settling water
            b.finish()
        }
        Sfx::Eat => {
            let mut b = Buf::new(0.12);
            b.noise(&mut r, 0.0, 0.004, 0.08, 0.4, 0.5); // one crunch (the app repeats it)
            b.tone(0.0, 0.002, 0.05, 300.0, 180.0, 1, 0.15);
            b.finish()
        }
        Sfx::Burp => {
            let mut b = Buf::new(0.3);
            b.tone(0.0, 0.02, 0.22, 90.0, 60.0, 2, 0.35);
            b.wobble(22.0, 0.6);
            b.finish()
        }
        // ── ranged / explosions ─────────────────────────────────────────────────────────────────
        Sfx::BowShoot => {
            let mut b = Buf::new(0.25);
            b.tone(0.0, 0.001, 0.12, 420.0, 300.0, 1, 0.4); // string twang
            b.noise(&mut r, 0.01, 0.02, 0.15, 0.6, 0.2); // arrow whoosh
            b.finish()
        }
        Sfx::ArrowHit => {
            let mut b = Buf::new(0.1);
            b.tone(0.0, 0.001, 0.06, 300.0, 180.0, 1, 0.4); // thock
            b.noise(&mut r, 0.0, 0.002, 0.03, 0.4, 0.3);
            b.finish()
        }
        Sfx::CreeperHiss => {
            // THE gameplay cue: a 1.4 s white-noise crescendo. Loud and unmistakable.
            let mut b = Buf::new(1.4);
            b.noise(&mut r, 0.0, 1.25, 0.15, 0.85, 0.85);
            b.finish()
        }
        Sfx::Explosion => {
            let mut b = Buf::new(1.2);
            b.noise(&mut r, 0.0, 0.002, 0.25, 0.9, 0.9); // initial crack
            b.noise(&mut r, 0.02, 0.05, 1.0, 0.08, 1.0); // deep rolling boom
            b.tone(0.0, 0.005, 0.5, 70.0, 30.0, 0, 0.6);
            b.finish()
        }
        // ── mob voices ──────────────────────────────────────────────────────────────────────────
        Sfx::ZombieGroan => {
            let mut b = Buf::new(0.7);
            b.tone(0.0, 0.1, 0.55, 110.0, 70.0, 2, 0.3);
            b.wobble(9.0, 0.6);
            b.finish()
        }
        Sfx::SkeletonRattle => {
            let mut b = Buf::new(0.5);
            for k in 0..7 {
                b.noise(&mut r, 0.06 * k as f32, 0.002, 0.03, 0.55, 0.35);
            }
            b.finish()
        }
        Sfx::CowMoo => {
            let mut b = Buf::new(0.8);
            b.tone(0.0, 0.12, 0.6, 130.0, 85.0, 2, 0.35);
            b.wobble(7.0, 0.35);
            b.finish()
        }
        Sfx::PigOink => {
            let mut b = Buf::new(0.3);
            b.tone(0.0, 0.02, 0.1, 220.0, 140.0, 2, 0.35);
            b.tone(0.14, 0.02, 0.1, 240.0, 150.0, 2, 0.3);
            b.wobble(28.0, 0.5);
            b.finish()
        }
        Sfx::SheepBaa => {
            let mut b = Buf::new(0.6);
            b.tone(0.0, 0.05, 0.5, 300.0, 210.0, 1, 0.3);
            b.wobble(16.0, 0.75); // the bleat
            b.finish()
        }
        Sfx::ChickenCluck => {
            let mut b = Buf::new(0.35);
            b.tone(0.0, 0.005, 0.07, 700.0, 420.0, 1, 0.3);
            b.tone(0.16, 0.005, 0.1, 620.0, 380.0, 1, 0.3);
            b.finish()
        }
        Sfx::WolfBark => {
            let mut b = Buf::new(0.25);
            b.tone(0.0, 0.004, 0.12, 340.0, 160.0, 2, 0.4);
            b.noise(&mut r, 0.0, 0.005, 0.08, 0.5, 0.35);
            b.finish()
        }
        // ── pickups / UI chimes ─────────────────────────────────────────────────────────────────
        Sfx::ItemPop => {
            let mut b = Buf::new(0.12);
            b.tone(0.0, 0.004, 0.08, 500.0, 900.0, 0, 0.3); // rising blip
            b.finish()
        }
        Sfx::XpDing => {
            let mut b = Buf::new(0.3);
            b.tone(0.0, 0.002, 0.2, 1320.0, 1310.0, 0, 0.22);
            b.tone(0.03, 0.002, 0.22, 1980.0, 1975.0, 0, 0.16);
            b.finish()
        }
    }
}

/// A generative ambient music segment: slow-attack chord pads (sine + a whisper of triangle an
/// octave up) over a gentle root drone. `bright` picks the day (major, lifted) vs night (minor,
/// darker + sparser) palette; `variant` rotates the progression. ~24 s per segment.
pub fn music_segment(bright: bool, variant: u64) -> Vec<f32> {
    let mut b = Buf::new(24.0);
    // Chord tables as semitone offsets from the root (A2 = 110 Hz day, F2 ≈ 87.3 Hz night).
    let (root, chords): (f32, [[i32; 3]; 4]) = if bright {
        (110.0, [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 9]]) // A F#m-ish lift
    } else {
        (87.31, [[0, 3, 7], [-2, 3, 5], [0, 3, 8], [-4, 0, 3]]) // Fm colors
    };
    let semi = |n: i32| root * (2f32).powf(n as f32 / 12.0);
    for (ci, chord) in chords.iter().enumerate() {
        let at = ci as f32 * 6.0;
        let rot = (ci as u64 + variant) as usize % chords.len();
        let voiced = chords[rot];
        let use_chord = if variant % 2 == 0 { *chord } else { voiced };
        for (vi, &n) in use_chord.iter().enumerate() {
            let f = semi(n + 12); // pads an octave up from the root drone
            let amp = 0.045 - vi as f32 * 0.008;
            b.tone(at, 2.5, 3.4, f, f * 1.001, 0, amp);
            if bright {
                b.tone(at + 0.8, 2.0, 3.0, f * 2.0, f * 2.002, 1, amp * 0.3); // shimmer
            }
        }
        // Root drone under each chord.
        b.tone(at, 2.0, 4.0, semi(use_chord[0]), semi(use_chord[0]), 0, 0.05);
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_is_sane() {
        for &sfx in Sfx::ALL.iter() {
            let s = synthesize(sfx);
            assert!(!s.is_empty(), "{sfx:?} empty");
            assert!(s.iter().all(|v| v.is_finite() && v.abs() <= 1.0), "{sfx:?} out of range");
            let rms = (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
            assert!(rms > 0.005, "{sfx:?} is near-silent (rms {rms})");
        }
        // Deterministic across calls.
        assert_eq!(synthesize(Sfx::StepStone), synthesize(Sfx::StepStone));
        for bright in [true, false] {
            let m = music_segment(bright, 0);
            assert!(m.len() > (20.0 * SR) as usize);
            assert!(m.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
        }
    }
}
