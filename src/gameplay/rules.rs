//! Game rules (P6): difficulty. Peaceful / Easy / Normal / Hard, affecting hostile spawning, the
//! damage hostile mobs deal, and starvation. A world property (persisted in `level.bin`).

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Difficulty {
    Peaceful,
    Easy,
    #[default]
    Normal,
    Hard,
}

impl Difficulty {
    /// Multiplier applied to ALL hostile-sourced damage (mob melee, arrows, creeper blasts) in
    /// `Player::take_hit`. Environmental damage (fall / lava / drowning / starvation) calls
    /// `apply_damage` directly and is NEVER scaled. Peaceful = 0 (and hostiles don't spawn anyway).
    /// Minecraft scales mob melee most strongly; scaling arrows/explosions too is a small simplification.
    pub fn damage_mult(self) -> f32 {
        match self {
            Self::Peaceful => 0.0,
            Self::Easy => 0.5,
            Self::Normal => 1.0,
            Self::Hard => 1.5,
        }
    }

    /// HEALTH floor below which starvation can't push the player (Minecraft): Easy stops at 10 HP,
    /// Normal at 1 HP, Hard can starve to death (0). Peaceful never starves (handled separately).
    pub fn starve_floor(self) -> f32 {
        match self {
            Self::Peaceful => 20.0,
            Self::Easy => 10.0,
            Self::Normal => 1.0,
            Self::Hard => 0.0,
        }
    }

    /// Hostile mobs spawn unless the world is Peaceful.
    pub fn spawns_hostiles(self) -> bool {
        self != Self::Peaceful
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Peaceful => "Peaceful",
            Self::Easy => "Easy",
            Self::Normal => "Normal",
            Self::Hard => "Hard",
        }
    }

    /// Cycle for the in-game toggle: Peaceful → Easy → Normal → Hard → Peaceful.
    pub fn next(self) -> Self {
        match self {
            Self::Peaceful => Self::Easy,
            Self::Easy => Self::Normal,
            Self::Normal => Self::Hard,
            Self::Hard => Self::Peaceful,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Peaceful => 0,
            Self::Easy => 1,
            Self::Normal => 2,
            Self::Hard => 3,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Peaceful,
            1 => Self::Easy,
            3 => Self::Hard,
            _ => Self::Normal,
        }
    }

    /// Parse a `VOXELCRAFT_DIFFICULTY` value (peaceful|easy|normal|hard), or None if unrecognized.
    pub fn from_env(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "peaceful" | "p" | "0" => Some(Self::Peaceful),
            "easy" | "e" | "1" => Some(Self::Easy),
            "normal" | "n" | "2" => Some(Self::Normal),
            "hard" | "h" | "3" => Some(Self::Hard),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_roundtrip() {
        assert_eq!(Difficulty::default(), Difficulty::Normal);
        assert_eq!(Difficulty::Easy.damage_mult(), 0.5);
        assert_eq!(Difficulty::Hard.damage_mult(), 1.5);
        assert_eq!(Difficulty::Peaceful.damage_mult(), 0.0);
        assert_eq!(Difficulty::Easy.starve_floor(), 10.0);
        assert_eq!(Difficulty::Hard.starve_floor(), 0.0);
        assert!(!Difficulty::Peaceful.spawns_hostiles());
        assert!(Difficulty::Hard.spawns_hostiles());
        for d in [Difficulty::Peaceful, Difficulty::Easy, Difficulty::Normal, Difficulty::Hard] {
            assert_eq!(Difficulty::from_u8(d.as_u8()), d);
        }
        // The cycle visits all four.
        let mut d = Difficulty::Peaceful;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            seen.insert(d);
            d = d.next();
        }
        assert_eq!(seen.len(), 4);
        assert_eq!(Difficulty::from_env("HARD"), Some(Difficulty::Hard));
        assert_eq!(Difficulty::from_env("nonsense"), None);
    }
}
