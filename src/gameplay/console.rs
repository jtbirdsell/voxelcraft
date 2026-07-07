//! S16: the slash-command console — the PURE half (parsing + item-name resolution), unit-testable
//! without a GPU. Execution (mutating the session) lives in `app::run_command`, which is a thin
//! match over [`Command`]. Single-player only: the player is the operator, no permission model.

use crate::item::{self, ItemId};
use crate::rules::GameMode;

/// One `/tp` coordinate: absolute, or `~`/`~n` relative to the player's current axis value.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Coord {
    Abs(f32),
    Rel(f32),
}

impl Coord {
    /// Resolve against the player's current value on this axis.
    pub fn resolve(self, current: f32) -> f32 {
        match self {
            Coord::Abs(v) => v,
            Coord::Rel(d) => current + d,
        }
    }
}

/// A parsed console command, ready for the app to apply to the session.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Command {
    Gamemode(GameMode),
    Give { item: ItemId, count: u32 },
    Tp([Coord; 3]),
    /// Day fraction 0..=1 (0.25 sunrise / 0.5 noon / 0.75 sunset / 0.0 midnight).
    TimeSet(f32),
    Seed,
    Help,
}

/// Everything `/help` prints (also the parser's own usage reference).
pub const HELP: &str = "/gamemode <s|c>  /give <item> [n]  /tp <x y z> (~ rel)  /time <day|noon|night|midnight|0..1>  /seed";



/// Parse one console line. `resolve` turns an item name/id into an ItemId (injected so the parser
/// stays pure; the app passes a closure over the session's creative palette).
pub fn parse(input: &str, resolve: impl Fn(&str) -> Option<ItemId>) -> Result<Command, String> {
    let s = input.trim();
    let Some(s) = s.strip_prefix('/') else {
        return Err("Commands start with / (try /help)".into());
    };
    let mut tokens = s.split_whitespace();
    let cmd = tokens.next().unwrap_or("");
    let args: Vec<&str> = tokens.collect();
    match cmd {
        "help" => Ok(Command::Help),
        "seed" => Ok(Command::Seed),
        "gamemode" | "gm" => match args.first().copied() {
            Some("survival" | "s" | "0") => Ok(Command::Gamemode(GameMode::Survival)),
            Some("creative" | "c" | "1") => Ok(Command::Gamemode(GameMode::Creative)),
            _ => Err("Usage: /gamemode <survival|creative>".into()),
        },
        "give" => {
            let Some(name) = args.first().copied() else {
                return Err("Usage: /give <item> [count]".into());
            };
            let Some(item) = resolve(name) else {
                return Err(format!("Unknown item {name:?}"));
            };
            let count = match args.get(1) {
                Some(c) => c.parse::<u32>().map_err(|_| format!("Bad count {c:?}"))?,
                None => 1,
            };
            if count == 0 {
                return Err("Count must be at least 1".into());
            }
            // A full inventory's worth of THIS item (unstackables cap at 36, not 36*64 —
            // the overflow spills as world entities, so the cap is also an entity-count guard).
            let max = item::max_stack(item) as u32 * 36;
            Ok(Command::Give { item, count: count.min(max) })
        }
        "tp" | "teleport" => {
            if args.len() != 3 {
                return Err("Usage: /tp <x> <y> <z> (~ or ~n = relative)".into());
            }
            let mut coords = [Coord::Rel(0.0); 3];
            for (i, a) in args.iter().enumerate() {
                coords[i] = if let Some(rest) = a.strip_prefix('~') {
                    if rest.is_empty() {
                        Coord::Rel(0.0)
                    } else {
                        Coord::Rel(rest.parse::<f32>().map_err(|_| format!("Bad offset {a:?}"))?)
                    }
                } else {
                    Coord::Abs(a.parse::<f32>().map_err(|_| format!("Bad coordinate {a:?}"))?)
                };
            }
            Ok(Command::Tp(coords))
        }
        "time" => {
            // Accept both the vanilla-style "/time set day" and the shorthand "/time day".
            let arg = match args.as_slice() {
                ["set", v] => *v,
                [v] => *v,
                _ => return Err("Usage: /time <day|noon|night|midnight|0..1>".into()),
            };
            let t = match arg {
                "day" => 0.30, // matches the bed-wake dawn (S11)
                "noon" => 0.5,
                "night" => 0.80, // just past sunset — hostiles spawn
                "midnight" => 0.0,
                v => v.parse::<f32>().map_err(|_| format!("Bad time {v:?}"))?,
            };
            if !(0.0..=1.0).contains(&t) {
                return Err("Time is a day fraction 0..1 (0.25 sunrise, 0.5 noon, 0.75 sunset)".into());
            }
            Ok(Command::TimeSet(t))
        }
        "" => Err("Empty command (try /help)".into()),
        other => Err(format!("Unknown command /{other} (try /help)")),
    }
}

/// Resolve an item query — a numeric id (must be known) or a display name matched
/// case/punctuation-insensitively ("Iron Ingot" == "iron_ingot" == "IRON INGOT").
pub fn resolve_item(palette: &[ItemId], query: &str) -> Option<ItemId> {
    if let Ok(id) = query.parse::<ItemId>() {
        return item::is_known(id).then_some(id);
    }
    let qn = normalize(query);
    let hit = |id: &ItemId| normalize(item::item_name(*id)) == qn;
    // On a name tie, prefer the non-block entry: "/give wheat" means the crafting material,
    // not the WHEAT_CROP block that shares its display name.
    palette
        .iter()
        .copied()
        .filter(|&id| id > crate::block::MAX_BLOCK)
        .find(hit)
        .or_else(|| palette.iter().copied().find(hit))
}

/// Lowercase, spaces/underscores fold to '_', everything non-alphanumeric else is dropped.
fn normalize(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c {
            ' ' | '_' => Some('_'),
            c if c.is_ascii_alphanumeric() => Some(c.to_ascii_lowercase()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block;

    fn no_items(_: &str) -> Option<ItemId> {
        None
    }

    #[test]
    fn s16_parse_commands() {
        assert_eq!(parse("/help", no_items), Ok(Command::Help));
        assert_eq!(parse("  /seed  ", no_items), Ok(Command::Seed));
        assert_eq!(parse("/gamemode creative", no_items), Ok(Command::Gamemode(GameMode::Creative)));
        assert_eq!(parse("/gm s", no_items), Ok(Command::Gamemode(GameMode::Survival)));
        assert!(parse("/gamemode hardcore", no_items).is_err());
        // Plain chat text is rejected with a hint, not silently eaten.
        assert!(parse("hello world", no_items).unwrap_err().contains("start with /"));
        assert!(parse("/dance", no_items).unwrap_err().contains("Unknown command"));
    }

    #[test]
    fn s16_parse_tp_and_time() {
        let c = parse("/tp 10 96 -20", no_items).unwrap();
        assert_eq!(c, Command::Tp([Coord::Abs(10.0), Coord::Abs(96.0), Coord::Abs(-20.0)]));
        let c = parse("/tp ~ ~10 ~-3.5", no_items).unwrap();
        assert_eq!(c, Command::Tp([Coord::Rel(0.0), Coord::Rel(10.0), Coord::Rel(-3.5)]));
        assert_eq!(Coord::Rel(10.0).resolve(90.0), 100.0);
        assert_eq!(Coord::Abs(7.0).resolve(90.0), 7.0);
        assert!(parse("/tp 1 2", no_items).is_err());
        assert!(parse("/tp a b c", no_items).is_err());
        assert_eq!(parse("/time set day", no_items), Ok(Command::TimeSet(0.30)));
        assert_eq!(parse("/time noon", no_items), Ok(Command::TimeSet(0.5)));
        assert_eq!(parse("/time 0.75", no_items), Ok(Command::TimeSet(0.75)));
        assert!(parse("/time 2.0", no_items).is_err());
        assert!(parse("/time", no_items).is_err());
    }

    #[test]
    fn s16_give_and_item_resolution() {
        let palette = item::creative_palette();
        let resolve = |q: &str| resolve_item(&palette, q);
        // Names in every writing style; numeric ids; unknowns rejected.
        assert_eq!(resolve("stone"), Some(block::STONE));
        assert_eq!(resolve("Iron_Ingot"), Some(item::IRON_INGOT));
        assert_eq!(resolve("MUSHROOM STEW".trim()), Some(item::MUSHROOM_STEW));
        assert_eq!(resolve(&block::CRAFTING_TABLE.to_string()), Some(block::CRAFTING_TABLE));
        assert_eq!(resolve("no_such_thing"), None);
        let c = parse("/give stone 65", resolve).unwrap();
        assert_eq!(c, Command::Give { item: block::STONE, count: 65 });
        let c = parse("/give diamond", resolve).unwrap();
        assert!(matches!(c, Command::Give { count: 1, .. }));
        assert!(parse("/give stone 0", resolve).is_err());
        assert!(parse("/give nothing_burger", resolve).is_err());
        // The clamp keeps a silly count from flooding the world: a stack-of-64 item caps at a
        // full inventory (36*64); an unstackable tool caps at 36 (one per slot).
        let c = parse("/give stone 999999", resolve).unwrap();
        assert_eq!(c, Command::Give { item: block::STONE, count: 36 * 64 });
        let c = parse("/give diamond_pickaxe 999999", resolve).unwrap();
        assert_eq!(c, Command::Give { item: item::DIAMOND_PICKAXE, count: 36 });
        // The name tie-break: "wheat" is the crafting material, not the crop block.
        assert_eq!(resolve("wheat"), Some(item::WHEAT));
    }
}
