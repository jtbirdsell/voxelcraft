//! World data + generation subsystem. `chunk` holds the former top-level `world.rs` and is
//! re-exported flat so `crate::world::Chunk` (and friends) keep resolving after the M33-G1 grouping.

pub(crate) mod block;
pub(crate) mod chunk;
pub(crate) mod light;
pub(crate) mod worker;
pub(crate) mod worldgen;

pub(crate) use chunk::*;
