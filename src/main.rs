//! Voxelcraft: an interactive voxel sandbox.
//!
//! Controls:
//!   WASD move · mouse look · Space up/jump · Left-Shift down · Left-Ctrl sprint/boost
//!   F toggle fly/walk · Left-click break · Right-click place · 1-9 / scroll select block · Esc quit
//!
//! Set VOXELCRAFT_SHOT=path.png to render one frame offscreen (headless verification).

mod app;
mod game;
mod gameplay;
mod gfx;
mod mesh;
mod persistence;
mod ui;
mod world;

// M33-G1 grouping shim: re-export each grouped submodule at the crate root so the existing
// `crate::<module>` call sites keep resolving unchanged. New code can use the grouped paths
// directly (e.g. `crate::gfx::camera`, `crate::world::worldgen`, `crate::gameplay::player`).
pub(crate) use gameplay::{container, crafting, entity, item, player, raycast, smelting};
pub(crate) use gfx::{
    camera, capture, device as gpu, dlss, environment, frame_gen, frustum, renderer, rt_probe,
    rt_spike, texture, voxel_volume,
};
pub(crate) use mesh::mesher;
pub(crate) use ui::{font, overlay};
pub(crate) use world::{block, light, worker, worldgen};

use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,voxelcraft=info"),
    )
    .init();

    // Headless persistence round-trip check (no window needed).
    if std::env::var("VOXELCRAFT_PERSIST_TEST").is_ok() {
        app::persist_selftest();
        return;
    }

    // Hardware-ray-tracing capability probe (no window needed).
    if std::env::var("VOXELCRAFT_RT_PROBE").is_ok() {
        rt_probe::probe();
        return;
    }

    // Hardware-ray-tracing spike: render one frame on the RT cores (no window needed).
    if std::env::var("VOXELCRAFT_RT_SPIKE").is_ok() {
        rt_spike::run();
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = app::App::new();
    event_loop.run_app(&mut app).unwrap();
}
