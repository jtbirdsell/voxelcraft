//! Voxelcraft — M4: an interactive voxel sandbox.
//!
//! Controls:
//!   WASD move · mouse look · Space up/jump · Left-Shift down · Left-Ctrl sprint/boost
//!   F toggle fly/walk · Left-click break · Right-click place · 1-9 / scroll select block · Esc quit
//!
//! Set VOXELCRAFT_SHOT=path.png to render one frame offscreen (headless verification).

mod block;
mod camera;
mod capture;
mod crafting;
mod entity;
mod environment;
mod font;
mod frustum;
mod game;
mod gpu;
mod item;
mod light;
mod mesher;
mod overlay;
mod persistence;
mod player;
mod raycast;
mod renderer;
mod rt_probe;
mod rt_spike;
mod smelting;
mod texture;
mod voxel_volume;
mod worker;
mod world;
mod worldgen;
mod app;

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
