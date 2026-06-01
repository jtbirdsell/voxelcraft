//! GPU / rendering subsystem (M33-G1 grouping; `foo.rs + foo/` facade pattern). Submodules are also
//! re-exported at the crate root (see `main.rs`) so existing `crate::camera`-style paths still resolve.

pub(crate) mod camera;
pub(crate) mod capture;
pub(crate) mod device;
// M33-G9b: the DLSS / Frame Generation modules are real only with the `dlss` feature; otherwise a
// zero-cost stub (uninhabited types, constructors return None) compiles in their place so the engine's
// `Option<DlssRender>`/`Option<FrameGen>` plumbing stays valid and `cargo build --no-default-features`
// needs no NVIDIA SDK. The `#[path]` alias keeps the module name `dlss`/`frame_gen` either way.
#[cfg(feature = "dlss")]
pub(crate) mod dlss;
#[cfg(not(feature = "dlss"))]
#[path = "gfx/dlss_stub.rs"]
pub(crate) mod dlss;
#[cfg(feature = "frame-generation")]
pub(crate) mod frame_gen;
#[cfg(not(feature = "frame-generation"))]
#[path = "gfx/frame_gen_stub.rs"]
pub(crate) mod frame_gen;
pub(crate) mod environment;
pub(crate) mod frustum;
pub(crate) mod graph;
pub(crate) mod renderer;
pub(crate) mod rt;
pub(crate) mod rt_probe;
pub(crate) mod rt_spike;
pub(crate) mod texture;
pub(crate) mod viewmodel;
pub(crate) mod voxel_volume;
