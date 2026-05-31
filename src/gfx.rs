//! GPU / rendering subsystem (M33-G1 grouping; `foo.rs + foo/` facade pattern). Submodules are also
//! re-exported at the crate root (see `main.rs`) so existing `crate::camera`-style paths still resolve.

pub(crate) mod camera;
pub(crate) mod capture;
pub(crate) mod device;
pub(crate) mod environment;
pub(crate) mod frustum;
pub(crate) mod renderer;
pub(crate) mod rt_probe;
pub(crate) mod rt_spike;
pub(crate) mod texture;
pub(crate) mod voxel_volume;
