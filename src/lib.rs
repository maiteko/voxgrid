mod coord;
mod coord_bbox;
pub mod cube;
mod log_dim;

pub use coord::*;
pub use coord_bbox::*;
pub use log_dim::*;

#[cfg(feature = "index64")]
mod config_index {
    /// i64 type used for coordinates.
    use glam::*;
    pub type Index = i64;
    pub type UIndex = u64;
    pub type IndexVec = I64Vec3;
    pub type UIndexVec = U64Vec3;
    pub type FIndex = f64;
    pub type FIndexVec = DVec3;
}

#[cfg(not(feature = "index64"))]
mod config_index {
    /// i32 type used for coordinates by default
    /// this is because Voxelis and Godot both use i32
    /// as the base integer
    use glam::*;
    pub type Index = i32;
    pub type UIndex = u32;
    pub type IndexVec = IVec3;
    pub type UIndexVec = UVec3;
    pub type FIndex = f32;
    pub type FIndexVec = Vec3;
}

pub use config_index::*;
