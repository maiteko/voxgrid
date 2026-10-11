#![doc = include_str!("../README.md")]

mod coord;
mod coord_bbox;
pub mod cube;
#[cfg(feature = "quantized")]
pub mod quantized;
mod tree_dim;
#[cfg(feature = "voxgrid")]
mod vox_grid;

pub use coord::*;
pub use coord_bbox::CoordBBox;
pub use tree_dim::*;
#[cfg(feature = "voxgrid")]
pub use vox_grid::*;

mod config_index {
    /// 32-bit index configuration used by default, because Voxelis and Godot both
    /// use `i32` as their base integer type.
    use glam::*;
    pub type Index = i32;
    pub type UIndex = u32;
    pub type IndexVec = IVec3;
    pub type UIndexVec = UVec3;
    pub type FIndex = f64;
    pub type FIndexVec = DVec3;
}

pub use config_index::*;
