//! # voxgrid
//!
//! Coordinate and bounding-box math for voxel grids and log-scaled voxel trees,
//! aimed at Godot / Voxelis-style engines.
//!
//! The crate provides three layers of abstractions:
//!
//! - [`Coord`] — a 3D integer coordinate (a thin wrapper around a `glam` integer
//!   vector) with a rich set of arithmetic, comparison, and conversion helpers.
//! - [`CoordBBox`] — an axis-aligned bounding box built from two `Coord` corners,
//!   with enclosing/intersect/expand operations and coordinate iterators.
//! - [`NodeDim`] / [`TreeDim`] — log-scale voxel-tree node dimensions used to index
//!   and traverse multi-resolution sparse voxel trees.
//!
//! The [`cube`] module gathers the static lookup tables (edges, corners, normals,
//! neighborhoods, side normals) that voxel and mesh-generation code relies on.
//!
//! ## Index type
//!
//! By default coordinates use `i32`, matching both Voxelis and Godot. Enable the
//! `index64` feature to switch the whole crate to `i64`. See the crate features in
//! `Cargo.toml`.
//!
//! ## Feature flags
//!
//! - `index64` — use `i64`/`u64` coordinates instead of `i32`/`u32`.
//! - `serde` — derive `Serialize`/`Deserialize` for the public types.
//! - `bytemuck` — derive `Pod`/`Zeroable` and enable byte-slice conversions.

mod coord;
mod coord_bbox;
pub mod cube;
mod tree_dim;
mod vox_grid;

pub use coord::*;
pub use coord_bbox::*;
pub use tree_dim::*;
pub use vox_grid::*;

#[cfg(feature = "index64")]
mod config_index {
    /// 64-bit index configuration, enabled by the `index64` feature.
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
    /// 32-bit index configuration used by default, because Voxelis and Godot both
    /// use `i32` as their base integer type.
    use glam::*;
    pub type Index = i32;
    pub type UIndex = u32;
    pub type IndexVec = IVec3;
    pub type UIndexVec = UVec3;
    pub type FIndex = f32;
    pub type FIndexVec = Vec3;
}

pub use config_index::*;
