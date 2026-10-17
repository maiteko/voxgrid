# VoxGrid

Coordinate and bounding-box math for sparse voxel grids and log-scaled voxel
trees, following the [OpenVDB](https://github.com/AcademySoftwareFoundation/OpenVDB)
model. VoxGrid is a collection of utility types and lookup tables — it is **not**
itself a voxel data structure.

## Types

- [`Coord`] — a signed 3D integer coordinate (a `glam` newtype) with the
  arithmetic, comparison, and `From`/operator conversions a voxel tree needs.
- [`LocalCoord`] — its unsigned counterpart: a pure index into a local frame
  (a [`NodeLevel`] child/voxel grid or a [`CoordBBox`]). It carries no meaning
  on its own and never converts to or from floating-point types.
- [`CoordBBox`] — an inclusive, axis-aligned bounding box over [`Coord`], with
  enclosing/intersection/iteration and global↔local conversion.
- [`NodeDim`] / [`TreeDim`] — the per-level geometry and the level chain of a
  multi-resolution sparse voxel tree: parent/child navigation, child-coordinate
  enumeration, and overlap queries.
- [`cube`] — static lookup tables over a unit cube's 8 corners, 12 edges, 6
  faces, and 3×3×3 Moore neighborhood, for voxel and mesh generation.
- [`vox_grid`] — read/write/background/slice traits for sparse and local grids:
  [`VoxGrid`] (read), [`VoxGridMut`] (write), [`VoxGridBackground`] (background
  value), and [`VoxGridSlice`] / [`VoxGridSliceMut`] (shared views).

## Index type

Coordinates are in `i32`/`u32`/`f64`, matching Voxelis and Godot. This is a hard limit outside
using 128 bit types for float/length. An `f32` doesn't have the precision to fully index the i32 coordinates. Similarly, `i64`/`u64` can't be indexed properly at the root level, because a `usize` cannot contain the full length (off by one). 


## Feature flags

| Feature     | Default | Effect |
|-------------|:-------:|--------|
| *(none)*    |   ✓     | N/A |
| `serde`     |         | Derive `Serialize`/`Deserialize` on the public types. |
| `bytemuck`   |          | Derive `Pod`/`Zeroable` and enable byte-slice conversions. |

## Building & testing

```bash
cargo build
cargo test --all-features
cargo doc --no-deps --all-features
```

## Status

This crate is being pulled out of a larger project and refactored to stand
alone. `Coord` and `CoordBBox` are close to feature complete; the rest of the
API is still being cleaned up.

Planned:

- SDF interactions and integration with `sdfu`.
- Integrations with various other libraries.
