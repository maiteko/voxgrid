# VoxGrid

Coordinate and bounding-box math for sparse voxel grids and log-scaled voxel
trees, following the [OpenVDB](https://github.com/AcademySoftwareFoundation/OpenVDB)
model. VoxGrid is a collection of utility types and lookup tables — it is **not**
itself a voxel data structure.

While initially based on OpenVDB's log indexing, it differs in a couple of key ways:

- Local indexing has a distinct type (`LocalCoord`) based on `u32`, this allows local indexing
  to work across the entire `Coord/i32` range. At the sparse root node 
  `LocalCoord::ORIGIN/MIN == Coord::MIN`, `LocalCoord::MAX == Coord::MAX`, and
  `(LocalCoord::MAX / 2 + 1) == Coord::ORIGIN`.
- Indexing for specific NodeLevels are defined through `TreeDim`, pulling it out
  of specific node implementations, and avoiding some of the const generic limitations
  of stable rust. Where possible `TreeDim` is const compatible to provide the same benefits
  as OpenVDB's Template implementation.
- Arbitrary local to global indexing through `CoordBBox`, allowing `TreeDim` to be bypassed for
  indexing several cases. 


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

Index Coordinates are in `i32`/`u32`. In order to have the precision to cover the entire 
32-bit range/size Vertex Coordinates are `f64` and size values are `u64`.


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

