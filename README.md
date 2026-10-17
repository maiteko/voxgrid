# VoxGrid

Coordinate and bounding-box math for sparse voxel grids and log-scaled voxel
trees, following the [OpenVDB](https://github.com/AcademySoftwareFoundation/OpenVDB)
model. VoxGrid is a collection of utility types and lookup tables — it is **not**
itself a voxel data structure.

While initially based on OpenVDB's log indexing, it differs in a few key ways:

- Local indexing has a distinct type ([`LocalCoord`]) based on `u32`, this allows local indexing
  to work across the entire `Coord/i32` range, including at the sparse root node. At root: 
  `LocalCoord::ORIGIN/MIN == Coord::MIN`, `LocalCoord::MAX == Coord::MAX`, and
  `(LocalCoord::MAX / 2 + 1) == Coord::ORIGIN`.
- The [`LocalCoord`] also allows for local indexing into a [`CoordBBox`] of an arbitrary size,
  supporting up to `CoordBBox::new(Coord::MIN, Coord::MAX)` with `bbox.local_to_global/global_to_local`
- Indexing for specific [`NodeLevels`](NodeLevel) are defined through [`TreeDim`], pulling it out
  of specific node implementations, and avoiding some of the `const generic` limitations
  of stable rust. Where possible [`TreeDim`] and [`NodeLevel`] are const compatible to provide similar benefits
  to OpenVDB's Template implementation.



## Types

- [`Coord`] — a signed 3D integer coordinate (a `glam` newtype) with the
  arithmetic, comparison, and `From`/operator conversions a voxel tree needs.
- [`LocalCoord`] — [`Coord's`](Coord) unsigned counterpart: a pure index into a local frame
  (a [`NodeLevel`] child/voxel grid or a [`CoordBBox`]). It carries no meaning
  without a frame of reference (`CoordBBox` or `NodeLevel`) and never converts 
  directly to or from floating-point types.
- [`CoordBBox`] — an inclusive, axis-aligned bounding box over [`Coord`], with
  enclosing/intersection/iteration and global↔local conversion.
- [`NodeLevel`] / [`TreeDim`] — the per-level geometry and the level chain of a
  multi-resolution sparse voxel tree: parent/child navigation, child-coordinate
  enumeration, and overlap queries.
- [`cube`] — static lookup tables over a unit cube's 8 corners, 12 edges, 6
  faces, and 3×3×3 Moore neighborhood, for voxel and mesh generation.
- [`vox_grid`] — read/write/background/slice traits for sparse and local grids:
  [`VoxGrid`] (read), [`VoxGridMut`] (write), [`VoxGridBackground`] (background
  value), and [`VoxGridSlice`] / [`VoxGridSliceMut`] (shared views).

## Index type

- Index Coordinates ([`Coord`], [`LocalCoord`]) are in `i32`/`u32`.
- Sizes are given as [`glam::U64Vec3`] so they can support the size
  of the `i32` range (`u32::MAX + 1`)
- Float types are `f64` to provide the precision to access the entire
  i32 range.

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

This crate is being pulled from a larger project, and refactored to stand alone.

The crate is mostly stable, with ongoing fixes to documentation, test completeness,
bugfixes, and api name changes for the sake of clarity.