# VoxGrid

VoxGrid is targeted at general-purpose utilities used in sparse voxel data structures. It is not, itself, a voxel data structure.

Instead, it provides common types, traits, and tables that are needed when building voxel worlds, including:

- Coordinate and bounding-box math for **voxel grids** and **log-scaled voxel trees**, based largely on
  the system used by OpenVDB
- _planned_: 
  - Quantized types for consistent hashable float storage
  - SDF interactions, integration with sdfu
  - Integrations with various other libraries

## Coord

`Coord` is based on the OpenVDB types of the same name, implementing a lot of the same core functionality, and providing some new functionality. `Coord` is defined as a thin wrapper around glam IVec3, and provides several conversion/arithmetic implementations specific to a voxel tree/bbox. It also implements conversions between various glam vec types and array/tuple triplets.

There is an `index64` feature which enables `i64`/`u64` with the intent of representing float types with `f128`, however, there is no `f128` glam type, so it currently uses `f64`. This means that float indexes cannot represent the entire integer coordinate range, and floating-point conversions are subject to precision/overflow error for large magnitudes (see `index64`). Long term, a 128-bit float index will be explored for the `index64` path, but it is not a priority right now.

## LocalCoord

`LocalCoord` is an unsigned coordinate (a thin wrapper around a `glam` unsigned vector, `UVec3`/`U64Vec3`) used to address a *local* frame of reference — the child/voxel grid of a `NodeDim`/`NodeLevel`, or a `CoordBBox`. It is a pure index and carries no meaning on its own; the conversion to and from a global [`Coord`] is only well-defined relative to that frame, which is why the conversion lives on `CoordBBox`/`NodeDim` rather than on the type itself. Without it, root/sparse `NodeLevel`s lack the precision to index their entire child/voxel space.

Because it is a pure index, `LocalCoord` does not convert to or from floating-point types.

## CoordBBox

`CoordBBox` is also based on the OpenVDB type of the same name, providing a two-point, axis-aligned bounding box. It provides tools for expanding, enclosing, checking overlaps, and converting between global and bbox local coordinates via `local_to_global`/`global_to_local`, which return a `Result` and error when the coordinate lies outside the box's range.

## TreeDim

It implements the core indexing strategy of OpenVDB as a tiered list of `NodeDim`s. Together they allow converting indexes between tier levels, provide child/voxel arithmetic at the node level, and offer additional functionality. 

OpenVDB's implementation (const expr arguments through template parameters) in Rust is at best extremely cumbersome, and at worst impossible in stable Rust. This struct is a compromise to pull it out of generics while keeping the logic const-compatible. 

## Cube

Static lookup tables over the 8 corners, 12 edges, 6 faces, and 26 face/diagonal
neighbors of a unit cube, built with `lazy_static` and exposed as public
constants. The corner/edge/face numbering conventions and axis orientation are
documented in `src/cube.rs`.

Key tables:

- `EDGES` — the 12 undirected edges as vertex-index pairs.
- `EDGE_INTERSECTIONS` — the classic 256-entry marching-cubes edge mask.
- `CORNER_OFFSETS`, `CORNER_NORMALS`, `EDGE_NORMALS`.
- `SIDE_NORMALS`, `SIDE_TANGENTS`, `SIDE_CORNERS`, `SIDE_EDGES`,
  `SIDE_NEIGHBORING_DISTANCES`.
- `MOORE_NEIGHBORHOOD_3D` and `MOORE_NEIGHBORHOOD_3D_SHELL_2`.
- Enums `Side`, `CardinalSide`, `Edge`, `Corner`, `Neighbor`.

## Index type

The core library is built around i32 indexes (i32, IVec3, UVec3, Vec3) by default as this is the limit on several
game engines/voxel libraries. i64 indexes (i64, I64Vec3, U64Vec3, DVec3) can be enabled with the `index64` feature.

## Feature flags

| Feature   | Default | Effect |
|-----------|:-------:|--------|
| *(none)*  |   ✓     | `i32`/`u32`/`f64` coordinates. |
| `index64` |         | Use `i64`/`u64`/`f64` coordinates instead. |
| `serde`   |         | Derive `Serialize`/`Deserialize` on the public types (and enable `glam/serde` and `half/serde`). |
| `bytemuck`|         | Derive `Pod`/`Zeroable` and enable byte-slice conversions (and enable `glam/bytemuck`). |

## Examples

### `Coord` & `LocalCoord`

```rust
use voxgrid::{Coord, LocalCoord};

// Coord is a signed 3D integer coordinate (i32 by default, i64 under the
// `index64` feature) that derefs to a glam integer vector.
let a = Coord::new(1, 2, 3);
let b = Coord::new(4, 5, 6);
assert_eq!(a + b, Coord::new(5, 7, 9));
assert_eq!(b - a, Coord::new(3, 3, 3));
assert_eq!(-a, Coord::new(-1, -2, -3));

// Floats convert through round_half_up, which snaps .5 away from zero so the
// grid stays consistent around the origin.
assert_eq!(Coord::from([0.5_f64, 0.5, 0.5]), Coord::new(1, 1, 1));
assert_eq!(Coord::from([-0.5_f64, -0.5, -0.5]), Coord::ORIGIN);

// LocalCoord is the unsigned index version. It carries no meaning on its own; it
// only addresses a local frame (a CoordBBox or NodeLevel), so it never converts
// to or from floating-point types.
let local = LocalCoord::new(1, 2, 3).offset_by(4, 5, 6);
assert_eq!(local, LocalCoord::new(5, 7, 9));
```

### `CoordBBox`

```rust
use voxgrid::{Coord, CoordBBox, LocalCoord};

// A two-corner, inclusive, axis-aligned bounding box.
let bbox = CoordBBox::new(Coord::new(-10, -5, 0), Coord::new(10, 5, 10));
assert_eq!(bbox.volume(), 21 * 11 * 11);
assert!(bbox.coord_is_inside(&Coord::new(0, 0, 0)));

// Enclosing grows the box to include a point.
let mut grown = CoordBBox::new(Coord::ORIGIN, Coord::new(4, 4, 4));
grown.enclose_point(&Coord::new(-3, -3, -3));
assert_eq!(grown.min, Coord::new(-3, -3, -3));

// global <-> local conversion is relative to the box's origin; it errors when
// the coordinate lies outside the box.
let local = bbox.global_to_local(&Coord::new(0, 0, 0)).unwrap();
assert_eq!(local, LocalCoord::new(10, 5, 0));
assert_eq!(bbox.local_to_global(&LocalCoord::ORIGIN).unwrap(), bbox.min);
assert!(bbox.global_to_local(&Coord::new(11, 0, 0)).is_err());
```

### `TreeDim` & `NodeLevel`

```rust
use voxgrid::{TreeDim, UIndex};

// TreeDim::octree() builds a uniform 2x2x2 sparse tree over MAX_DEPTH levels.
let tree: TreeDim = TreeDim::octree();

// The root is sparse, so its voxel length is the entire index space.
assert!(tree.root().is_sparse);
assert_eq!(tree.root().voxel_length, UIndex::MAX as usize);

// first_child is the top non-root level; its voxel length is 2^7 = 128.
assert_eq!(tree.first_child().voxel_length, 128);

// A NodeLevel derefs to its NodeDim and can walk the tree down to the leaf.
let mut node = tree.root();
while let Some(child) = node.child() {
    node = child;
}
assert_eq!(node, tree.leaf());
assert!(node.is_leaf);
assert_eq!(node.voxel_length, 2);

// A custom tree is built from each level's log2 child size.
let custom = TreeDim::<4>::new(&[2, 3, 4]);
assert_eq!(custom.leaf().voxel_length, 4); // 2^2 leaf
assert_eq!(custom.root().node_level, 3);
```

## Building & testing

```bash
cargo build          # default (i32) index
cargo build --features index64
cargo test --all-features
cargo doc --no-deps --all-features
```

# Note:

While some features are mostly feature complete (Coord, CoordBBox), this is actively being pulled from a larger project and refactored to work as a standalone library. The api is being reworked/cleaned up in the process.
