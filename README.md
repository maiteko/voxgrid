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

## CoordBBox

It is also based on the OpenVDB type of the same name, providing a two-point, axis-aligned bounding box. It provides tools for expanding, enclosing, checking overlaps, and converting between global and bbox local coordinates.

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
| *(none)*  |   ✓     | `i32`/`u32` coordinates. |
| `index64` |         | Use `i64`/`u64` (and `f64`/`DVec3`) coordinates instead. |
| `serde`   |         | Derive `Serialize`/`Deserialize` on the public types (and enable `glam/serde` and `half/serde`). |
| `bytemuck`|         | Derive `Pod`/`Zeroable` and enable byte-slice conversions (and enable `glam/bytemuck`). |

## Examples

```rust
use voxgrid::*;

let a = Coord::new(1, 2, 3);
let b = Coord::new(4, 5, 6);

assert_eq!(a + b, Coord::new(5, 7, 9));
assert_eq!(b - a, Coord::new(3, 3, 3));
assert_eq!(-a, Coord::new(-1, -2, -3));

let mut bbox = CoordBBox::new(a, b);

bbox.enclose_point(&Coord::ORIGIN);
assert_eq!(bbox, CoordBBox::new(Coord::ORIGIN, b));


const TREE_DIM: TreeDim<8> = TreeDim::octree();
// Root's voxel length is the entire index space, positive and negative
assert_eq!(TREE_DIM.root().voxel_length, UIndex::MAX as usize);
// octree sets all child nodes up to 1, with tree_depth == MAX_DEPTH
// This means the first child is an octree with axis voxel length of 2^7
assert_eq!(TREE_DIM.child(TREE_DIM.root()).voxel_length, 128);
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
