# voxgrid

Coordinate and bounding-box math for **voxel grids** and **log-scaled voxel trees**,
built for Godot / Voxelis-style engines.

The crate layers three abstractions on top of [`glam`](https://docs.rs/glam):

- **`Coord`** — a 3D integer coordinate (a thin wrapper over a `glam` integer
  vector) with a large set of arithmetic, comparison, and conversion helpers.
- **`CoordBBox`** — an axis-aligned bounding box built from two `Coord` corners,
  with enclose/intersect/expand/translate operations and coordinate iterators.
- **`NodeDim` / `TreeDim`** — log-scale voxel-tree node dimensions for indexing and
  traversing multi-resolution sparse voxel trees.

The public **`cube`** module gathers the static lookup tables (edges, corners,
normals, neighborhoods, side normals) that voxel and mesh-generation code relies
on.

## Index type

Coordinates default to **`i32`**, matching both Voxelis and Godot. Enable the
`index64` feature to switch the whole crate to `i64`.

## Feature flags

| Feature   | Default | Effect |
|-----------|:-------:|--------|
| *(none)*  |   ✓     | `i32`/`u32` coordinates. |
| `index64` |         | Use `i64`/`u64` (and `f64`/`DVec3`) coordinates instead. |
| `serde`   |         | Derive `Serialize`/`Deserialize` on the public types (and enable `glam/serde`). |
| `bytemuck`|         | Derive `Pod`/`Zeroable` and enable byte-slice conversions (and enable `glam/bytemuck`). |

## Modules

### `Coord`

A newtype over a `glam` integer vector (`IndexVec`). It implements `Deref` /
`DerefMut`, so `.x`, `.y`, `.z` access and `glam` vector math work directly.

```rust
use voxgrid::Coord;

let a = Coord::new(1, 2, 3);
let b = Coord::new(4, 5, 6);

assert_eq!(a + b, Coord::new(5, 7, 9));
assert_eq!(b - a, Coord::new(3, 3, 3));
assert_eq!(-a, Coord::new(-1, -2, -3));
```

Highlights:

- `offset` / `offset_by` / `single_offset` / `single_offset_by` — move a point by
  a per-axis or uniform delta (mutating or returning a new value).
- `min_component` / `max_component` / `component_lt` … — per-component min/max and
  comparisons.
- `min_idx` / `max_idx` — index (0/1/2) of the smallest/largest component.
- `abs`, `sum`, `zyx`, `any_match`, `match_axes`, `unit_clamp`.
- `From`/`Into` conversions to and from the common `glam` integer and
  floating-point vector types (`IVec3`, `UVec3`, `Vec3`, `DVec3`, …), 3-element
  arrays (`[i32; 3]`, `[f32; 3]`, …), and `(x, y, z)` tuples. Conversions from
  floating-point types round half-up.
- Ordering via `Ord`/`PartialOrd` (lexicographic, x then y then z).
- `Display` renders as `(x, y, z)`.
- With the `bytemuck` feature: `bytes_of`, `as_slice`, `as_slice_unsigned`, …
- Implements the `CoordRound` trait (`round_half_up`, `ceil`, `floor`) for the
  `glam` vector types.

### `CoordBBox`

An axis-aligned box defined by inclusive `min` and `max` corners.

```rust
use voxgrid::{Coord, CoordBBox};

let bbox = CoordBBox::create_cube(&Coord::ORIGIN, 4); // 4x4x4 starting at origin
assert_eq!(bbox.volume(), 64);
assert!(bbox.coord_is_inside(&Coord::new(3, 3, 3)));
assert!(!bbox.coord_is_inside(&Coord::new(4, 4, 4)));
```

Highlights:

- `create_cube`, `create_centered_box`, `inf`, `reset`, `reset_with`,
  `reset_to_cube`.
- `volume`, `axis_dims`, `get_center`, `min_extent` / `max_extent`,
  `is_divisible`, `empty` / `has_volume`.
- `coord_is_inside`, `bbox_is_inside`, `has_overlap`, `is_boundary_coord`,
  `is_boundary_bbox`.
- `expand` / `expand_by` / `enclose_point` / `enclose_bbox` / `expand_cube` /
  `intersect` / `translate` / `move_min` / `move_max` / `pad_by`.
- `get_corner_points` — fill a caller-supplied buffer with the 8 corners.
- `local_to_global` / `global_to_local`.
- `range_x` / `range_y` / `range_z` — `RangeInclusive` over each axis.
- `boundary_direction` — for a coordinate on the box's boundary, a per-axis
  `{-1, 0, 1}` vector; `touching_neighbors` — every non-zero combination of that
  direction.
- Bitwise/shift operators: `&`, `|`, `<<`, `>>` (and their `..._assign` forms).
- The `Default` value is the empty box (`min == MAX`, `max == MIN`), handy as an
  enclose-accumulator.

**Iterators** — `BoxIterator<const ZYX_ORDERING>` walks the coordinates the box
spans. `CoordBBox::to_xyz_iter()` yields `XYZIterator` (`x` innermost, cache
friendly for z-major data); `to_zyx_iter()` yields `ZYXIterator` (`x`
outermost). `to_iter()` is an alias for the ZYX iterator.

### `NodeDim` / `TreeDim`

Log-scaled voxel-tree geometry. Sizes are stored as **log2** values, so
parent/child relationships are integer bit shifts instead of divisions.

- `NodeDim` describes one node level: `log_dim`, `sum_child_dims`, `total_dim`,
   `node_level`, `child_length`, `voxel_length`, `coord_dim_mask`, `is_sparse`,
   `is_leaf`. It maps a global coordinate to its owning node
   (`node_origin`/`node_offset`/`same_origin`), reports `voxel_bbox`/`local_bbox`,
  and allocates `make_child_buffer` / `make_voxel_buffer`.
- `TreeDim<const TREE_DEPTH = 8>` chains `TREE_DEPTH` `NodeDim` levels from the
  leaf (index 0) to the root (index `TREE_DEPTH - 1`). Navigation:
  `leaf`, `root`, `child`, `parent`, `node_level_size`; plus
  `child_coords`, `overlapping_children`, `child_touching_neighbors`,
  `enclose_children`, and `child_count_in_bbox`.

```rust
use voxgrid::TreeDim;

// A uniform 2x2x2 octree (child dim 1), 8 levels deep.
let tree = TreeDim::<8>::sparse_octree();
let leaf = tree.leaf();

assert_eq!(leaf.node_level, 1);
assert!(leaf.is_leaf);
assert_eq!(leaf.voxel_length, 2); // a 2x2x2 node
assert_eq!(tree.root().node_level, 8);
```

Sparse trees (built via `NodeDim::sparse` / `TreeDim::sparse_octree`) have no
fixed node bounds — `node_origin`, `voxel_bbox`, and `local_bbox` collapse to the
origin / an infinite box, since sparse nodes are keyed by their child
coordinate.

### `cube`

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
- Enums `Side`, `SideAxis`, `Edge`, `Corner`, `Neighbor`, and the `dir_to_side`
  helper.

## Axis conventions

Corners follow the Godot 4 right-handed convention (documented in `cube.rs`):

```
    6-------7
   /|       /|
  / |      / |  Corners
 2-------3   |
 |   4----|--5
 |/       | /     y z
 |/       |/       |/   Godot 4 axis convention (right handed)
 0-------1       o--x
```

A TODO in `cube.rs` notes a planned switch to the Minecraft naming
(`West/East/North/South/Down/Up`), which `SideAxis` already aliases.

## Building & testing

```bash
cargo build          # default (i32) index
cargo build --features index64
cargo test --all-features
cargo doc --no-deps --all-features
```

## License

Licensed under either of:

- MIT (see [`LICENSE-MIT`](LICENSE-MIT))
- Apache-2.0 (see [`LICENSE-APACHE`](LICENSE-APACHE))

at your option.
