## Reference

`Side`, `Edge`, `Corner`, and `Neighbor` are `#[repr(usize)]` enums whose
variant values match the indexes of their tables, so they derive `IntoPrimitive`
and can index the tables directly. Each also provides infallible `get_*`
accessors and an `all_*()` iterator over the valid variants:

```rust
use voxgrid::cube::Side;
use voxgrid::Coord;

// The variant value of a `Side` is its index in the `SIDE_*` tables.
assert_eq!(Side::East as usize, 0);
assert_eq!(Side::North as usize, 5);

// Infallible accessors read the corresponding table entry.
assert_eq!(Side::East.get_normal(), Coord::new(1, 0, 0));
assert_eq!(Side::Up.get_normal(), Coord::new(0, 1, 0));

// Reverse-lookup the face whose normal matches a direction.
assert_eq!(Side::from_face_direction(Coord::new(0, 0, 1)), Some(Side::North));

// Iterate the valid variants.
assert_eq!(Side::all_sides().count(), 6);
```

### Tables

| Constant | Description |
|----------|-------------|
| `SIDE_COUNT`, `EDGE_COUNT`, `CORNER_COUNT`, `NEIGHBOR_COUNT`, `ORDERED_MOORE_AREA_3D_COUNT` | Lengths of the corresponding tables. |
| `EDGES` | The 12 undirected edges as corner-index pairs (`U8Vec2`). |
| `EDGE_INTERSECTIONS` | The 256-entry marching-cubes edge mask. |
| `CORNER_OFFSETS`, `CORNER_NORMALS` | Corner positions and outward normals (`[Vec3; 8]`). |
| `EDGE_NORMALS` | Outward normals for each edge (`[Vec3; 12]`). |
| `SIDE_NORMALS`, `SIDE_TANGENTS` | Face normals and tangent-frame 4-vectors, per `Side`. |
| `SIDE_CORNERS`, `SIDE_EDGES` | The four `Corner`s/`Edge`s bounding each face, counter-clockwise from outside. |
| `SIDE_NEIGHBORING_DISTANCES` | Distance to a face-adjacent neighbor (`1.0` on a unit grid). |
| `SIDE_TRIANGLES`, `SIDE_TRIANGLES_FLIPPED` | Triangle indices splitting each face along its two diagonals. |
| `OPPOSITE_SIDE` | For each face, the index of the opposite face. |
| `MOORE_NEIGHBORHOOD_3D` | The 26 non-center voxels of a 3×3×3 neighborhood. |
| `MOORE_NEIGHBORHOOD_3D_SHELL_2` | Non-center voxels of a 5×5×5 block (radius 2). |
| `ORDERED_MOORE_AREA_3D` | The full 3×3×3 area (27 voxels; the center sits at index 13). |

`MooreNeighbor` pairs each voxel's integer `n_offset` with its Euclidean
`n_distance`.
