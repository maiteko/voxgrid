//! Tables in the cube module all respect the following conventions.
//!
//! Many of these tables are refactored from [godot_voxel's](https://github.com/Zylann/godot_voxel) cube_tables constants
//! with the following caveats:
//!
//! 1. reordered to match the EDGES generation below, making it consistent with the
//!    EDGE_INTERSECTION conventions
//! 2. reordered to a right handed system (+z = north)
//! 3. using cardinal direction naming for sides
//!
//! EDGES and EDGE_INTERSECTIONS generation adapted from mikolalysenko's
//! [surfacenets.js](https://github.com/mikolalysenko/mikolalysenko.github.com/blob/master/Isosurface/js/surfacenets.js)
//! example.
//!
//! Corner numbering (index = x + 2*y + 4*z):
//!
//! <pre>
//!    6-------7
//!   /|      /|
//!  / |     / |  Corners
//! 2-------3  |
//! |  4----|--5
//! | /     | /     y z
//! |/      |/      |/   coordinates assume right handed conventions
//! 0-------1       o--x
//! </pre>
//!
//! Sides are ordered according to the Side enum (`East`, `West`, `Down`, `Up`,
//! `South`, `North`).
//!
//! <pre>
//!     o---11----o
//!    /|        /|
//!   6 9       7 10
//!  /  |      /  |
//! o----5----o   |
//! |   o---8-|---o
//! 1  /      3  /
//! | 2       | 4
//! |/        |/
//! o----0----o
//! </pre>

use crate::Coord;
use glam::{U8Vec2, Vec3};
use lazy_static::lazy_static;
use num_enum::IntoPrimitive;
use std::{panic, slice::Iter};

const SQRT_2: f64 = 1.4142135;
const SQRT_3: f64 = 1.7320508;

/// Number of `Side` variants; also the length of the `SIDE_*` tables.
pub const SIDE_COUNT: usize = 6;

/// Number of `Edge` variants; also the length of the `EDGES`/`EDGE_NORMALS`
/// tables.
pub const EDGE_COUNT: usize = 12;

/// Number of `Corner` variants; also the length of the `CORNER_*` tables.
pub const CORNER_COUNT: usize = 8;

/// Number of `Neighbor` variants; also the length of `MOORE_NEIGHBORHOOD_3D`.
pub const NEIGHBOR_COUNT: usize = 26;

/// Number of voxels in the full 3x3x3 Moore area, including the center; the
/// length of `ORDERED_MOORE_AREA_3D`.
pub const ORDERED_MOORE_AREA_3D_COUNT: usize = 27;

/// The six faces of a cube. The variants index the `SIDE_NORMALS`,
/// `SIDE_TANGENTS`, `SIDE_CORNERS`, and `SIDE_EDGES` tables (see `SIDE_COUNT`).
///
/// Variants are ordered as in `Cube::SideAxis` (positive-x first), and use the
/// Godot 4 (right-handed) axis naming:
/// `East` = +x, `West` = -x, `Down` = -y, `Up` = +y, `South` = -z, `North` = +z.
#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, IntoPrimitive)]
pub enum Side {
    East = 0,
    West,
    Down,
    Up,
    South,
    North,
}

const VALID_SIDES: [Side; SIDE_COUNT] = [
    Side::East,
    Side::West,
    Side::Down,
    Side::Up,
    Side::South,
    Side::North,
];

impl Side {
    /// Returns the face normal for this side.
    pub fn get_normal(&self) -> Coord {
        SIDE_NORMALS[*self as usize]
    }

    /// Returns the tangent-frame 4-vector for this side.
    pub fn get_tangent(&self) -> [f32; 4] {
        SIDE_TANGENTS[*self as usize]
    }

    /// Returns the four `Corner`s bounding this side.
    pub fn get_corners(&self) -> [Corner; 4] {
        SIDE_CORNERS[*self as usize]
    }

    /// Returns the four `Edge`s bounding this side.
    pub fn get_edges(&self) -> [Edge; 4] {
        SIDE_EDGES[*self as usize]
    }

    /// Returns the distance to a face-adjacent neighbor for this side.
    pub fn get_neighboring_distance(&self) -> f32 {
        SIDE_NEIGHBORING_DISTANCES[*self as usize]
    }

    pub fn all_sides() -> Iter<'static, Side> {
        VALID_SIDES.iter()
    }

    /// Return the `Side` whose face normal matches the unit-clamped direction
    /// `d`, or `None` if `d` is not one of the six axis directions.
    pub fn from_face_direction(d: Coord) -> Option<Side> {
        let d = d.unit_clamp();
        // Use the absolute sum so negative-axis directions (West, Down, South)
        // are accepted: `sum` on a coordinate with a negative component would
        // otherwise wrap to a large unsigned value.
        if d.abs().sum() != 1 {
            return None;
        }

        for side in Self::all_sides() {
            if side.get_normal() == d {
                return Some(*side);
            }
        }

        panic!("face direction {} did not match any sides", d);
    }
}

/// <pre>
///     o---11----o
///    /|        /|
///   6 9       7 10   Face Edges
///  /  |      /  |
/// o----5----o   |
/// |   o---8-|---o
/// 1  /      3  /
/// | 2       | 4
/// |/        |/
/// o----0----o
/// </pre>
/// The twelve edges of a cube, indexed into `EDGES` and `EDGE_NORMALS`.
///
/// Each edge is named after its two *constant* axes, listed in `z`, `x`, `y`
/// order (skipping the axis the edge runs along), using the axis names
/// `North` = +z, `South` = -z, `West` = -x, `East` = +x, `Down` = -y, `Up` = +y.
/// The variant order matches the `EDGES` table, where edge `i` connects corner
/// `i ^ (1 << j)` for the generated iteration order.
#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, IntoPrimitive)]
pub enum Edge {
    SouthDown = 0,
    SouthWest,
    WestDown,
    SouthEast,
    EastDown,
    SouthUp,
    WestUp,
    EastUp,
    NorthDown,
    NorthWest,
    NorthEast,
    NorthUp,
}

const VALID_EDGES: [Edge; EDGE_COUNT] = [
    Edge::SouthDown,
    Edge::SouthWest,
    Edge::WestDown,
    Edge::SouthEast,
    Edge::EastDown,
    Edge::SouthUp,
    Edge::WestUp,
    Edge::EastUp,
    Edge::NorthDown,
    Edge::NorthWest,
    Edge::NorthEast,
    Edge::NorthUp,
];

impl Edge {
    /// Returns the edge (a pair of corner indices) for this edge.
    pub fn get_edge(&self) -> U8Vec2 {
        EDGES[*self as usize]
    }

    /// Returns the outward-facing normal for this edge.
    pub fn get_normal(&self) -> Vec3 {
        EDGE_NORMALS[*self as usize]
    }

    pub fn all_edges() -> Iter<'static, Edge> {
        VALID_EDGES.iter()
    }
}

/// The eight corners of a cube, indexed into `CORNER_OFFSETS` and
/// `CORNER_NORMALS`.
///
/// A corner index is `x + 2*y + 4*z` (x, y, z each 0 or 1), the same bit
/// ordering used by `EDGES` and `CORNER_OFFSETS`. Each corner is named after its
/// three axes in `z`, `x`, `y` order, using the axis names `North` = +z,
/// `South` = -z, `West` = -x, `East` = +x, `Down` = -y, `Up` = +y
/// (Godot 4 right-handed, so `North` is towards +z):
///
/// <pre>
/// 0: SouthWestDown   (0,0,0)
/// 1: SouthEastDown   (1,0,0)
/// 2: SouthWestUp     (0,1,0)
/// 3: SouthEastUp     (1,1,0)
/// 4: NorthWestDown   (0,0,1)
/// 5: NorthEastDown   (1,0,1)
/// 6: NorthWestUp     (0,1,1)
/// 7: NorthEastUp     (1,1,1)
/// </pre>
#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, IntoPrimitive)]
pub enum Corner {
    SouthWestDown = 0,
    SouthEastDown,
    SouthWestUp,
    SouthEastUp,
    NorthWestDown,
    NorthEastDown,
    NorthWestUp,
    NorthEastUp,
}

const VALID_CORNERS: [Corner; CORNER_COUNT] = [
    Corner::SouthWestDown,
    Corner::SouthEastDown,
    Corner::SouthWestUp,
    Corner::SouthEastUp,
    Corner::NorthWestDown,
    Corner::NorthEastDown,
    Corner::NorthWestUp,
    Corner::NorthEastUp,
];

impl Corner {
    /// Returns the corner's local-space offset.
    pub fn get_offset(&self) -> Vec3 {
        CORNER_OFFSETS[*self as usize]
    }

    /// Returns the outward-facing normal for this corner.
    pub fn get_normal(&self) -> Vec3 {
        CORNER_NORMALS[*self as usize]
    }

    pub fn all_corners() -> Iter<'static, Corner> {
        VALID_CORNERS.iter()
    }
}

/// The 26 face/edge/corner-adjacent voxels of a 3x3x3 Moore neighborhood,
/// ordered by z (South to North), then y (Down to Up), then x (West to East).
/// The center is excluded.
#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, IntoPrimitive)]
pub enum Neighbor {
    /// (-1, -1, -1)
    SouthWestDown = 0,
    /// (0, -1, -1)
    SouthDown,
    /// (1, -1, -1)
    SouthEastDown,
    /// (-1, 0, -1)
    SouthWest,
    /// (0, 0, -1)
    South,
    /// (1, 0, -1)
    SouthEast,
    /// (-1, 1, -1)
    SouthWestUp,
    /// (0, 1, -1)
    SouthUp,
    /// (1, 1, -1)
    SouthEastUp,
    /// (-1, -1, 0)
    WestDown,
    /// (0, -1, 0)
    Down,
    /// (1, -1, 0)
    EastDown,
    /// (-1, 0, 0)
    West,
    // Center (0,0,0)
    /// (1, 0, 0)
    East,
    /// (-1, 1, 0)
    WestUp,
    /// (0, 1, 0)
    Up,
    /// (1, 1, 0)
    EastUp,
    /// (-1, -1, 1)
    NorthWestDown,
    /// (0, -1, 1)
    NorthDown,
    /// (1, -1, 1)
    NorthEastDown,
    /// (-1, 0, 1)
    NorthWest,
    /// (0, 0, 1)
    North,
    /// (1, 0, 1)
    NorthEast,
    /// (-1, 1, 1)
    NorthWestUp,
    /// (0, 1, 1)
    NorthUp,
    /// (1, 1, 1)
    NorthEastUp,
}

const VALID_NEIGHBORS: [Neighbor; NEIGHBOR_COUNT] = [
    Neighbor::SouthWestDown,
    Neighbor::SouthDown,
    Neighbor::SouthEastDown,
    Neighbor::SouthWest,
    Neighbor::South,
    Neighbor::SouthEast,
    Neighbor::SouthWestUp,
    Neighbor::SouthUp,
    Neighbor::SouthEastUp,
    Neighbor::WestDown,
    Neighbor::Down,
    Neighbor::EastDown,
    Neighbor::West,
    Neighbor::East,
    Neighbor::WestUp,
    Neighbor::Up,
    Neighbor::EastUp,
    Neighbor::NorthWestDown,
    Neighbor::NorthDown,
    Neighbor::NorthEastDown,
    Neighbor::NorthWest,
    Neighbor::North,
    Neighbor::NorthEast,
    Neighbor::NorthWestUp,
    Neighbor::NorthUp,
    Neighbor::NorthEastUp,
];

impl Neighbor {
    /// Returns the Moore-neighborhood entry for this neighbor.
    pub fn get_neighbor(&self) -> MooreNeighbor {
        MOORE_NEIGHBORHOOD_3D[*self as usize]
    }

    pub fn all_neighbors() -> Iter<'static, Neighbor> {
        VALID_NEIGHBORS.iter()
    }
}

/// A neighbor voxel in a Moore neighborhood: its integer `n_offset` from the
/// origin and its Euclidean `n_distance` to that offset.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct MooreNeighbor {
    pub n_offset: Coord,
    pub n_distance: f64,
}

lazy_static! {
   /// list of the 12 undirected edges of a unit cube as pairs of vertex indices. Each U8Vec2
   /// (a, b) is an edge between vertex a and vertex b. Vertex numbering follows the convention:
   /// index = x + 2*y + 4*z (Z outermost, Y middle, X innermost).
   ///
   /// The table is built by iterating i=0..7, j=0..2 and connecting corner `i` with
   /// `i^(1<<j)` when they differ. Edges only appear where one endpoint has the bit set.
   ///
   /// Corner Diagram
   ///
   /// <pre>
   ///      z
   ///     /
   ///    6----7
   ///   /|   /|
   ///  2----3 |
   ///  | 4--|-5
   ///  |/   |/
   ///  0----1   --> x
   ///  y (up)
   /// </pre>
   ///
   /// Edges by iteration order (i^(1<<j) convention):
   /// 0: (0,1)    1: (0,2)    2: (0,4)
   /// 3: (1,3)    4: (1,5)
   /// 5: (2,3)    6: (2,6)
   /// 7: (3,7)
   /// 8: (4,5)    9: (4,6)
   /// 10:(5,7)    11:(6,7)
   ///
    pub static ref EDGES: [U8Vec2; EDGE_COUNT] = {
        let mut cube_edges = Vec::<U8Vec2>::with_capacity(EDGE_COUNT);

        for i in 0..8 {
            for j in 0..3 {
                let p = i^(1<<j);
                if i <= p {
                    cube_edges.push(U8Vec2 { x: i, y: p });
                }
            }
        }

        cube_edges.try_into().unwrap()
    };

    /// lookup table mapping each cube vertex inside/outside configuration
    /// (8-bit index, 0..255) to a 12-bit edge-intersection bitmask (u16).
    ///
    /// - Input index i (0..255): bit v of i is 1 if vertex v is "inside" (e.g., value > isolevel),
    ///   0 if "outside".
    /// - Output value: a 12-bit mask where bit j is set iff edge j (from CUBE_EDGES[j]) has
    ///   endpoints with different signs (one inside, one outside) and therefore is intersected
    ///   by the isosurface.
    ///
    /// Example:
    /// - i = 0  (0000_0000) -> 0b0000_0000_0000 (no edges)
    /// - i = 255(1111_1111) -> 0b1111_1111_1111 (all edges)
    ///
    ///
    ///     o---11----o
    ///    /|        /|
    ///   6 9       7 10   Face Edges
    ///  /  |      /  |
    /// o----5----o   |
    /// |   o---8-|---o
    /// 1  /      3  /
    /// | 2       | 4
    /// |/        |/
    /// o----0----o
    ///
    /// This table lets algorithms quickly determine which edges of a cube are crossed
    /// by the isosurface without recomputing per-edge vertex comparisons.
    ///
    pub static ref EDGE_INTERSECTIONS: [u16;256] = {
        let mut edge_table: [u16;256] = [0;256];

        for i in 0..256 {
            let mut edge_mask = 0;
            for j in 0..12 {
                let a = (i & (1<<EDGES[j].x)) != 0;
                let b = (i & (1<<EDGES[j].y)) != 0;

                edge_mask |= if a != b {
                    1 << j
                } else {
                    0
                }
            }
            edge_table[i] = edge_mask;
        }

        edge_table
    };

    /// coordinates of the 8 cube corner positions in local voxel space.
    /// Each entry is a Vector3(x, y, z) with coordinates in {0.0, 1.0}. The ordering
    /// matches CUBE_EDGES and EDGE_INTERSECTIONS vertex numbering (vertices 0..7):
    ///
    /// Generated with Z outermost, Y middle, X innermost:
    /// index = x + 2*y + 4*z
    ///
    ///    6-------7
    ///   /|      /|
    ///  / |     / |  Corners
    /// 2-------3  |
    /// |  4----|--5
    /// | /     | /     y z
    /// |/      |/      |/   Godot 4 axis convention (right handed)
    /// 0-------1       o--x
    ///
    /// Index mapping:
    ///   0: (0, 0, 0)    1: (1, 0, 0)
    ///   2: (0, 1, 0)    3: (1, 1, 0)
    ///   4: (0, 0, 1)    5: (1, 0, 1)
    ///   6: (0, 1, 1)    7: (1, 1, 1)
    pub static ref CORNER_OFFSETS: [Vec3; CORNER_COUNT] = {
        let mut corners = Vec::<Vec3>::with_capacity(CORNER_COUNT);

        // z outermost, y middle, x innermost -> index = x + 2*y + 4*z
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    corners.push(Vec3 { x: x as f32, y: y as f32, z: z as f32 })
                }
            }
        }

        corners.try_into().unwrap()
    };

    /// Builds a table of edge normals pointing away from the cube
    pub static ref EDGE_NORMALS: [Vec3; EDGE_COUNT] = {
        let mut normals = Vec::<Vec3>::with_capacity(EDGE_COUNT);

        for edge in *EDGES {
            let ca = CORNER_OFFSETS[edge.x as usize];
            let cb = CORNER_OFFSETS[edge.y as usize];

            // find the axis along which the two corners differ -> edge direction
            let dx = cb.x - ca.x;
            let dy = cb.y - ca.y;
            let dz = cb.z - ca.z;

            // For axes constant along the edge, take sign = if coord==0 -> -1 else +1.
            // Set the axis along the edge to 0.
            let nx = if dx != 0. { 0. } else { if ca.x == 0. { -1. } else { 1. } };
            let ny = if dy != 0. { 0. } else { if ca.y == 0. { -1. } else { 1. } };
            let nz = if dz != 0. { 0. } else { if ca.z == 0. { -1. } else { 1. } };

            normals.push(Vec3 { x: nx as f32, y: ny as f32, z: nz as f32})
        }

        normals.try_into().unwrap()
    };


    /// Normals of corner pointing away from centroid
    pub static ref CORNER_NORMALS: [Vec3; CORNER_COUNT] = {
        let mut normals = Vec::<Vec3>::with_capacity(CORNER_COUNT);

        for corner in *CORNER_OFFSETS {

            let nx = if corner.x == 0. { -1 } else { 1 };
            let ny = if corner.y == 0. { -1 } else { 1 };
            let nz = if corner.z == 0. { -1 } else { 1 };

            normals.push(Vec3 { x: nx as f32, y: ny as f32, z: nz as f32})
        }

        normals.try_into().unwrap()
    };

    /// The 26 voxels of a 3x3x3 Moore neighborhood around the origin (center
    /// excluded), each with its `Coord` offset and Euclidean distance.
    pub static ref MOORE_NEIGHBORHOOD_3D: [MooreNeighbor; NEIGHBOR_COUNT] = {
        let mut neighbors = Vec::<MooreNeighbor>::with_capacity(NEIGHBOR_COUNT);

        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    if [x, y, z] == [0, 0, 0] {
                        continue
                    }

                    let neighbor = MooreNeighbor {
                        n_offset: Coord::new( x, y, z ),
                        n_distance: match x.abs()+y.abs()+z.abs() {
                            1 => 1.,
                            2 => SQRT_2,
                            3 => SQRT_3,
                            x => panic!("added offsets should be between 1 and 3. Got {}", x),
                        }
                    };

                    neighbors.push(neighbor)
                }
            }
        }

        neighbors.try_into().unwrap()
    };


    /// The non-center voxels of a 5x5x5 block (offsets -2..=2), each with its
    /// `Coord` offset and Euclidean distance.
    pub static ref MOORE_NEIGHBORHOOD_3D_SHELL_2: Box<[MooreNeighbor]> = {
        let mut neighbors = Vec::<MooreNeighbor>::new();

        for z in -2..=2 {
            for y in -2..=2 {
                for x in -2..=2 {
                    if x == 0 && y == 0 && z == 0 || x * x + y * y + z * z > 16 {
                        continue
                    }

                    let neighbor = MooreNeighbor {
                        n_offset: Coord::new( x, y, z ),
                        n_distance: ((x * x + y * y + z * z) as f64).sqrt(),
                    };

                    neighbors.push(neighbor)
                }
            }
        }
        neighbors.into_boxed_slice()
     };

    /// The 27 voxels of a 3x3x3 Moore area around the origin, *including* the
    /// center, ordered by z (outer), then y, then x (innermost) — the
    /// `g_ordered_moore_area_3d` layout, which lets multithreaded code iterate
    /// blocks in a fixed order to avoid deadlocks. The center sits at index 13.
    pub static ref ORDERED_MOORE_AREA_3D: Box<[MooreNeighbor]> = {
        let mut neighbors = Vec::<MooreNeighbor>::with_capacity(ORDERED_MOORE_AREA_3D_COUNT);

        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    let neighbor = MooreNeighbor {
                        n_offset: Coord::new( x, y, z ),
                        n_distance: match x.abs()+y.abs()+z.abs() {
                            0 => 0.,
                            1 => 1.,
                            2 => SQRT_2,
                            3 => SQRT_3,
                            x => panic!("added offsets should be between 0 and 3. Got {}", x),
                         }
                     };

                    neighbors.push(neighbor)
                 }
             }
         }

        neighbors.into_boxed_slice()
     };
}

/// The unit face normal for each `Side`, in `Side` order
/// (`East`, `West`, `Down`, `Up`, `South`, `North`).
pub const SIDE_NORMALS: [Coord; SIDE_COUNT] = [
    Coord::new(1, 0, 0),  // EAST  (+x)
    Coord::new(-1, 0, 0), // WEST  (-x)
    Coord::new(0, -1, 0), // DOWN  (-y)
    Coord::new(0, 1, 0),  // UP    (+y)
    Coord::new(0, 0, -1), // SOUTH (-z)
    Coord::new(0, 0, 1),  // NORTH (+z)
];

/// A tangent-frame 4-vector per `Side`: the first three components give the
/// tangent direction for that face; the fourth is a constant `1.0`.
pub const SIDE_TANGENTS: [[f32; 4]; SIDE_COUNT] = [
    // East   (+1,0,0): tangent along -z
    [0.0, 0.0, -1.0, 1.0],
    // West   (-1,0,0): tangent along +z
    [0.0, 0.0, 1.0, 1.0],
    // Down   (0,-1,0): tangent along +x
    [1.0, 0.0, 0.0, 1.0],
    // Up     (0,+1,0): tangent along -x
    [-1.0, 0.0, 0.0, 1.0],
    // South  (0,0,-1): tangent along +x
    [1.0, 0.0, 0.0, 1.0],
    // North  (0,0,+1): tangent along -x
    [-1.0, 0.0, 0.0, 1.0],
];

/// The four `Corner`s bounding each face, in `Side` order.
///
/// Each row lists the face's corners counter-clockwise as seen from outside the
/// cube. The triangle indices for a face come from `SIDE_TRIANGLES`.
pub const SIDE_CORNERS: [[Corner; 4]; SIDE_COUNT] = [
    [
        // EAST
        Corner::NorthEastDown,
        Corner::SouthEastDown,
        Corner::SouthEastUp,
        Corner::NorthEastUp,
    ],
    [
        // WEST
        Corner::SouthWestDown,
        Corner::NorthWestDown,
        Corner::NorthWestUp,
        Corner::SouthWestUp,
    ],
    [
        // DOWN
        Corner::SouthWestDown,
        Corner::SouthEastDown,
        Corner::NorthEastDown,
        Corner::NorthWestDown,
    ],
    [
        // UP
        Corner::SouthEastUp,
        Corner::SouthWestUp,
        Corner::NorthWestUp,
        Corner::NorthEastUp,
    ],
    [
        // SOUTH
        Corner::SouthEastDown,
        Corner::SouthWestDown,
        Corner::SouthWestUp,
        Corner::SouthEastUp,
    ],
    [
        // NORTH
        Corner::NorthWestDown,
        Corner::NorthEastDown,
        Corner::NorthEastUp,
        Corner::NorthWestUp,
    ],
];

/// The four `Edge`s bounding each face, in `Side` order.
///
/// Edge `i` of a face connects `SIDE_CORNERS[side][i]` with
/// `SIDE_CORNERS[side][(i + 1) % 4]`, so the two together walk the face
/// counter-clockwise as seen from outside.
///
/// <pre>
///     o---11----o
///    /|        /|
///   6 9       7 10
///  /  |      /  |
/// o----5----o   |
/// |   o---8-|---o
/// 1  /      3  /
/// | 2       | 4
/// |/        |/
/// o----0----o
/// </pre>
pub const SIDE_EDGES: [[Edge; 4]; SIDE_COUNT] = [
    [
        // EAST
        Edge::EastDown,
        Edge::SouthEast,
        Edge::EastUp,
        Edge::NorthEast,
    ],
    [
        // WEST
        Edge::WestDown,
        Edge::NorthWest,
        Edge::WestUp,
        Edge::SouthWest,
    ],
    [
        // DOWN
        Edge::SouthDown,
        Edge::EastDown,
        Edge::NorthDown,
        Edge::WestDown,
    ],
    [
        // UP
        Edge::SouthUp,
        Edge::WestUp,
        Edge::NorthUp,
        Edge::EastUp,
    ],
    [
        // SOUTH
        Edge::SouthDown,
        Edge::SouthWest,
        Edge::SouthUp,
        Edge::SouthEast,
    ],
    [
        // NORTH
        Edge::NorthDown,
        Edge::NorthEast,
        Edge::NorthUp,
        Edge::NorthWest,
    ],
];

/// The distance to a face-adjacent neighbor for each `Side` (`1.0` on a unit
/// voxel grid).
pub const SIDE_NEIGHBORING_DISTANCES: [f32; SIDE_COUNT] = [1.0; SIDE_COUNT];

/// The two triangles making up each face, as indices into `SIDE_CORNERS`, in the
/// same order as `Cube::g_side_quad_triangles`: `{0, 2, 1, 0, 3, 2}`. The
/// connectivity is identical for every face, so a single array is shared.
///
/// The winding puts the face seam along the `(c2, c3)` / `(c0, c1)` diagonal.
pub const SIDE_TRIANGLES: [u32; 6] = [0, 2, 1, 0, 3, 2];

/// The same two triangles as `SIDE_TRIANGLES` but with the winding reversed,
/// so the face seam runs along the `(c1, c2)` / `(c0, c3)` diagonal instead.
/// Use this when a face's UV seam should sit on the other diagonal.
pub const SIDE_TRIANGLES_FLIPPED: [u32; 6] = [0, 1, 2, 0, 3, 2];

/// For each `Side`, the index (in `Side` order) of the opposite face. Mirrors
/// `Cube::g_opposite_side`: `East`↔`West`, `Down`↔`Up`, `South`↔`North`.
pub const OPPOSITE_SIDE: [Side; SIDE_COUNT] = [
    Side::West,  // EAST
    Side::East,  // WEST
    Side::Up,    // DOWN
    Side::Down,  // UP
    Side::North, // SOUTH
    Side::South, // NORTH
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_normals_match_axis_convention() {
        for side in Side::all_sides() {
            let expected = match side {
                Side::East => (1, 0, 0),
                Side::West => (-1, 0, 0),
                Side::Down => (0, -1, 0),
                Side::Up => (0, 1, 0),
                Side::South => (0, 0, -1),
                Side::North => (0, 0, 1),
            };
            assert_eq!(
                side.get_normal(),
                Coord::new(expected.0, expected.1, expected.2),
                "normal of {:?}",
                side,
            );
        }
    }

    #[test]
    fn from_face_direction_roundtrips() {
        for side in Side::all_sides() {
            assert_eq!(Side::from_face_direction(side.get_normal()), Some(*side),);
        }
    }

    #[test]
    fn opposite_side_is_involution() {
        for i in 0..SIDE_COUNT {
            let j = OPPOSITE_SIDE[i] as usize;
            assert_eq!(
                OPPOSITE_SIDE[j] as usize, i,
                "opposite of opposite of {} should be {}",
                i, i,
            );
            assert_eq!(
                SIDE_NORMALS[j], -SIDE_NORMALS[i],
                "opposite of {} should have negated normal",
                i,
            );
        }
    }

    #[test]
    fn corner_names_match_offsets() {
        // index = x + 2*y + 4*z, with North = +z, South = -z.
        let cases = [
            (Corner::SouthWestDown, (0, 0, 0)),
            (Corner::SouthEastDown, (1, 0, 0)),
            (Corner::SouthWestUp, (0, 1, 0)),
            (Corner::SouthEastUp, (1, 1, 0)),
            (Corner::NorthWestDown, (0, 0, 1)),
            (Corner::NorthEastDown, (1, 0, 1)),
            (Corner::NorthWestUp, (0, 1, 1)),
            (Corner::NorthEastUp, (1, 1, 1)),
        ];
        for (c, e) in cases {
            let p = CORNER_OFFSETS[c as usize];
            assert_eq!((p.x as i32, p.y as i32, p.z as i32), e, "offset of {:?}", c,);
        }
    }

    #[test]
    fn corner_normals_match_offsets() {
        for c in Corner::all_corners() {
            let p = CORNER_OFFSETS[*c as usize];
            let n = CORNER_NORMALS[*c as usize];
            let expect = (
                if p.x == 0. { -1 } else { 1 },
                if p.y == 0. { -1 } else { 1 },
                if p.z == 0. { -1 } else { 1 },
            );
            assert_eq!(
                (n.x as i32, n.y as i32, n.z as i32),
                expect,
                "normal of {:?}",
                c,
            );
        }
    }

    #[test]
    fn edge_names_match_normals() {
        let cases = [
            (Edge::SouthDown, (0, -1, -1)),
            (Edge::SouthWest, (-1, 0, -1)),
            (Edge::WestDown, (-1, -1, 0)),
            (Edge::SouthEast, (1, 0, -1)),
            (Edge::EastDown, (1, -1, 0)),
            (Edge::SouthUp, (0, 1, -1)),
            (Edge::WestUp, (-1, 1, 0)),
            (Edge::EastUp, (1, 1, 0)),
            (Edge::NorthDown, (0, -1, 1)),
            (Edge::NorthWest, (-1, 0, 1)),
            (Edge::NorthEast, (1, 0, 1)),
            (Edge::NorthUp, (0, 1, 1)),
        ];
        for (e, n) in cases {
            let p = EDGE_NORMALS[e as usize];
            assert_eq!((p.x as i32, p.y as i32, p.z as i32), n, "normal of {:?}", e,);
        }
    }

    #[test]
    fn edge_normals_match_edges() {
        for e in Edge::all_edges() {
            let edge = EDGES[*e as usize];
            let ca = CORNER_OFFSETS[edge.x as usize];
            let cb = CORNER_OFFSETS[edge.y as usize];
            let nx = if (cb.x - ca.x).abs() < 0.5 {
                if ca.x == 0. { -1. } else { 1. }
            } else {
                0.
            };
            let ny = if (cb.y - ca.y).abs() < 0.5 {
                if ca.y == 0. { -1. } else { 1. }
            } else {
                0.
            };
            let nz = if (cb.z - ca.z).abs() < 0.5 {
                if ca.z == 0. { -1. } else { 1. }
            } else {
                0.
            };
            let got = EDGE_NORMALS[*e as usize];
            assert_eq!(
                (got.x as i32, got.y as i32, got.z as i32),
                (nx as i32, ny as i32, nz as i32),
                "normal of {:?}",
                e,
            );
        }
    }

    #[test]
    fn side_edges_walk_side_corners() {
        for side in Side::all_sides() {
            let corners = SIDE_CORNERS[usize::from(*side)];
            let edges = SIDE_EDGES[usize::from(*side)];
            for i in 0..4 {
                let a = usize::from(corners[i]);
                let b = usize::from(corners[(i + 1) % 4]);
                let edge = EDGES[usize::from(edges[i])];
                assert!(
                    (usize::from(edge.x) == a && usize::from(edge.y) == b)
                        || (usize::from(edge.x) == b && usize::from(edge.y) == a),
                    "edge {} of {:?} should connect corners {} and {}",
                    i,
                    side,
                    a,
                    b,
                );
            }
        }
    }

    #[test]
    fn face_edges_point_outward() {
        for side in Side::all_sides() {
            let n = SIDE_NORMALS[usize::from(*side)];
            let (axis, sign) = if n.x != 0 {
                (0, n.x)
            } else if n.y != 0 {
                (1, n.y)
            } else {
                (2, n.z)
            };
            for e in SIDE_EDGES[usize::from(*side)] {
                let en = EDGE_NORMALS[usize::from(e)];
                let comp = match axis {
                    0 => en.x,
                    1 => en.y,
                    _ => en.z,
                };
                assert_eq!(
                    comp as i32, sign,
                    "edge {:?} of {:?} should have outward normal component",
                    e, side,
                );
            }
        }
    }

    #[test]
    fn faces_use_all_edges_and_corners() {
        let mut edge_counts = [0u32; 12];
        let mut corner_counts = [0u32; 8];
        for side in Side::all_sides() {
            for c in SIDE_CORNERS[usize::from(*side)] {
                corner_counts[usize::from(c)] += 1;
            }
            for e in SIDE_EDGES[usize::from(*side)] {
                edge_counts[usize::from(e)] += 1;
            }
        }
        for c in 0..8 {
            assert_eq!(
                corner_counts[c], 3,
                "corner {} should be shared by exactly 3 faces",
                c,
            );
        }
        for e in 0..12 {
            assert_eq!(
                edge_counts[e], 2,
                "edge {} should be shared by exactly 2 faces",
                e,
            );
        }
    }

    #[test]
    fn side_triangles_reference_valid_corners() {
        for &t in SIDE_TRIANGLES.iter() {
            assert!((0..4).contains(&t), "triangle index {} out of range", t,);
        }
    }

    #[test]
    fn moore_neighborhood_has_26_noncenter_voxels() {
        assert_eq!(MOORE_NEIGHBORHOOD_3D.len(), 26);
        for n in MOORE_NEIGHBORHOOD_3D.iter() {
            let x = n.n_offset.x;
            let y = n.n_offset.y;
            let z = n.n_offset.z;
            assert!((x, y, z) != (0, 0, 0), "neighborhood must exclude center");
            let man = x.abs() + y.abs() + z.abs();
            let dist = match man {
                1 => 1.0_f64,
                2 => SQRT_2,
                3 => SQRT_3,
                other => panic!("unexpected manhattan distance {}", other),
            };
            assert!((n.n_distance - dist).abs() < 1e-9);
        }
    }

    #[test]
    fn ordered_moore_area_has_27_voxels_including_center() {
        let area = &*ORDERED_MOORE_AREA_3D;
        assert_eq!(area.len(), 27);
        assert_eq!(
            area[13].n_offset,
            Coord::ORIGIN,
            "center must be at index 13"
        );
        for i in 0..27 {
            // z-outer, y-middle, x-inner: index = 9*(z+1) + 3*(y+1) + (x+1).
            let z = (i / 9) as i32 - 1;
            let y = (i % 9) as i32 / 3 - 1;
            let x = (i % 3) as i32 - 1;
            assert_eq!(
                area[i].n_offset,
                Coord::new(x, y, z),
                "ordered area index {}",
                i,
            );
        }
    }
}
