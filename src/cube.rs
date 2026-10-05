use crate::Coord;
use glam::{U8Vec2, Vec3};
use lazy_static::lazy_static;
use std::panic;

const SQRT_2: f64 = 1.4142135;
const SQRT_3: f64 = 1.7320508;

// The following tables respect the following conventions
//
//    6-------7
//   /|      /|
//  / |     / |  Corners
// 2-------3  |
// |  4----|--5
// | /     | /     y z
// |/      |/      |/   Godot 4 axis convention (right handed)
// 0-------1       o--x
//
// Sides are ordered according to the Voxel::Side enum.
//
//     o---11----o
//    /|        /|
//   6 9       7 10   Face Edges
//  /  |      /  |
// o----5----o   |
// |   o---8-|---o
// 1  /      3  /
// | 2       | 4
// |/        |/
// o----0----o
//
// Edges are ordered according to the Voxel::Edge enum (only g_edge_inormals!).
//

// Index convention used in some lookup tables
#[repr(usize)]
#[derive(Copy, Clone, Debug, Default)]
pub enum Side {
    Left = 0,
    Right,
    Bottom,
    Top,
    Back,
    Front,

    Count,
    #[default]
    Unknown,
}

impl From<usize> for Side {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Left,
            1 => Self::Right,
            2 => Self::Bottom,
            3 => Self::Top,
            4 => Self::Back,
            5 => Self::Front,
            6.. => Self::Unknown,
        }
    }
}

// TODO We should use this naming system, taken from Minecraft:
// - West: -X
// - East: +X
// - North: -Z
// - South: +Z
// - Down: -Y
// - Up: +Y

// Alias to the above for clarity, fixing some interpretation problems regarding the side_normals table...
#[repr(usize)]
#[derive(Copy, Clone, Debug, Default)]
pub enum SideAxis {
    /// Negative X
    West = 0,
    /// Positive X
    East,
    /// Negative Y
    Down,
    /// Positive Y
    Up,
    /// Negative Z
    North,
    /// Positive Z
    South,

    Count,
    #[default]
    Unknown,
}

impl From<usize> for SideAxis {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::West,
            1 => Self::East,
            2 => Self::Down,
            3 => Self::Up,
            4 => Self::North,
            5 => Self::South,
            _ => Self::Unknown,
        }
    }
}

impl From<SideAxis> for Side {
    fn from(value: SideAxis) -> Self {
        match value {
            SideAxis::West => Self::Left,
            SideAxis::East => Self::Right,
            SideAxis::Down => Self::Bottom,
            SideAxis::Up => Self::Top,
            SideAxis::North => Self::Back,
            SideAxis::South => Self::Front,
            _ => Self::Unknown,
        }
    }
}

impl From<SideAxis> for usize {
    fn from(value: SideAxis) -> Self {
        value as usize
    }
}

impl From<Side> for usize {
    fn from(value: Side) -> Self {
        value as usize
    }
}

impl From<Side> for SideAxis {
    fn from(value: Side) -> Self {
        match value {
            Side::Left => Self::West,
            Side::Right => Self::East,
            Side::Bottom => Self::Down,
            Side::Top => Self::Up,
            Side::Back => Self::North,
            Side::Front => Self::South,
            _ => Self::Unknown,
        }
    }
}

// Index into CUBE_EDGES table
#[repr(usize)]
#[derive(Copy, Clone, Debug, Default)]
pub enum Edge {
    SouthDown = 0,
    SouthWest,
    WestDown,
    SouthEast,
    EastDown,
    SouthUp,
    EastUp,
    WestUp,
    NorthDown,
    NorthWest,
    NorthEast,
    NorthUp,
    Count,
    #[default]
    Unknown,
}

// Index convention used in some lookup tables
#[repr(usize)]
#[derive(Copy, Clone, Debug, Default)]
pub enum Corner {
    SouthWestDown = 0,
    SouthEastDown,
    SouthEastup,
    SouthWestup,
    NorthWestDown,
    NorthEastDown,
    NorthWestUp,
    NorthEastUp,

    Count,
    #[default]
    Unknown,
}

#[repr(usize)]
#[derive(Copy, Clone, Debug, Default)]
pub enum Neighbor {
    SouthWestDown = 0,
    SouthDown,
    SouthEastDown,
    SouthWest,
    South,
    SouthEast,
    SouthWestUp,
    SouthUp,
    SouthEastUp,
    WestDown,
    Down,
    EastDown,
    West,
    // Center,
    East,
    WestUp,
    Up,
    EastUp,
    NorthWestDown,
    NorthDown,
    NorthEastDown,
    NorthWest,
    North,
    NorthEast,
    NorthWestUp,
    NorthUp,
    NorthEastUp,

    Count,
    #[default]
    Unknown,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct MooreNeighbor {
    pub n_offset: Coord,
    pub n_distance: f64,
}

lazy_static! {
   /// CUBE_EDGES: list of the 12 undirected edges of a unit cube as pairs of vertex indices. Each tuple
   /// (a, b) is an edge between vertex a and vertex b. Vertex numbering follows the convention:
   /// index = x + 4*y + 16*z (Z outermost → Y middle → X innermost).
   ///
   /// The table is built by iterating i=0..7, j=0..2 and connecting corner `i` with
   /// `i^(1<<j)` when they differ. Edges only appear where one endpoint has the bit set.
   ///
   ///
   ///
   /// <pre>
   ///  Corner Diagram
   ///
   ///     z
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
    pub static ref CUBE_EDGES: [U8Vec2;12] = {
        let mut cube_edges = Vec::<U8Vec2>::with_capacity(12);

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

    /// EDGE_TABLE: lookup table mapping each cube vertex inside/outside configuration
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
                let a = (i & (1<<CUBE_EDGES[j].x)) != 0;
                let b = (i & (1<<CUBE_EDGES[j].y)) != 0;

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

    /// VOXEL_CORNERS: coordinates of the 8 cube corner positions in local voxel space.
    /// Each entry is a Vector3(x, y, z) with coordinates in {0.0, 1.0}. The ordering
    /// matches CUBE_EDGES and EDGE_INTERSECTIONS vertex numbering (vertices 0..7):
    ///
    /// Generated with Z outermost, Y middle, X innermost:
    /// index = x + 4*z + 16*y
    ///
    /// <pre>
    ///    6-------7
    ///   /|      /|
    ///  / |     / |  Corners
    /// 2-------3  |
    /// |  4----|--5
    /// | /     | /     y z
    /// |/      |/      |/   Godot 4 axis convention (right handed)
    /// 0-------1       o--x
    /// </pre>
    ///
    /// Index mapping:
    ///   0: (0, 0, 0)    1: (1, 0, 0)
    ///   2: (0, 1, 0)    3: (1, 1, 0)
    ///   4: (0, 0, 1)    5: (1, 0, 1)
    ///   6: (0, 1, 1)    7: (1, 1, 1)
    pub static ref VOXEL_CORNER_OFFSETS: [Vec3; 8] = {
        let mut corners = Vec::<Vec3>::with_capacity(8);

        // z outermost, y middle, X innermost -> index = x + 4*y + 16*z (XYZ ordering)
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
    pub static ref EDGE_NORMALS: [Vec3; Edge::Count as usize] = {
        let mut normals = Vec::<Vec3>::with_capacity(Edge::Count as usize);

        for edge in *CUBE_EDGES {
            let ca = VOXEL_CORNER_OFFSETS[edge.x as usize];
            let cb = VOXEL_CORNER_OFFSETS[edge.y as usize];

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
    ///
    ///
    pub static ref CORNER_NORMALS: [Vec3; Corner::Count as usize] = {
        let mut normals = Vec::<Vec3>::with_capacity(Corner::Count as usize);

        for corner in *VOXEL_CORNER_OFFSETS {

            let nx = if corner.x == 0. { -1 } else { 1 };
            let ny = if corner.y == 0. { -1 } else { 1 };
            let nz = if corner.z == 0. { -1 } else { 1 };

            normals.push(Vec3 { x: nx as f32, y: ny as f32, z: nz as f32})
        }

        normals.try_into().unwrap()
    };

    pub static ref MOORE_NEIGHBORHOOD_3D: [MooreNeighbor; Neighbor::Count as usize] = {
        let mut neighbors = Vec::<MooreNeighbor>::with_capacity(Neighbor::Count as usize);

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
}

pub const SIDE_NORMALS: [Coord; Side::Count as usize] = [
    Coord::new(-1, 0, 0), // LEFT
    Coord::new(1, 0, 0),  // RIGHT
    //
    Coord::new(0, -1, 0), // BOTTOM
    Coord::new(0, 1, 0),  // TOP
    //
    Coord::new(0, 0, -1), // BACK
    Coord::new(0, 0, 1),  // FRONT
];

pub const SIDE_TANGENTS: [[f32; 4]; Side::Count as usize] = [
    // Left  (-1,0,0): tangent along +z
    [0.0, 0.0, 1.0, 1.0],
    // Right (+1,0,0): tangent along -z
    [0.0, 0.0, -1.0, 1.0],
    // Bottom (0,-1,0): tangent along +x
    [1.0, 0.0, 0.0, 1.0],
    // Top (0,+1,0): tangent along +x
    [1.0, 0.0, 0.0, 1.0],
    // Back (0,0,-1): tangent along +x
    [1.0, 0.0, 0.0, 1.0],
    // Front(0,0,+1): tangent along -x
    [-1.0, 0.0, 0.0, 1.0],
];

pub const SIDE_CORNERS: [[usize; 4]; Side::Count as usize] = [
    [2, 0, 6, 4], // WEST
    [1, 3, 5, 7], // EAST
    [2, 3, 0, 1], // DOWN
    [4, 5, 6, 7], // UP
    [3, 2, 7, 6], // NORTH
    [0, 1, 4, 5], // SOUTH
];

pub const SIDE_EDGES: [[usize; 4]; Side::Count as usize] = [
    [1, 2, 6, 9],   // WEST
    [3, 4, 7, 10],  // EAST
    [0, 1, 3, 5],   // DOWN
    [8, 9, 10, 11], // UP
    [5, 6, 7, 11],  // NORTH
    [0, 2, 4, 8],   // SOUTH
];

pub const SIDE_NEIGHBORING_DISTANCES: [f32; 6] = [1.0; 6];

pub fn dir_to_side(d: Coord) -> Option<Side> {
    for i in 0..Side::Count as usize {
        if SIDE_NORMALS[i] == d {
            return Some(i.into());
        }
    }

    None
}
