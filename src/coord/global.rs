//! 3D integer coordinates.
//!
//! [`Coord`] is a sparse voxel tree coordinate initially based on OpenVDB Coord.
//! It is defined as a newtype over a `glam` integer vector (`IndexVec`) that provides
//! the core vector functionality. It also defines several functions specific to indexing a voxel
//! grid, and managing CoordBBox (such as component_gt/ge/lt/le).
//! It derefs to the underlying vector, so `glam` component access (`.x`,
//! `.y`, `.z`) and vector math are available directly.
//!
//! The module also defines the [`CoordRound`] rounding trait and, via macros, a
//! large family of `From`/operator conversions between `Coord` and the common
//! `glam` integer and floating-point vector types, 3-element arrays, and tuples.
//!
//! Coord is opinionated when it comes to conversions to properly align to the grid:
//!
//! Float conversions/operators convert to f64, then uses round_half_up
//! on the result. Rounding away from zero (standard rust round) can cause inconsistent indexing
//! around origin:
//!
//! .round():
//!
//! ```text
//! coord a: (-0.5, -0.5, -0.5) -> (-1.0, -1.0, -1.0)
//! coord b: (0.5, 0.5, 0.5) -> (1.0, 1.0, 1.0)
//! distance: 2.0
//!
//! coord a: (0.5, 0.5, 0.5) -> (1.0, 1.0, 1.0)
//! coord b: (1.5, 1.5, 1.5) -> (2.0, 2.0, 2.0)
//! distance: 1.0
//! ```
//!
//! round_half_up:
//!
//! ```text
//! coord a: (-0.5, -0.5, -0.5) -> (0, 0, 0)
//! coord b: (0.5, 0.5, 0.5) -> (1.0, 1.0, 1.0)
//! distance: 1.0
//!
//! coord a: (0.5, 0.5, 0.5) -> (1.0, 1.0, 1.0)
//! coord b: (1.5, 1.5, 1.5) -> (2.0, 2.0, 2.0)
//! distance: 1.0
//! ```
//! signed/unsigned integer conversion (`Coord` to `LocalCoord`) occurs through TreeDim or CoordBBox

use glam::Vec3Swizzles;
use num::{Integer, NumCast};
use num_traits::ToPrimitive;
use num_traits::real::Real;
use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

#[cfg(feature = "bytemuck")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use crate::*;

/// A 3D integer coordinate for addressing a voxel grid.
///
/// `Coord` wraps a `glam` integer vector (`IndexVec`) and implements `Deref`/
/// `DerefMut`, so the underlying `.x`, `.y`, `.z` fields and `glam` vector
/// operations are usable directly. The element type is `i32` by default and
/// `i64` under the `index64` feature.
///
/// ```
/// use voxgrid::Coord;
///
/// let p = Coord::new(1, 2, 3).offset_by(4, 5, 6);
/// assert_eq!(p, Coord::new(5, 7, 9));
/// ```
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(Zeroable, Pod))]
#[repr(C)]
pub struct Coord(pub IndexVec);

impl DerefMut for Coord {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for Coord {
    type Target = IndexVec;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
/// Methods for offsetting, clamping, and comparing `Coord` values.
impl Coord {
    /// The smallest representable coordinate: every component at the index type's minimum.
    pub const MIN: Self = Self(IndexVec {
        x: Index::MIN,
        y: Index::MIN,
        z: Index::MIN,
    });
    /// The largest representable coordinate: every component at the index type's maximum.
    pub const MAX: Self = Self(IndexVec {
        x: Index::MAX,
        y: Index::MAX,
        z: Index::MAX,
    });
    /// The grid origin, `(0, 0, 0)`.
    pub const ORIGIN: Self = Self(IndexVec { x: 0, y: 0, z: 0 });

    /// Construct a [`Coord`] from its three index components.
    pub const fn new(x: Index, y: Index, z: Index) -> Self {
        Self(IndexVec { x, y, z })
    }

    pub fn from_int<T: Integer + NumCast>(v: T) -> Self {
        #[cfg(feature = "index64")]
        let v = v.to_i64().unwrap();

        #[cfg(not(feature = "index64"))]
        let v = v.to_i32().unwrap();

        Self::new(v, v, v)
    }

    pub fn from_float<T: Real + NumCast>(v: T) -> Self {
        let v = round_half_up(v);
        #[cfg(feature = "index64")]
        let v = v.to_i64().unwrap();

        #[cfg(not(feature = "index64"))]
        let v = v.to_i32().unwrap();

        Self::new(v, v, v)
    }

    #[cfg(feature = "bytemuck")]
    /// Returns a raw byte slice over the coordinate's fields. Requires the `bytemuck` feature.
    pub fn bytes_of(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    #[cfg(feature = "bytemuck")]
    /// Returns a mutable raw byte slice over the coordinate's fields. Requires the `bytemuck` feature.
    pub fn bytes_of_mut(&mut self) -> &mut [u8] {
        bytemuck::bytes_of_mut(self)
    }

    #[cfg(feature = "bytemuck")]
    /// Views the coordinate's fields as a slice of index values. Requires the `bytemuck` feature.
    pub fn as_slice(&self) -> &[Index] {
        bytemuck::cast_slice(self.bytes_of())
    }

    #[cfg(feature = "bytemuck")]
    /// Views the coordinate's fields as a mutable slice of index values. Requires the `bytemuck` feature.
    pub fn as_slice_mut(&mut self) -> &mut [Index] {
        bytemuck::cast_slice_mut(self.bytes_of_mut())
    }

    #[cfg(feature = "bytemuck")]
    /// Reinterprets the components as the unsigned index type, as used for hashing and indexing. Requires the `bytemuck` feature.
    pub fn as_slice_unsigned(&self) -> &[UIndex] {
        bytemuck::cast_slice(self.as_slice())
    }

    /// Returns the components as a three-element index array.
    pub fn as_array(&self) -> [Index; 3] {
        [self.x, self.y, self.z]
    }

    /// Returns the components as a three-element unsigned-index array.
    pub fn as_array_unsigned(&self) -> [UIndex; 3] {
        [self.x as UIndex, self.y as UIndex, self.z as UIndex]
    }

    /// Offsets this coordinate in place by the per-axis deltas `(dx, dy, dz)`, returning `&mut self` for chaining.
    pub const fn offset(&mut self, dx: Index, dy: Index, dz: Index) -> &mut Self {
        self.0.x += dx;
        self.0.y += dy;
        self.0.z += dz;

        self
    }

    /// Offsets this coordinate in place by `n` on every axis, returning `&mut self` for chaining.
    pub const fn single_offset(&mut self, n: Index) -> &mut Self {
        self.offset(n, n, n)
    }

    /// Returns a new [`Coord`] offset by the per-axis deltas `(dx, dy, dz)`.
    pub const fn offset_by(&self, dx: Index, dy: Index, dz: Index) -> Self {
        Self::new(self.0.x + dx, self.0.y + dy, self.0.z + dz)
    }

    /// Returns a new [`Coord`] offset by `n` on every axis.
    pub const fn single_offset_by(&self, n: Index) -> Self {
        self.offset_by(n, n, n)
    }

    /// Returns a new [`Coord`] with the per-component minimum of this and `other`.
    pub fn min_component(&self, other: &Self) -> Coord {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Returns a new [`Coord`] with the per-component maximum of this and `other`.
    pub fn max_component(&self, other: &Self) -> Coord {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Returns `true` if any component is less than the corresponding component of `other`.
    pub fn component_lt(&self, other: &Self) -> bool {
        self.x < other.x || self.y < other.y || self.z < other.z
    }

    /// Returns `true` if any component is less than or equal to the corresponding component of `other`.
    pub fn component_le(&self, other: &Self) -> bool {
        self.x <= other.x || self.y <= other.y || self.z <= other.z
    }

    /// Returns `true` if any component is greater than the corresponding component of `other`.
    pub fn component_gt(&self, other: &Self) -> bool {
        self.x > other.x || self.y > other.y || self.z > other.z
    }

    /// Returns `true` if any component is greater than or equal to the corresponding component of `other`.
    pub fn component_ge(&self, other: &Self) -> bool {
        self.x >= other.x || self.y >= other.y || self.z >= other.z
    }

    /// Returns the axis index of the smallest component: `0` for `x`, `1` for `y`, `2` for `z`.
    pub fn min_idx(&self) -> usize {
        let mut idx = 0;

        #[cfg(feature = "bytemuck")]
        let slice = self.as_slice();

        #[cfg(not(feature = "bytemuck"))]
        let slice = self.as_array();

        for i in 1..3 {
            if slice[i] < slice[idx] {
                idx = i
            }
        }
        idx
    }

    /// Returns the axis index of the largest component: `0` for `x`, `1` for `y`, `2` for `z`.
    pub fn max_idx(&self) -> usize {
        let mut idx = 0;

        #[cfg(feature = "bytemuck")]
        let slice = self.as_slice();

        #[cfg(not(feature = "bytemuck"))]
        let slice = self.as_array();

        for i in 1..3 {
            if slice[i] > slice[idx] {
                idx = i
            }
        }
        idx
    }

    #[inline]
    /// Returns a coordinate with each component made non-negative.
    pub fn abs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline]
    /// adds all coordinates together and returns as a usize
    pub fn sum(&self) -> usize {
        (self.x + self.y + self.z) as usize
    }

    /// Returns a coordinate with the `x` and `z` components swapped (a `glam` `zyx` swizzle).
    pub fn zyx(&self) -> Coord {
        Self(self.0.zyx())
    }

    /// Returns `true` if this and `other` agree on at least one axis.
    pub fn any_match(&self, other: &Coord) -> bool {
        self.x == other.x || self.y == other.y || self.z == other.z
    }

    /// Returns a coordinate that is `1` on each axis where this and `other` agree and `0` otherwise.
    pub fn match_axes(&self, other: &Coord) -> Coord {
        Self::new(
            (self.x == other.x) as Index,
            (self.y == other.y) as Index,
            (self.z == other.z) as Index,
        )
    }

    /// Returns a coordinate with each component clamped to the range `[-1, 1]`.
    pub fn unit_clamp(&self) -> Coord {
        Self::new(
            self.x.clamp(-1, 1),
            self.y.clamp(-1, 1),
            self.z.clamp(-1, 1),
        )
    }
}

impl PartialOrd for Coord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.x.partial_cmp(&other.x) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        match self.y.partial_cmp(&other.y) {
            Some(core::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.z.partial_cmp(&other.z)
    }
}

impl Ord for Coord {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.x.cmp(&other.x) {
            Ordering::Equal => (),
            ord => return ord,
        }
        match self.y.cmp(&other.y) {
            Ordering::Equal => (),
            ord => return ord,
        }
        self.z.cmp(&other.z)
    }
}

impl Display for Coord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("({}, {}, {})", self.x, self.y, self.z))
    }
}

impl std::ops::AddAssign<Coord> for Coord {
    fn add_assign(&mut self, rhs: Coord) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::Add<Coord> for Coord {
    type Output = Coord;

    fn add(self, rhs: Coord) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::SubAssign<Coord> for Coord {
    fn sub_assign(&mut self, rhs: Coord) {
        self.0 -= rhs.0
    }
}

impl std::ops::Sub<Coord> for Coord {
    type Output = Coord;

    fn sub(self, rhs: Coord) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<Coord> for Coord {
    type Output = Coord;

    fn mul(self, rhs: Coord) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::MulAssign<Coord> for Coord {
    fn mul_assign(&mut self, rhs: Coord) {
        self.0 *= rhs.0
    }
}

impl std::ops::Div<Coord> for Coord {
    type Output = Coord;

    fn div(self, rhs: Coord) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl std::ops::DivAssign<Coord> for Coord {
    fn div_assign(&mut self, rhs: Coord) {
        self.0 /= rhs.0
    }
}

impl std::ops::Neg for Coord {
    type Output = Coord;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

add_glam_float_type!(Coord, glam::Vec3, f32, glam::DVec3, f64);
add_glam_int_type!(
    Coord,
    Index,
    glam::I8Vec3,
    i8,
    glam::I16Vec3,
    i16,
    glam::IVec3,
    i32,
    glam::I64Vec3,
    i64,
    glam::U8Vec3,
    u8,
    glam::U16Vec3,
    u16,
    glam::UVec3,
    u32,
    glam::U64Vec3,
    u64
);
add_int_conversions!(
    Coord, Index, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize
);
add_int_ops!(
    Coord, Index, to_i64, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize
);
add_float_conversions!(Coord, f32, f64);
add_float_ops!(Coord, f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    // ============== Construction Tests ==============
    #[test]
    fn test_coord_construction() {
        // Test new()
        let coord = Coord::new(1, 2, 3);
        assert_eq!(coord.x, 1);
        assert_eq!(coord.y, 2);
        assert_eq!(coord.z, 3);

        // Test ORIGIN constant
        assert_eq!(Coord::ORIGIN, Coord::new(0, 0, 0));
        assert_eq!(Coord::ORIGIN.x, 0);
        assert_eq!(Coord::ORIGIN.y, 0);
        assert_eq!(Coord::ORIGIN.z, 0);

        // Test MIN and MAX constants
        assert_eq!(Coord::MIN.x, Index::MIN);
        assert_eq!(Coord::MIN.y, Index::MIN);
        assert_eq!(Coord::MIN.z, Index::MIN);
        assert_eq!(Coord::MAX.x, Index::MAX);
        assert_eq!(Coord::MAX.y, Index::MAX);
        assert_eq!(Coord::MAX.z, Index::MAX);
    }

    #[test]
    fn test_coord_copy_clone() {
        let coord = Coord::new(5, 10, 15);
        let copied = coord; // Copy trait
        let cloned = coord.clone(); // Clone trait

        assert_eq!(copied, coord);
        assert_eq!(cloned, coord);

        // Modifying copy shouldn't affect original
        let mut copy = coord;
        copy.x = 100;
        assert_eq!(coord.x, 5);
        assert_eq!(copy.x, 100);
    }

    #[test]
    fn test_coord_default() {
        let default_coord = Coord::default();
        assert_eq!(default_coord, Coord::ORIGIN);
    }

    // ============== Coordinate Arithmetic Tests ==============
    #[test]
    fn test_coord_addition() {
        let a = Coord::new(1, 2, 3);
        let b = Coord::new(4, 5, 6);
        let sum = a + b;

        assert_eq!(sum.x, 5);
        assert_eq!(sum.y, 7);
        assert_eq!(sum.z, 9);

        // Test AddAssign
        let mut c = a;
        c += b;
        assert_eq!(c, sum);
    }

    #[test]
    fn test_coord_subtraction() {
        let a = Coord::new(10, 20, 30);
        let b = Coord::new(4, 5, 6);
        let diff = a - b;

        assert_eq!(diff.x, 6);
        assert_eq!(diff.y, 15);
        assert_eq!(diff.z, 24);

        // Test SubAssign
        let mut c = a;
        c -= b;
        assert_eq!(c, diff);
    }

    #[test]
    fn test_coord_negation() {
        let coord = Coord::new(5, 10, -15);
        let neg = -coord;

        assert_eq!(neg.x, -5);
        assert_eq!(neg.y, -10);
        assert_eq!(neg.z, 15);

        // Negation of origin
        let neg_origin = -Coord::ORIGIN;
        assert_eq!(neg_origin, Coord::ORIGIN);
    }

    #[test]
    fn test_coord_shifts() {
        let coord = Coord::new(8, 16, 32);

        // Test Shl
        let shifted_left: IndexVec = *coord << 1;
        assert_eq!(shifted_left.x, 16);
        assert_eq!(shifted_left.y, 32);
        assert_eq!(shifted_left.z, 64);

        // Test Shr
        let shifted_right: IndexVec = *coord >> 2;
        assert_eq!(shifted_right.x, 2);
        assert_eq!(shifted_right.y, 4);
        assert_eq!(shifted_right.z, 8);

        // Test ShlAssign
        let mut c = coord;
        c <<= 1;
        assert_eq!(c.x, 16);
        assert_eq!(c.y, 32);
        assert_eq!(c.z, 64);

        // Test ShrAssign
        let mut d = coord;
        d >>= 2;
        assert_eq!(d.x, 2);
        assert_eq!(d.y, 4);
        assert_eq!(d.z, 8);
    }

    // ============== Coordinate Offset Tests ==============
    #[test]
    fn test_coord_offset() {
        let mut coord = Coord::new(1, 2, 3);

        // Test offset() - in-place modification
        coord.offset(10, 20, 30);
        assert_eq!(coord.x, 11);
        assert_eq!(coord.y, 22);
        assert_eq!(coord.z, 33);

        // Test offset_by() - creates new Coord
        let offset_coord = Coord::new(5, 5, 5).offset_by(1, 2, 3);
        assert_eq!(offset_coord.x, 6);
        assert_eq!(offset_coord.y, 7);
        assert_eq!(offset_coord.z, 8);

        // Test single_offset() - in-place, same delta for all axes
        let mut s = Coord::ORIGIN;
        s.single_offset(5);
        assert_eq!(s, Coord::new(5, 5, 5));

        // Test single_offset_by() - creates new Coord with same delta
        let single = Coord::new(0, 0, 0).single_offset_by(10);
        assert_eq!(single, Coord::new(10, 10, 10));
    }

    #[test]
    fn test_coord_single_offset_by() {
        let coord = Coord::new(1, 1, 1);
        let result = coord.single_offset_by(5);
        assert_eq!(result, Coord::new(6, 6, 6));
    }

    // ============== Component Operations Tests ==============
    #[test]
    fn test_coord_min_component() {
        let a = Coord::new(5, 10, 15);
        let b = Coord::new(3, 12, 8);

        let min = a.min_component(&b);
        assert_eq!(min.x, 3);
        assert_eq!(min.y, 10);
        assert_eq!(min.z, 8);

        // Same comparison
        let same = a.min_component(&a);
        assert_eq!(same, a);
    }

    #[test]
    fn test_coord_max_component() {
        let a = Coord::new(5, 10, 15);
        let b = Coord::new(3, 12, 8);

        let max = a.max_component(&b);
        assert_eq!(max.x, 5);
        assert_eq!(max.y, 12);
        assert_eq!(max.z, 15);

        // Same comparison
        let same = a.max_component(&a);
        assert_eq!(same, a);
    }

    #[test]
    fn test_coord_component_less_than() {
        let a = Coord::new(5, 5, 5);
        let b = Coord::new(3, 10, 3); // x < a.x, y > a.y, z < a.z

        // a is less than b only if ANY component is less
        assert!(a.component_lt(&b));
        assert!(b.component_lt(&a));

        let smaller = Coord::ORIGIN;
        let larger = Coord::new(1, 1, 1);

        assert!(!larger.component_lt(&smaller))
    }

    #[test]
    fn test_coord_min_max_idx() {
        let coord = Coord::new(10, 5, 15);

        assert_eq!(coord.min_idx(), 1); // y=5 is smallest
        assert_eq!(coord.max_idx(), 2); // z=15 is largest

        let equal = Coord::new(5, 5, 5);
        // When all equal, should return first i64
        assert_eq!(equal.min_idx(), 0);
        assert_eq!(equal.max_idx(), 0);

        let reverse = Coord::new(15, 10, 5);
        assert_eq!(reverse.min_idx(), 2); // z=5 is smallest
        assert_eq!(reverse.max_idx(), 0); // x=15 is largest
    }

    #[test]
    fn test_coord_abs() {
        let coord = Coord::new(-5, 10, -15);
        let abs = coord.abs();

        assert_eq!(abs.x, 5);
        assert_eq!(abs.y, 10);
        assert_eq!(abs.z, 15);

        // Abs of non-negative values
        let positive = Coord::new(5, 10, 15);
        assert_eq!(positive.abs(), positive);

        // Abs of origin
        assert_eq!(Coord::ORIGIN.abs(), Coord::ORIGIN);
    }

    #[test]
    fn test_coord_sum() {
        let coord = Coord::new(10, 20, 30);
        assert_eq!(coord.sum(), 60);

        let with_negative = Coord::new(-10, 20, -5);
        assert_eq!(with_negative.sum(), 5);

        // Sum of origin
        assert_eq!(Coord::ORIGIN.sum(), 0);
    }

    // ============== Byte/Array Conversion Tests ==============
    #[test]
    #[cfg(feature = "bytemuck")]
    fn test_coord_byte_conversion() {
        let coord = Coord::new(1, 2, 3);

        // Test bytes_of
        let bytes = coord.bytes_of();
        assert_eq!(bytes.len(), std::mem::size_of::<Coord>());

        // Test bytes_of_mut
        let mut mutable_coord = coord;
        let bytes_mut = mutable_coord.bytes_of_mut();
        assert_eq!(bytes_mut.len(), std::mem::size_of::<Coord>());
    }

    #[test]
    #[cfg(feature = "bytemuck")]
    fn test_coord_as_slice() {
        let coord = Coord::new(1, 2, 3);

        let slice = coord.as_slice();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);

        // Test as_slice_mut
        let mut mutable_coord = coord;
        let slice_mut = mutable_coord.as_slice_mut();
        slice_mut[0] = 100;
        slice_mut[1] = 200;
        slice_mut[2] = 300;
        assert_eq!(mutable_coord, Coord::new(100, 200, 300));
    }

    #[test]
    fn test_coord_as_array() {
        let coord = Coord::new(1, 2, 3);

        let array = coord.as_array();
        assert_eq!(array[0], 1);
        assert_eq!(array[1], 2);
        assert_eq!(array[2], 3);

        // Test unsigned array
        let unsigned = Coord::new(1, 2, 3).as_array_unsigned();
        assert_eq!(unsigned[0], 1);
        assert_eq!(unsigned[1], 2);
        assert_eq!(unsigned[2], 3);
    }

    // ============== Conversion Tests ==============
    #[test]
    fn test_coord_from_index_tuple() {
        let tuple: (Index, Index, Index) = (5, 10, 15);
        let coord: Coord = tuple.into();

        assert_eq!(coord.x, 5);
        assert_eq!(coord.y, 10);
        assert_eq!(coord.z, 15);
    }

    #[test]
    fn test_coord_to_index_tuple() {
        let coord = Coord::new(5, 10, 15);
        let tuple: (Index, Index, Index) = coord.into();

        assert_eq!(tuple.0, 5);
        assert_eq!(tuple.1, 10);
        assert_eq!(tuple.2, 15);
    }

    // ============== Comparison Tests ==============
    #[test]
    fn test_coord_ordering() {
        let a = Coord::new(1, 2, 3);
        let b = Coord::new(1, 2, 4);
        let c = Coord::new(2, 2, 3);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&a), Ordering::Equal);
        assert_eq!(c.cmp(&a), Ordering::Greater);
    }

    #[test]
    fn test_coord_hash() {
        use std::collections::HashSet;

        let coord1 = Coord::new(1, 2, 3);
        let coord2 = Coord::new(1, 2, 3);
        let coord3 = Coord::new(4, 5, 6);

        let mut set = HashSet::new();
        set.insert(coord1);
        set.insert(coord2);
        set.insert(coord3);

        // coord1 and coord2 should be considered equal for hashing
        assert_eq!(set.len(), 2);
        assert!(set.contains(&coord1));
        assert!(set.contains(&coord2));
        assert!(set.contains(&coord3));
    }

    // ============== PartialOrd Tests ==============
    #[test]
    fn test_coord_partial_cmp() {
        let a = Coord::new(1, 2, 3);
        let b = Coord::new(1, 2, 3);
        let c = Coord::new(1, 2, 4);

        assert!(a == b);
        assert!(b >= a);
        assert!(c > b);
        assert!(a < c);
    }

    // ============== Operator Overloads ==============
    #[test]
    fn test_coord_bitwise_operations() {
        let coord = Coord::new(15, 15, 15); // 1111 in binary

        // BitAnd: 15 & 5 (0101) = 5 (0101)
        let and_result = Coord(*coord & 5);
        assert_eq!(and_result, Coord::new(5, 5, 5));

        // BitOr: 15 | 1 (0001) = 15
        let or_result = Coord(*coord | 1);
        assert_eq!(or_result, Coord::new(15, 15, 15));

        // BitXor: 15 ^ 3 (0011) = 12 (1100)
        let xor_result = Coord(*coord ^ 3);
        assert_eq!(xor_result, Coord::new(12, 12, 12));

        // BitAndAssign
        let mut and_assign = coord;
        and_assign &= 5;
        assert_eq!(and_assign, Coord::new(5, 5, 5));

        // BitOrAssign
        let mut or_assign = coord;
        or_assign |= 1;
        assert_eq!(or_assign, Coord::new(15, 15, 15));

        // BitXorAssign
        let mut xor_assign = coord;
        xor_assign ^= 3;
        assert_eq!(xor_assign, Coord::new(12, 12, 12));
    }

    // ============== Debug and String Representation ==============
    #[test]
    fn test_coord_display() {
        let coord = Coord::new(1, 2, 3);
        let debug_str = format!("{:?}", coord);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
    }
}
