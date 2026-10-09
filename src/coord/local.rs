//! 3D unsigned integer coordinates.
//!
//! [`LocalCoord`] is an unsigned coordinate used to index into a local frame of reference
//! such as a `NodeDim`/`NodeLevel` child or voxel grid, or a `CoordBBox`.
//!
//! Without this, root/sparse `NodeLevel`s would not have the precision to index into all of their
//! children/voxels
//!
//! Because it is a pure index it does not convert to or from floating-point types. The module
//! defines, via macros, a large family of `From`/operator conversions between `LocalCoord` and the
//! common `glam` unsigned integer vector types, 3-element arrays, and tuples.
//!
//! LocalCoord has no value without a frame of reference (bbox or node level); the conversion
//! between a [`Coord`] and a `LocalCoord` is only well-defined relative to that frame, and is
//! provided by `CoordBBox`/`NodeDim` rather than by the type itself.

use glam::Vec3Swizzles;
use num::{Integer, NumCast};
use num_traits::ToPrimitive;
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

/// A 3D unsigned integer Local Coordinate for addressing a voxel grid. local to a bbox or node space
///
/// `LocalCoord` wraps a `glam` integer vector (`UIndexVec`) and implements `Deref`/
/// `DerefMut`, so the underlying `.x`, `.y`, `.z` fields and `glam` vector
/// operations are usable directly. The element type is `u32` by default and
/// `u64` under the `index64` feature.
///
/// ```
/// use voxgrid::LocalCoord;
///
/// let p = LocalCoord::new(1, 2, 3).offset_by(4, 5, 6);
/// assert_eq!(p, LocalCoord::new(5, 7, 9));
/// ```
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bytemuck", derive(Zeroable, Pod))]
#[repr(C)]
pub struct LocalCoord(pub UIndexVec);

impl DerefMut for LocalCoord {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for LocalCoord {
    type Target = UIndexVec;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
/// Methods for offsetting, clamping, and comparing `LocalCoord` values.
impl LocalCoord {
    /// The smallest representable coordinate: every component at the UIndex type's minimum.
    pub const MIN: Self = Self(UIndexVec {
        x: UIndex::MIN,
        y: UIndex::MIN,
        z: UIndex::MIN,
    });
    /// The largest representable coordinate: every component at the UIndex type's maximum.
    pub const MAX: Self = Self(UIndexVec {
        x: UIndex::MAX,
        y: UIndex::MAX,
        z: UIndex::MAX,
    });
    /// The grid origin, `(0, 0, 0)`.
    pub const ORIGIN: Self = Self(UIndexVec { x: 0, y: 0, z: 0 });

    /// Construct a [`LocalCoord`] from its three UIndex components.
    pub const fn new(x: UIndex, y: UIndex, z: UIndex) -> Self {
        Self(UIndexVec { x, y, z })
    }

    pub fn from_int<T: Integer + NumCast>(v: T) -> Self {
        #[cfg(feature = "index64")]
        let v = v.to_u64().unwrap();

        #[cfg(not(feature = "index64"))]
        let v = v.to_u32().unwrap();

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
    /// Views the coordinate's fields as a slice of UIndex values. Requires the `bytemuck` feature.
    pub fn as_slice(&self) -> &[UIndex] {
        bytemuck::cast_slice(self.bytes_of())
    }

    #[cfg(feature = "bytemuck")]
    /// Views the coordinate's fields as a mutable slice of UIndex values. Requires the `bytemuck` feature.
    pub fn as_slice_mut(&mut self) -> &mut [UIndex] {
        bytemuck::cast_slice_mut(self.bytes_of_mut())
    }

    /// Returns the components as a three-element UIndex array.
    pub fn as_array(&self) -> [UIndex; 3] {
        [self.x, self.y, self.z]
    }

    /// Offsets this coordinate in place by the per-axis deltas `(dx, dy, dz)`, returning `&mut self` for chaining.
    pub const fn offset(&mut self, dx: UIndex, dy: UIndex, dz: UIndex) -> &mut Self {
        self.0.x += dx;
        self.0.y += dy;
        self.0.z += dz;

        self
    }

    /// Offsets this coordinate in place by `n` on every axis, returning `&mut self` for chaining.
    pub const fn single_offset(&mut self, n: UIndex) -> &mut Self {
        self.offset(n, n, n)
    }

    /// Returns a new [`LocalCoord`] offset by the per-axis deltas `(dx, dy, dz)`.
    pub const fn offset_by(&self, dx: UIndex, dy: UIndex, dz: UIndex) -> Self {
        Self::new(self.0.x + dx, self.0.y + dy, self.0.z + dz)
    }

    /// Returns a new [`LocalCoord`] offset by `n` on every axis.
    pub const fn single_offset_by(&self, n: UIndex) -> Self {
        self.offset_by(n, n, n)
    }

    /// Returns a new [`LocalCoord`] with the per-component minimum of this and `other`.
    pub fn min_component(&self, other: &Self) -> LocalCoord {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Returns a new [`LocalCoord`] with the per-component maximum of this and `other`.
    pub fn max_component(&self, other: &Self) -> LocalCoord {
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

    /// Returns the axis UIndex of the smallest component: `0` for `x`, `1` for `y`, `2` for `z`.
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

    /// Returns the axis UIndex of the largest component: `0` for `x`, `1` for `y`, `2` for `z`.
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
    /// adds all coordinates together and returns as a usize
    pub fn sum(&self) -> usize {
        (self.x + self.y + self.z) as usize
    }

    /// Returns a coordinate with the `x` and `z` components swapped (a `glam` `zyx` swizzle).
    pub fn zyx(&self) -> LocalCoord {
        Self(self.0.zyx())
    }

    /// Returns `true` if this and `other` agree on at least one axis.
    pub fn any_match(&self, other: &LocalCoord) -> bool {
        self.x == other.x || self.y == other.y || self.z == other.z
    }

    /// Returns a coordinate that is `1` on each axis where this and `other` agree and `0` otherwise.
    pub fn match_axes(&self, other: &LocalCoord) -> LocalCoord {
        Self::new(
            (self.x == other.x) as UIndex,
            (self.y == other.y) as UIndex,
            (self.z == other.z) as UIndex,
        )
    }
}

impl PartialOrd for LocalCoord {
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

impl Ord for LocalCoord {
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

impl Display for LocalCoord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("({}, {}, {})", self.x, self.y, self.z))
    }
}

impl std::ops::AddAssign<LocalCoord> for LocalCoord {
    fn add_assign(&mut self, rhs: LocalCoord) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::Add<LocalCoord> for LocalCoord {
    type Output = LocalCoord;

    fn add(self, rhs: LocalCoord) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl std::ops::SubAssign<LocalCoord> for LocalCoord {
    fn sub_assign(&mut self, rhs: LocalCoord) {
        self.0 -= rhs.0
    }
}

impl std::ops::Sub<LocalCoord> for LocalCoord {
    type Output = LocalCoord;

    fn sub(self, rhs: LocalCoord) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl std::ops::Mul<LocalCoord> for LocalCoord {
    type Output = LocalCoord;

    fn mul(self, rhs: LocalCoord) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::MulAssign<LocalCoord> for LocalCoord {
    fn mul_assign(&mut self, rhs: LocalCoord) {
        self.0 *= rhs.0
    }
}

impl std::ops::Div<LocalCoord> for LocalCoord {
    type Output = LocalCoord;

    fn div(self, rhs: LocalCoord) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl std::ops::DivAssign<LocalCoord> for LocalCoord {
    fn div_assign(&mut self, rhs: LocalCoord) {
        self.0 /= rhs.0
    }
}

add_glam_int_type!(
    LocalCoord,
    UIndex,
    glam::U8Vec3,
    u8,
    glam::U16Vec3,
    u16,
    glam::UVec3,
    u32,
    glam::U64Vec3,
    u64
);

add_int_conversions!(LocalCoord, UIndex, u8, u16, u32, u64, usize);
add_int_ops!(LocalCoord, UIndex, to_u64, u8, u16, u32, u64, usize);

#[cfg(test)]
mod tests {
    use super::*;

    // ============== Construction Tests ==============
    #[test]
    fn test_local_coord_construction() {
        // Test new()
        let coord = LocalCoord::new(1, 2, 3);
        assert_eq!(coord.x, 1);
        assert_eq!(coord.y, 2);
        assert_eq!(coord.z, 3);

        // Test ORIGIN constant
        assert_eq!(LocalCoord::ORIGIN, LocalCoord::new(0, 0, 0));
        assert_eq!(LocalCoord::ORIGIN.x, 0);
        assert_eq!(LocalCoord::ORIGIN.y, 0);
        assert_eq!(LocalCoord::ORIGIN.z, 0);

        // Test MIN and MAX constants
        assert_eq!(LocalCoord::MIN.x, UIndex::MIN);
        assert_eq!(LocalCoord::MIN.y, UIndex::MIN);
        assert_eq!(LocalCoord::MIN.z, UIndex::MIN);
        assert_eq!(LocalCoord::MAX.x, UIndex::MAX);
        assert_eq!(LocalCoord::MAX.y, UIndex::MAX);
        assert_eq!(LocalCoord::MAX.z, UIndex::MAX);
    }

    #[test]
    fn test_local_coord_copy_clone() {
        let coord = LocalCoord::new(5, 10, 15);
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
    fn test_local_coord_default() {
        let default_local_coord = LocalCoord::default();
        assert_eq!(default_local_coord, LocalCoord::ORIGIN);
    }

    // ============== coordinate Arithmetic Tests ==============
    #[test]
    fn test_local_coord_addition() {
        let a = LocalCoord::new(1, 2, 3);
        let b = LocalCoord::new(4, 5, 6);
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
    fn test_local_coord_subtraction() {
        let a = LocalCoord::new(10, 20, 30);
        let b = LocalCoord::new(4, 5, 6);
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
    fn test_local_coord_shifts() {
        let coord = LocalCoord::new(8, 16, 32);

        // Test Shl
        let shifted_left = coord << 1u32;
        assert_eq!(shifted_left.x, 16);
        assert_eq!(shifted_left.y, 32);
        assert_eq!(shifted_left.z, 64);

        // Test Shr
        let shifted_right = coord >> 2u32;
        assert_eq!(shifted_right.x, 2);
        assert_eq!(shifted_right.y, 4);
        assert_eq!(shifted_right.z, 8);

        // Test ShlAssign
        let mut c = coord;
        c <<= 1u32;
        assert_eq!(c.x, 16);
        assert_eq!(c.y, 32);
        assert_eq!(c.z, 64);

        // Test ShrAssign
        let mut d = coord;
        d >>= 2u32;
        assert_eq!(d.x, 2);
        assert_eq!(d.y, 4);
        assert_eq!(d.z, 8);
    }

    // ============== coordinate Offset Tests ==============
    #[test]
    fn test_local_coord_offset() {
        let mut local_coord = LocalCoord::new(1, 2, 3);

        // Test offset() - in-place modification
        local_coord.offset(10, 20, 30);
        assert_eq!(local_coord.x, 11);
        assert_eq!(local_coord.y, 22);
        assert_eq!(local_coord.z, 33);

        // Test offset_by() - creates new LocalCoord
        let offset_local_coord = LocalCoord::new(5, 5, 5).offset_by(1, 2, 3);
        assert_eq!(offset_local_coord.x, 6);
        assert_eq!(offset_local_coord.y, 7);
        assert_eq!(offset_local_coord.z, 8);

        // Test single_offset() - in-place, same delta for all axes
        let mut s = LocalCoord::ORIGIN;
        s.single_offset(5);
        assert_eq!(s, LocalCoord::new(5, 5, 5));

        // Test single_offset_by() - creates new LocalCoord with same delta
        let single = LocalCoord::new(0, 0, 0).single_offset_by(10);
        assert_eq!(single, LocalCoord::new(10, 10, 10));
    }

    #[test]
    fn test_local_coord_single_offset_by() {
        let local_coord = LocalCoord::new(1, 1, 1);
        let result = local_coord.single_offset_by(5);
        assert_eq!(result, LocalCoord::new(6, 6, 6));
    }

    // ============== Component Operations Tests ==============
    #[test]
    fn test_local_coord_min_component() {
        let a = LocalCoord::new(5, 10, 15);
        let b = LocalCoord::new(3, 12, 8);

        let min = a.min_component(&b);
        assert_eq!(min.x, 3);
        assert_eq!(min.y, 10);
        assert_eq!(min.z, 8);

        // Same comparison
        let same = a.min_component(&a);
        assert_eq!(same, a);
    }

    #[test]
    fn test_local_coord_max_component() {
        let a = LocalCoord::new(5, 10, 15);
        let b = LocalCoord::new(3, 12, 8);

        let max = a.max_component(&b);
        assert_eq!(max.x, 5);
        assert_eq!(max.y, 12);
        assert_eq!(max.z, 15);

        // Same comparison
        let same = a.max_component(&a);
        assert_eq!(same, a);
    }

    #[test]
    fn test_local_coord_component_less_than() {
        let a = LocalCoord::new(5, 5, 5);
        let b = LocalCoord::new(3, 10, 3); // x < a.x, y > a.y, z < a.z

        // a is less than b only if ANY component is less
        assert!(a.component_lt(&b));
        assert!(b.component_lt(&a));

        let smaller = LocalCoord::ORIGIN;
        let larger = LocalCoord::new(1, 1, 1);

        assert!(!larger.component_lt(&smaller))
    }

    #[test]
    fn test_local_coord_min_max_idx() {
        let local_coord = LocalCoord::new(10, 5, 15);

        assert_eq!(local_coord.min_idx(), 1); // y=5 is smallest
        assert_eq!(local_coord.max_idx(), 2); // z=15 is largest

        let equal = LocalCoord::new(5, 5, 5);
        // When all equal, should return first i64
        assert_eq!(equal.min_idx(), 0);
        assert_eq!(equal.max_idx(), 0);

        let reverse = LocalCoord::new(15, 10, 5);
        assert_eq!(reverse.min_idx(), 2); // z=5 is smallest
        assert_eq!(reverse.max_idx(), 0); // x=15 is largest
    }

    #[test]
    fn test_local_coord_sum() {
        let local_coord = LocalCoord::new(10, 20, 30);
        assert_eq!(local_coord.sum(), 60);

        // Sum of origin
        assert_eq!(LocalCoord::ORIGIN.sum(), 0);
    }

    // ============== Byte/Array Conversion Tests ==============
    #[test]
    #[cfg(feature = "bytemuck")]
    fn test_local_coord_byte_conversion() {
        let local_coord = LocalCoord::new(1, 2, 3);

        // Test bytes_of
        let bytes = local_coord.bytes_of();
        assert_eq!(bytes.len(), std::mem::size_of::<LocalCoord>());

        // Test bytes_of_mut
        let mut mutable_local_coord = local_coord;
        let bytes_mut = mutable_local_coord.bytes_of_mut();
        assert_eq!(bytes_mut.len(), std::mem::size_of::<LocalCoord>());
    }

    #[test]
    #[cfg(feature = "bytemuck")]
    fn test_local_coord_as_slice() {
        let local_coord = LocalCoord::new(1, 2, 3);

        let slice = local_coord.as_slice();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);

        // Test as_slice_mut
        let mut mutable_local_coord = local_coord;
        let slice_mut = mutable_local_coord.as_slice_mut();
        slice_mut[0] = 100;
        slice_mut[1] = 200;
        slice_mut[2] = 300;
        assert_eq!(mutable_local_coord, LocalCoord::new(100, 200, 300));
    }

    #[test]
    fn test_local_coord_as_array() {
        let local_coord = LocalCoord::new(1, 2, 3);

        let array = local_coord.as_array();
        assert_eq!(array[0], 1);
        assert_eq!(array[1], 2);
        assert_eq!(array[2], 3);
    }

    // ============== Conversion Tests ==============
    #[test]
    fn test_local_coord_from_uindex_tuple() {
        let tuple: (UIndex, UIndex, UIndex) = (5, 10, 15);
        let local_coord: LocalCoord = tuple.into();

        assert_eq!(local_coord.x, 5);
        assert_eq!(local_coord.y, 10);
        assert_eq!(local_coord.z, 15);
    }

    #[test]
    fn test_local_coord_to_uindex_tuple() {
        let local_coord = LocalCoord::new(5, 10, 15);
        let tuple: (UIndex, UIndex, UIndex) = local_coord.into();

        assert_eq!(tuple.0, 5);
        assert_eq!(tuple.1, 10);
        assert_eq!(tuple.2, 15);
    }

    // ============== Comparison Tests ==============
    #[test]
    fn test_local_coord_ordering() {
        let a = LocalCoord::new(1, 2, 3);
        let b = LocalCoord::new(1, 2, 4);
        let c = LocalCoord::new(2, 2, 3);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&a), Ordering::Equal);
        assert_eq!(c.cmp(&a), Ordering::Greater);
    }

    #[test]
    fn test_local_coord_hash() {
        use std::collections::HashSet;

        let local_coord1 = LocalCoord::new(1, 2, 3);
        let local_coord2 = LocalCoord::new(1, 2, 3);
        let local_coord3 = LocalCoord::new(4, 5, 6);

        let mut set = HashSet::new();
        set.insert(local_coord1);
        set.insert(local_coord2);
        set.insert(local_coord3);

        // LocalCoord1 and local_coord2 should be considered equal for hashing
        assert_eq!(set.len(), 2);
        assert!(set.contains(&local_coord1));
        assert!(set.contains(&local_coord2));
        assert!(set.contains(&local_coord3));
    }

    // ============== PartialOrd Tests ==============
    #[test]
    fn test_local_coord_partial_cmp() {
        let a = LocalCoord::new(1, 2, 3);
        let b = LocalCoord::new(1, 2, 3);
        let c = LocalCoord::new(1, 2, 4);

        assert!(a == b);
        assert!(b >= a);
        assert!(c > b);
        assert!(a < c);
    }

    // ============== Operator Overloads ==============
    #[test]
    fn test_local_coord_bitwise_operations() {
        let local_coord = LocalCoord::new(15, 15, 15); // 1111 in binary

        // BitAnd: 15 & 5 (0101) = 5 (0101)
        let and_result = LocalCoord(*local_coord & 5);
        assert_eq!(and_result, LocalCoord::new(5, 5, 5));

        // BitOr: 15 | 1 (0001) = 15
        let or_result = LocalCoord(*local_coord | 1);
        assert_eq!(or_result, LocalCoord::new(15, 15, 15));

        // BitXor: 15 ^ 3 (0011) = 12 (1100)
        let xor_result = LocalCoord(*local_coord ^ 3);
        assert_eq!(xor_result, LocalCoord::new(12, 12, 12));

        // BitAndAssign
        let mut and_assign = local_coord;
        and_assign &= 5u32;
        assert_eq!(and_assign, LocalCoord::new(5, 5, 5));

        // BitOrAssign
        let mut or_assign = local_coord;
        or_assign |= 1u32;
        assert_eq!(or_assign, LocalCoord::new(15, 15, 15));

        // BitXorAssign
        let mut xor_assign = local_coord;
        xor_assign ^= 3u32;
        assert_eq!(xor_assign, LocalCoord::new(12, 12, 12));
    }

    // ============== Debug and String Representation ==============
    #[test]
    fn test_local_coord_display() {
        let local_coord = LocalCoord::new(1, 2, 3);
        let debug_str = format!("{:?}", local_coord);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
    }
}
