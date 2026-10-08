//! 3D integer LocalCoordinates.
//!
//! [`LocalCoord`] is an unsigned coordinate used to index into a local frame of reference
//! such as with a `NodeLevel` or `CoordBBox`.
//!
//! Without this, root/sparse `NodeLevel`s would not have the precision to index into all of their
//! children/voxels
//!
//! The module also defines the [`LocalCoordRound`] rounding trait and, via macros, a
//! large family of `From`/operator conversions between `LocalCoord` and the common
//! `glam` unsigned integer vector types, 3-element arrays, and tuples.
//!
//! LocalCoord has no value without a frame of reference (bbox or node level), and does not convert
//! to/from floating types, it is purely an index

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

/// A 3D integer LocalCoordinate for addressing a voxel grid.
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
    /// The smallest representable LocalCoordinate: every component at the UIndex type's minimum.
    pub const MIN: Self = Self(UIndexVec {
        x: UIndex::MIN,
        y: UIndex::MIN,
        z: UIndex::MIN,
    });
    /// The largest representable LocalCoordinate: every component at the UIndex type's maximum.
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
    /// Returns a raw byte slice over the LocalCoordinate's fields. Requires the `bytemuck` feature.
    pub fn bytes_of(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    #[cfg(feature = "bytemuck")]
    /// Returns a mutable raw byte slice over the LocalCoordinate's fields. Requires the `bytemuck` feature.
    pub fn bytes_of_mut(&mut self) -> &mut [u8] {
        bytemuck::bytes_of_mut(self)
    }

    #[cfg(feature = "bytemuck")]
    /// Views the LocalCoordinate's fields as a slice of UIndex values. Requires the `bytemuck` feature.
    pub fn as_slice(&self) -> &[UIndex] {
        bytemuck::cast_slice(self.bytes_of())
    }

    #[cfg(feature = "bytemuck")]
    /// Views the LocalCoordinate's fields as a mutable slice of UIndex values. Requires the `bytemuck` feature.
    pub fn as_slice_mut(&mut self) -> &mut [UIndex] {
        bytemuck::cast_slice_mut(self.bytes_of_mut())
    }

    /// Returns the components as a three-element UIndex array.
    pub fn as_array(&self) -> [UIndex; 3] {
        [self.x, self.y, self.z]
    }

    /// Offsets this LocalCoordinate in place by the per-axis deltas `(dx, dy, dz)`, returning `&mut self` for chaining.
    pub const fn offset(&mut self, dx: UIndex, dy: UIndex, dz: UIndex) -> &mut Self {
        self.0.x += dx;
        self.0.y += dy;
        self.0.z += dz;

        self
    }

    /// Offsets this LocalCoordinate in place by `n` on every axis, returning `&mut self` for chaining.
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
    /// adds all LocalCoordinates together and returns as a usize
    pub fn sum(&self) -> usize {
        (self.x + self.y + self.z) as usize
    }

    /// Returns a LocalCoordinate with the `x` and `z` components swapped (a `glam` `zyx` swizzle).
    pub fn zyx(&self) -> LocalCoord {
        Self(self.0.zyx())
    }

    /// Returns `true` if this and `other` agree on at least one axis.
    pub fn any_match(&self, other: &LocalCoord) -> bool {
        self.x == other.x || self.y == other.y || self.z == other.z
    }

    /// Returns a LocalCoordinate that is `1` on each axis where this and `other` agree and `0` otherwise.
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

super::add_glam_int_type!(
    LocalCoord,
    glam::U8Vec3,
    u8,
    glam::U16Vec3,
    u16,
    glam::UVec3,
    u32,
    glam::U64Vec3,
    u64
);

add_int_conversions!(LocalCoord, u8, u16, u32, u64, usize);
add_int_ops!(LocalCoord, u8, u16, u32, u64, usize);

#[cfg(test)]
mod tests {
    use super::*;

    // ============== Construction Tests ==============
    #[test]
    fn test_LocalCoord_construction() {
        // Test new()
        let LocalCoord = LocalCoord::new(1, 2, 3);
        assert_eq!(LocalCoord.x, 1);
        assert_eq!(LocalCoord.y, 2);
        assert_eq!(LocalCoord.z, 3);

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
    fn test_LocalCoord_copy_clone() {
        let LocalCoord = LocalCoord::new(5, 10, 15);
        let copied = LocalCoord; // Copy trait
        let cloned = LocalCoord.clone(); // Clone trait

        assert_eq!(copied, LocalCoord);
        assert_eq!(cloned, LocalCoord);

        // Modifying copy shouldn't affect original
        let mut copy = LocalCoord;
        copy.x = 100;
        assert_eq!(LocalCoord.x, 5);
        assert_eq!(copy.x, 100);
    }

    #[test]
    fn test_LocalCoord_default() {
        let default_LocalCoord = LocalCoord::default();
        assert_eq!(default_LocalCoord, LocalCoord::ORIGIN);
    }

    // ============== LocalCoordinate Arithmetic Tests ==============
    #[test]
    fn test_LocalCoord_addition() {
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
    fn test_LocalCoord_subtraction() {
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
    fn test_LocalCoord_negation() {
        let LocalCoord = LocalCoord::new(5, 10, -15);
        let neg = -LocalCoord;

        assert_eq!(neg.x, -5);
        assert_eq!(neg.y, -10);
        assert_eq!(neg.z, 15);

        // Negation of origin
        let neg_origin = -LocalCoord::ORIGIN;
        assert_eq!(neg_origin, LocalCoord::ORIGIN);
    }

    #[test]
    fn test_LocalCoord_shifts() {
        let LocalCoord = LocalCoord::new(8, 16, 32);

        // Test Shl
        let shifted_left: UIndexVec = *LocalCoord << 1;
        assert_eq!(shifted_left.x, 16);
        assert_eq!(shifted_left.y, 32);
        assert_eq!(shifted_left.z, 64);

        // Test Shr
        let shifted_right: UIndexVec = *LocalCoord >> 2;
        assert_eq!(shifted_right.x, 2);
        assert_eq!(shifted_right.y, 4);
        assert_eq!(shifted_right.z, 8);

        // Test ShlAssign
        let mut c = LocalCoord;
        c <<= 1;
        assert_eq!(c.x, 16);
        assert_eq!(c.y, 32);
        assert_eq!(c.z, 64);

        // Test ShrAssign
        let mut d = LocalCoord;
        d >>= 2;
        assert_eq!(d.x, 2);
        assert_eq!(d.y, 4);
        assert_eq!(d.z, 8);
    }

    // ============== LocalCoordinate Offset Tests ==============
    #[test]
    fn test_LocalCoord_offset() {
        let mut LocalCoord = LocalCoord::new(1, 2, 3);

        // Test offset() - in-place modification
        LocalCoord.offset(10, 20, 30);
        assert_eq!(LocalCoord.x, 11);
        assert_eq!(LocalCoord.y, 22);
        assert_eq!(LocalCoord.z, 33);

        // Test offset_by() - creates new LocalCoord
        let offset_LocalCoord = LocalCoord::new(5, 5, 5).offset_by(1, 2, 3);
        assert_eq!(offset_LocalCoord.x, 6);
        assert_eq!(offset_LocalCoord.y, 7);
        assert_eq!(offset_LocalCoord.z, 8);

        // Test single_offset() - in-place, same delta for all axes
        let mut s = LocalCoord::ORIGIN;
        s.single_offset(5);
        assert_eq!(s, LocalCoord::new(5, 5, 5));

        // Test single_offset_by() - creates new LocalCoord with same delta
        let single = LocalCoord::new(0, 0, 0).single_offset_by(10);
        assert_eq!(single, LocalCoord::new(10, 10, 10));
    }

    #[test]
    fn test_LocalCoord_single_offset_by() {
        let LocalCoord = LocalCoord::new(1, 1, 1);
        let result = LocalCoord.single_offset_by(5);
        assert_eq!(result, LocalCoord::new(6, 6, 6));
    }

    // ============== Component Operations Tests ==============
    #[test]
    fn test_LocalCoord_min_component() {
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
    fn test_LocalCoord_max_component() {
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
    fn test_LocalCoord_component_less_than() {
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
    fn test_LocalCoord_min_max_idx() {
        let LocalCoord = LocalCoord::new(10, 5, 15);

        assert_eq!(LocalCoord.min_idx(), 1); // y=5 is smallest
        assert_eq!(LocalCoord.max_idx(), 2); // z=15 is largest

        let equal = LocalCoord::new(5, 5, 5);
        // When all equal, should return first i64
        assert_eq!(equal.min_idx(), 0);
        assert_eq!(equal.max_idx(), 0);

        let reverse = LocalCoord::new(15, 10, 5);
        assert_eq!(reverse.min_idx(), 2); // z=5 is smallest
        assert_eq!(reverse.max_idx(), 0); // x=15 is largest
    }

    #[test]
    fn test_LocalCoord_abs() {
        let LocalCoord = LocalCoord::new(-5, 10, -15);
        let abs = LocalCoord.abs();

        assert_eq!(abs.x, 5);
        assert_eq!(abs.y, 10);
        assert_eq!(abs.z, 15);

        // Abs of non-negative values
        let positive = LocalCoord::new(5, 10, 15);
        assert_eq!(positive.abs(), positive);

        // Abs of origin
        assert_eq!(LocalCoord::ORIGIN.abs(), LocalCoord::ORIGIN);
    }

    #[test]
    fn test_LocalCoord_sum() {
        let LocalCoord = LocalCoord::new(10, 20, 30);
        assert_eq!(LocalCoord.sum(), 60);

        let with_negative = LocalCoord::new(-10, 20, -5);
        assert_eq!(with_negative.sum(), 5);

        // Sum of origin
        assert_eq!(LocalCoord::ORIGIN.sum(), 0);
    }

    // ============== Byte/Array Conversion Tests ==============
    #[test]
    #[cfg(feature = "bytemuck")]
    fn test_LocalCoord_byte_conversion() {
        let LocalCoord = LocalCoord::new(1, 2, 3);

        // Test bytes_of
        let bytes = LocalCoord.bytes_of();
        assert_eq!(bytes.len(), std::mem::size_of::<LocalCoord>());

        // Test bytes_of_mut
        let mut mutable_LocalCoord = LocalCoord;
        let bytes_mut = mutable_LocalCoord.bytes_of_mut();
        assert_eq!(bytes_mut.len(), std::mem::size_of::<LocalCoord>());
    }

    #[test]
    #[cfg(feature = "bytemuck")]
    fn test_LocalCoord_as_slice() {
        let LocalCoord = LocalCoord::new(1, 2, 3);

        let slice = LocalCoord.as_slice();
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);

        // Test as_slice_mut
        let mut mutable_LocalCoord = LocalCoord;
        let slice_mut = mutable_LocalCoord.as_slice_mut();
        slice_mut[0] = 100;
        slice_mut[1] = 200;
        slice_mut[2] = 300;
        assert_eq!(mutable_LocalCoord, LocalCoord::new(100, 200, 300));
    }

    #[test]
    fn test_LocalCoord_as_array() {
        let LocalCoord = LocalCoord::new(1, 2, 3);

        let array = LocalCoord.as_array();
        assert_eq!(array[0], 1);
        assert_eq!(array[1], 2);
        assert_eq!(array[2], 3);

        // Test unsigned array
        let unsigned = LocalCoord::new(1, 2, 3).as_array_unsigned();
        assert_eq!(unsigned[0], 1);
        assert_eq!(unsigned[1], 2);
        assert_eq!(unsigned[2], 3);
    }

    // ============== Conversion Tests ==============
    #[test]
    fn test_LocalCoord_from_UIndex_tuple() {
        let tuple: (UIndex, UIndex, UIndex) = (5, 10, 15);
        let LocalCoord: LocalCoord = tuple.into();

        assert_eq!(LocalCoord.x, 5);
        assert_eq!(LocalCoord.y, 10);
        assert_eq!(LocalCoord.z, 15);
    }

    #[test]
    fn test_LocalCoord_to_UIndex_tuple() {
        let LocalCoord = LocalCoord::new(5, 10, 15);
        let tuple: (UIndex, UIndex, UIndex) = LocalCoord.into();

        assert_eq!(tuple.0, 5);
        assert_eq!(tuple.1, 10);
        assert_eq!(tuple.2, 15);
    }

    // ============== Comparison Tests ==============
    #[test]
    fn test_LocalCoord_ordering() {
        let a = LocalCoord::new(1, 2, 3);
        let b = LocalCoord::new(1, 2, 4);
        let c = LocalCoord::new(2, 2, 3);

        assert_eq!(a.cmp(&b), Ordering::Less);
        assert_eq!(b.cmp(&a), Ordering::Greater);
        assert_eq!(a.cmp(&a), Ordering::Equal);
        assert_eq!(c.cmp(&a), Ordering::Greater);
    }

    #[test]
    fn test_LocalCoord_hash() {
        use std::collections::HashSet;

        let LocalCoord1 = LocalCoord::new(1, 2, 3);
        let LocalCoord2 = LocalCoord::new(1, 2, 3);
        let LocalCoord3 = LocalCoord::new(4, 5, 6);

        let mut set = HashSet::new();
        set.insert(LocalCoord1);
        set.insert(LocalCoord2);
        set.insert(LocalCoord3);

        // LocalCoord1 and LocalCoord2 should be considered equal for hashing
        assert_eq!(set.len(), 2);
        assert!(set.contains(&LocalCoord1));
        assert!(set.contains(&LocalCoord2));
        assert!(set.contains(&LocalCoord3));
    }

    // ============== PartialOrd Tests ==============
    #[test]
    fn test_LocalCoord_partial_cmp() {
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
    fn test_LocalCoord_bitwise_operations() {
        let LocalCoord = LocalCoord::new(15, 15, 15); // 1111 in binary

        // BitAnd: 15 & 5 (0101) = 5 (0101)
        let and_result = LocalCoord(*LocalCoord & 5);
        assert_eq!(and_result, LocalCoord::new(5, 5, 5));

        // BitOr: 15 | 1 (0001) = 15
        let or_result = LocalCoord(*LocalCoord | 1);
        assert_eq!(or_result, LocalCoord::new(15, 15, 15));

        // BitXor: 15 ^ 3 (0011) = 12 (1100)
        let xor_result = LocalCoord(*LocalCoord ^ 3);
        assert_eq!(xor_result, LocalCoord::new(12, 12, 12));

        // BitAndAssign
        let mut and_assign = LocalCoord;
        and_assign &= 5;
        assert_eq!(and_assign, LocalCoord::new(5, 5, 5));

        // BitOrAssign
        let mut or_assign = LocalCoord;
        or_assign |= 1;
        assert_eq!(or_assign, LocalCoord::new(15, 15, 15));

        // BitXorAssign
        let mut xor_assign = LocalCoord;
        xor_assign ^= 3;
        assert_eq!(xor_assign, LocalCoord::new(12, 12, 12));
    }

    // ============== Debug and String Representation ==============
    #[test]
    fn test_LocalCoord_display() {
        let LocalCoord = LocalCoord::new(1, 2, 3);
        let debug_str = format!("{:?}", LocalCoord);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
        assert!(debug_str.contains("3"));
    }
}
