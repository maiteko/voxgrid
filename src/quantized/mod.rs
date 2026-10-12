//! Hashable, integer-backed representation of floating-point values.
//!
//! The types defined here are very opinionated to address a couple of
//! specific needs in dealing with Signed Distance Fields
//!
//! 1. Ordinary floating-point types do not implement `Hash`/`Eq` with the
//!    and rounding errors cause naive hashing to reduce reasonable matches.
//!    `Quantized` stores an integer code. values are rounded to the nearest
//!    multiple of `1 / 1024`, making hashing more consistent. This helps with
//!    tiling values values.
//! 2. Having clear MIN/MAX sentinels simplifies logic around narrow band
//!    Signed Distance Fields, while maintaining the ability to perform CSG
//!    operations
//!
//! NaN values are rejected with an Err. +/- infinity are converted to i32::MIN/MAX.
use anyhow::{Result, bail};
use half::f16;
use num_traits::{Float, PrimInt};
use num_traits::{FromPrimitive, NumCast, ToPrimitive};
use paste::paste;
use std::fmt::Display;
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Defines how a floating-point type is encoded in and decoded from
/// an integer storage type.
///
/// Implementations determine the supported distance range, sentinel
/// behavior, and conversion details for a particular float/storage pair.
pub trait QuantizedRepr<F, I>
where
    F: Float,
    I: PrimInt,
{
    /// Encodes a float value.
    ///
    /// Implementations should reject non-finite values and must ensure
    /// ordinary encoded values do not collide with any reserved sentinel
    /// codes.
    fn encode(value: F) -> Result<I>;

    /// Decodes an integer code to a float value.
    fn decode(value: I) -> F;

    /// Clamps a storage code to the reserved MIN/MAX sentinel when it is
    /// outside the representable distance range, so that out-of-range values
    /// collapse to a single tileable value instead of a distinct code each.
    fn clamp_to_sentinel(value: I) -> I;
}

/// A floating-point value stored as a hashable integer code.
///
/// `F` is the float type exposed by the representation, and `I` is its
/// integer storage type. The actual representation policy is supplied by
/// a [`QuantizedRepr`] implementation.
///
/// The default resolution in the implementations below is `1 / 1024`.
pub struct Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    value: I,
    _float: PhantomData<F>,
}

impl<F, I> Display for Quantized<F, I>
where
    F: Float + Display,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::decode(self.value))
    }
}

impl<F, I> std::fmt::Debug for Quantized<F, I>
where
    F: Float + std::fmt::Debug,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", Self::decode(self.value))
    }
}

impl<F, I: Copy> Copy for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
}

impl<F, I: Copy> Clone for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, I: PartialEq> PartialEq for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F, I: Eq> Eq for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
}

impl<F, I: Hash> Hash for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<F, I: PartialOrd> PartialOrd for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F, I: Ord> Ord for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<F, I: Default> Default for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn default() -> Self {
        Self {
            value: I::default(),
            _float: PhantomData,
        }
    }
}

impl<F, I> Quantized<F, I>
where
    F: Float,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    /// Encodes `value` using the specified representation.
    pub fn from_float(value: F) -> Result<Self> {
        Ok(Self {
            value: Self::encode(value)?,
            _float: PhantomData,
        })
    }

    /// Decodes this value using the specified representation.
    pub fn into_float(self) -> F {
        Self::decode(self.value)
    }

    /// Returns the stored integer code.
    pub fn raw(&self) -> &I {
        &self.value
    }
}

/// The number of integer steps per unit of the represented float.
const SCALE: f64 = 1024.0;

/// Implements a float/integer representation pair using an explicit
/// supported-distance cutoff.
///
/// The cutoff leaves room below the integer endpoint for scaling and
/// rounding, so normal values do not encode to the reserved MIN/MAX codes.
macro_rules! impl_quantized_repr {
    ($float:ty, $storage:ty, $limit:expr, $scaled_limit:expr) => {
        impl QuantizedRepr<$float, $storage> for Quantized<$float, $storage> {
            fn encode(value: $float) -> Result<$storage> {
                if value.is_nan() {
                    bail!("NaN not supported");
                }

                let quantized = match value as f64 {
                    v if v <= -$limit => return Ok(<$storage>::MIN),
                    v if v >= $limit => return Ok(<$storage>::MAX),
                    v => (v * SCALE).round() as $storage,
                };

                Ok(quantized)
            }

            fn decode(value: $storage) -> $float {
                match value {
                    v if v == <$storage>::MIN => <$float>::MIN,
                    v if v == <$storage>::MAX => <$float>::MAX,
                    v => (v as f64 / SCALE) as $float,
                }
            }

            /// Clamps a code to the reserved sentinel when it is outside the
            /// representable distance range, so out-of-range values tile.
            fn clamp_to_sentinel(v: $storage) -> $storage {
                const SCALED_LIMIT: $storage = $scaled_limit;
                match v {
                    v if v <= -SCALED_LIMIT => <$storage>::MIN,
                    v if v >= SCALED_LIMIT => <$storage>::MAX,
                    _ => v,
                }
            }
        }
    };
}

// These limits are deliberately conservative. The i32/f32 and i64/f64
// pairings have much more storage range than the f16/i16 pairing.
impl_quantized_repr!(f32, i32, 2_097_151.0, 2_097_151 * 1024);
impl_quantized_repr!(
    f64,
    i64,
    9_007_199_254_740_991.0,
    9_007_199_254_740_991 * 1024
);

/// Representation for `f16` values stored as `i16`.
///
/// The input is widened before scaling, avoiding intermediate `f16`
/// overflow. The supported distance cutoff is determined by `i16` storage.
impl QuantizedRepr<f16, i16> for Quantized<f16, i16> {
    fn encode(value: f16) -> Result<i16> {
        let value = match value.to_f32() {
            v if v <= -31.0 => return Ok(i16::MIN),
            v if v >= 31.0 => return Ok(i16::MAX),
            v if v.is_nan() => bail!("Nan not supported"),
            v => v,
        };

        let scaled = (value as f64 * SCALE).round();
        Ok(scaled as i16)
    }

    fn decode(value: i16) -> f16 {
        match value {
            v if v == i16::MIN => f16::MIN,
            v if v == i16::MAX => f16::MAX,
            v => f16::from_f64(v as f64 / SCALE),
        }
    }

    /// Clamps a code to the reserved sentinel when it is outside the
    /// representable distance range, so out-of-range values tile.
    fn clamp_to_sentinel(v: i16) -> i16 {
        const SCALED_LIMIT: i16 = 31 * 1024;
        match v {
            v if v <= -SCALED_LIMIT => i16::MIN,
            v if v >= SCALED_LIMIT => i16::MAX,
            _ => v,
        }
    }
}

impl<F, I> NumCast for Quantized<F, I>
where
    F: NumCast + Float,
    I: Copy + PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let float = <F as NumCast>::from(n)?;
        let value = <Self as QuantizedRepr<F, I>>::encode(float).ok()?;

        Some(Self {
            value,
            _float: PhantomData,
        })
    }
}

impl<F, I> ToPrimitive for Quantized<F, I>
where
    F: ToPrimitive + Float,
    I: Copy + PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn to_i64(&self) -> Option<i64> {
        <Self as QuantizedRepr<F, I>>::decode(self.value).to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        <Self as QuantizedRepr<F, I>>::decode(self.value).to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        <Self as QuantizedRepr<F, I>>::decode(self.value).to_f64()
    }
}

impl<F, I> FromPrimitive for Quantized<F, I>
where
    F: FromPrimitive + Float,
    I: Copy + PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn from_i64(n: i64) -> Option<Self> {
        let float = F::from_i64(n)?;
        let value = <Self as QuantizedRepr<F, I>>::encode(float).ok()?;

        Some(Self {
            value,
            _float: PhantomData,
        })
    }

    fn from_u64(n: u64) -> Option<Self> {
        let float = F::from_u64(n)?;
        let value = <Self as QuantizedRepr<F, I>>::encode(float).ok()?;

        Some(Self {
            value,
            _float: PhantomData,
        })
    }
}

/// Re-encodes a computed float value into this representation.
///
/// This is the "encode" half of the decode/compute/encode pattern used by the
/// arithmetic and math operators below.
impl<F, I> Quantized<F, I>
where
    I: Copy + PrimInt,
    F: Float,
    Self: QuantizedRepr<F, I>,
{
    /// Wraps a computed `F` value by encoding it into the storage type.
    fn encode_value(value: F) -> Result<Self> {
        <Self as QuantizedRepr<F, I>>::encode(value).map(|value| Self {
            value,
            _float: PhantomData,
        })
    }
}

/// Helpers that decode a value, run a `f64` operation, and re-encode the result.
///
/// The computation is performed in `f64` and converted back to `F` before the
/// final encode, so the encode can still fail when the result is not
/// representable in the storage range.
impl<F, I> Quantized<F, I>
where
    F: FromPrimitive + ToPrimitive + Float,
    I: Copy + PrimInt,
    Self: QuantizedRepr<F, I>,
{
    /// Decodes this value, applies a unary `f64` operation, and re-encodes it.
    fn map_float(&self, f: impl FnOnce(f64) -> f64) -> Self {
        let decoded = <Self as QuantizedRepr<F, I>>::decode(self.value);
        let value = decoded.to_f64().unwrap();
        let result = F::from_f64(f(value)).unwrap();
        Self::encode_value(result).unwrap()
    }

    /// Decodes two values, applies a binary `f64` operation, and re-encodes it.
    fn map_pair(&self, other: &Self, f: impl FnOnce(f64, f64) -> f64) -> Self {
        let a = <Self as QuantizedRepr<F, I>>::decode(self.value)
            .to_f64()
            .unwrap();
        let b = <Self as QuantizedRepr<F, I>>::decode(other.value)
            .to_f64()
            .unwrap();
        let result = F::from_f64(f(a, b)).unwrap();

        Self::encode_value(result).expect("Shouldn't be Nan")
    }

    /// Hypotenuse of the two decoded values.
    pub fn hypot(&self, other: &Self) -> Self {
        self.map_pair(other, f64::hypot)
    }

    /// `atan2` of the two decoded values.
    pub fn atan2(&self, other: &Self) -> Self {
        self.map_pair(other, f64::atan2)
    }

    /// Raises the decoded value to the power of another decoded value.
    pub fn pow(&self, other: &Self) -> Self {
        self.map_pair(other, f64::powf)
    }

    /// Raises the decoded value to an integer power.
    pub fn powi(&self, n: i32) -> Self {
        self.map_float(|v| v.powi(n))
    }
}

/// Standard unary math operations. Each decodes the value, computes the
/// matching `f64` function, and re-encodes the result. Non-representable
/// results (NaN, infinity, or outside the storage range) are reported as errors.
macro_rules! impl_quantized_unary_math {
    ($($method:ident),*) => {
        impl<F, I> Quantized<F, I>
        where
            F: FromPrimitive + ToPrimitive + Float,
            I: Copy + PrimInt,
            Self: QuantizedRepr<F, I>,
          {
            $(
                /// Applies the corresponding `f64` method to the decoded value.
                pub fn $method(&self) -> Self {
                    self.map_float(f64::$method)
                }
            )*
        }
    };
}

impl_quantized_unary_math!(
    abs, signum, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh,
    floor, ceil, round, trunc, ln, log2, log10, exp, exp2
);

/// For implementing operators that can be accomplished without a conversion
macro_rules! impl_quantized_add_sub {
    ($trait:ident, $method:ident, $op:tt) => {
        paste! {
            impl<F, I> $trait for Quantized<F, I>
            where
                I: Copy + $trait<Output = I> + PrimInt,
                F: Float,
                Self: QuantizedRepr<F, I>,
            {
                type Output = Self;

                fn $method(self, rhs: Self) -> Self {
                    Self {
                        value: Self::clamp_to_sentinel(self.value $op rhs.value),
                        _float: Default::default(),
                    }
                }
            }

            impl<F, I> [<$trait Assign>] for Quantized<F, I>
            where
                I: Copy + [<$trait>]<Output = I> + PrimInt,
                F: Float,
                Self: QuantizedRepr<F, I>,
            {
                fn [<$method _assign>](&mut self, rhs: Self) {
                    self.value = Self::clamp_to_sentinel(self.value $op rhs.value);
                }
            }
        }
    };
}

impl_quantized_add_sub!(Add, add, +);
impl_quantized_add_sub!(Sub, sub, -);

/// For implementing operators that can be accomplished without a conversion
macro_rules! impl_quantized_mul_div {
    ($trait:ident, $method:ident, $op:tt) => {
        paste! {
            impl<F, I> $trait for Quantized<F, I>
            where
                F: Copy + $trait<Output = F> + Float,
                I: Copy + PrimInt,
                Self: QuantizedRepr<F, I>,
            {
                type Output = Self;

                fn $method(self, rhs: Self) -> Self {
                    Self::from_float(
                        Self::decode(self.value) $op Self::decode(rhs.value)
                    ).unwrap()
                }
            }

            impl<F, I> [<$trait Assign>] for Quantized<F, I>
            where
                F: [<$trait>]<Output = F> + Float,
                I: Copy + PrimInt,
                Self: QuantizedRepr<F, I>,
            {
                    fn [<$method _assign>](&mut self, rhs: Self) {
                        self.value = Self::encode(
                            Self::decode(self.value) $op Self::decode(rhs.value)
                        ).unwrap();
                    }
            }
        }
    };
}
impl_quantized_mul_div!(Mul, mul, *);
impl_quantized_mul_div!(Div, div, /);

/// Unary negation. Decodes the operand, negates it, and re-encodes the result.
impl<F, I> Neg for Quantized<F, I>
where
    I: PrimInt + Copy + Neg<Output = I>,
    F: Float,
{
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            _float: Default::default(),
        }
    }
}

/// Quantized `f16` values stored in `i16`.
pub type Quant16 = Quantized<f16, i16>;

/// Quantized `f32` values stored in `i32`.
pub type Quant32 = Quantized<f32, i32>;

/// Quantized `f64` values stored in `i64`.
pub type Quant64 = Quantized<f64, i64>;

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use half::f16;

    #[test]
    fn arithmetic_operators() -> Result<()> {
        let a = Quant32::from_float(4.0_f32)?;
        let b = Quant32::from_float(2.0_f32)?;

        assert_eq!((a + b).into_float(), 6.0_f32);
        assert_eq!((a - b).into_float(), 2.0_f32);
        assert_eq!((a * b).into_float(), 8.0_f32);
        assert_eq!((a / b).into_float(), 2.0_f32);
        assert_eq!((-b).into_float(), -2.0_f32);

        let mut x = Quant32::from_float(3.0_f32)?;
        x += b;
        assert_eq!(x.into_float(), 5.0_f32);

        let mut y = Quant32::from_float(8.0_f32)?;
        y *= b;
        assert_eq!(y.into_float(), 16.0_f32);

        Ok(())
    }

    #[test]
    fn unary_math() -> Result<()> {
        assert_eq!(Quant32::from_float(9.0_f32)?.sqrt().into_float(), 3.0_f32);
        assert_eq!(Quant32::from_float(-3.0_f32)?.abs().into_float(), 3.0_f32);
        assert_eq!(Quant32::from_float(5.0_f32)?.signum().into_float(), 1.0_f32);

        Ok(())
    }

    #[test]
    fn binary_math() -> Result<()> {
        let a = Quant32::from_float(3.0_f32)?;
        let b = Quant32::from_float(4.0_f32)?;

        assert_eq!(a.hypot(&b).into_float(), 5.0_f32);
        assert_eq!(a.pow(&b).into_float(), 81.0_f32);

        Ok(())
    }

    #[test]
    fn quant16_operators() -> Result<()> {
        let a = Quant16::from_float(f16::from_f32(1.5))?;
        let b = Quant16::from_float(f16::from_f32(2.0))?;

        assert_eq!((a + b).into_float(), f16::from_f32(3.5));

        let mut x = Quant16::from_float(f16::from_f32(3.0))?;
        x += b;
        assert_eq!(x.into_float(), f16::from_f32(5.0));

        Ok(())
    }
}

#[cfg(feature = "voxelis")]
use voxelis::VoxelTrait;
#[cfg(feature = "voxelis")]
impl<F, I> VoxelTrait for Quantized<F, I>
where
    F: std::fmt::Debug + std::fmt::Display + Float,
    I: Default + Hash + PrimInt,
    Self: std::fmt::Debug + std::fmt::Display,
{
}
