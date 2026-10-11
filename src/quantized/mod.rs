//! Hashable, integer-backed representation of floating-point values.
//!
//! Ordinary floating-point types do not implement `Hash`/`Eq` with the
//! semantics needed here. `Quantized` stores an integer code instead.
//! Finite values are rounded to the nearest multiple of `1 / 1024`.
//!
//! The MIN/MAX storage codes are reserved as sentinels and decode to the
//! float type's finite MIN/MAX values. Finite values beyond the supported
//! distance range also encode as those sentinels. NaN and infinities are
//! rejected.
//!
//! Warning: unstable, incomplete, and untested

use anyhow::{Context, Result, bail};
use half::f16;
use num_traits::{FromPrimitive, NumCast, ToPrimitive};
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
pub trait QuantizedRepr<F, I> {
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
pub struct Quantized<F, I> {
    value: I,
    _float: PhantomData<fn() -> F>,
}

impl<F, I: Copy> Copy for Quantized<F, I> {}

impl<F, I: Copy> Clone for Quantized<F, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, I: PartialEq> PartialEq for Quantized<F, I> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F, I: Eq> Eq for Quantized<F, I> {}

impl<F, I: Hash> Hash for Quantized<F, I> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<F, I: PartialOrd> PartialOrd for Quantized<F, I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F, I: Ord> Ord for Quantized<F, I> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<F, I: Default> Default for Quantized<F, I> {
    fn default() -> Self {
        Self {
            value: I::default(),
            _float: PhantomData,
        }
    }
}

impl<F, I> Quantized<F, I> {
    /// Encodes `value` using the specified representation.
    pub fn from_float<R>(value: F) -> Result<Self>
    where
        R: QuantizedRepr<F, I>,
    {
        Ok(Self {
            value: R::encode(value)?,
            _float: PhantomData,
        })
    }

    /// Decodes this value using the specified representation.
    pub fn into_float<R>(self) -> F
    where
        R: QuantizedRepr<F, I>,
    {
        R::decode(self.value)
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
        impl QuantizedRepr<$float, $storage> for () {
            fn encode(value: $float) -> Result<$storage> {
                if value.is_nan() || value.is_infinite() {
                    bail!("NaN and infinity are not supported");
                }

                // Preserve the float extrema as the SDF sentinel values.
                if value == <$float>::MIN {
                    return Ok(<$storage>::MIN);
                }
                if value == <$float>::MAX {
                    return Ok(<$storage>::MAX);
                }

                let value64 = value as f64;

                // Clamp values outside the distance range to the corresponding
                // sentinel so they tile, instead of erroring on the storage
                // boundary.
                let code = match value64 {
                    v if v <= -$limit => <$storage>::MIN,
                    v if v >= $limit => <$storage>::MAX,
                    _ => (value64 * SCALE).round() as $storage,
                };

                Ok(code)
            }

            fn decode(value: $storage) -> $float {
                if value == <$storage>::MIN {
                    return <$float>::MIN;
                }
                if value == <$storage>::MAX {
                    return <$float>::MAX;
                }

                (value as f64 / SCALE) as $float
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
impl QuantizedRepr<f16, i16> for () {
    fn encode(value: f16) -> Result<i16> {
        let value32 = match value {
            v if v == f16::MAX => return Ok(i16::MAX),
            v if v == f16::MIN => return Ok(i16::MIN),
            v if v.is_nan() || v.is_infinite() => bail!("Nan and infinity are not supported"),
            v => v,
        }
        .to_f32();

        // Stay a margin inside the i16 endpoint so rounding cannot use
        // the reserved sentinel codes.
        if value32.abs() >= 31.0 {
            return Ok(if value32.is_sign_negative() {
                i16::MIN
            } else {
                i16::MAX
            });
        }

        let scaled = (value32 as f64 * SCALE).round();
        Ok(scaled as i16)
    }

    fn decode(value: i16) -> f16 {
        if value == i16::MIN {
            return f16::MIN;
        }
        if value == i16::MAX {
            return f16::MAX;
        }

        f16::from_f32(value as f32 / SCALE as f32)
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
    F: NumCast,
    I: Copy,
    (): QuantizedRepr<F, I>,
{
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let float = <F as NumCast>::from(n)?;
        let value = <() as QuantizedRepr<F, I>>::encode(float).ok()?;

        Some(Self {
            value,
            _float: PhantomData,
        })
    }
}

impl<F, I> ToPrimitive for Quantized<F, I>
where
    F: ToPrimitive,
    I: Copy,
    (): QuantizedRepr<F, I>,
{
    fn to_i64(&self) -> Option<i64> {
        <() as QuantizedRepr<F, I>>::decode(self.value).to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        <() as QuantizedRepr<F, I>>::decode(self.value).to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        <() as QuantizedRepr<F, I>>::decode(self.value).to_f64()
    }
}

impl<F, I> FromPrimitive for Quantized<F, I>
where
    F: FromPrimitive,
    I: Copy,
    (): QuantizedRepr<F, I>,
{
    fn from_i64(n: i64) -> Option<Self> {
        let float = F::from_i64(n)?;
        let value = <() as QuantizedRepr<F, I>>::encode(float).ok()?;

        Some(Self {
            value,
            _float: PhantomData,
        })
    }

    fn from_u64(n: u64) -> Option<Self> {
        let float = F::from_u64(n)?;
        let value = <() as QuantizedRepr<F, I>>::encode(float).ok()?;

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
    I: Copy,
    (): QuantizedRepr<F, I>,
{
    /// Wraps a computed `F` value by encoding it into the storage type.
    fn encode_value(value: F) -> Result<Self> {
        <() as QuantizedRepr<F, I>>::encode(value).map(|value| Self {
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
    F: FromPrimitive + ToPrimitive,
    I: Copy,
    (): QuantizedRepr<F, I>,
{
    /// Decodes this value, applies a unary `f64` operation, and re-encodes it.
    fn map_float(&self, f: impl FnOnce(f64) -> f64) -> Result<Self> {
        let decoded = <() as QuantizedRepr<F, I>>::decode(self.value);
        let value = decoded
            .to_f64()
            .context("decoded value is not representable as f64")?;
        let result = F::from_f64(f(value))
            .context("computed value is not representable in the float type")?;

        Self::encode_value(result)
    }

    /// Decodes two values, applies a binary `f64` operation, and re-encodes it.
    fn map_pair(&self, other: &Self, f: impl FnOnce(f64, f64) -> f64) -> Result<Self> {
        let a = <() as QuantizedRepr<F, I>>::decode(self.value)
            .to_f64()
            .context("decoded value is not representable as f64")?;
        let b = <() as QuantizedRepr<F, I>>::decode(other.value)
            .to_f64()
            .context("decoded value is not representable as f64")?;
        let result = F::from_f64(f(a, b))
            .context("computed value is not representable in the float type")?;

        Self::encode_value(result)
    }

    /// Hypotenuse of the two decoded values.
    pub fn hypot(&self, other: &Self) -> Result<Self> {
        self.map_pair(other, f64::hypot)
    }

    /// `atan2` of the two decoded values.
    pub fn atan2(&self, other: &Self) -> Result<Self> {
        self.map_pair(other, f64::atan2)
    }

    /// Raises the decoded value to the power of another decoded value.
    pub fn pow(&self, other: &Self) -> Result<Self> {
        self.map_pair(other, f64::powf)
    }

    /// Raises the decoded value to an integer power.
    pub fn powi(&self, n: i32) -> Result<Self> {
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
            F: FromPrimitive + ToPrimitive,
            I: Copy,
             (): QuantizedRepr<F, I>,
        {
            $(
                /// Applies the corresponding `f64` method to the decoded value.
                pub fn $method(&self) -> Result<Self> {
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

/// Binary arithmetic operators. Each decodes both operands, applies the
/// floating-point operation, and re-encodes the result.
///
/// The result is a `Result` because the final encode can fail when the outcome
/// is not representable in the storage range.
macro_rules! impl_quantized_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<F, I> $trait for Quantized<F, I>
        where
            F: $trait<Output = F>,
            I: Copy,
             (): QuantizedRepr<F, I>,
        {
            type Output = Result<Self>;

            fn $method(self, rhs: Self) -> Result<Self> {
                let a = <() as QuantizedRepr<F, I>>::decode(self.value);
                let b = <() as QuantizedRepr<F, I>>::decode(rhs.value);

                Self::encode_value(a $op b)
            }
        }
    };
}

impl_quantized_binop!(Add, add, +);
impl_quantized_binop!(Sub, sub, -);
impl_quantized_binop!(Mul, mul, *);
impl_quantized_binop!(Div, div, /);

/// Unary negation. Decodes the operand, negates it, and re-encodes the result.
impl<F, I> Neg for Quantized<F, I>
where
    F: Neg<Output = F>,
    I: Copy,
    (): QuantizedRepr<F, I>,
{
    type Output = Result<Self>;

    fn neg(self) -> Result<Self> {
        Self::encode_value(-<() as QuantizedRepr<F, I>>::decode(self.value))
    }
}

/// In-place arithmetic operators. Each decodes both operands, applies the
/// operation, and writes the re-encoded result back into `self`.
///
/// If the result cannot be encoded (it is outside the storage range), `self`
/// is left unchanged.
macro_rules! impl_quantized_assign {
    ($assign:ident, $binary:ident, $method:ident, $op:tt) => {
        impl<F, I> $assign for Quantized<F, I>
        where
            F: $binary<Output = F>,
            I: Copy,
             (): QuantizedRepr<F, I>,
        {
            fn $method(&mut self, rhs: Self) {
                let a = <() as QuantizedRepr<F, I>>::decode(self.value);
                let b = <() as QuantizedRepr<F, I>>::decode(rhs.value);

                if let Ok(next) = Self::encode_value(a $op b) {
                    *self = next;
                 }
            }
        }
    };
}

impl_quantized_assign!(AddAssign, Add, add_assign, +);
impl_quantized_assign!(SubAssign, Sub, sub_assign, -);
impl_quantized_assign!(MulAssign, Mul, mul_assign, *);
impl_quantized_assign!(DivAssign, Div, div_assign, /);

/// Quantized `f16` values stored in `i16`.
pub type Quant16 = Quantized<f16, i16>;

/// Quantized `f32` values stored in `i32`.
pub type Quant32 = Quantized<f32, i32>;

/// Quantized `f64` values stored in `i64`.
pub type Quant64 = Quantized<f64, i64>;

#[cfg(test)]
mod tests {
    use super::{Quant16, Quant32};
    use anyhow::Result;
    use half::f16;

    #[test]
    fn arithmetic_operators() -> Result<()> {
        let a = Quant32::from_float::<()>(4.0_f32)?;
        let b = Quant32::from_float::<()>(2.0_f32)?;

        assert_eq!((a + b)?.into_float::<()>(), 6.0_f32);
        assert_eq!((a - b)?.into_float::<()>(), 2.0_f32);
        assert_eq!((a * b)?.into_float::<()>(), 8.0_f32);
        assert_eq!((a / b)?.into_float::<()>(), 2.0_f32);
        assert_eq!((-b)?.into_float::<()>(), -2.0_f32);

        let mut x = Quant32::from_float::<()>(3.0_f32)?;
        x += b;
        assert_eq!(x.into_float::<()>(), 5.0_f32);

        let mut y = Quant32::from_float::<()>(8.0_f32)?;
        y *= b;
        assert_eq!(y.into_float::<()>(), 16.0_f32);

        Ok(())
    }

    #[test]
    fn unary_math() -> Result<()> {
        assert_eq!(
            Quant32::from_float::<()>(9.0_f32)?
                .sqrt()?
                .into_float::<()>(),
            3.0_f32
        );
        assert_eq!(
            Quant32::from_float::<()>(-3.0_f32)?
                .abs()?
                .into_float::<()>(),
            3.0_f32
        );
        assert_eq!(
            Quant32::from_float::<()>(5.0_f32)?
                .signum()?
                .into_float::<()>(),
            1.0_f32
        );

        Ok(())
    }

    #[test]
    fn binary_math() -> Result<()> {
        let a = Quant32::from_float::<()>(3.0_f32)?;
        let b = Quant32::from_float::<()>(4.0_f32)?;

        assert_eq!(a.hypot(&b)?.into_float::<()>(), 5.0_f32);
        assert_eq!(a.pow(&b)?.into_float::<()>(), 81.0_f32);

        Ok(())
    }

    #[test]
    fn quant16_operators() -> Result<()> {
        let a = Quant16::from_float::<()>(f16::from_f32(1.5))?;
        let b = Quant16::from_float::<()>(f16::from_f32(2.0))?;

        assert_eq!((a + b)?.into_float::<()>(), f16::from_f32(3.5));

        let mut x = Quant16::from_float::<()>(f16::from_f32(3.0))?;
        x += b;
        assert_eq!(x.into_float::<()>(), f16::from_f32(5.0));

        Ok(())
    }
}
