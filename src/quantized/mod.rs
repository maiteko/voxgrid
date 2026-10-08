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

use anyhow::{Result, bail};
use half::f16;
use num_traits::{FromPrimitive, NumCast, ToPrimitive};
use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
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

/// Rounds to the nearest integer, with halfway cases rounded away from zero.
fn round_half_up(x: f64) -> f64 {
    if x.is_sign_negative() {
        (x - 0.5).ceil()
    } else {
        (x + 0.5).floor()
    }
}

/// Implements a float/integer representation pair using an explicit
/// supported-distance cutoff.
///
/// The cutoff leaves room below the integer endpoint for scaling and
/// rounding, so normal values do not encode to the reserved MIN/MAX codes.
macro_rules! impl_quantized_repr {
    ($float:ty, $storage:ty, $limit:expr) => {
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

                // Clamp values outside the representable distance range to
                // the corresponding sentinel.
                if value64.abs() >= $limit {
                    return Ok(if value64.is_sign_negative() {
                        <$storage>::MIN
                    } else {
                        <$storage>::MAX
                    });
                }

                let scaled = round_half_up(value64 * SCALE);

                // Keep the sentinel codes exclusively for sentinel values.
                if scaled <= <$storage>::MIN as f64 || scaled >= <$storage>::MAX as f64 {
                    bail!("quantized value is outside the storage range");
                }

                Ok(scaled as $storage)
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
        }
    };
}

// These limits are deliberately conservative. The i32/f32 and i64/f64
// pairings have much more storage range than the f16/i16 pairing.
impl_quantized_repr!(f32, i32, 2_097_151.0);
impl_quantized_repr!(f64, i64, 9_007_199_254_740_991.0);

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

/// Quantized `f16` values stored in `i16`.
pub type Quant16 = Quantized<f16, i16>;

/// Quantized `f32` values stored in `i32`.
pub type Quant32 = Quantized<f32, i32>;

/// Quantized `f64` values stored in `i64`.
pub type Quant64 = Quantized<f64, i64>;
