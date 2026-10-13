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
use std::{
    fmt::{self, Debug, Display},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use anyhow::{Result, bail};
use half::f16;
use num_traits::{
    Float, FromPrimitive, Num, NumCast, One, Pow, PrimInt, Signed, ToPrimitive, Zero,
};

const SCALE: f64 = 1024.0;

/// Representation policy for a quantized float/storage pair.
///
/// The storage minimum and maximum are reserved for negative and positive
/// infinity. Implementations must keep ordinary encoded values away from
/// those codes.
pub trait QuantizedRepr<F, I>
where
    F: Float,
    I: PrimInt,
{
    /// Largest supported absolute input value. Inputs at or beyond this
    /// limit are rejected.
    const LIMIT: f64;

    /// Largest supported absolute storage code for ordinary values.
    const SCALED_LIMIT: I;

    /// Encodes a finite value. Implementations must reject values that cannot
    /// be represented and must not return either reserved sentinel code.
    fn encode(value: F) -> Result<I>;

    /// Decodes a storage code. The reserved minimum and maximum codes
    /// represent negative and positive infinity.
    fn decode(value: I) -> F;
}

/// A floating-point value stored as a hashable integer code.
///
/// The representation policy is supplied by a [`QuantizedRepr`] implementation.
/// The default resolution in the implementations below is `1 / 1024`.
pub struct Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    value: I,
    _float: PhantomData<F>,
}

impl<F, I> Quantized<F, I>
where
    F: Float,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    /// Encodes `value` using this type's representation.
    pub fn from_float(value: F) -> Result<Self> {
        Ok(Self::from_raw(<Self as QuantizedRepr<F, I>>::encode(
            value,
        )?))
    }

    /// Decodes this value using this type's representation.
    pub fn into_float(self) -> F {
        <Self as QuantizedRepr<F, I>>::decode(self.value)
    }

    /// Returns the stored integer code.
    pub fn raw(&self) -> &I {
        &self.value
    }

    fn from_raw(value: I) -> Self {
        Self {
            value,
            _float: PhantomData,
        }
    }

    fn from_result(value: F) -> Self {
        Self::from_float(value).expect("result is not representable as Quantized")
    }

    fn map(self, f: impl FnOnce(F) -> F) -> Self {
        Self::from_result(f(self.into_float()))
    }

    fn zip(self, other: Self, f: impl FnOnce(F, F) -> F) -> Self {
        Self::from_result(f(self.into_float(), other.into_float()))
    }
}

impl<F, I> Display for Quantized<F, I>
where
    F: Float + Display,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&<Self as QuantizedRepr<F, I>>::decode(self.value), f)
    }
}

impl<F, I> Debug for Quantized<F, I>
where
    F: Float + Debug,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&<Self as QuantizedRepr<F, I>>::decode(self.value), f)
    }
}

impl<F, I> Copy for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
}

impl<F, I> Clone for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<F, I> PartialEq for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F, I> Eq for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
}

impl<F, I> Hash for Quantized<F, I>
where
    F: Float,
    I: PrimInt + Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<F, I> PartialOrd for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F, I> Ord for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<F, I> Default for Quantized<F, I>
where
    F: Float,
    I: PrimInt + Zero,
    Self: QuantizedRepr<F, I>,
{
    fn default() -> Self {
        Self::from_raw(I::zero())
    }
}

/// Encodes a finite float using a fixed scale and a representation cutoff.
fn encode_scaled<F, I>(value: F, limit: f64) -> Result<I>
where
    F: Float + ToPrimitive,
    I: PrimInt + NumCast + FromPrimitive,
{
    if !value.is_finite() {
        bail!("non-finite values are not supported");
    }

    let value = value
        .to_f64()
        .ok_or_else(|| anyhow::anyhow!("value cannot be converted to f64"))?;

    if value <= -limit || value >= limit {
        bail!("value is outside the representable range");
    }

    let scaled = (value * SCALE).round();
    let code =
        I::from_f64(scaled).ok_or_else(|| anyhow::anyhow!("scaled value is not representable"))?;

    if code == I::min_value() || code == I::max_value() {
        bail!("encoded value collides with a reserved sentinel");
    }

    Ok(code)
}

/// Decodes a scaled integer code. Integer endpoints are reserved sentinels.
fn decode_scaled<F, I>(value: I) -> F
where
    F: Float + FromPrimitive,
    I: PrimInt + ToPrimitive,
{
    if value == I::min_value() {
        F::neg_infinity()
    } else if value == I::max_value() {
        F::infinity()
    } else {
        F::from_f64(value.to_f64().unwrap() / SCALE).unwrap()
    }
}

impl QuantizedRepr<f32, i32> for Quantized<f32, i32> {
    const LIMIT: f64 = 2_097_151.0;
    const SCALED_LIMIT: i32 = 2_097_151 * 1024;

    fn encode(value: f32) -> Result<i32> {
        encode_scaled(value, Self::LIMIT)
    }

    fn decode(value: i32) -> f32 {
        decode_scaled(value)
    }
}

impl QuantizedRepr<f64, i64> for Quantized<f64, i64> {
    const LIMIT: f64 = 9_007_199_254_740_991.0;
    const SCALED_LIMIT: i64 = 9_007_199_254_740_991 * 1024;

    fn encode(value: f64) -> Result<i64> {
        encode_scaled(value, Self::LIMIT)
    }

    fn decode(value: i64) -> f64 {
        decode_scaled(value)
    }
}

impl QuantizedRepr<f16, i16> for Quantized<f16, i16> {
    const LIMIT: f64 = 31.0;
    const SCALED_LIMIT: i16 = 31 * 1024;

    fn encode(value: f16) -> Result<i16> {
        encode_scaled(value, Self::LIMIT)
    }

    fn decode(value: i16) -> f16 {
        decode_scaled(value)
    }
}

impl<F, I> NumCast for Quantized<F, I>
where
    F: Float + NumCast,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        Self::from_float(F::from(n)?).ok()
    }
}

impl<F, I> ToPrimitive for Quantized<F, I>
where
    F: Float + ToPrimitive,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn to_i64(&self) -> Option<i64> {
        self.clone().into_float().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.clone().into_float().to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        self.clone().into_float().to_f64()
    }
}

impl<F, I> FromPrimitive for Quantized<F, I>
where
    F: Float + FromPrimitive,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn from_i64(n: i64) -> Option<Self> {
        Self::from_float(F::from_i64(n)?).ok()
    }

    fn from_u64(n: u64) -> Option<Self> {
        Self::from_float(F::from_u64(n)?).ok()
    }

    fn from_f64(n: f64) -> Option<Self> {
        Self::from_float(F::from_f64(n)?).ok()
    }
}

impl<F, I> Zero for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn zero() -> Self {
        Self::from_raw(I::zero())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }
}

impl<F, I> One for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    fn one() -> Self {
        Self::from_result(F::one())
    }
}

impl<F, I> Num for Quantized<F, I>
where
    F: Float + NumCast,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    type FromStrRadixErr = <F as Num>::FromStrRadixErr;

    fn from_str_radix(s: &str, radix: u32) -> std::result::Result<Self, Self::FromStrRadixErr> {
        let value = F::from_str_radix(s, radix)?;
        Ok(Self::from_result(value))
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<F, I> $trait for Quantized<F, I>
        where
            F: Float,
            I: PrimInt,
            Self: QuantizedRepr<F, I>,
        {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                self.zip(rhs, |a, b| a $op b)
            }
        }
    };
}

impl_binary_op!(Add, add, +);
impl_binary_op!(Sub, sub, -);
impl_binary_op!(Mul, mul, *);
impl_binary_op!(Div, div, /);
impl_binary_op!(Rem, rem, %);

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl<F, I> $trait for Quantized<F, I>
        where
            F: Float,
            I: PrimInt,
            Self: QuantizedRepr<F, I>,
        {
            fn $method(&mut self, rhs: Self) {
                *self = (*self).$op_method(rhs);
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, Add, add);
impl_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_assign_op!(DivAssign, div_assign, Div, div);
impl_assign_op!(RemAssign, rem_assign, Rem, rem);

impl<F, I> Neg for Quantized<F, I>
where
    F: Float,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.map(Neg::neg)
    }
}

impl<F, I> Pow<Self> for Quantized<F, I>
where
    F: Float + Pow<F, Output = F>,
    I: PrimInt,
    Self: QuantizedRepr<F, I>,
{
    type Output = Self;

    fn pow(self, rhs: Self) -> Self::Output {
        Self::from_result(self.into_float().pow(rhs.into_float()))
    }
}

impl<F, I> Signed for Quantized<F, I>
where
    F: Float,
    I: PrimInt + Signed,
    Self: QuantizedRepr<F, I>,
{
    fn abs(&self) -> Self {
        Self::from_raw(self.value.abs())
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Self::from_raw(self.value.abs_sub(&other.value))
    }

    fn signum(&self) -> Self {
        Self::from_raw(self.value.signum())
    }

    fn is_positive(&self) -> bool {
        self.value.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.value.is_negative()
    }
}

impl<F, I> Float for Quantized<F, I>
where
    F: Float + FromPrimitive + ToPrimitive,
    I: PrimInt + NumCast + Signed,
    Self: QuantizedRepr<F, I>,
{
    fn nan() -> Self {
        panic!("Quantized does not support NaN")
    }

    fn infinity() -> Self {
        Self::from_raw(I::max_value())
    }

    fn neg_infinity() -> Self {
        Self::from_raw(I::min_value())
    }

    fn neg_zero() -> Self {
        Self::zero()
    }

    fn min_value() -> Self {
        Self::from_raw(I::min_value())
    }

    fn min_positive_value() -> Self {
        Self::zero()
    }

    fn max_value() -> Self {
        Self::from_raw(I::max_value())
    }

    fn epsilon() -> Self {
        Self::from_result(F::epsilon())
    }

    fn is_nan(self) -> bool {
        false
    }

    fn is_infinite(self) -> bool {
        false
    }

    fn is_finite(self) -> bool {
        true
    }

    fn is_normal(self) -> bool {
        self.into_float().is_normal()
    }

    fn classify(self) -> std::num::FpCategory {
        self.into_float().classify()
    }

    fn floor(self) -> Self {
        self.map(Float::floor)
    }

    fn ceil(self) -> Self {
        self.map(Float::ceil)
    }

    fn round(self) -> Self {
        self.map(Float::round)
    }

    fn trunc(self) -> Self {
        self.map(Float::trunc)
    }

    fn fract(self) -> Self {
        self.map(Float::fract)
    }

    fn abs(self) -> Self {
        self.map(Float::abs)
    }

    fn signum(self) -> Self {
        self.map(Float::signum)
    }

    fn is_sign_positive(self) -> bool {
        self.into_float().is_sign_positive()
    }

    fn is_sign_negative(self) -> bool {
        self.into_float().is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        Self::from_result(self.into_float().mul_add(a.into_float(), b.into_float()))
    }

    fn recip(self) -> Self {
        self.map(Float::recip)
    }

    fn powi(self, n: i32) -> Self {
        Self::from_result(self.into_float().powi(n))
    }

    fn powf(self, n: Self) -> Self {
        Self::from_result(self.into_float().powf(n.into_float()))
    }

    fn sqrt(self) -> Self {
        self.map(Float::sqrt)
    }

    fn exp(self) -> Self {
        self.map(Float::exp)
    }

    fn exp2(self) -> Self {
        self.map(Float::exp2)
    }

    fn ln(self) -> Self {
        self.map(Float::ln)
    }

    fn log(self, base: Self) -> Self {
        Self::from_result(self.into_float().log(base.into_float()))
    }

    fn log2(self) -> Self {
        self.map(Float::log2)
    }

    fn log10(self) -> Self {
        self.map(Float::log10)
    }

    fn max(self, other: Self) -> Self {
        Self::from_result(self.into_float().max(other.into_float()))
    }

    fn min(self, other: Self) -> Self {
        Self::from_result(self.into_float().min(other.into_float()))
    }

    fn abs_sub(self, other: Self) -> Self {
        Self::from_raw(self.value.abs_sub(&other.value))
    }

    fn cbrt(self) -> Self {
        self.map(Float::cbrt)
    }

    fn hypot(self, other: Self) -> Self {
        Self::from_result(self.into_float().hypot(other.into_float()))
    }

    fn sin(self) -> Self {
        self.map(Float::sin)
    }

    fn cos(self) -> Self {
        self.map(Float::cos)
    }

    fn tan(self) -> Self {
        self.map(Float::tan)
    }

    fn asin(self) -> Self {
        self.map(Float::asin)
    }

    fn acos(self) -> Self {
        self.map(Float::acos)
    }

    fn atan(self) -> Self {
        self.map(Float::atan)
    }

    fn atan2(self, other: Self) -> Self {
        Self::from_result(self.into_float().atan2(other.into_float()))
    }

    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = self.into_float().sin_cos();
        (Self::from_result(sin), Self::from_result(cos))
    }

    fn exp_m1(self) -> Self {
        self.map(Float::exp_m1)
    }

    fn ln_1p(self) -> Self {
        self.map(Float::ln_1p)
    }

    fn sinh(self) -> Self {
        self.map(Float::sinh)
    }

    fn cosh(self) -> Self {
        self.map(Float::cosh)
    }

    fn tanh(self) -> Self {
        self.map(Float::tanh)
    }

    fn asinh(self) -> Self {
        self.map(Float::asinh)
    }

    fn acosh(self) -> Self {
        self.map(Float::acosh)
    }

    fn atanh(self) -> Self {
        self.map(Float::atanh)
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        self.into_float().integer_decode()
    }
}

pub type Quant16 = Quantized<f16, i16>;
pub type Quant32 = Quantized<f32, i32>;
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

        assert_eq!(a.hypot(b).into_float(), 5.0_f32);
        assert_eq!(a.pow(b).into_float(), 81.0_f32);

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
