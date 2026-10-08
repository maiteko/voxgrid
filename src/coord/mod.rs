use glam::{DVec3, Vec3};
use num_traits::{NumCast, real::Real};

/// Rounds `value` to the nearest integer, with ties rounding up to the nearest integer.
pub fn round_half_up<T: Real + NumCast>(value: T) -> T {
    (value + <T as NumCast>::from(0.5).unwrap()).floor()
}

/// Per-component rounding and truncation for floating-point `glam` vectors.
pub trait CoordRound {
    /// Rounds each component to the nearest integer, with ties rounding up.
    fn round_half_up(&self) -> Self;
    /// Rounds each component up to the nearest integer.
    fn ceil(&self) -> Self;
    /// Rounds each component down to the nearest integer.
    fn floor(&self) -> Self;
    /// Rounds ties away from zero
    fn round(&self) -> Self;
}

macro_rules! glam_coord_round {
    ($($vec_type:ty, $int:ty),+) => {
        $(
            impl CoordRound for $vec_type {
                fn round_half_up(&self) -> Self {
                    Self {
                        x: round_half_up(self.x),
                        y: round_half_up(self.y),
                        z: round_half_up(self.z),
                    }
                }

                fn ceil(&self) -> Self {
                    Self {
                        x: self.x.ceil(),
                        y: self.y.ceil(),
                        z: self.z.ceil(),
                    }
                }

                fn floor(&self) -> Self {
                    Self {
                        x: self.x.floor(),
                        y: self.y.floor(),
                        z: self.z.floor(),
                    }
                }

                fn round(&self) -> Self {
                    Self {
                        x: self.x.round(),
                        y: self.y.round(),
                        z: self.z.round(),
                    }
                }
            }
        )+
    };
}

glam_coord_round!(Vec3, f32, DVec3, f64);

macro_rules! add_glam_int_type {
    ($coord:ty, $($vec_type:ty, $int:ty),+) => {
        $(
            impl From<$vec_type> for $coord {
                fn from(v: $vec_type) -> Self {
                    Self(IndexVec {
                        x: v.x as Index,
                        y: v.y as Index ,
                        z: v.z as Index,
                    })
                }
            }

            impl From<&$vec_type> for $coord {
                fn from(v: &$vec_type) -> Self {
                    Self(IndexVec {
                        x: v.x as Index,
                        y: v.y as Index ,
                        z: v.z as Index,
                    })
                }
            }

            impl From<$coord> for $vec_type {
                fn from(c: $coord) -> Self {
                    <$vec_type>::new(
                        c.x as $int,
                        c.y as $int,
                        c.z as $int,
                    )
                }
            }

            impl From<&$coord> for $vec_type {
                fn from(c: &$coord) -> Self {
                    <$vec_type>::new(
                        c.x as $int,
                        c.y as $int,
                        c.z as $int,
                    )
                }
            }
        )+
    };
}

macro_rules! add_glam_float_type {
    ($coord:ty, $($vec_type:ty, $float:ty),+) => {
        $(
            impl From<$vec_type> for $coord {
                fn from(v: $vec_type) -> Self {
                    let v = v.round_half_up();
                    Self(IndexVec::new(v.x as Index, v.y as Index, v.z as Index))
                }
            }

            impl From<&$vec_type> for $coord {
                fn from(v: &$vec_type) -> Self {
                    let v = v.round_half_up();
                    Self(IndexVec::new(v.x as Index, v.y as Index, v.z as Index))
                }
            }

            impl From<$coord> for $vec_type {
                fn from(c: $coord) -> Self {
                    <$vec_type>::new(
                        c.x as $float,
                        c.y as $float,
                        c.z as $float,
                    )
                }
            }

            impl From<&$coord> for $vec_type {
                fn from(c: &$coord) -> Self {
                    <$vec_type>::new(
                        c.x as $float,
                        c.y as $float,
                        c.z as $float,
                    )
                }
            }
        )+
    };
}

macro_rules! int_conversions {
    ($coord:ty, $($int:ty),+) => {
         $(
             impl From<$coord> for ($int, $int, $int) {
                 fn from(value: $coord) -> Self {
                     (value.x as $int, value.y as $int, value.z as $int)
                 }
             }

             impl From<&$coord> for ($int, $int, $int) {
                 fn from(value: &$coord) -> Self {
                     value.into()
                 }
             }

             impl From<($int, $int, $int)> for $coord {
                 fn from(value: ($int, $int, $int)) -> Self {
                     Self(IndexVec {
                         x: value.0 as Index,
                         y: value.1 as Index,
                         z: value.2 as Index,
                     })
                 }
             }

             impl From<&($int, $int, $int)> for $coord {
                 fn from(value: &($int, $int, $int)) -> Self {
                     value.into()
                 }
             }

             impl From<[$int;3]> for $coord {
                 fn from(vec: [$int;3]) -> Self {
                     Self::new(vec[0] as Index, vec[1] as Index, vec[2] as Index)
                 }
             }

             impl From<&[$int;3]> for $coord {
                 fn from(vec: &[$int;3]) -> Self {
                     Self::new(vec[0] as Index, vec[1] as Index, vec[2] as Index)
                 }
             }

             impl From<$coord> for [$int;3] {
                 fn from($coord: $coord) -> Self {
                     [$coord.x as $int, $coord.y as $int, $coord.z as $int]
                 }
             }

             impl From<&$coord> for [$int;3] {
                 fn from($coord: &$coord) -> Self {
                     [$coord.x as $int, $coord.y as $int, $coord.z as $int]
                 }
             }

             #[doc = concat!("Converts ", stringify!($int), " v into a `$coord` with values (v as Index, v as Index, v as Index)")]
             impl From<$int> for $coord {
                 fn from(v: $int) -> Self {
                     Self::new(v as Index, v as Index, v as Index)
                 }
             }
         )+
    }
}

macro_rules! int_ops {
    ($coord:ty, $(($int:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt)),+ $(,)?) => {
        $(
            impl std::ops::$trait<[$int; 3]> for $coord {
                type Output = Self;

                fn $method(self, rhs: [$int; 3]) -> Self::Output {
                    let x = (self.x.to_i64().unwrap() $op rhs[0].to_i64().unwrap()) as Index;
                    let y = (self.y.to_i64().unwrap() $op rhs[1].to_i64().unwrap()) as Index;
                    let z = (self.z.to_i64().unwrap() $op rhs[2].to_i64().unwrap()) as Index;

                    Self::new(x, y, z)
                }
            }

            impl std::ops::$trait<&[$int]> for $coord {
                type Output = Self;

                fn $method(self, rhs: &[$int]) -> Self::Output {
                    let mut v = self.as_array();

                    for i in 0..rhs.len() {
                        v[i] = (v[i].to_i64().unwrap() $op rhs[i].to_i64().unwrap()) as Index;
                    }

                    v.into()
                }
            }

            impl std::ops::$trait<$int> for $coord {
                type Output = Self;

                fn $method(self, rhs: $int) -> Self::Output {
                    let x = (self.x.to_i64().unwrap() $op rhs.to_i64().unwrap()) as Index;
                    let y = (self.y.to_i64().unwrap() $op rhs.to_i64().unwrap()) as Index;
                    let z = (self.z.to_i64().unwrap() $op rhs.to_i64().unwrap()) as Index;

                    Self::new(x, y, z)
                }
            }

            impl std::ops::$assign_trait<[$int; 3]> for $coord {
                fn $assign_method(&mut self, rhs: [$int; 3]) {
                    self.x = (self.x.to_i64().unwrap() $op rhs[0].to_i64().unwrap()) as Index;
                    self.y = (self.y.to_i64().unwrap() $op rhs[1].to_i64().unwrap()) as Index;
                    self.z = (self.z.to_i64().unwrap() $op rhs[2].to_i64().unwrap()) as Index;
                }
            }

            impl std::ops::$assign_trait<&[$int]> for $coord {
                fn $assign_method(&mut self, rhs: &[$int]) {
                    let mut v = self.as_array();

                    for i in 0..rhs.len() {
                        v[i] = (v[i].to_i64().unwrap() $op rhs[i].to_i64().unwrap()) as Index;
                    }

                    let x = self.x $op rhs[0] as Index;
                    let y = self.y $op rhs[1] as Index;
                    let z = self.z $op rhs[2] as Index;

                    *self = Self::new(x, y, z)
                }
            }

            impl std::ops::$assign_trait<$int> for $coord {
                fn $assign_method(&mut self, rhs: $int) {
                    let x = (self.x.to_i64().unwrap() $op rhs.to_i64().unwrap()) as Index;
                    let y = (self.y.to_i64().unwrap() $op rhs.to_i64().unwrap()) as Index;
                    let z = (self.z.to_i64().unwrap() $op rhs.to_i64().unwrap()) as Index;

                    *self = Self::new(x, y, z)
                }
            }
        )+
    };
}

macro_rules! add_int_ops {
    ($coord:ty, $index:ty, $($int:ty),+) => {
        $(
            int_ops!(
                ($coord, $int, Add, add, AddAssign, add_assign, +),
                ($coord, $int, Sub, sub, SubAssign, sub_assign, -),
                ($coord, $int, Mul, mul, MulAssign, mul_assign, *),
                ($coord, $int, Div, div, DivAssign, div_assign, /),
                ($coord, $int, BitAnd, bitand, BitAndAssign, bitand_assign, &),
                ($coord, $int, BitOr, bitor, BitOrAssign, bitor_assign, |),
                ($coord, $int, BitXor, bitxor, BitXorAssign, bitxor_assign, ^),
                ($coord, $int, Shr, shr, ShrAssign, shr_assign, >>),
                ($coord, $int, Shl, shl, ShlAssign, shl_assign, <<),
            );

            impl std::cmp::PartialEq<[$int;3]> for $coord {
                fn eq(&self, rhs: &[$int;3]) -> bool {
                    self.x == rhs[0] as $index && self.y == rhs[1] as $index && self.z == rhs[2] as $index
                }
            }

            impl std::cmp::PartialOrd<[$int;3]> for $coord {
                fn partial_cmp(&self, rhs: &[$int;3]) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(&rhs.into()))
                }
            }
        )+
    };
}

macro_rules! float_conversions {
    ($coord:ty, $($float:ty),+) => {
         $(
             impl From<$coord> for ($float, $float, $float) {
                 fn from(value: $coord) -> Self {
                     (value.x as $float, value.y as $float, value.z as $float)
                 }
             }

             impl From<&$coord> for ($float, $float, $float) {
                 fn from(value: &$coord) -> Self {
                     value.into()
                 }
             }

             impl From<($float, $float, $float)> for $coord {
                 fn from(value: ($float, $float, $float)) -> Self {
                     Self(IndexVec {
                         x: round_half_up(value.0.to_f64().unwrap()) as Index,
                         y: round_half_up(value.1.to_f64().unwrap()) as Index,
                         z: round_half_up(value.2.to_f64().unwrap()) as Index,
                     })
                 }
             }

             impl From<&($float, $float, $float)> for $coord {
                 fn from(value: &($float, $float, $float)) -> Self {
                     value.into()
                 }
             }

             impl From<[$float;3]> for $coord {
                 fn from(vec: [$float;3]) -> Self {
                     Self::new(
                         round_half_up(vec[0].to_f64().unwrap()) as Index,
                         round_half_up(vec[1].to_f64().unwrap()) as Index,
                         round_half_up(vec[2].to_f64().unwrap()) as Index
                     )
                 }
             }

             impl From<&[$float;3]> for $coord {
                 fn from(vec: &[$float;3]) -> Self {
                     Self::new(
                         round_half_up(vec[0].to_f64().unwrap()) as Index,
                         round_half_up(vec[1].to_f64().unwrap()) as Index,
                         round_half_up(vec[2].to_f64().unwrap()) as Index
                     )
                 }
             }

             impl From<$coord> for [$float;3] {
                 fn from($coord: $coord) -> Self {
                     [$coord.x as $float, $coord.y as $float, $coord.z as $float]
                 }
             }

             impl From<&$coord> for [$float;3] {
                 fn from($coord: &$coord) -> Self {
                     [$coord.x as $float, $coord.y as $float, $coord.z as $float]
                 }
             }

             #[doc = concat!("Converts ", stringify!($float), " v floato a `$coord` with values (v as Index, v as Index, v as Index)")]
             impl From<$float> for $coord {
                 fn from(v: $float) -> Self {
                     Self::new(v as Index, v as Index, v as Index)
                 }
             }
         )+
    }
}

macro_rules! float_ops {
    ($coord:ty, $(($float:ty, $trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt, $conv:ident)),+ $(,)?) => {
        $(
            impl std::ops::$trait<[$float; 3]> for $coord {
                type Output = Self;

                fn $method(self, rhs: [$float; 3]) -> Self::Output {
                    let x = round_half_up((self.x.to_f64().unwrap() $op rhs[0].to_f64().unwrap())) as Index;
                    let y = round_half_up((self.y.to_f64().unwrap() $op rhs[1].to_f64().unwrap())) as Index;
                    let z = round_half_up((self.z.to_f64().unwrap() $op rhs[2].to_f64().unwrap())) as Index;

                    Self::new(x, y, z)
                }
            }

            impl std::ops::$trait<&[$float]> for $coord {
                type Output = Self;

                fn $method(self, rhs: &[$float]) -> Self::Output {
                    let mut v = self.as_array();

                    for i in 0..rhs.len() {
                        v[i] = round_half_up((v[i].to_f64().unwrap() $op rhs[i].to_f64().unwrap())) as Index;
                    }

                    v.into()
                }
            }

            impl std::ops::$trait<$float> for $coord {
                type Output = Self;

                fn $method(self, rhs: $float) -> Self::Output {
                    let x = round_half_up((self.x.to_f64().unwrap() $op rhs.to_f64().unwrap())) as Index;
                    let y = round_half_up((self.y.to_f64().unwrap() $op rhs.to_f64().unwrap())) as Index;
                    let z = round_half_up((self.z.to_f64().unwrap() $op rhs.to_f64().unwrap())) as Index;

                    Self::new(x, y, z)
                }
            }

            impl std::ops::$assign_trait<[$float; 3]> for $coord {
                fn $assign_method(&mut self, rhs: [$float; 3]) {
                    self.x = round_half_up((self.x.to_f64().unwrap() $op rhs[0].to_f64().unwrap())) as Index;
                    self.y = round_half_up((self.y.to_f64().unwrap() $op rhs[1].to_f64().unwrap())) as Index;
                    self.z = round_half_up((self.z.to_f64().unwrap() $op rhs[2].to_f64().unwrap())) as Index;
                }
            }

            impl std::ops::$assign_trait<&[$float]> for $coord {
                fn $assign_method(&mut self, rhs: &[$float]) {
                    let mut v = self.as_array();

                    for i in 0..rhs.len() {
                        v[i] = round_half_up((v[i].to_f64().unwrap() $op rhs[i].to_f64().unwrap())) as Index;
                    }

                    let x = round_half_up(self.x.to_f64() $op rhs[0]) as Index;
                    let y = round_half_up(self.y.to_f64() $op rhs[1]) as Index;
                    let z = round_half_up(self.z.to_f64() $op rhs[2]) as Index;

                    *self = Self::new(x, y, z)
                }
            }

            impl std::ops::$assign_trait<$float> for $coord {
                fn $assign_method(&mut self, rhs: $float) {
                    let x = round_half_up((self.x.to_f64().unwrap() $op rhs.to_f64().unwrap())) as Index;
                    let y = round_half_up((self.y.to_f64().unwrap() $op rhs.to_f64().unwrap())) as Index;
                    let z = round_half_up((self.z.to_f64().unwrap() $op rhs.to_f64().unwrap())) as Index;

                    *self = Self::new(x, y, z)
                }
            }
        )+
    };
}

macro_rules! add_float_ops {
    ($coord:ty, $($float:ty),+) => {
        $(
            float_ops!(
                ($coord, $float, Add, add, AddAssign, add_assign, +),
                ($coord, $float, Sub, sub, SubAssign, sub_assign, -),
                ($coord, $float, Mul, mul, MulAssign, mul_assign, *),
                ($coord, $float, Div, div, DivAssign, div_assign, /),
            );

            impl std::cmp::PartialEq<[$float;3]> for $coord {
                fn eq(&self, rhs: &[$float;3]) -> bool {
                    self.x == rhs[0] as Index && self.y == rhs[1] as Index && self.z == rhs[2] as Index
                }
            }

            impl std::cmp::PartialOrd<[$float;3]> for $coord {
                fn partial_cmp(&self, rhs: &[$float;3]) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(&rhs.into()))
                }
            }
        )+
    };
}

mod global;
mod local;
pub use global::Coord;
pub use local::LocalCoord;
