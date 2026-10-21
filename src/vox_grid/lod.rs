use super::super::*;
use super::{VoxGrid, VoxGridBackground};
use std::marker::PhantomData;

/// A coarse, read-only view of a [`VoxGrid`]: every `2^shift`-th voxel.
///
/// Coordinate `c` of the view reads the grid's voxel at `c << shift`, so the view is the
/// grid point-sampled on a lattice of spacing `stride() = 2^shift`. Tile reads go through
/// [`VoxGrid::for_each_tile_strided`], so a grid that overrides it is read efficiently.
///
/// ```
/// use voxgrid::{Coord, CoordBBox, VoxGrid, VoxGridBackground, VoxGridLod};
///
/// struct Ramp;
/// impl VoxGridBackground<i32> for Ramp {
///     fn background_value(&self) -> i32 { 0 }
/// }
/// impl VoxGrid<i32> for Ramp {
///     fn get_voxel(&self, p: &Coord) -> Option<i32> { Some(p.x) }
/// }
///
/// let coarse = VoxGridLod::new(&Ramp, 2);
/// assert_eq!(coarse.stride(), 4);
/// assert_eq!(coarse.get_value(&Coord::new(3, 0, 0)), 12);
/// assert_eq!(coarse.get_value(&Coord::new(-1, 5, 0)), -4);
/// ```
pub struct VoxGridLod<'a, G: ?Sized, T> {
    grid: &'a G,
    shift: u32,
    _marker: PhantomData<T>,
}

impl<'a, G: VoxGrid<T> + ?Sized, T> VoxGridLod<'a, G, T> {
    /// The view of `grid` at spacing `2^shift`.
    pub fn new(grid: &'a G, shift: u32) -> Self {
        Self {
            grid,
            shift,
            _marker: PhantomData,
        }
    }

    /// Log2 of the lattice spacing.
    pub fn shift(&self) -> u32 {
        self.shift
    }

    /// The lattice spacing, in grid voxels.
    pub fn stride(&self) -> CIndex {
        1 << self.shift
    }

    /// The grid voxel that coarse coordinate `c` reads.
    pub fn to_fine(&self, c: &Coord) -> Coord {
        Coord::new(c.x << self.shift, c.y << self.shift, c.z << self.shift)
    }
}

impl<G: VoxGrid<T> + ?Sized, T> VoxGridBackground<T> for VoxGridLod<'_, G, T> {
    fn background_value(&self) -> T {
        self.grid.background_value()
    }
}

impl<G: VoxGrid<T> + ?Sized, T> VoxGrid<T> for VoxGridLod<'_, G, T> {
    fn get_voxel(&self, pos: &Coord) -> Option<T> {
        self.grid.get_voxel(&self.to_fine(pos))
    }

    fn get_value(&self, pos: &Coord) -> T {
        self.grid.get_value(&self.to_fine(pos))
    }

    fn for_each_tile(&self, bbox: &CoordBBox, f: &mut dyn FnMut(&CoordBBox, &T)) {
        self.grid.for_each_tile_strided(bbox, self.shift, f);
    }

    fn for_each_tile_strided(
        &self,
        bbox: &CoordBBox,
        shift: u32,
        f: &mut dyn FnMut(&CoordBBox, &T),
    ) {
        self.grid.for_each_tile_strided(bbox, self.shift + shift, f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseBlock;

    struct Lumpy;
    impl VoxGridBackground<i32> for Lumpy {
        fn background_value(&self) -> i32 {
            -1
        }
    }
    impl VoxGrid<i32> for Lumpy {
        fn get_voxel(&self, p: &Coord) -> Option<i32> {
            (p.y != 0).then_some(p.x * 100 + p.y * 10 + p.z)
        }
    }

    #[test]
    fn strided_tiles_point_sample() {
        let bbox = CoordBBox::new(Coord::new(-3, -2, -1), Coord::new(2, 1, 2));
        for shift in 0..3 {
            let view = VoxGridLod::new(&Lumpy, shift);
            let mut seen = DenseBlock::filled(bbox, 0u32, 0);
            view.for_each_tile(&bbox, &mut |tile, v| {
                for c in tile.to_xyz_iter() {
                    assert_eq!(*v, Lumpy.get_value(&view.to_fine(&c)), "at {c}");
                    *seen.get_mut(&c).unwrap() += 1;
                }
            });
            assert!(seen.as_slice().iter().all(|&n| n == 1));
        }
    }

    #[test]
    fn nested_views_compose() {
        let outer = VoxGridLod::new(&Lumpy, 1);
        let inner = VoxGridLod::new(&outer, 2);
        let c = Coord::new(-1, 1, 2);
        assert_eq!(inner.get_value(&c), Lumpy.get_value(&Coord::new(-8, 8, 16)));
        let mut tiles = Vec::new();
        inner.for_each_tile(&CoordBBox::new(c, c), &mut |t, v| tiles.push((*t, *v)));
        assert_eq!(tiles, vec![(CoordBBox::new(c, c), inner.get_value(&c))]);
    }
}
