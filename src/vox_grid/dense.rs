use super::super::*;
use super::{VoxGrid, VoxGridBackground};

/// A dense copy of the voxel values in a [`CoordBBox`].
///
/// Values are stored `x` fastest, then `y`, then `z`. Reads outside the box return
/// `None` from [`VoxGrid::get_voxel`], so [`VoxGrid::get_value`] falls back to the
/// background.
///
/// Build one from any grid with [`VoxGrid::read_dense`]; it fills one tile at a time
/// (see [`VoxGrid::for_each_tile`]), so grids with uniform regions copy them in bulk.
///
/// ```
/// use voxgrid::{Coord, CoordBBox, DenseBlock, VoxGrid};
///
/// let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(3, 1, 1));
/// let mut block = DenseBlock::filled(bbox, 0u8, 0u8);
/// block.fill(&CoordBBox::new(Coord::new(2, 0, 0), Coord::new(9, 9, 9)), &7);
///
/// assert_eq!(block.dims(), [4, 2, 2]);
/// assert_eq!(block.get_value(&Coord::new(1, 0, 0)), 0);
/// assert_eq!(block.get_value(&Coord::new(3, 1, 1)), 7);
/// assert_eq!(block.get(&Coord::new(4, 0, 0)), None);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DenseBlock<T> {
    bbox: CoordBBox,
    dims: [usize; 3],
    data: Vec<T>,
    background: T,
}

impl<T: Clone> DenseBlock<T> {
    /// A block over `bbox` with every voxel set to `value`. `background` is what reads
    /// outside the box resolve to.
    pub fn filled(bbox: CoordBBox, value: T, background: T) -> Self {
        let d = bbox.axis_dims();
        let dims = [d.x as usize, d.y as usize, d.z as usize];
        Self {
            bbox,
            dims,
            data: vec![value; dims[0] * dims[1] * dims[2]],
            background,
        }
    }

    /// Copy the values of `grid` inside `bbox`, one tile at a time.
    pub fn read<G: VoxGrid<T> + ?Sized>(grid: &G, bbox: &CoordBBox) -> Self {
        let background = grid.background_value();
        let mut block = Self::filled(*bbox, background.clone(), background);
        grid.for_each_tile(bbox, &mut |tile, value| block.fill(tile, value));
        block
    }

    /// Set every voxel of `bbox` that lies inside this block to `value`.
    pub fn fill(&mut self, bbox: &CoordBBox, value: &T) {
        let mut clip = *bbox;
        clip.intersect(&self.bbox);
        if clip.empty() {
            return;
        }
        let x0 = (clip.min.x - self.bbox.min.x) as usize;
        let x1 = (clip.max.x - self.bbox.min.x) as usize + 1;
        for z in clip.min.z..=clip.max.z {
            for y in clip.min.y..=clip.max.y {
                let row = self.row_start(y, z);
                self.data[row + x0..row + x1].fill(value.clone());
            }
        }
    }
}

impl<T> DenseBlock<T> {
    /// The box this block covers.
    pub fn bbox(&self) -> &CoordBBox {
        &self.bbox
    }

    /// Voxels per axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// The values, `x` fastest, then `y`, then `z`.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The values, mutably, in the same order as [`as_slice`](Self::as_slice).
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Take the values, in the same order as [`as_slice`](Self::as_slice).
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// The index of `pos` in [`as_slice`](Self::as_slice), or `None` outside the box.
    pub fn index(&self, pos: &Coord) -> Option<usize> {
        if !self.bbox.coord_is_inside(pos) || self.bbox.empty() {
            return None;
        }
        Some(self.row_start(pos.y, pos.z) + (pos.x - self.bbox.min.x) as usize)
    }

    /// The value at `pos`, or `None` outside the box.
    pub fn get(&self, pos: &Coord) -> Option<&T> {
        self.index(pos).map(|i| &self.data[i])
    }

    /// The value at `pos` mutably, or `None` outside the box.
    pub fn get_mut(&mut self, pos: &Coord) -> Option<&mut T> {
        self.index(pos).map(|i| &mut self.data[i])
    }

    fn row_start(&self, y: CIndex, z: CIndex) -> usize {
        let j = (y - self.bbox.min.y) as usize;
        let k = (z - self.bbox.min.z) as usize;
        self.dims[0] * (j + self.dims[1] * k)
    }
}

impl<T: Clone> VoxGridBackground<T> for DenseBlock<T> {
    fn background_value(&self) -> T {
        self.background.clone()
    }
}

impl<T: Clone> VoxGrid<T> for DenseBlock<T> {
    fn limits(&self) -> CoordBBox {
        self.bbox
    }

    fn get_voxel(&self, pos: &Coord) -> Option<T> {
        self.get(pos).cloned()
    }

    fn for_each_tile(&self, bbox: &CoordBBox, f: &mut dyn FnMut(&CoordBBox, &T)) {
        // Outside the block is background; inside, every voxel is its own tile.
        let b = clip_outside(bbox, &self.bbox, &self.background, f);
        if b.empty() {
            return;
        }
        for z in b.min.z..=b.max.z {
            for y in b.min.y..=b.max.y {
                for x in b.min.x..=b.max.x {
                    let p = Coord::new(x, y, z);
                    f(&CoordBBox::new(p, p), &self.data[self.index(&p).unwrap()]);
                }
            }
        }
    }
}

/// Report the part of `bbox` outside `inner` to `f` as up to six `outside` tiles, and
/// return the part inside `inner` (possibly empty).
fn clip_outside<T>(
    bbox: &CoordBBox,
    inner: &CoordBBox,
    outside: &T,
    f: &mut dyn FnMut(&CoordBBox, &T),
) -> CoordBBox {
    if bbox.empty() {
        return CoordBBox::default();
    }
    let mut rest = *bbox;
    for axis in 0..3 {
        // Slab below `inner` on this axis.
        if rest.min[axis] < inner.min[axis] {
            let mut slab = rest;
            slab.max[axis] = (inner.min[axis] - 1).min(rest.max[axis]);
            if slab.has_volume() {
                f(&slab, outside);
            }
            rest.min[axis] = inner.min[axis];
        }
        // Slab above `inner` on this axis.
        if rest.max[axis] > inner.max[axis] {
            let mut slab = rest;
            slab.min[axis] = (inner.max[axis] + 1).max(rest.min[axis]);
            if slab.has_volume() {
                f(&slab, outside);
            }
            rest.max[axis] = inner.max[axis];
        }
        if rest.empty() {
            return CoordBBox::default();
        }
    }
    rest
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid that only implements `get_voxel`, so it uses the default tiles.
    struct Lumpy;
    impl VoxGridBackground<i32> for Lumpy {
        fn background_value(&self) -> i32 {
            -1
        }
    }
    impl VoxGrid<i32> for Lumpy {
        fn get_voxel(&self, p: &Coord) -> Option<i32> {
            (p.x != 0).then_some(p.x * 100 + p.y * 10 + p.z)
        }
    }

    fn assert_tiles_partition<G: VoxGrid<i32>>(grid: &G, bbox: &CoordBBox) {
        let mut seen = DenseBlock::filled(*bbox, 0u32, 0);
        grid.for_each_tile(bbox, &mut |tile, value| {
            assert!(tile.has_volume());
            assert!(bbox.bbox_is_inside(tile), "{tile:?} leaves {bbox:?}");
            for p in tile.to_xyz_iter() {
                assert_eq!(*value, grid.get_value(&p), "at {p}");
                *seen.get_mut(&p).unwrap() += 1;
            }
        });
        assert!(
            seen.as_slice().iter().all(|&n| n == 1),
            "tiles overlap or miss"
        );
    }

    #[test]
    fn default_tiles_partition_the_box() {
        let bbox = CoordBBox::new(Coord::new(-2, -1, 0), Coord::new(2, 1, 3));
        assert_tiles_partition(&Lumpy, &bbox);

        let mut count = 0;
        Lumpy.for_each_tile(&bbox, &mut |_, _| count += 1);
        assert_eq!(count, bbox.volume());
    }

    #[test]
    fn empty_box_has_no_tiles() {
        let mut count = 0;
        Lumpy.for_each_tile(&CoordBBox::default(), &mut |_, _| count += 1);
        assert_eq!(count, 0);
        assert_eq!(Lumpy.read_dense(&CoordBBox::default()).as_slice().len(), 0);
    }

    #[test]
    fn read_dense_matches_get_value() {
        let bbox = CoordBBox::new(Coord::new(-1, -2, -3), Coord::new(3, 2, 1));
        let block = Lumpy.read_dense(&bbox);
        assert_eq!(block.dims(), [5, 5, 5]);
        for p in bbox.to_xyz_iter() {
            assert_eq!(block.get_value(&p), Lumpy.get_value(&p));
        }
        // x fastest, then y, then z.
        assert_eq!(block.as_slice()[1], Lumpy.get_value(&Coord::new(0, -2, -3)));
        assert_eq!(
            block.as_slice()[5],
            Lumpy.get_value(&Coord::new(-1, -1, -3))
        );
        assert_eq!(
            block.as_slice()[25],
            Lumpy.get_value(&Coord::new(-1, -2, -2))
        );
        // Outside reads the background.
        assert_eq!(block.get_value(&Coord::new(9, 9, 9)), -1);
    }

    #[test]
    fn fill_clips_to_the_block() {
        let bbox = CoordBBox::new(Coord::new(0, 0, 0), Coord::new(3, 3, 3));
        let mut block = DenseBlock::filled(bbox, 0, 0);
        block.fill(
            &CoordBBox::new(Coord::new(-5, 2, 2), Coord::new(1, 9, 9)),
            &4,
        );
        block.fill(
            &CoordBBox::new(Coord::new(7, 7, 7), Coord::new(9, 9, 9)),
            &5,
        );
        let filled = block.as_slice().iter().filter(|&&v| v == 4).count();
        assert_eq!(filled, 2 * 2 * 2);
        assert!(block.as_slice().iter().all(|&v| v != 5));
    }

    #[test]
    fn dense_block_tiles_split_inside_and_outside() {
        let block = Lumpy.read_dense(&CoordBBox::new(Coord::new(0, 0, 0), Coord::new(2, 2, 2)));
        let query = CoordBBox::new(Coord::new(-3, -1, 1), Coord::new(4, 3, 5));
        assert_tiles_partition(&block, &query);
        // A box fully outside is one background tile.
        let mut tiles = Vec::new();
        let far = CoordBBox::new(Coord::new(10, 10, 10), Coord::new(12, 12, 12));
        block.for_each_tile(&far, &mut |t, v| tiles.push((*t, *v)));
        assert_eq!(tiles, vec![(far, -1)]);
    }
}
