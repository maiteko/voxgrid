use super::*;

mod dense;
mod lod;
mod slice;

pub use dense::DenseBlock;
pub use lod::VoxGridLod;
pub use slice::{VoxGridSlice, VoxGridSliceMut};

/// Trait for providing a background value when a voxel position is empty
pub trait VoxGridBackground<T> {
    fn background_value(&self) -> T;
}

/// The core read surface of a voxel grid, addressed in world [`Coord`]s.
///
/// A grid may be sparse or dense; the difference shows up in `bgv` and in whether
/// `get_voxel` can return `None`. See the **Voxel vs Value** note below.
///
/// Splitting reads from [`VoxGridMut`] is deliberate: a read-only or shared view
/// (e.g. [`VoxGridSlice`]) can implement this trait without being able to mutate,
/// which keeps it usable as a bound anywhere a grid is only read.
///
/// ## Voxel vs Value
///
/// - Voxel: the data at a location in a grid, in sparse trees it may not exist.
/// - Value: the resolved value of that location, defaulting to `bgv()` (then
///   `T::default()`) when no voxel exists.
pub trait VoxGrid<T>: VoxGridBackground<T> {
    /// Accessible limits of the grid in world coords.
    fn limits(&self) -> CoordBBox {
        CoordBBox::new(Coord::MIN, Coord::MAX)
    }

    /// Returns the voxel at `pos`, or `None` if it doesn't exist.
    fn get_voxel(&self, pos: &Coord) -> Option<T>;

    /// Returns the resolved value at position,
    /// or the background value if it doesn't exist
    fn get_value(&self, pos: &Coord) -> T {
        self.get_voxel(pos).unwrap_or(self.background_value())
    }

    /// Visit the values inside `bbox` as uniform tiles.
    ///
    /// `f` is called with disjoint, non-empty boxes that together cover `bbox` exactly,
    /// each with the one resolved value (as [`get_value`](Self::get_value) would return)
    /// shared by every voxel in it. The order is unspecified.
    ///
    /// The default reports each voxel as its own tile. Grids with uniform regions (tree
    /// nodes, unloaded space) should override it to report each region once, which lets
    /// bulk readers like [`read_dense`](Self::read_dense) skip per-voxel lookups.
    fn for_each_tile(&self, bbox: &CoordBBox, f: &mut dyn FnMut(&CoordBBox, &T)) {
        if bbox.empty() {
            return;
        }
        for z in bbox.min.z..=bbox.max.z {
            for y in bbox.min.y..=bbox.max.y {
                for x in bbox.min.x..=bbox.max.x {
                    let p = Coord::new(x, y, z);
                    f(&CoordBBox::new(p, p), &self.get_value(&p));
                }
            }
        }
    }

    /// [`for_each_tile`](Self::for_each_tile) on the coarse lattice of every `2^shift`-th
    /// voxel.
    ///
    /// Coarse coordinate `c` stands for the voxel at `c << shift`. `bbox` and the tiles are
    /// in coarse coordinates, and each tile's value is the [`get_value`](Self::get_value) of
    /// the voxels its coarse coordinates stand for. Voxels between lattice points are not
    /// read. This is point sampling, which suits distance fields: the coarse lattice holds
    /// the exact distances at its points.
    ///
    /// The default calls `for_each_tile` for `shift == 0`, and otherwise reports each
    /// coarse coordinate as its own tile. Tree-backed grids should override it to stop
    /// descending at nodes one coarse voxel wide.
    fn for_each_tile_strided(
        &self,
        bbox: &CoordBBox,
        shift: u32,
        f: &mut dyn FnMut(&CoordBBox, &T),
    ) {
        if shift == 0 {
            return self.for_each_tile(bbox, f);
        }
        if bbox.empty() {
            return;
        }
        for z in bbox.min.z..=bbox.max.z {
            for y in bbox.min.y..=bbox.max.y {
                for x in bbox.min.x..=bbox.max.x {
                    let c = Coord::new(x, y, z);
                    let p = Coord::new(x << shift, y << shift, z << shift);
                    f(&CoordBBox::new(c, c), &self.get_value(&p));
                }
            }
        }
    }

    /// Copy the values inside `bbox` into a [`DenseBlock`].
    ///
    /// The default fills the block from [`for_each_tile`](Self::for_each_tile).
    fn read_dense(&self, bbox: &CoordBBox) -> DenseBlock<T>
    where
        T: Clone,
    {
        DenseBlock::read(self, bbox)
    }
}

/// Mutating access to a [`VoxGrid`], in world [`Coord`]s.
pub trait VoxGridMut<T>: VoxGridBackground<T> {
    /// Sets the voxel at `pos` to `v`, or clears it.
    /// how "clear" works is implementation dependent
    fn set_voxel(&mut self, pos: &Coord, v: Option<T>);

    /// Sets the value at pos
    fn set_value(&mut self, pos: &Coord, v: T)
    where
        T: PartialEq,
    {
        if v == self.background_value() {
            self.set_voxel(pos, None);
        } else {
            self.set_voxel(pos, Some(v));
        }
    }
}
