use super::*;

mod slice;

pub use slice::{VoxGridSlice, VoxGridSliceMut};

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
pub trait VoxGrid<T> {
    /// Background value for a sparse grid, or `None` for a dense grid.
    ///
    /// Sparse grids require a bgv to default to when checking values.
    /// Non sparse grids should always return `None`.
    fn bgv(&self) -> Option<T>;

    /// Accessible limits of the grid in world coords.
    ///
    /// A sparse tree has limits `Coord::MIN` to `Coord::MAX`; bounded grids
    /// (and local grids) override this.
    fn limits(&self) -> CoordBBox {
        CoordBBox::new(Coord::MIN, Coord::MAX)
    }

    /// Returns the voxel at `pos`, or `None` if it doesn't exist.
    fn get_voxel(&self, pos: &Coord) -> Option<T>;

    /// Returns a reference to the voxel at `pos`, if it exists.
    fn get_voxel_ref(&self, pos: &Coord) -> Option<&T>;

    /// Returns the resolved value at `pos`: the voxel, else `bgv()`, else
    /// `T::default()`.
    fn get_value(&self, pos: &Coord) -> T
    where
        T: Default,
    {
        self.get_voxel(pos)
            .unwrap_or(self.bgv().unwrap_or_default())
    }
}

/// Mutating access to a [`VoxGrid`], in world [`Coord`]s.
pub trait VoxGridMut<T>: VoxGrid<T> {
    /// Returns a mutable reference to the voxel at `pos`, if it exists.
    fn get_voxel_mut(&mut self, pos: &Coord) -> Option<&mut T>;

    /// Sets the voxel at `pos` to `v`, or clears it by setting `None`.
    fn set_voxel(&mut self, pos: &Coord, v: Option<T>);

    /// Sets the value at `pos` to `v`, or clears the voxel if `v == bgv()`.
    fn set_value(&mut self, pos: &Coord, v: T)
    where
        T: PartialEq,
    {
        if let Some(bgv) = self.bgv()
            && bgv == v
        {
            self.set_voxel(pos, None);
        } else {
            self.set_voxel(pos, Some(v));
        }
    }
}
