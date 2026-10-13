use super::super::*;
use super::{VoxGrid, VoxGridMut};
use std::marker::PhantomData;

/// A shared, read-only view into a [`VoxGrid`], restricted to a [`CoordBBox`].
///
/// The slice borrows the underlying grid and records a box. As a [`VoxGrid`] it
/// forwards world-coord access to the grid and reports its box as its
/// [`limits`](VoxGrid::limits). Because it is read-only it composes as a `VoxGrid`
/// bound without needing mutation rights over the parent.
///
/// The restriction to `bbox` is advertised through [`limits`](Self::limits); like
/// the parent's own access, in-bounds reads forward the parent's sparse result,
/// so the window does not fabricate values for absent voxels.
#[derive(Copy, Clone, Debug)]
pub struct VoxGridSlice<'a, G, T> {
    grid: &'a G,
    bbox: CoordBBox,
    // Captures `T` (otherwise it would be unused) without affecting variance.
    _marker: PhantomData<T>,
}

impl<'a, G, T> VoxGridSlice<'a, G, T> {
    /// A new slice over `grid` restricted to `bbox`.
    pub fn new(grid: &'a G, bbox: CoordBBox) -> Self {
        Self {
            grid,
            bbox,
            _marker: PhantomData,
        }
    }

    /// The box this slice is restricted to.
    pub fn bbox(&self) -> &CoordBBox {
        &self.bbox
    }
}

impl<'a, G, T> VoxGrid<T> for VoxGridSlice<'a, G, T>
where
    G: VoxGrid<T>,
{
    fn bgv(&self) -> Option<T> {
        self.grid.bgv()
    }

    fn limits(&self) -> CoordBBox {
        self.bbox
    }

    fn get_voxel(&self, pos: &Coord) -> Option<T> {
        self.grid.get_voxel(pos)
    }

    fn get_voxel_ref(&self, pos: &Coord) -> Option<&T> {
        self.grid.get_voxel_ref(pos)
    }
}

/// A mutable view into a [`VoxGrid`], restricted to a [`CoordBBox`].
///
/// Behaves like [`VoxGridSlice`] but takes `&mut`, so it implements both
/// [`VoxGridMut`] and forwards writes to the parent.
#[derive(Debug)]
pub struct VoxGridSliceMut<'a, G, T> {
    grid: &'a mut G,
    bbox: CoordBBox,
    // Captures `T` (otherwise it would be unused) without affecting variance.
    _marker: PhantomData<T>,
}

impl<'a, G, T> VoxGridSliceMut<'a, G, T> {
    /// A new mutable slice over `grid` restricted to `bbox`.
    pub fn new(grid: &'a mut G, bbox: CoordBBox) -> Self {
        Self {
            grid,
            bbox,
            _marker: PhantomData,
        }
    }

    /// The box this slice is restricted to.
    pub fn bbox(&self) -> &CoordBBox {
        &self.bbox
    }
}

impl<'a, G, T> VoxGrid<T> for VoxGridSliceMut<'a, G, T>
where
    G: VoxGrid<T>,
{
    fn bgv(&self) -> Option<T> {
        self.grid.bgv()
    }

    fn limits(&self) -> CoordBBox {
        self.bbox
    }

    fn get_voxel(&self, pos: &Coord) -> Option<T> {
        self.grid.get_voxel(pos)
    }

    fn get_voxel_ref(&self, pos: &Coord) -> Option<&T> {
        self.grid.get_voxel_ref(pos)
    }
}

impl<'a, G, T> VoxGridMut<T> for VoxGridSliceMut<'a, G, T>
where
    G: VoxGridMut<T>,
{
    fn get_voxel_mut(&mut self, pos: &Coord) -> Option<&mut T> {
        self.grid.get_voxel_mut(pos)
    }

    fn set_voxel(&mut self, pos: &Coord, v: Option<T>) {
        self.grid.set_voxel(pos, v);
    }
}
