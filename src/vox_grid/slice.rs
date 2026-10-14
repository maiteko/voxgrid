use super::super::*;
use super::{VoxGrid, VoxGridMut};
use std::marker::PhantomData;

/// A shared, read-only view into a [`VoxGrid`], restricted to a [`CoordBBox`].
///
/// The slice borrows the underlying grid and records a box. converts local bbox coordinates
/// to global, and forwards them to the underlying grid
///
/// The global `bbox` is advertised through [`bbox`](Self::bbox);
#[derive(Copy, Clone, Debug)]
pub struct VoxGridSlice<'a, G, T> {
    grid: &'a G,
    bbox: CoordBBox,
    // Captures `T` (otherwise it would be unused) without affecting variance.
    _marker: PhantomData<T>,
}

impl<'a, G, T> VoxGridSlice<'a, G, T>
where
    G: VoxGrid<T>,
{
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

    /// Create a new read reference with bbox size modified by padding
    pub fn pad(&self, padding: i32) -> Self {
        Self::new(self.grid, self.bbox.expand_by(padding))
    }

    pub fn get_voxel(&self, pos: &LocalCoord) -> Option<T> {
        self.grid.get_voxel(&self.bbox().local_to_global(pos).ok()?)
    }

    pub fn get_value(&self, pos: &LocalCoord) -> T {
        self.grid
            .get_value(&self.bbox().local_to_global(pos).unwrap())
    }
}

/// A mutable view into a [`VoxGrid`], restricted to a [`CoordBBox`].
#[derive(Debug)]
pub struct VoxGridSliceMut<'a, G, T> {
    grid: &'a mut G,
    bbox: CoordBBox,
    // Captures `T` (otherwise it would be unused) without affecting variance.
    _marker: PhantomData<T>,
}

impl<'a, G, T> VoxGridSliceMut<'a, G, T>
where
    G: VoxGridMut<T> + VoxGrid<T>,
{
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

    /// Pads bbox in place
    pub fn pad(&mut self, padding: i32) {
        self.bbox.expand(padding);
    }

    pub fn get_voxel(&self, pos: &LocalCoord) -> Option<T> {
        self.grid.get_voxel(&self.bbox().local_to_global(pos).ok()?)
    }

    pub fn get_value(&self, pos: &LocalCoord) -> T {
        self.grid
            .get_value(&self.bbox().local_to_global(pos).unwrap())
    }

    pub fn set_voxel(&mut self, pos: &LocalCoord, v: Option<T>) {
        self.grid
            .set_voxel(&self.bbox().local_to_global(pos).unwrap(), v);
    }

    pub fn set_value(&mut self, pos: &LocalCoord, v: T)
    where
        T: PartialEq,
    {
        self.grid
            .set_value(&self.bbox().local_to_global(pos).unwrap(), v);
    }
}
