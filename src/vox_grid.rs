#![cfg(feature = "sdf-grid")]
pub mod sdf;

use super::*;

/// Defines operations on a voxel grid which may or may not be sparse
///
/// ## Voxel vs Value
///
/// - Voxel: the data at a location in a grid, in sparse trees it may not exist.
/// - Value: the resolved value of that location, if no voxel exists it defaults to the value
///   returned by self.bgv() when requested
pub trait VoxGrid<T> {
    /// provide a background value.
    /// Sparse grids require a bgv to default to when checking values
    /// Non sparse grids should always return None
    fn bgv(&self) -> Option<T>;

    /// CoordBBox defining the accesible limits of the VoxGrid in WorldCoords.
    /// A sparse tree has limits Coord::MIN to Coord::MAX
    fn limits(&self) -> CoordBBox {
        CoordBBox::new(Coord::MIN, Coord::MAX)
    }

    /// Returns the value or None if it doesn't exist
    /// Non sparse grids always return Some(v),
    /// bypassing defaults of get_value
    fn get_voxel(&self, pos: &Coord) -> Option<T>;

    /// Returns a reference to a voxel if it exists
    fn get_voxel_ref(&self, pos: &Coord) -> Option<&T>;

    /// returns a mutable reference to a voxel if it exists
    fn get_voxel_mut(&self, pos: &Coord) -> Option<&mut T>;

    /// Returns the value, bgv, or T::default()
    fn get_value(&self, pos: &Coord) -> T
    where
        T: Default,
    {
        self.get_voxel(pos)
            .unwrap_or(self.bgv().unwrap_or_default())
    }

    /// Sets the value at pos, setting None or Some(bgv)
    fn set_voxel(&mut self, pos: &Coord, v: Option<T>);

    /// Sets the value to v or None if v == bgv
    fn set_value(&mut self, pos: &Coord, v: T)
    where
        T: PartialEq,
    {
        if let Some(bgv) = self.bgv()
            && bgv == v
        {
            self.set_voxel(pos, None);
        } else {
            self.set_voxel(pos, Some(v))
        }
    }

    /// Returns the voxel at bbox local coordinates
    fn get_bbox_voxel(&self, bbox: &CoordBBox, pos: &Coord) -> Option<T> {
        self.get_voxel(&bbox.local_to_global(pos))
    }

    /// Returns the voxel ref at bbox local coordinates
    fn get_bbox_voxel_ref(&self, bbox: &CoordBBox, pos: &Coord) -> Option<&T> {
        self.get_voxel_ref(&bbox.local_to_global(pos))
    }

    /// Returns the voxel ref at bbox local coordinates
    fn get_bbox_voxel_mut(&self, bbox: &CoordBBox, pos: &Coord) -> Option<&mut T> {
        self.get_voxel_mut(&bbox.local_to_global(pos))
    }

    /// Returns value at bbox local coordinates
    fn get_bbox_value(&self, bbox: &CoordBBox, pos: &Coord) -> T
    where
        T: Default,
    {
        self.get_value(&bbox.local_to_global(pos))
    }

    /// Set voxel at bbox local coordinates
    fn set_bbox_voxel(&mut self, bbox: &CoordBBox, pos: &Coord, v: Option<T>) {
        self.set_voxel(&bbox.local_to_global(pos), v);
    }

    /// Set value at bbox local coordinates
    fn set_bbox_value(&mut self, bbox: &CoordBBox, pos: &Coord, v: T)
    where
        T: PartialEq,
    {
        self.set_value(&bbox.local_to_global(pos), v);
    }
}
