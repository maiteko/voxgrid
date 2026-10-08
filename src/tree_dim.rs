//! Log-scale voxel-tree dimensions.
//!
//! This module models the per-level geometry of a multi-resolution voxel tree
//! (a sparse voxel tree / mip chain) in terms of *log2* sizes rather than raw
//! sizes, so
//! that parent/child relationships are integer bit shifts instead of
//! divisions.
//!
//! - [`NodeDim`] describes a single node level: the log2 split size of the node
//!    (`log_dim`), the accumulated log2 size of its children (`sum_child_dims`),
//!   and the derived quantities (`total_dim`, `voxel_length`, `coord_dim_mask`,
//!    ...). It maps a global coordinate to its owning node and builds the
//!   child/voxel buffers.
//! - [`TreeDim`] chains `MAX_DEPTH` `NodeDim` levels and provides the parent/child
//!   navigation, child-coordinate enumeration, and overlap queries used to
//!   traverse the tree. Levels are stored **root-first**: the root is at index
//!    0 and the leaf at the highest used index (`node_dims[tree_depth -
//!    node_level]`).
//!
//! Sparse trees (built with [`NodeDim::sparse`] / [`TreeDim::octree`])
//! have no fixed node bounds: `node_origin`, `voxel_bbox`, and `local_bbox`
//! collapse to the origin / an infinite box, since sparse nodes are keyed by
//! their child coordinate rather than by a fixed grid slot.

use std::ops::Deref;

use super::*;

/// A chain of `MAX_DEPTH` node levels, stored **root-first**: index 0 is the
/// root and the highest used index (`tree_depth - 1`) is the leaf.
///
/// `node_dims` is indexed by `tree_depth - node_level`, so `node_dims[0]` is the
/// root (`node_level == tree_depth`) and `node_dims[tree_depth - 1]` is the leaf
/// (`node_level == 1`). Build a tree from the log2 child size of each level with
/// [`TreeDim::new`], or use the [`TreeDim::octree`] helper for a uniform 2x2x2
/// split at every level (a sparse octree, child dim 1).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TreeDim<const MAX_DEPTH: usize = 8> {
    /// Index zero is the root level; the leaf is at index `tree_depth - 1`.
    node_dims: [NodeDim; MAX_DEPTH],
    tree_depth: usize,
}

impl<const MAX_DEPTH: usize> TreeDim<MAX_DEPTH> {
    /// Build a [`TreeDim`] level by level from each level's log2 child size.
    ///
    /// `child_log_dims[k]` gives the log2 child size of the node at `node_level`
    /// `k + 1`; the levels are stored root-first, with the leaf at the highest
    /// used index. The root (top level) is created implicitly with a `log_dim`
    /// of `63 - child_sum`, so the tree spans the full coordinate range. The
    /// depth is capped at `MAX_DEPTH`, so any `child_log_dims` past it are
    /// ignored.
    ///
    /// ```
    /// use voxgrid::TreeDim;
    ///
    /// // leaf log_dim 2 (4x4x4 voxels), one intermediate level, then a root.
    /// let tree = TreeDim::<4>::new(&[2, 3, 4]);
    /// assert_eq!(tree.leaf().log_dim, 2);
    /// assert_eq!(tree.root().node_level, 3);
    /// assert!(tree.root().is_sparse);
    /// ```
    pub const fn new(child_log_dims: &[u8]) -> Self {
        let mut node_dims = [NodeDim::no_child(0); MAX_DEPTH];

        // min is not currently supported in const contexts
        let tree_depth = if child_log_dims.len() > MAX_DEPTH {
            MAX_DEPTH
        } else {
            child_log_dims.len()
        };

        let mut child_sum = 0;
        let mut node_level = 1;

        while node_level < tree_depth {
            let log_dim = child_log_dims[node_level - 1];

            node_dims[tree_depth - node_level] =
                NodeDim::new(log_dim, child_sum, node_level as u8, false);

            child_sum += log_dim;
            node_level += 1;
        }

        node_dims[0] = NodeDim::new(63 - child_sum, child_sum, tree_depth as u8, true);

        Self {
            node_dims,
            tree_depth,
        }
    }

    /// Build a sparse voxel-tree dimension with a uniform 2x2x2 split at every
    /// level except the root (a sparse octree; the non-root levels have child
    /// dim 1).
    ///
    /// ```
    /// use voxgrid::TreeDim;
    ///
    /// let tree: TreeDim = TreeDim::octree();
    /// assert_eq!(tree.leaf().node_level, 1);
    /// assert_eq!(tree.root().node_level, 8);
    /// assert_eq!(tree.leaf().voxel_length, 2); // a 2x2x2 leaf
    /// ```
    pub const fn octree() -> Self {
        let child_log_dims: [u8; MAX_DEPTH] = [1; MAX_DEPTH];
        Self::new(&child_log_dims)
    }

    #[inline]
    /// The [`NodeDim`] for a 1-based `node_level`, or `None` if it is `0` or above `tree_depth`.
    pub const fn get_node_level_size(&self, node_level: usize) -> Option<&NodeDim> {
        if node_level > self.tree_depth || node_level == 0 {
            None
        } else {
            Some(&self.node_dims[self.tree_depth - node_level])
        }
    }

    #[inline]
    /// The [`NodeLevel`] at a 1-based `node_level`, or `None` if it is `0` or outside the tree's depth.
    pub const fn get_node_level<'a>(
        &'a self,
        node_level: usize,
    ) -> Option<NodeLevel<'a, MAX_DEPTH>> {
        if let Some(node) = self.get_node_level_size(node_level) {
            Some(NodeLevel {
                node_dim: node,
                tree_dim: self,
            })
        } else {
            None
        }
    }

    #[inline]
    /// Like [`TreeDim::get_node_level_size`] but panics when `node_level` is
    /// outside `(0, tree_depth]`.
    pub const fn node_level_size(&self, node_level: usize) -> &NodeDim {
        assert!(
            node_level > 0 && node_level <= self.tree_depth,
            "node level out of range"
        );

        &self.node_dims[self.tree_depth - node_level]
    }

    #[inline]
    /// Like [`TreeDim::get_node_level`] but panics when `node_level` is outside `(0, tree_depth]`.
    pub const fn node_level<'a>(&'a self, node_level: usize) -> NodeLevel<'a, MAX_DEPTH> {
        NodeLevel {
            node_dim: self.node_level_size(node_level),
            tree_dim: self,
        }
    }

    #[inline]
    /// The leaf level (`node_dims[tree_depth - 1]`).
    pub const fn leaf<'a>(&'a self) -> NodeLevel<'a, MAX_DEPTH> {
        self.node_level(1)
    }

    #[inline]
    /// The root level (`node_dims[0]`).
    pub const fn root<'a>(&'a self) -> NodeLevel<'a, MAX_DEPTH> {
        self.node_level(self.tree_depth)
    }

    #[inline]
    /// The first child level of the root (`node_level == tree_depth - 1`).
    pub const fn first_child<'a>(&'a self) -> NodeLevel<'a, MAX_DEPTH> {
        self.node_level(self.tree_depth - 1)
    }

    #[inline]
    /// The [`NodeLevel`] at 0-based `idx` (the leaf is `idx == 0`), via [`TreeDim::node_level`].
    pub const fn at_index<'a>(&'a self, idx: usize) -> NodeLevel<'a, MAX_DEPTH> {
        self.node_level(idx + 1)
    }
}

/// The geometry of a single node level in a log-scaled voxel tree.
///
/// Sizes are stored as log2 values so that node/child/voxel dimensions and the
/// coordinate bit-mask needed to extract a local index are simple bit
/// operations. Construct via [`NodeDim::new`], or the convenience constructors
/// [`NodeDim::no_child`] and [`NodeDim::sparse`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeDim {
    /// Log2 of this level's own split size; `child_length` is `2^log_dim`.
    pub log_dim: u8,
    /// Accumulated log2 side length of every level below this node; `total_dim` is `log_dim + sum_child_dims`.
    pub sum_child_dims: u8,
    /// `log_dim + sum_child_dims`, so `voxel_length` is `2^total_dim`.
    pub total_dim: u8,
    /// One-based level index; `1` is the leaf and `MAX_DEPTH` is the root.
    pub node_level: u8,
    /// `2^log_dim`, the number of child nodes per axis at this level.
    pub child_length: usize,
    /// `2^total_dim`, the node's side length in voxels.
    pub voxel_length: usize,
    /// `child_length - 1`, the bit mask that extracts a child-level local index.
    pub coord_dim_mask: usize,
    /// `true` for sparse nodes, keyed by child coordinate rather than a fixed grid slot.
    pub is_sparse: bool,
    /// `true` for the leaf level, where `node_level == 1` and there are no children.
    pub is_leaf: bool,
}

impl Default for NodeDim {
    fn default() -> Self {
        Self::no_child(0)
    }
}

impl NodeDim {
    /// Construct a [`NodeDim`], deriving `total_dim`, `child_length`,
    /// `voxel_length`, `coord_dim_mask`, and `is_leaf` (true when
    /// `node_level == 1`).
    ///
    /// ```
    /// use voxgrid::NodeDim;
    ///
    /// // log2 split of 3 => 8 children; accumulated 2 => 16 voxels on a side.
    /// let n = NodeDim::new(3, 2, 5, false);
    /// assert_eq!(n.total_dim, 5);
    /// assert_eq!(n.child_length, 8);
    /// assert_eq!(n.voxel_length, 32);
    /// assert_eq!(n.coord_dim_mask, 7);
    /// assert!(!n.is_leaf);
    /// ```
    pub const fn new(log_dim: u8, sum_child_dims: u8, node_level: u8, is_sparse: bool) -> Self {
        let total_dim = log_dim + sum_child_dims;

        let (child_length, voxel_length) = if is_sparse {
            let voxel_length = UIndex::MAX as usize;
            let child_length = voxel_length / (1_usize << sum_child_dims);
            (child_length, voxel_length)
        } else {
            let child_length = 1_usize << log_dim as usize;
            let voxel_length = 1_usize << total_dim as usize;
            (child_length, voxel_length)
        };

        let coord_dim_mask = child_length - 1;

        Self {
            log_dim,
            sum_child_dims,
            total_dim,
            node_level,
            child_length,
            voxel_length,
            coord_dim_mask,
            is_sparse,
            is_leaf: node_level == 1,
        }
    }

    /// A leaf node: `sum_child_dims == 0`, `node_level == 1`, and not sparse.
    pub const fn no_child(log_dim: u8) -> Self {
        Self::new(log_dim, 0, 1, false)
    }

    /// Like [`NodeDim::new`] but with `is_sparse == true`, so the node is keyed by its child coordinate.
    pub const fn sparse(log_dim: u8, sum_child_dims: u8, node_level: u8) -> Self {
        Self::new(log_dim, sum_child_dims, node_level, true)
    }

    #[inline]
    /// The node origin of `pos`: [`Coord::ORIGIN`] for sparse nodes, otherwise `pos`
    /// aligned down to the node grid (its low `total_dim` bits cleared).
    pub fn node_origin(&self, pos: &Coord) -> Coord {
        if self.is_sparse {
            Coord::ORIGIN
        } else {
            (**pos & !(((1 as Index) << self.total_dim) - 1) as Index).into()
        }
    }

    #[inline]
    /// The node index of `pos`, i.e. `node_origin(pos) / voxel_length`.
    pub fn node_offset(&self, pos: &Coord) -> Coord {
        self.node_origin(pos) / self.voxel_length
    }

    #[inline]
    /// Whether `left` and `right` fall within the same node.
    pub fn same_origin(&self, left: &Coord, right: &Coord) -> bool {
        self.node_origin(left) == self.node_origin(right)
    }

    #[inline]
    /// The node's side length in voxels on every axis (`[voxel_length; 3]`).
    pub fn voxel_dims(&self) -> Coord {
        [self.voxel_length; 3].into()
    }

    #[inline]
    /// The node's side length in child nodes on every axis (`[child_length; 3]`).
    pub fn child_dims(&self) -> Coord {
        [self.child_length; 3].into()
    }

    #[inline]
    /// The [`CoordBBox`] of the node at `pos`; the infinite box for sparse nodes.
    pub fn voxel_bbox(&self, pos: &Coord) -> CoordBBox {
        if self.is_sparse {
            CoordBBox {
                min: Coord::MIN,
                max: Coord::MAX,
            }
        } else {
            let pos = self.node_origin(pos);

            CoordBBox {
                min: pos,
                max: pos.single_offset_by((self.voxel_length - 1) as Index),
            }
        }
    }

    #[inline]
    /// The local child-index box `[0, child_length - 1]`, independent of `pos`.
    pub fn local_bbox(&self) -> CoordBBox {
        CoordBBox::new(
            Coord::ORIGIN,
            Coord::ORIGIN.single_offset_by((self.child_length - 1) as Index),
        )
    }

    /// Fill a `Vec<T>` of `child_length^3` elements, each produced by `f`.
    pub fn make_child_buffer<T, F: FnMut() -> T>(&self, f: &mut F) -> Vec<T> {
        let size = self.child_length.pow(3);
        let mut result = Vec::with_capacity(size);
        result.resize_with(size, f);
        result
    }

    /// Fill a `Vec<T>` of `voxel_length^3` elements, each produced by `f`.
    pub fn make_voxel_buffer<T, F: FnMut() -> T>(&self, f: &mut F) -> Vec<T> {
        let size = self.voxel_length.pow(3);
        let mut result = Vec::with_capacity(size);
        result.resize_with(size, f);
        result
    }

    /// The child-level local coordinate of `pos`: shifted right by `sum_child_dims`
    /// and masked by `coord_dim_mask`.
    pub fn global_to_local_child(&self, pos: &Coord) -> LocalCoord {
        let pos = UIndexVec::from(pos);
        let shifted = pos >> self.sum_child_dims;
        let mask = self.coord_dim_mask as UIndex;

        (shifted & mask).into()
    }

    /// The voxel-level local coordinate of `pos`, masked to `voxel_length - 1`.
    pub fn global_to_local_voxel(&self, pos: &Coord) -> LocalCoord {
        let pos = UIndexVec::from(pos);
        let mask = (self.voxel_length - 1) as UIndex;

        (pos & mask).into()
    }

    /// Call `f` with the node origin of each of the six face-adjacent nodes of `pos`.
    pub fn for_each_face_neighbor(&self, pos: &Coord, mut f: impl FnMut(&Coord)) {
        use crate::cube::SIDE_NORMALS;

        let voxel_length = self.voxel_length as Index;

        for side_normal in SIDE_NORMALS {
            let offset: Coord = [
                side_normal.x * voxel_length,
                side_normal.y * voxel_length,
                side_normal.z * voxel_length,
            ]
            .into();

            f(&self.node_origin(&(*pos + offset)));
        }
    }
}

/// A borrowed view of one level of a [`TreeDim`] tree.
///
/// It holds a reference to the owning [`TreeDim`] (`tree_dim`) and to that level's
/// [`NodeDim`] (`node_dim`), and derefs to the `NodeDim` so its fields and methods
/// are reachable directly. On top of that it offers parent/child navigation within
/// the tree ([`NodeLevel::child`], [`NodeLevel::parent`], [`NodeLevel::root`],
/// [`NodeLevel::leaf`]) plus the child-enumeration and overlap queries
/// ([`NodeLevel::child_coords`], [`NodeLevel::overlapping_children`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeLevel<'a, const MAX_DEPTH: usize> {
    tree_dim: &'a TreeDim<MAX_DEPTH>,
    node_dim: &'a NodeDim,
}

impl<'a, const MAX_DEPTH: usize> NodeLevel<'a, MAX_DEPTH> {
    #[inline]
    /// The root level of the owning tree (see [`TreeDim::root`]).
    pub fn root(&self) -> Self {
        self.tree_dim.root()
    }

    #[inline]
    /// The leaf level of the owning tree (see [`TreeDim::leaf`]).
    pub fn leaf(&self) -> Self {
        self.tree_dim.leaf()
    }

    #[inline]
    /// The level one below this one (`node_level - 1`), or `None` if this is the leaf.
    pub const fn child(&self) -> Option<NodeLevel<'a, MAX_DEPTH>> {
        self.tree_dim
            .get_node_level((self.node_dim.node_level - 1) as usize)
    }

    #[inline]
    /// The level one above this one (`node_level + 1`), or `None` if this is the root.
    pub const fn parent(&self) -> Option<NodeLevel<'a, MAX_DEPTH>> {
        self.tree_dim
            .get_node_level((self.node_dim.node_level + 1) as usize)
    }

    #[inline]
    /// Returns the "child's" voxel length
    ///
    /// In this context, each voxel of a leaf node is an independent "child"
    /// with voxel length 1
    pub fn child_stride(&self) -> usize {
        match self.child() {
            Some(c) => c.voxel_length,
            None => 1,
        }
    }

    #[inline]
    /// The box, in child-offset units, covering the children of the node at `pos`.
    ///
    /// It is the node's [`NodeDim::voxel_bbox`] with each corner floored to the
    /// [`NodeLevel::child_stride`].
    pub fn child_offset_bbox(&self, pos: &Coord) -> CoordBBox {
        let child_stride = self.child_stride();

        let mut bbox = self.voxel_bbox(pos);

        bbox.min /= child_stride;
        bbox.max /= child_stride;

        bbox
    }

    /// The global [`Coord`] of each child origin inside the node at `origin`.
    pub fn child_coords(&self, origin: &Coord) -> Vec<Coord> {
        let local_bbox = self.local_bbox();
        let child_stride = self.child_stride();
        let mut child_iter = local_bbox.to_xyz_iter();
        let origin = self.node_origin(origin);
        self.make_child_buffer(&mut || {
            let offset = child_iter
                .next()
                .expect("buffer size and iterator size should match")
                * child_stride;

            origin + offset
        })
    }

    /// Grow `bbox` to enclose the child nodes of this level it touches.
    pub fn enclose_children(&self, mut bbox: CoordBBox) -> CoordBBox {
        if let Some(child) = self.child() {
            bbox.enclose_bbox(&child.voxel_bbox(&child.node_origin(&bbox.min)));

            bbox.enclose_bbox(&child.voxel_bbox(&child.node_origin(&bbox.max)));
        }

        bbox
    }

    #[inline]

    /// The number of child nodes of this level that touch `bbox`.
    ///
    /// The bbox is transformed to enclose all children, ensuring
    /// it aligns with self.child_stride()
    pub fn child_count_in_bbox(&self, bbox: &CoordBBox) -> usize {
        let bbox = self.enclose_children(*bbox);
        let child_stride = self.child_stride();

        let [x, y, z]: [usize; 3] = (bbox.axis_dims() / child_stride).into();

        x * y * z
    }

    /// The child [`Coord`]s of every child node of this level overlapping `bbox`.
    pub fn overlapping_children(&self, bbox: &CoordBBox) -> Vec<Coord> {
        let regions_bbox = self.enclose_children(*bbox);
        let child_stride = self.child_stride();
        let capacity = self.child_count_in_bbox(&regions_bbox);

        let mut result = Vec::with_capacity(capacity);

        for x in regions_bbox.range_x().step_by(child_stride) {
            for y in regions_bbox.range_y().step_by(child_stride) {
                for z in regions_bbox.range_z().step_by(child_stride) {
                    result.push([x, y, z].into());
                }
            }
        }

        result
    }

    /// The [`Coord`]s of a boundary child's neighbors across the boundary line.
    ///
    /// Empty when `pos` is not a boundary child of this level (or the level has no
    /// children). Used to check whether a boundary-neighbor cell has been
    /// loaded/generated so child's SDF or mesh can be computed.
    pub fn child_boundary_neighbors(&self, pos: &Coord) -> Vec<Coord> {
        let child = match self.child() {
            Some(c) => c,
            None => return Vec::new(),
        };
        let pos = child.node_origin(pos);
        let bbox = self.child_offset_bbox(&pos);
        let child_stride = self.child_stride();
        let offset = pos / child_stride;

        bbox.touching_neighbors(&offset)
            .into_iter()
            .map(|neighbor| (neighbor + offset) * child_stride)
            .collect()
    }
}

impl<'a, const MAX_DEPTH: usize> Deref for NodeLevel<'a, MAX_DEPTH> {
    type Target = NodeDim;

    fn deref(&self) -> &Self::Target {
        self.node_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- NodeDim construction ------------------------------------------------
    #[test]
    fn test_new_derives_fields() {
        let n = NodeDim::new(3, 2, 5, false);
        assert_eq!(n.log_dim, 3);
        assert_eq!(n.sum_child_dims, 2);
        assert_eq!(n.total_dim, 5); // log_dim + sum_child_dims
        assert_eq!(n.child_length, 8); // 2^3
        assert_eq!(n.voxel_length, 32); // 2^5
        assert_eq!(n.coord_dim_mask, 7); // child_length - 1
        assert_eq!(n.node_level, 5);
        assert!(!n.is_sparse);
        assert!(!n.is_leaf); // node_level != 1
    }

    #[test]
    fn test_is_leaf_only_at_level_one() {
        assert!(NodeDim::new(1, 0, 1, false).is_leaf);
        assert!(!NodeDim::new(1, 0, 2, false).is_leaf);
    }

    #[test]
    fn test_no_child_is_leaf() {
        let n = NodeDim::no_child(2);
        assert_eq!(n.log_dim, 2);
        assert_eq!(n.sum_child_dims, 0);
        assert_eq!(n.node_level, 1);
        assert!(n.is_leaf);
        assert!(!n.is_sparse);
        assert_eq!(n.voxel_length, 4);
        assert_eq!(n.child_length, 4);
        assert_eq!(n.coord_dim_mask, 3);
    }

    #[test]
    fn test_sparse_sets_flag_and_keeps_geometry() {
        let n = NodeDim::sparse(1, 2, 3);
        assert!(n.is_sparse);
        assert_eq!(n.total_dim, 3); // log_dim + sum_child_dims
        assert_eq!(n.node_level, 3);
        assert!(!n.is_leaf);
    }

    #[test]
    fn test_default_is_all_no_child_zero() {
        let d = NodeDim::default();
        assert_eq!(d, NodeDim::no_child(0));
        assert_eq!(d.child_length, 1);
        assert_eq!(d.voxel_length, 1);
        assert_eq!(d.coord_dim_mask, 0);
        assert!(d.is_leaf);
    }

    // ---- NodeDim geometry ----------------------------------------------------

    #[test]
    fn test_node_origin_dense_aligns_down() {
        // total_dim 2 => voxel_length 4 => clear the low two bits.
        let n = NodeDim::new(1, 1, 2, false);
        assert_eq!(n.node_origin(&Coord::new(5, 7, 3)), Coord::new(4, 4, 0));
        assert_eq!(
            n.node_origin(&Coord::new(-5, -7, -3)),
            Coord::new(-8, -8, -4)
        );
    }

    #[test]
    fn test_node_origin_sparse_is_origin() {
        let n = NodeDim::sparse(1, 2, 3);
        assert_eq!(n.node_origin(&Coord::new(7, 3, 11)), Coord::ORIGIN);
    }

    #[test]
    fn test_node_offset_is_grid_index() {
        let n = NodeDim::new(1, 1, 2, false); // voxel_length 4
        assert_eq!(n.node_offset(&Coord::new(5, 7, 3)), Coord::new(1, 1, 0));
        assert_eq!(n.node_offset(&Coord::new(12, 8, 0)), Coord::new(3, 2, 0));
    }

    #[test]
    fn test_same_origin() {
        let n = NodeDim::new(1, 1, 2, false); // voxel_length 4
        assert!(n.same_origin(&Coord::new(1, 0, 0), &Coord::new(3, 0, 0)));
        assert!(n.same_origin(&Coord::new(0, 0, 3), &Coord::new(0, 2, 3)));
        assert!(!n.same_origin(&Coord::new(0, 0, 0), &Coord::new(4, 0, 0)));
    }

    #[test]
    fn test_voxel_and_child_dims() {
        let n = NodeDim::new(1, 1, 2, false); // total_dim 2, log_dim 1
        assert_eq!(n.voxel_dims(), Coord::new(4, 4, 4));
        assert_eq!(n.child_dims(), Coord::new(2, 2, 2));
    }

    #[test]
    fn test_voxel_bbox_dense() {
        let n = NodeDim::new(1, 1, 2, false); // voxel_length 4
        let bbox = n.voxel_bbox(&Coord::new(5, 7, 3));
        assert_eq!(bbox.min, Coord::new(4, 4, 0));
        assert_eq!(bbox.max, Coord::new(7, 7, 3));
    }

    #[test]
    fn test_voxel_bbox_sparse_is_infinite() {
        let n = NodeDim::sparse(1, 2, 3);
        assert_eq!(n.voxel_bbox(&Coord::new(1, 2, 3)), CoordBBox::inf());
    }

    #[test]
    fn test_local_bbox() {
        let n = NodeDim::new(2, 0, 1, false); // child_length 4
        let bbox = n.local_bbox();
        assert_eq!(bbox.min, Coord::ORIGIN);
        assert_eq!(bbox.max, Coord::new(3, 3, 3));
    }

    #[test]
    fn test_global_to_local_child_shifts_and_masks() {
        // log_dim 1, sum_child_dims 2 => extract bit 2.
        let n = NodeDim::new(1, 2, 3, false);
        assert_eq!(
            n.global_to_local_child(&Coord::new(4, 4, 4)),
            Coord::new(1, 1, 1)
        );
        assert_eq!(
            n.global_to_local_child(&Coord::new(8, 8, 8)),
            Coord::new(0, 0, 0)
        );
        assert_eq!(n.global_to_local_child(&Coord::ORIGIN), Coord::ORIGIN);
    }

    #[test]
    fn test_global_to_local_voxel_masks_low_bits() {
        // total_dim 3 => voxel_length 8 => mask 7.
        let n = NodeDim::new(1, 2, 3, false);
        assert_eq!(
            n.global_to_local_voxel(&Coord::new(10, 10, 10)),
            Coord::new(2, 2, 2)
        );
        assert_eq!(
            n.global_to_local_voxel(&Coord::new(15, 15, 15)),
            Coord::new(7, 7, 7)
        );
        assert_eq!(n.global_to_local_voxel(&Coord::new(8, 8, 8)), Coord::ORIGIN);
    }

    #[test]
    fn test_child_local_differs_from_voxel_local() {
        // The child path shifts by sum_child_dims while the voxel path does not,
        // so they diverge when sum_child_dims != 0.
        let n = NodeDim::new(1, 2, 3, false);
        let pos = Coord::new(10, 10, 10);
        assert_eq!(n.global_to_local_child(&pos), Coord::ORIGIN);
        assert_eq!(n.global_to_local_voxel(&pos), Coord::new(2, 2, 2));
        assert_ne!(n.global_to_local_child(&pos), n.global_to_local_voxel(&pos));
    }

    #[test]
    fn test_make_buffers_have_expected_length() {
        let n = NodeDim::new(2, 0, 1, false); // child_length 4, voxel_length 4
        assert_eq!(n.make_child_buffer(&mut || 0).len(), 64); // 4^3
        assert_eq!(n.make_voxel_buffer(&mut || 0).len(), 64); // 4^3
    }

    #[test]
    fn test_for_each_face_neighbor() {
        let n = NodeDim::new(3, 0, 1, false); // voxel_length 8
        let mut origins: Vec<Coord> = Vec::new();
        n.for_each_face_neighbor(&Coord::ORIGIN, |o| origins.push(*o));

        assert_eq!(origins.len(), 6);
        let expected = [
            Coord::new(-8, 0, 0),
            Coord::new(8, 0, 0),
            Coord::new(0, -8, 0),
            Coord::new(0, 8, 0),
            Coord::new(0, 0, -8),
            Coord::new(0, 0, 8),
        ];
        for e in expected {
            assert!(origins.contains(&e));
        }
    }

    // ---- TreeDim layout ------------------------------------------------------

    #[test]
    fn test_new_layout_three_levels() {
        let tree = TreeDim::<4>::new(&[2, 3, 4]);
        assert_eq!(tree.tree_depth, 3);

        let root = tree.root();
        let leaf = tree.leaf();
        assert_eq!(root.node_level, 3);
        assert!(root.is_sparse);
        assert_eq!(root.log_dim, 58); // 63 - child_sum (2 + 3)
        assert_eq!(root.sum_child_dims, 5);
        assert_eq!(root.total_dim, 63);

        assert_eq!(leaf.node_level, 1);
        assert!(leaf.is_leaf);
        assert!(!leaf.is_sparse);
        assert_eq!(leaf.log_dim, 2); // child_log_dims[0]
        assert_eq!(leaf.sum_child_dims, 0);
        assert_eq!(leaf.voxel_length, 4);
        let child = root.child().unwrap();
        // Tiling invariant: total_dim(level) = total_dim(child) + log_dim(level).
        assert_eq!(root.total_dim, child.total_dim + root.log_dim);
        assert_eq!(
            root.child().unwrap().total_dim,
            leaf.total_dim + root.child().unwrap().log_dim
        );
    }

    #[test]
    fn test_new_caps_depth_at_max_depth() {
        // More entries than MAX_DEPTH: the extras are ignored.
        let tree = TreeDim::<2>::new(&[2, 3, 4, 5]);
        assert_eq!(tree.tree_depth, 2);
        assert_eq!(tree.leaf().log_dim, 2); // only child_log_dims[0] is used
        assert_eq!(tree.root().node_level, 2);
    }

    #[test]
    fn test_octree_layout() {
        let tree: TreeDim = TreeDim::octree(); // MAX_DEPTH 8, uniform log_dim 1
        assert_eq!(tree.tree_depth, 8);

        let root = tree.root();
        let leaf = tree.leaf();
        assert_eq!(root.node_level, 8);
        assert_eq!(leaf.node_level, 1);
        assert!(leaf.is_leaf);
        assert!(root.is_sparse);
        assert_eq!(root.log_dim, 56); // 63 - 7 non-root levels
        assert_eq!(leaf.voxel_length, 2);

        // Every level 1..=8 is addressable; 0 and 9 are out of range.
        for lvl in 1..=8usize {
            assert!(tree.get_node_level_size(lvl).is_some());
            assert_eq!(tree.node_level_size(lvl).node_level, lvl as u8);
        }
        assert!(tree.get_node_level_size(0).is_none());
        assert!(tree.get_node_level_size(9).is_none());

        // Walk down from the root to the leaf.
        let mut node = tree.root();
        for _ in 0..7 {
            node = node.child().unwrap();
        }
        assert_eq!(node.node_level, 1);
        assert!(node.is_leaf);
        assert_eq!(node, tree.leaf());
    }

    #[test]
    fn test_get_node_level_size_bounds() {
        let tree = TreeDim::<3>::new(&[2, 3]); // tree_depth 2
        assert!(tree.get_node_level_size(0).is_none());
        assert!(tree.get_node_level_size(3).is_none()); // 3 > tree_depth
        assert!(tree.get_node_level_size(1).is_some());
        assert!(tree.get_node_level_size(2).is_some());
    }

    #[test]
    fn test_get_and_node_level_size_agree() {
        let tree = TreeDim::<3>::new(&[2, 3]);
        for lvl in 1..=tree.tree_depth {
            assert_eq!(
                tree.get_node_level_size(lvl),
                Some(tree.node_level_size(lvl))
            );
        }
    }

    #[test]
    fn test_child_parent_round_trip() {
        let tree = TreeDim::<4>::new(&[2, 3, 4]); // levels 1..=3
        let l2 = tree.get_node_level(2).unwrap();
        assert_eq!(l2.child(), tree.get_node_level(1));
        assert_eq!(l2.parent(), tree.get_node_level(3));
        assert_eq!(tree.root().parent(), None); // nothing above the root
    }

    #[test]
    #[should_panic(expected = "node level out of range")]
    fn test_node_level_size_zero_panics() {
        let tree = TreeDim::<3>::new(&[2, 3]);
        let _ = tree.node_level_size(0);
    }

    #[test]
    #[should_panic(expected = "node level out of range")]
    fn test_node_level_size_above_tree_depth_panics() {
        // tree_depth is 2 but MAX_DEPTH is 3; level 3 must panic instead of
        // underflowing the index `tree_depth - node_level`.
        let tree = TreeDim::<3>::new(&[2, 3]);
        let _ = tree.node_level_size(3);
    }

    #[test]
    fn test_child_of_leaf_panics() {
        let tree = TreeDim::<3>::new(&[2, 3]);
        let _ = tree.leaf().child();
    }

    // ---- child enumeration / overlap queries (octree level 2, stride 2) ------

    #[test]
    fn test_child_voxel_length() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap();
        assert_eq!(l2.child_stride(), 2); // level 1 voxel_length
        assert_eq!(l2.child().unwrap(), tree.leaf());
    }

    #[test]
    fn test_child_coords_octree_level2() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap();
        let coords = l2.child_coords(&Coord::ORIGIN);
        assert_eq!(coords.len(), 8); // 2 children per axis

        let mut got = coords;
        got.sort();
        let mut want = vec![
            Coord::new(0, 0, 0),
            Coord::new(0, 0, 2),
            Coord::new(0, 2, 0),
            Coord::new(0, 2, 2),
            Coord::new(2, 0, 0),
            Coord::new(2, 0, 2),
            Coord::new(2, 2, 0),
            Coord::new(2, 2, 2),
        ];
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn test_child_coords_at_offset() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap();
        let coords = l2.child_coords(&Coord::new(4, 4, 4));
        assert_eq!(coords.len(), 8);
        for c in &coords {
            for v in c.as_array() {
                assert!(v == 4 || v == 6); // origin 4 + {0, 2}
            }
        }
    }

    #[test]
    fn test_child_offset_bbox() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap();
        // level-2 voxel box at (4,4,4) is [4..7]^3; divide by stride 2.
        let bbox = l2.child_offset_bbox(&Coord::new(4, 4, 4));
        assert_eq!(bbox.min, Coord::new(2, 2, 2));
        assert_eq!(bbox.max, Coord::new(3, 3, 3));
    }

    #[test]
    fn test_enclose_children_grows_to_child_grid() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap();

        // A single point at the origin expands to the leaf voxel box [0..1]^3.
        let point = CoordBBox::new(Coord::ORIGIN, Coord::ORIGIN);
        let enclosed = l2.enclose_children(point);
        assert_eq!(enclosed.min, Coord::ORIGIN);
        assert_eq!(enclosed.max, Coord::new(1, 1, 1));

        // [2..5]^3 already spans whole child nodes, so it is unchanged.
        let aligned = CoordBBox::new(Coord::new(2, 2, 2), Coord::new(5, 5, 5));
        assert_eq!(l2.enclose_children(aligned), aligned);
    }

    #[test]
    fn test_child_count_in_bbox() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap(); // stride 2
        assert_eq!(
            l2.child_count_in_bbox(&CoordBBox::new(Coord::ORIGIN, Coord::new(1, 1, 1))),
            1
        );
        assert_eq!(
            l2.child_count_in_bbox(&CoordBBox::new(Coord::ORIGIN, Coord::new(3, 3, 3))),
            8
        );
        assert_eq!(
            l2.child_count_in_bbox(&CoordBBox::new(Coord::ORIGIN, Coord::new(7, 7, 7))),
            64
        );
    }

    #[test]
    fn test_overlapping_children() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap(); // stride 2

        // A single point overlaps exactly one child node.
        assert_eq!(
            l2.overlapping_children(&CoordBBox::new(Coord::ORIGIN, Coord::ORIGIN)),
            vec![Coord::ORIGIN]
        );

        // The full level-2 box overlaps all 8 child nodes.
        let all = l2.overlapping_children(&CoordBBox::new(Coord::ORIGIN, Coord::new(3, 3, 3)));
        assert_eq!(all.len(), 8);
        let mut got = all;
        got.sort();
        let mut want = vec![
            Coord::new(0, 0, 0),
            Coord::new(0, 0, 2),
            Coord::new(0, 2, 0),
            Coord::new(0, 2, 2),
            Coord::new(2, 0, 0),
            Coord::new(2, 0, 2),
            Coord::new(2, 2, 0),
            Coord::new(2, 2, 2),
        ];
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn test_child_touching_neighbors() {
        let tree: TreeDim = TreeDim::octree();
        let l2 = tree.get_node_level(2).unwrap(); // stride 2

        // The origin's node touches 7 child nodes in the negative octant.
        let at_origin = l2.child_boundary_neighbors(&Coord::ORIGIN);
        assert_eq!(at_origin.len(), 7);
        let mut got = at_origin;
        got.sort();
        let mut want = vec![
            Coord::new(-2, 0, 0),
            Coord::new(0, -2, 0),
            Coord::new(0, 0, -2),
            Coord::new(-2, -2, 0),
            Coord::new(-2, 0, -2),
            Coord::new(0, -2, -2),
            Coord::new(-2, -2, -2),
        ];
        want.sort();
        assert_eq!(got, want);

        // Off origin, the child at grid index (2,2,2) touches 7 neighbours.
        let at_offset = l2.child_boundary_neighbors(&Coord::new(4, 4, 4));
        assert_eq!(at_offset.len(), 7);
        assert!(at_offset.contains(&Coord::new(2, 2, 2)));
        assert!(at_offset.contains(&Coord::new(2, 4, 4)));
        assert!(at_offset.contains(&Coord::new(4, 2, 4)));
    }
}
