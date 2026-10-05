use super::*;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LogDim {
    pub log_dim: u8,
    pub sum_child_dims: u8,
    pub total_dim: u8,
    pub node_level: u8,
    pub child_length: usize,
    pub voxel_length: usize,
    pub coord_dim_mask: usize,
    pub is_sparse: bool,
    pub is_leaf: bool,
}

impl LogDim {
    pub const fn new(log_dim: u8, sum_child_dims: u8, node_level: u8, is_sparse: bool) -> Self {
        let total_dim = log_dim + sum_child_dims;
        let child_length = 1_usize << log_dim as usize;
        let voxel_length = 1_usize << total_dim as usize;
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

    pub const fn no_child(log_dim: u8) -> Self {
        Self::new(log_dim, 0, 1, false)
    }

    pub const fn sparse(log_dim: u8, sum_child_dims: u8, node_level: u8) -> Self {
        Self::new(log_dim, sum_child_dims, node_level, true)
    }

    #[inline]
    pub fn node_origin(&self, pos: &Coord) -> Coord {
        if self.is_sparse {
            Coord::ORIGIN
        } else {
            (**pos & !(((1 as Index) << self.total_dim) - 1) as Index).into()
        }
    }

    #[inline]
    pub fn node_offset(&self, pos: &Coord) -> Coord {
        self.node_origin(pos) / self.voxel_length
    }

    #[inline]
    pub fn same_origin(&self, left: &Coord, right: &Coord) -> bool {
        self.node_origin(left) == self.node_origin(right)
    }

    #[inline]
    pub fn voxel_dims(&self) -> Coord {
        [self.voxel_length; 3].into()
    }

    #[inline]
    pub fn child_dims(&self) -> Coord {
        [self.child_length; 3].into()
    }

    #[inline]
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
    pub fn local_bbox(&self) -> CoordBBox {
        CoordBBox::new(
            Coord::ORIGIN,
            Coord::ORIGIN.single_offset_by((self.child_length - 1) as Index),
        )
    }

    pub fn make_child_buffer<T, F: FnMut() -> T>(&self, f: &mut F) -> Vec<T> {
        let size = self.child_length.pow(3);
        let mut result = Vec::with_capacity(size);
        result.resize_with(size, f);
        result
    }

    pub fn make_voxel_buffer<T, F: FnMut() -> T>(&self, f: &mut F) -> Vec<T> {
        let size = self.voxel_length.pow(3);
        let mut result = Vec::with_capacity(size);
        result.resize_with(size, f);
        result
    }

    pub fn global_to_local_child(&self, pos: &Coord) -> Coord {
        let pos = UIndexVec::from(pos);
        let shifted = pos >> self.sum_child_dims;
        let mask = self.coord_dim_mask as UIndex;

        (shifted & mask).into()
    }

    pub fn global_to_local_voxel(&self, pos: &Coord) -> Coord {
        let pos = UIndexVec::from(pos);
        let mask = (self.voxel_length - 1) as UIndex;

        (pos & mask).into()
    }

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TreeDim<const TREE_DEPTH: usize = 8> {
    /// Index zero is the leaf level.
    pub node_dims: [LogDim; TREE_DEPTH],
}

impl<const TREE_DEPTH: usize> TreeDim<TREE_DEPTH> {
    pub const fn new(child_log_dims: [u8; TREE_DEPTH]) -> Self {
        let mut node_dims = [LogDim::no_child(0); TREE_DEPTH];

        let mut child_sum = 0;
        let mut index = 0;

        while index < TREE_DEPTH {
            let is_root = index + 1 == TREE_DEPTH;

            let log_dim = if is_root {
                63 - child_sum
            } else {
                child_log_dims[index]
            };

            node_dims[index] = LogDim::new(log_dim, child_sum, (index + 1) as u8, is_root);

            child_sum += log_dim;
            index += 1;
        }

        Self { node_dims }
    }

    pub const fn sparse_octree() -> Self {
        let child_log_dims: [u8; TREE_DEPTH] = [1; TREE_DEPTH];
        Self::new(child_log_dims)
    }

    #[inline]
    pub fn get_node_level_size(&self, node_level: usize) -> Option<&LogDim> {
        if node_level == 0 || node_level > TREE_DEPTH {
            None
        } else {
            Some(&self.node_dims[node_level - 1])
        }
    }

    #[inline]
    pub fn node_level_size(&self, node_level: usize) -> &LogDim {
        assert!(
            node_level > 0 && node_level <= TREE_DEPTH,
            "node level out of range"
        );

        &self.node_dims[node_level - 1]
    }

    #[inline]
    pub fn leaf(&self) -> &LogDim {
        &self.node_dims[0]
    }

    #[inline]
    pub fn root(&self) -> &LogDim {
        &self.node_dims[TREE_DEPTH - 1]
    }

    #[inline]
    pub fn child(&self, node: &LogDim) -> &LogDim {
        assert!(node.node_level > 1);

        &self.node_dims[node.node_level as usize - 2]
    }

    #[inline]
    pub fn parent(&self, node: &LogDim) -> Option<&LogDim> {
        let index = node.node_level as usize;

        self.node_dims.get(index)
    }

    #[inline]
    pub fn child_voxel_length(&self, node: &LogDim) -> usize {
        self.child(node).voxel_length
    }

    #[inline]
    pub fn child_offset_bbox(&self, node: &LogDim, pos: &Coord) -> CoordBBox {
        let child_stride = self.child_voxel_length(node);
        let mut bbox = node.voxel_bbox(pos);

        bbox.min /= child_stride;
        bbox.max /= child_stride;

        bbox
    }

    pub fn child_coords(&self, node: &LogDim, origin: &Coord) -> Vec<Coord> {
        let local_bbox = node.local_bbox();
        let child_stride = self.child_voxel_length(node);
        let mut child_iter = local_bbox.to_xyz_iter();

        node.make_child_buffer(&mut || {
            let offset = child_iter
                .next()
                .expect("buffer size and iterator size should match")
                * child_stride;

            *origin + offset
        })
    }

    pub fn enclose_children(&self, node: &LogDim, mut bbox: CoordBBox) -> CoordBBox {
        let child = self.child(node);

        bbox.enclose_bbox(&child.voxel_bbox(&child.node_origin(&bbox.min)));

        bbox.enclose_bbox(&child.voxel_bbox(&child.node_origin(&bbox.max)));

        bbox
    }

    #[inline]
    pub fn child_count_in_bbox(&self, node: &LogDim, bbox: &CoordBBox) -> usize {
        let child_stride = self.child_voxel_length(node);
        let [x, y, z]: [usize; 3] = (bbox.axis_dims() / child_stride).into();

        x * y * z
    }

    pub fn overlapping_children(&self, node: &LogDim, bbox: &CoordBBox) -> Vec<Coord> {
        let regions_bbox = self.enclose_children(node, *bbox);
        let child_stride = self.child_voxel_length(node);
        let capacity = self.child_count_in_bbox(node, &regions_bbox);

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

    pub fn child_touching_neighbors(&self, node: &LogDim, pos: &Coord) -> Vec<Coord> {
        let child = self.child(node);
        let pos = child.node_origin(pos);
        let bbox = self.child_offset_bbox(node, &pos);
        let child_stride = self.child_voxel_length(node);
        let offset = pos / child_stride;

        bbox.touching_neighbors(&offset)
            .into_iter()
            .map(|neighbor| (neighbor + offset) * child_stride)
            .collect()
    }
}
