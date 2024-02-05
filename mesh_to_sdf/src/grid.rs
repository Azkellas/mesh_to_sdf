use crate::point::Point;

/// Result of snapping a point to the grid.
/// If the point is inside the grid, the cell it is within is returned.
/// If the point is outside the grid, the cell index is the nearest cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapResult {
    /// The point is inside the grid.
    /// Cell index is the cell it is within.
    Inside([usize; 3]),
    /// The point is outside the grid
    /// Cell index is the cell it is the nearest from.
    Outside([usize; 3]),
}

/// Helper struct to represent a grid for grid sdf.
/// A grid is defined by three parameters:
/// - `first_cell`: the position of the center of the first cell.
/// - `cell_size`: the size of a cell (i.e. the size of a voxel).
/// - `cell_count`: the number of cells in each direction (i.e. the number of voxels in each direction).
///
/// Note that if you want to sample x in 0 1 2 .. 10, you need 11 cells in this direction and not 10.
///
/// - `cell_size` can be different in each direction and even negative.
/// - `cell_count` can be different in each direction
#[derive(Debug, Clone)]
pub struct Grid<V: Point> {
    /// The center of the first cell.
    first_cell: V,
    /// The size of a cell. A cell goes from `center - cell_size / 2` to `center + cell_size / 2`.
    cell_size: V,
    /// The number of cells in each direction.
    cell_count: [usize; 3],
}

impl<V: Point> Grid<V> {
    /// Create a new grid.
    /// - `first_cell` is the center of the first cell.
    /// - `cell_size` is the size of a cell. A cell goes from `center - cell_size / 2` to `center + cell_size / 2`.
    pub fn new(first_cell: V, cell_size: V, cell_count: [usize; 3]) -> Self {
        Self {
            first_cell,
            cell_size,
            cell_count,
        }
    }

    /// Create a new grid from a bounding box.
    /// - `min_cell` is the minimum corner of the bounding box.
    /// - `max_cell` is the maximum corner of the bounding box.
    /// - `cell_count` is the number of cells in each direction.
    ///
    /// The grid will be centered in the bounding box.
    /// The size of a cell will be `bounding_box_size / cell_count`.
    /// The first cell will be at `min_cell + cell_size / 2`.
    pub fn from_bounding_box(min_cell: &V, max_cell: &V, cell_count: &[usize; 3]) -> Self {
        let fcell_count = V::new(
            cell_count[0] as f32,
            cell_count[1] as f32,
            cell_count[2] as f32,
        );
        let cell_size = max_cell.sub(min_cell).comp_div(&fcell_count);
        // We add half a cell size to the first cell to center it.
        let first_cell = min_cell.add(&cell_size.fmul(0.5));

        Self {
            first_cell,
            cell_size,
            cell_count: *cell_count,
        }
    }

    /// Get the center of the first cell.
    pub fn get_first_cell(&self) -> V {
        self.first_cell
    }

    /// Get the center of the last cell.
    pub fn get_last_cell(&self) -> V {
        V::new(
            self.first_cell.x() + self.cell_count[0] as f32 * self.cell_size.x(),
            self.first_cell.y() + self.cell_count[1] as f32 * self.cell_size.y(),
            self.first_cell.z() + self.cell_count[2] as f32 * self.cell_size.z(),
        )
    }

    /// Get the size of a cell.
    pub fn get_cell_size(&self) -> V {
        self.cell_size
    }

    /// Get the number of cells in each direction.
    pub fn get_cell_count(&self) -> [usize; 3] {
        self.cell_count
    }

    /// Get the total  of cells.
    pub fn get_total_cell_count(&self) -> usize {
        self.cell_count[0] * self.cell_count[1] * self.cell_count[2]
    }

    /// Get bouding box.
    ///
    /// The bounding box is defined by the minimum and maximum corners.
    /// - The minimum corner is the center of the first cell minus half a cell size.
    /// - The maximum corner is the center of the last cell plus half a cell size.
    pub fn get_bounding_box(&self) -> (V, V) {
        let min = self.first_cell.sub(&self.cell_size.fmul(0.5));
        let max = V::new(
            min.x() + self.cell_count[0] as f32 * self.cell_size.x(),
            min.y() + self.cell_count[1] as f32 * self.cell_size.y(),
            min.z() + self.cell_count[2] as f32 * self.cell_size.z(),
        );

        (min, max)
    }

    /// Get the index of a cell in a grid.
    pub fn get_cell_idx(&self, cell: &[usize; 3]) -> usize {
        cell[2] + cell[1] * self.cell_count[2] + cell[0] * self.cell_count[1] * self.cell_count[2]
    }

    /// Get the position of a cell in a grid.
    pub fn get_cell_center(&self, cell: &[usize; 3]) -> V {
        V::new(
            self.first_cell.x() + cell[0] as f32 * self.cell_size.x(),
            self.first_cell.y() + cell[1] as f32 * self.cell_size.y(),
            self.first_cell.z() + cell[2] as f32 * self.cell_size.z(),
        )
    }

    /// Snap a point to the grid.
    /// Returns a `SnapResult` specifying if the point is inside or outside the grid.
    pub fn snap_point_to_grid(&self, point: &V) -> SnapResult {
        let cell = point
            .sub(&self.get_bounding_box().0)
            .comp_div(&self.cell_size);

        let cell = [
            cell.x().floor() as isize,
            cell.y().floor() as isize,
            cell.z().floor() as isize,
        ];

        let ires = [
            cell[0].clamp(0, self.cell_count[0] as isize - 1),
            cell[1].clamp(0, self.cell_count[1] as isize - 1),
            cell[2].clamp(0, self.cell_count[2] as isize - 1),
        ];

        let res = [ires[0] as usize, ires[1] as usize, ires[2] as usize];

        if ires != cell {
            SnapResult::Outside(res)
        } else {
            SnapResult::Inside(res)
        }
    }

    // TODO: provide functions to get distance from any point with interpolation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_bounding_box() {
        let min_cell = [-1.0, 0.0, 1.0];
        let max_cell = [0.0, 2.0, 5.0];
        let cell_count = [2, 2, 2];
        let grid = Grid::from_bounding_box(&min_cell, &max_cell, &cell_count);
        assert_eq!(grid.first_cell, [-0.75, 0.5, 2.]);
        assert_eq!(grid.cell_size, [0.5, 1., 2.]);
        assert_eq!(grid.cell_count, [2, 2, 2]);

        assert_eq!(grid.get_bounding_box(), (min_cell, max_cell));
    }

    #[test]
    fn test_snap_point_to_grid() {
        let min_cell = [0.0, 0.0, 0.0];
        let max_cell = [1.0, 1.0, 1.0];
        let cell_count = [2, 2, 2];
        let grid = Grid::from_bounding_box(&min_cell, &max_cell, &cell_count);

        assert_eq!(
            grid.snap_point_to_grid(&[0.4, 0.8, 0.1]),
            SnapResult::Inside([0, 1, 0])
        );

        assert_eq!(
            grid.snap_point_to_grid(&[-0.5, 0.8, 0.8]),
            SnapResult::Outside([0, 1, 1])
        );

        assert_eq!(
            grid.snap_point_to_grid(&[0.8, 0.8, 0.8]),
            SnapResult::Inside([1, 1, 1])
        );

        assert_eq!(
            grid.snap_point_to_grid(&[0.8, 1.5, 0.8]),
            SnapResult::Outside([1, 1, 1])
        );
    }

    #[test]
    fn test_get_cell_idx() {
        let min_cell = [0.0, 0.0, 0.0];
        let max_cell = [1.0, 1.0, 1.0];
        let cell_count = [2, 2, 2];
        let grid = Grid::from_bounding_box(&min_cell, &max_cell, &cell_count);

        assert_eq!(grid.get_cell_idx(&[0, 0, 0]), 0);
        assert_eq!(grid.get_cell_idx(&[0, 0, 1]), 1);
        assert_eq!(grid.get_cell_idx(&[0, 1, 0]), 2);
        assert_eq!(grid.get_cell_idx(&[0, 1, 1]), 3);
        assert_eq!(grid.get_cell_idx(&[1, 0, 0]), 4);
        assert_eq!(grid.get_cell_idx(&[1, 0, 1]), 5);
        assert_eq!(grid.get_cell_idx(&[1, 1, 0]), 6);
        assert_eq!(grid.get_cell_idx(&[1, 1, 1]), 7);
    }

    #[test]
    fn test_get_cell_center() {
        let min_cell = [0.0, 0.0, 0.0];
        let max_cell = [1.0, 1.0, 1.0];
        let cell_count = [2, 2, 2];
        let grid = Grid::from_bounding_box(&min_cell, &max_cell, &cell_count);

        assert_eq!(grid.get_cell_center(&[0, 0, 0]), [0.25, 0.25, 0.25]);
        assert_eq!(grid.get_cell_center(&[0, 0, 1]), [0.25, 0.25, 0.75]);
        assert_eq!(grid.get_cell_center(&[0, 1, 0]), [0.25, 0.75, 0.25]);
        assert_eq!(grid.get_cell_center(&[0, 1, 1]), [0.25, 0.75, 0.75]);
        assert_eq!(grid.get_cell_center(&[1, 0, 0]), [0.75, 0.25, 0.25]);
        assert_eq!(grid.get_cell_center(&[1, 0, 1]), [0.75, 0.25, 0.75]);
        assert_eq!(grid.get_cell_center(&[1, 1, 0]), [0.75, 0.75, 0.25]);
        assert_eq!(grid.get_cell_center(&[1, 1, 1]), [0.75, 0.75, 0.75]);
    }
}
