use std::boxed::Box;

use itertools::Itertools;
use ordered_float::NotNan;
use rayon::prelude::*;

mod geo;
mod point;

pub use point::Point;

/// Mesh Topology: how indices are stored.
pub enum Topology<'a, I>
where
    // I should be a u32 or u16
    I: Into<u32>,
{
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    /// If no indices are provided, they are supposed to be 0..vertices.len()
    TriangleList(Option<&'a [I]>),
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    /// If no indices are provided, they are supposed to be 0..vertices.len()
    TriangleStrip(Option<&'a [I]>),
}

impl<'a, I> Topology<'a, I>
where
    I: Into<u32>,
{
    /// Compute the triangles list
    /// Returns an iterator of tuples of 3 indices representing a triangle.
    fn get_triangles<V>(
        vertices: &[V],
        indices: &'a Topology<I>,
    ) -> Box<dyn Iterator<Item = (usize, usize, usize)> + 'a>
    where
        V: Point,
        I: Copy + Into<u32> + Sync + Send,
    {
        match indices {
            Topology::TriangleList(Some(indices)) => {
                Box::new(indices.iter().map(|x| (*x).into() as usize).tuples())
            }
            // TODO: test
            Topology::TriangleList(None) => Box::new((0..vertices.len()).tuples()),
            // TODO: test
            Topology::TriangleStrip(Some(indices)) => {
                Box::new(indices.iter().map(|x| (*x).into() as usize).tuple_windows())
            }
            // TODO: test
            Topology::TriangleStrip(None) => Box::new((0..vertices.len()).tuple_windows()),
        }
    }
}

/// Compare two signed distances, taking into account floating point errors and signs.
fn compare_distances(a: f32, b: f32) -> std::cmp::Ordering {
    // for a point to be inside, it has to be inside all normals of nearest triangles.
    // if one distance is positive, then the point is outside.
    // this check is sensible to floating point errors though
    // so it's not perfect, but it reduces the number of false positives considerably.
    // TODO: expose ulps and epsilon?
    if float_cmp::approx_eq!(f32, a.abs(), b.abs(), ulps = 2, epsilon = 1e-6) {
        // they are equals: return the one with the smallest distance, privileging positive distances.
        match (a.is_sign_negative(), b.is_sign_negative()) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => a.abs().partial_cmp(&b.abs()).unwrap(),
        }
    } else {
        // return the closest to 0.
        a.abs().partial_cmp(&b.abs()).unwrap()
    }
}
/// Generate a signed distance field from a mesh.
/// Compute the signed distance for each query point.
/// Query points are expected to be in the same space as the mesh.
pub fn generate_sdf<V, I>(vertices: &[V], indices: Topology<I>, query_points: &[V]) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    // For each query point, we compute the distance to each triangle.
    // sign is estimated by comparing the normal to the direction.
    // when two triangles give the same distance (wrt floating point errors),
    // we keep the one with positive distance since to be inside means to be inside all triangles.
    // whereas to be outside means to be outside at least one triangle.
    // see `compare_distances` for more details.
    query_points
        .par_iter()
        .map(|query| {
            Topology::get_triangles(vertices, &indices)
                .map(|(i, j, k)| (&vertices[i], &vertices[j], &vertices[k]))
                // point_triangle_signed_distance estimates sign with normals.
                .map(|(a, b, c)| geo::point_triangle_signed_distance(query, a, b, c))
                // find the closest triangle
                .min_by(|a, b| compare_distances(*a, *b))
                .expect("no triangle found") // TODO: handle error
        })
        .collect()
}

/// State for the binary heap.
/// Used in generate_grid_sdf.
#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    // signed distance to mesh.
    distance: NotNan<f32>,
    // current cell in grid.
    cell: (usize, usize, usize),
    // triangle that generated the distance.
    triangle: (usize, usize, usize),
}

impl Ord for State {
    /// We compare by distance first, then use cell and triangles as tie-breakers.
    /// Only the distance is important to reduce the number of steps.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        compare_distances(other.distance.into_inner(), self.distance.into_inner())
            .then_with(|| self.cell.cmp(&other.cell))
            .then_with(|| self.triangle.cmp(&other.triangle))
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Generate a signed distance field from a mesh for a grid.
///
/// A grid is defined by three parameters:
/// - `start_pos`: the position of the first cell.
/// - `cell_radius`: the size of a cell (e.g. the size of a voxel).
/// - `cell_count`: the number of cells in each direction (e.g. the number of voxels in each direction).
/// Note that if you want to sample x in 0 1 2 .. 10, you need 11 cells in this direction and not 10.
/// The start_cell is the center of the first cell and not a corner of the grid.
/// `cell_radius` can be different in each direction and even negative.
/// `cell_count` can be different in each direction
///
/// returns a vector of signed distances.
/// The grid is formatted as a 1D array, with the x axis first, then y, then z.
/// The index of a cell is `z + y * cell_count[2] + x * cell_count[1] * cell_count[2]`.
pub fn generate_grid_sdf<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    start_pos: &V,
    cell_radius: &V,
    cell_count: &[u32; 3],
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    // The generation works in the following way:
    // - init the grid with f32::MAX
    // - for each triangle in the mesh:
    //    - compute the bounding box of the triangle
    //    - for each cell in the bounding box:
    //        - compute the distance to the triangle
    //        - if the distance is smaller than the current distance in the grid, update the grid
    //          and add the cell to the heap.
    //
    // - while the heap is not empty:
    //    - pop the cell with the smallest distance (wrt sign)
    //    - for each neighbour cell:
    //        - compute the distance to the triangle
    //        - if the distance is smaller than the current distance in the grid, update the grid
    //          and add the cell to the heap.
    //
    // - return the grid.
    let total_cell_count = (cell_count[0] * cell_count[1] * cell_count[2]) as usize;
    let mut grid = vec![f32::MAX; total_cell_count];

    let mut heap = std::collections::BinaryHeap::new();

    // TODO: expose this in the api?
    let get_cell_idx = |cell: (usize, usize, usize)| {
        cell.2
            + cell.1 * cell_count[2] as usize
            + cell.0 * cell_count[1] as usize * cell_count[2] as usize
    };

    // debug step counter.
    let mut steps = 0;

    // init heap.
    Topology::get_triangles(vertices, &indices).for_each(|triangle| {
        let a = &vertices[triangle.0];
        let b = &vertices[triangle.1];
        let c = &vertices[triangle.2];

        // TODO: We can reduce the number of point here by following the triangle "slope" instead of the bounding box.
        // Like a bresenham algorithm but in 3D. Not sure how to do it though.
        // This would help a lot for large triangles.
        // But large triangles means not a lot of them so it should be ok without this optimisation.
        let bounding_box = geo::triangle_bounding_box(a, b, c);

        // The bounding box is snapped to the grid.
        // min is floored and max is ceiled to keep nearby cells.
        let min_cell = bounding_box.0.sub(start_pos).comp_div(cell_radius);
        let min_cell = [
            min_cell.x().floor() as usize,
            min_cell.y().floor() as usize,
            min_cell.z().floor() as usize,
        ];

        let max_cell = bounding_box.1.sub(start_pos).comp_div(cell_radius);
        let max_cell = [
            max_cell.x().ceil() as usize,
            max_cell.y().ceil() as usize,
            max_cell.z().ceil() as usize,
        ];

        // For each cell in the bounding box.
        for x in min_cell[0]..=max_cell[0] {
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    let cell = (x, y, z);
                    // TODO: expose this in point for better ergonomics and optimisation.
                    let cell_pos = V::new(
                        start_pos.x() + cell.0 as f32 * cell_radius.x(),
                        start_pos.y() + cell.1 as f32 * cell_radius.y(),
                        start_pos.z() + cell.2 as f32 * cell_radius.z(),
                    );

                    let cell_idx = get_cell_idx(cell);
                    if cell_idx >= total_cell_count {
                        continue;
                    }

                    let distance = geo::point_triangle_signed_distance(&cell_pos, a, b, c);
                    if compare_distances(distance, grid[cell_idx]).is_lt() {
                        // New smallest ditance: update the grid and add the cell to the heap.
                        steps += 1;

                        grid[cell_idx] = distance;
                        let state = State {
                            distance: NotNan::new(distance.abs()).unwrap(), // TODO: handle error
                            triangle,
                            cell,
                        };
                        heap.push(state);
                    }
                }
            }
        }
    });
    // First step is done: we have the closest triangle for each cell.
    // And a bit more since a triangle might erase the distance of another triangle later in the process.

    log::info!("[generate_grid_sdf] init steps: {}", steps);
    steps = 0;

    // Second step: propagate the distance to the neighbours.
    while let Some(State { triangle, cell, .. }) = heap.pop() {
        let a = &vertices[triangle.0];
        let b = &vertices[triangle.1];
        let c = &vertices[triangle.2];

        // check neighbour cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    // Compute neighbour cell.
                    // TODO: clean, this is ugly.
                    let neighbour_cell = (
                        cell.0 as isize + dx,
                        cell.1 as isize + dy,
                        cell.2 as isize + dz,
                    );
                    if neighbour_cell.0 < 0
                        || neighbour_cell.1 < 0
                        || neighbour_cell.2 < 0
                        || neighbour_cell.0 >= cell_count[0] as isize
                        || neighbour_cell.1 >= cell_count[1] as isize
                        || neighbour_cell.2 >= cell_count[2] as isize
                    {
                        continue;
                    }

                    let neighbour_cell = (
                        neighbour_cell.0 as usize,
                        neighbour_cell.1 as usize,
                        neighbour_cell.2 as usize,
                    );

                    let neighbour_cell_pos = V::new(
                        start_pos.x() + neighbour_cell.0 as f32 * cell_radius.x(),
                        start_pos.y() + neighbour_cell.1 as f32 * cell_radius.y(),
                        start_pos.z() + neighbour_cell.2 as f32 * cell_radius.z(),
                    );

                    let neighbour_cell_idx = get_cell_idx(neighbour_cell);

                    let distance =
                        geo::point_triangle_signed_distance(&neighbour_cell_pos, a, b, c);

                    if compare_distances(distance, grid[neighbour_cell_idx]).is_lt() {
                        // New smallest ditance: update the grid and add the cell to the heap.
                        steps += 1;

                        grid[neighbour_cell_idx] = distance;
                        let state = State {
                            distance: NotNan::new(distance.abs()).unwrap(), // TODO: handle error
                            triangle,
                            cell: neighbour_cell,
                        };
                        heap.push(state);
                    }
                }
            }
        }
    }
    log::info!("[generate_grid_sdf] propagation steps: {}", steps);

    grid
}
