//! ⚠️ This crate is still in its early stages. Expect the API to change.
//!
//! ---
//!
//! This crate provides two entry points:
//!
//! - [`generate_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
//! - [`generate_grid_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` on a grid with `cell_count` cells of size `cell_radius` starting at `start_pos`.
//!
//! ```
//! # use mesh_to_sdf::{generate_sdf, generate_grid_sdf, Topology};
//! let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
//! let indices: Vec<u32> = vec![0, 1, 2];
//!
//! let query_points: Vec<[f32; 3]> = vec![[0., 0., 0.]];
//!
//! // Query points are expected to be in the same space as the mesh.
//! let sdf: Vec<f32> = generate_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)),
//!     &query_points);
//!
//! for point in query_points.iter().zip(sdf.iter()) {
//!     println!("Distance to {:?}: {}", point.0, point.1);
//! }
//! # assert_eq!(sdf, vec![1.0]);
//!
//! // if you can, use generate_grid_sdf instead of generate_sdf.
//! let bounding_box_min = [0., 0., 0.];
//! let bounding_box_max = [10., 10., 10.];
//! let cell_radius = [1., 1., 1.];
//! let cell_count = [11, 11, 11]; // 0 1 2 .. 10 = 11 samples
//!
//! let sdf: Vec<f32> = generate_grid_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)),
//!     &bounding_box_min,
//!     &cell_radius,
//!     &cell_count);
//!
//! for x in 0..cell_count[0] {
//!     for y in 0..cell_count[1] {
//!         for z in 0..cell_count[2] {
//!             let index = z + y * cell_count[2] + x * cell_count[1] * cell_count[2];
//!             println!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
//!         }
//!     }
//! }
//! # assert_eq!(sdf[0], 1.0);
//! ```
//!
//! Indices can be of any type that implements `Into<u32>`, e.g. `u16` and `u32`. Topology can be list or strip. Triangle orientation does not matter.
//! For vertices, this library aims to be as generic as possible by providing a trait `Point` that can be implemented for any type. In a near future, this crate will provide implementation for most common libraries (`glam`, `nalgebra`, etc.). Such implementations are gated behind feature flags. By default, only `[f32; 3]` is provided. If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.
//!
//! #### Using your favorite library
//!
//! To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependencies. For example, to use `glam`:
//! ```toml
//! [dependencies]
//! mesh_to_sdf = { version = "0.1", features = ["glam"] }
//! ```
//!
//! Currently, the following libraries are supported:
//! - `cgmath` (`cgmath::Vector3<f32>`)
//! - `glam` (`glam::Vec3`)
//! - `mint` (`mint::Vector3<f32>` and `mint::Point3<f32>`)
//! - `nalgebra` (`nalgebra::Vector3<f32>` and `nalgebra::Point3<f32>`)
//! - and `[f32; 3]`
//!
//! #### Determining inside/outside
//!
//! As of now, sign is computed by checking the normals of the triangles. This is not robust and might lead to negative distances leaking outside the mesh in pyramidal shapes. A more robust solution is planned for the future.
//!
//! #### Benchmarks
//!
//! [`generate_grid_sdf`] is much faster than [`generate_sdf`] and should be used whenever possible. [`generate_sdf`] does not allocate memory (except for the result array) but is slow. A faster implementation is planned for the future.
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
/// ```
/// # use mesh_to_sdf::{generate_sdf, Topology};
/// let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let query_points: Vec<[f32; 3]> = vec![[0., 0., 0.]];
///
/// // Query points are expected to be in the same space as the mesh.
/// let sdf: Vec<f32> = generate_sdf(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     &query_points);
///
/// for point in query_points.iter().zip(sdf.iter()) {
///     println!("Distance to {:?}: {}", point.0, point.1);
/// }
///
/// # assert_eq!(sdf, vec![1.0]);
/// ```
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
    cell: [usize; 3],
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

/// Get the index of a cell in a grid.
pub fn get_cell_idx(cell: &[usize; 3], cell_count: &[usize; 3]) -> usize {
    cell[2] + cell[1] * cell_count[2] + cell[0] * cell_count[1] * cell_count[2]
}

/// Get the position of a cell in a grid.
pub fn get_cell_point<V: Point>(cell: &[usize; 3], start_pos: &V, cell_radius: &V) -> V {
    V::new(
        start_pos.x() + cell[0] as f32 * cell_radius.x(),
        start_pos.y() + cell[1] as f32 * cell_radius.y(),
        start_pos.z() + cell[2] as f32 * cell_radius.z(),
    )
}

/// Result of snapping a point to the grid.
/// If the point is inside the grid, the cell index is returned.
/// If the point is outside the grid, the cell index is the nearest cell.
pub enum SnapResult {
    /// The point is inside the grid.
    /// Cell index is the cell it is within.
    Inside([usize; 3]),
    /// The point is outside the grid
    /// Cell index is the cell it is the nearest from.
    Outside([usize; 3]),
}

/// Snap a point to the grid.
/// Returns None if the point is outside the grid.
pub fn snap_point_to_grid<V: Point>(
    point: &V,
    start_pos: &V,
    cell_radius: &V,
    cell_count: &[usize; 3],
) -> SnapResult {
    // TODO FIXME: I'm pretty sure this is incorrect
    // in other parts of the code, we suppose start_pos to be the center of the first cell.
    // here, we suppose it is the corner of the first cell.
    let cell = point.sub(start_pos).comp_div(cell_radius);

    let cell = [
        cell.x().floor() as isize,
        cell.y().floor() as isize,
        cell.z().floor() as isize,
    ];

    let ires = [
        cell[0].clamp(0, cell_count[0] as isize - 1),
        cell[1].clamp(0, cell_count[1] as isize - 1),
        cell[2].clamp(0, cell_count[2] as isize - 1),
    ];

    let res = [ires[0] as usize, ires[1] as usize, ires[2] as usize];

    if ires != cell {
        SnapResult::Outside(res)
    } else {
        SnapResult::Inside(res)
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
///
/// ```
/// # use mesh_to_sdf::{generate_grid_sdf, Topology};
/// let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let bounding_box_min = [0., 0., 0.];
/// let bounding_box_max = [10., 10., 10.];
/// let cell_radius = [1., 1., 1.];
/// let cell_count = [11, 11, 11]; // 0 1 2 .. 10 = 11 samples
///
/// let sdf: Vec<f32> = generate_grid_sdf(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     &bounding_box_min,
///     &cell_radius,
///     &cell_count);
///
/// for x in 0..cell_count[0] {
///     for y in 0..cell_count[1] {
///         for z in 0..cell_count[2] {
///             let index = z + y * cell_count[2] + x * cell_count[1] * cell_count[2];
///             println!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
///         }
///     }
/// }
/// # assert_eq!(sdf[0], 1.0);
/// ```
pub fn generate_grid_sdf<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    start_pos: &V,
    cell_radius: &V,
    cell_count: &[usize; 3],
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
    let total_cell_count = cell_count[0] * cell_count[1] * cell_count[2];
    let mut grid = vec![f32::MAX; total_cell_count];

    let mut heap = std::collections::BinaryHeap::new();

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
        let min_cell = match snap_point_to_grid(&bounding_box.0, start_pos, cell_radius, cell_count)
        {
            SnapResult::Inside(cell) => cell,
            SnapResult::Outside(cell) => cell,
        };
        let max_cell = match snap_point_to_grid(&bounding_box.1, start_pos, cell_radius, cell_count)
        {
            SnapResult::Inside(cell) => cell,
            SnapResult::Outside(cell) => cell,
        };
        // Add one to max_cell and remove one to min_cell to make sure we have all the cells.
        let min_cell = [
            if min_cell[0] == 0 { 0 } else { min_cell[0] - 1 },
            if min_cell[1] == 0 { 0 } else { min_cell[1] - 1 },
            if min_cell[2] == 0 { 0 } else { min_cell[2] - 1 },
        ];
        let max_cell = [
            (max_cell[0] + 1).min(cell_count[0] - 1),
            (max_cell[1] + 1).min(cell_count[1] - 1),
            (max_cell[2] + 1).min(cell_count[2] - 1),
        ];

        // For each cell in the bounding box.
        for cell in itertools::iproduct!(
            min_cell[0]..=max_cell[0],
            min_cell[1]..=max_cell[1],
            min_cell[2]..=max_cell[2]
        ) {
            let cell = [cell.0, cell.1, cell.2];
            let cell_idx = get_cell_idx(&cell, cell_count);
            if cell_idx >= total_cell_count {
                continue;
            }

            let cell_pos = get_cell_point(&cell, start_pos, cell_radius);

            let distance = geo::point_triangle_signed_distance(&cell_pos, a, b, c);
            if compare_distances(distance, grid[cell_idx]).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                steps += 1;

                grid[cell_idx] = distance;
                let state = State {
                    distance: NotNan::new(distance).unwrap(), // TODO: handle error
                    triangle,
                    cell,
                };
                heap.push(state);
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

        // Compute neighbours around the cell in the three directions.
        // Discard neighbours that are outside the grid.
        let neighbours = itertools::iproduct!(-1..=1, -1..=1, -1..=1)
            .map(|v| {
                (
                    cell[0] as isize + v.0,
                    cell[1] as isize + v.1,
                    cell[2] as isize + v.2,
                )
            })
            .filter(|&(x, y, z)| {
                x >= 0
                    && y >= 0
                    && z >= 0
                    && x < cell_count[0] as isize
                    && y < cell_count[1] as isize
                    && z < cell_count[2] as isize
            })
            .map(|(x, y, z)| [x as usize, y as usize, z as usize]);

        for neighbour_cell in neighbours {
            let neighbour_cell_pos = get_cell_point(&neighbour_cell, start_pos, cell_radius);

            let neighbour_cell_idx = get_cell_idx(&neighbour_cell, cell_count);

            let distance = geo::point_triangle_signed_distance(&neighbour_cell_pos, a, b, c);

            if compare_distances(distance, grid[neighbour_cell_idx]).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                steps += 1;

                grid[neighbour_cell_idx] = distance;
                let state = State {
                    distance: NotNan::new(distance).unwrap(), // TODO: handle error
                    triangle,
                    cell: neighbour_cell,
                };
                heap.push(state);
            }
        }
    }
    log::info!("[generate_grid_sdf] propagation steps: {}", steps);

    grid
}
