//! ⚠️ This crate is still in its early stages. Expect the API to change.
//!
//! ---
//!
//! This crate provides two entry points:
//!
//! - [`generate_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` at the points `query_points`.
//! - [`generate_grid_sdf`]: computes the signed distance field for the mesh defined by `vertices` and `indices` on a [Grid].
//!
//! ```
//! # use mesh_to_sdf::{generate_sdf, generate_grid_sdf, Topology, Grid};
//! // vertices are [f32; 3], but can be cgmath::Vector3<f32>, glam::Vec3, etc.
//! let vertices: Vec<[f32; 3]> = vec![[0.5, 1.5, 0.5], [1., 2., 3.], [1., 3., 7.]];
//! let indices: Vec<u32> = vec![0, 1, 2];
//!
//! // query points must be of the same type as vertices
//! let query_points: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]];
//!
//! // Query points are expected to be in the same space as the mesh.
//! let sdf: Vec<f32> = generate_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)), // TriangleList as opposed to TriangleStrip
//!     &query_points);
//!
//! for point in query_points.iter().zip(sdf.iter()) {
//!     // distance is positive outside the mesh and negative inside.
//!     println!("Distance to {:?}: {}", point.0, point.1);
//! }
//! # assert_eq!(sdf, vec![1.0]);
//!
//! // if you can, use generate_grid_sdf instead of generate_sdf as it's optimized and much faster.
//! let bounding_box_min = [0., 0., 0.];
//! let bounding_box_max = [10., 10., 10.];
//! let cell_count = [10, 10, 10];
//!
//! let grid = Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, &cell_count);
//!
//! let sdf: Vec<f32> = generate_grid_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)),
//!     &grid);
//!
//! for x in 0..cell_count[0] {
//!     for y in 0..cell_count[1] {
//!         for z in 0..cell_count[2] {
//!             let index = grid.get_cell_idx(&[x, y, z]);
//!             log::info!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
//!         }
//!     }
//! }
//! # assert_eq!(sdf[0], 1.0);
//! ```
//!
//! #### Mesh Topology
//!
//! Indices can be of any type that implements `Into<u32>`, e.g. `u16` and `u32`. Topology can be list or strip.
//! If the indices are not provided, they are supposed to be `0..vertices.len()`.
//!
//! For vertices, this library aims to be as generic as possible by providing a trait `Point` that can be implemented for any type.
//! Implementations for most common math libraries are gated behind feature flags. By default, only `[f32; 3]` is provided.
//! If you do not find your favorite library, feel free to implement the trait for it and submit a PR or open an issue.
//!
//! #### Using your favorite library
//!
//! To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependency. For example, to use `glam`:
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
//! As of now, sign is computed by checking the normals of the triangles. This is not robust and might lead to negative distances leaking outside the mesh in pyramidal shapes.
//! A more robust solution is planned for the future.
//!
//! #### Benchmarks
//!
//! [`generate_grid_sdf`] is much faster than [`generate_sdf`] and should be used whenever possible.
//! [`generate_sdf`] does not allocate memory (except for the result array) but is slow. A faster implementation is planned for the future.
use std::boxed::Box;

use itertools::Itertools;
use ordered_float::NotNan;
use rayon::prelude::*;

mod geo;
mod grid;
mod point;

pub use grid::{Grid, SnapResult};
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
    /// If no indices are provided, they are supposed to be `0..vertices.len()`
    TriangleList(Option<&'a [I]>),
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `1 2 3`, `2 3 4`, and `3 4 5`
    /// If no indices are provided, they are supposed to be `0..vertices.len()`
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
            Topology::TriangleList(None) => Box::new((0..vertices.len()).tuples()),
            Topology::TriangleStrip(Some(indices)) => {
                Box::new(indices.iter().map(|x| (*x).into() as usize).tuple_windows())
            }
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
        a.abs().partial_cmp(&b.abs()).expect("NaN distance")
    }
}
/// Generate a signed distance field from a mesh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
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
/// Used in [`generate_grid_sdf`].
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

/// Generate a signed distance field from a mesh for a grid.
/// See [Grid] for more details on how to create and use a grid.
///
/// Returns a vector of signed distances.
/// Cells outside the mesh will have a positive distance, and cells inside the mesh will have a negative distance.
/// ```
/// # use mesh_to_sdf::{generate_grid_sdf, Topology, Grid};
/// let vertices: Vec<[f32; 3]> = vec![[0.5, 1.5, 0.5], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let bounding_box_min = [0., 0., 0.];
/// let bounding_box_max = [10., 10., 10.];
/// let cell_count = [10, 10, 10];
///
/// let grid = Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, &cell_count);
///
/// let sdf: Vec<f32> = generate_grid_sdf(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     &grid);
///
/// for x in 0..cell_count[0] {
///     for y in 0..cell_count[1] {
///         for z in 0..cell_count[2] {
///             let index = grid.get_cell_idx(&[x, y, z]);
///             log::info!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index as usize]);
///         }
///     }
/// }
/// # assert_eq!(sdf[0], 1.0);
/// ```
pub fn generate_grid_sdf<V, I>(vertices: &[V], indices: Topology<I>, grid: &Grid<V>) -> Vec<f32>
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
    let mut distances = vec![f32::MAX; grid.get_total_cell_count()];

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
        let min_cell = match grid.snap_point_to_grid(&bounding_box.0) {
            SnapResult::Inside(cell) | SnapResult::Outside(cell) => cell,
        };
        let max_cell = match grid.snap_point_to_grid(&bounding_box.1) {
            SnapResult::Inside(cell) | SnapResult::Outside(cell) => cell,
        };
        // Add one to max_cell and remove one to min_cell to check nearby cells.
        let min_cell = [
            if min_cell[0] == 0 { 0 } else { min_cell[0] - 1 },
            if min_cell[1] == 0 { 0 } else { min_cell[1] - 1 },
            if min_cell[2] == 0 { 0 } else { min_cell[2] - 1 },
        ];
        let max_cell = [
            (max_cell[0] + 1).min(grid.get_cell_count()[0] - 1),
            (max_cell[1] + 1).min(grid.get_cell_count()[1] - 1),
            (max_cell[2] + 1).min(grid.get_cell_count()[2] - 1),
        ];

        // For each cell in the bounding box.
        for cell in itertools::iproduct!(
            min_cell[0]..=max_cell[0],
            min_cell[1]..=max_cell[1],
            min_cell[2]..=max_cell[2]
        ) {
            let cell = [cell.0, cell.1, cell.2];
            let cell_idx = grid.get_cell_idx(&cell);
            if cell_idx >= grid.get_total_cell_count() {
                continue;
            }

            let cell_pos = grid.get_cell_center(&cell);

            let distance = geo::point_triangle_signed_distance(&cell_pos, a, b, c);
            if compare_distances(distance, distances[cell_idx]).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                steps += 1;

                distances[cell_idx] = distance;
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
                    && x < grid.get_cell_count()[0] as isize
                    && y < grid.get_cell_count()[1] as isize
                    && z < grid.get_cell_count()[2] as isize
            })
            .map(|(x, y, z)| [x as usize, y as usize, z as usize]);

        for neighbour_cell in neighbours {
            let neighbour_cell_pos = grid.get_cell_center(&neighbour_cell);

            let neighbour_cell_idx = grid.get_cell_idx(&neighbour_cell);

            let distance = geo::point_triangle_signed_distance(&neighbour_cell_pos, a, b, c);

            if compare_distances(distance, distances[neighbour_cell_idx]).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                steps += 1;

                distances[neighbour_cell_idx] = distance;
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

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        // make sure generate_grid_sdf returns the same result as generate_sdf
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();
        let query_points = [
            cgmath::Vector3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(1.0, 1.0, 1.0),
            cgmath::Vector3::new(0.1, 0.2, 0.2),
        ];
        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
        );

        // pysdf [0.45216727 -0.6997909   0.45411023] # negative is outside in pysdf
        // mesh_to_sdf [-0.40961263  0.6929414  -0.46345082] # negative is inside in mesh_to_sdf
        let baseline = [-0.42, 0.69, -0.46];

        // make sure the results are close enough.
        // the results are not exactly the same because the algorithm is not the same and baselines might not be exact.
        // this is mostly to make sure the results are not completely off.
        for (sdf, baseline) in sdf.iter().zip(baseline.iter()) {
            assert!((sdf - baseline).abs() < 0.1);
        }
    }

    #[test]
    fn test_generate_grid() {
        // make sure generate_grid_sdf returns the same result as generate_sdf.
        // assumes generate_sdf is properly tested and correct.
        let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.], [2., 0., 0.]];
        let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3];
        let grid = Grid::from_bounding_box(&[0., 0., 0.], &[5., 5., 5.], &[5, 5, 5]);
        let mut query_points = Vec::new();
        for x in 0..grid.get_cell_count()[0] {
            for y in 0..grid.get_cell_count()[1] {
                for z in 0..grid.get_cell_count()[2] {
                    query_points.push(grid.get_cell_center(&[x, y, z]));
                }
            }
        }
        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(&indices)),
            &query_points,
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(&indices)),
            &grid,
        );

        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            assert_eq!(sdf, grid_sdf, "i: {}", i);
        }
    }

    #[test]
    fn test_topology() {
        let grid = Grid::from_bounding_box(&[0., 0., 0.], &[5., 5., 5.], &[5, 5, 5]);

        let v0 = [0., 1., 0.];
        let v1 = [1., 2., 3.];
        let v2 = [1., 3., 4.];
        let v3 = [2., 0., 0.];
        // triangles: 012 123 230

        let triangle_list_indices = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v3];
            let indices: Vec<u32> = vec![0, 1, 2, 1, 2, 3, 2, 3, 0];
            generate_grid_sdf(
                &vertices,
                crate::Topology::TriangleList(Some(&indices)),
                &grid,
            )
        };

        let triangle_list_none = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v1, v2, v3, v2, v3, v0];
            generate_grid_sdf(&vertices, Topology::TriangleList::<u32>(None), &grid)
        };

        let triangle_strip_indices = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v3];
            let indices: Vec<u32> = vec![0, 1, 2, 3, 0];
            generate_grid_sdf(&vertices, Topology::TriangleStrip(Some(&indices)), &grid)
        };

        let triangle_strip_none = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v3, v0];
            generate_grid_sdf(&vertices, Topology::TriangleStrip::<u32>(None), &grid)
        };

        let cell_count = grid.get_total_cell_count();
        for i in 0..cell_count {
            assert_eq!(triangle_list_indices[i], triangle_list_none[i]);
            assert_eq!(triangle_list_indices[i], triangle_strip_indices[i]);
            assert_eq!(triangle_list_indices[i], triangle_strip_none[i]);
        }
    }
}
