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
//! use mesh_to_sdf::{generate_sdf, generate_grid_sdf, SignMethod, AccelerationMethod, Topology, Grid};
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
//!     &query_points,
//!     AccelerationMethod::Bvh, // Use bvh to accelerate queries.
//!     SignMethod::Raycast, // How the sign is computed.
//! );                       // Raycast is robust but requires the mesh to be watertight.
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
//! let grid = Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, cell_count);
//!
//! let sdf: Vec<f32> = generate_grid_sdf(
//!     &vertices,
//!     Topology::TriangleList(Some(&indices)),
//!     &grid,
//!     SignMethod::Normal, // How the sign is computed.
//! );                      // Normal might leak negative distances outside the mesh
//!                         // but works for all meshes, even surfaces.
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
//! ---
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
//! ---
//!
//! #### Computing sign
//!
//! This crate provides two methods to compute the sign of the distance:
//! - [`SignMethod::Raycast`] (default): a robust method to compute the sign of the distance. It counts the number of intersections between a ray starting from the query point and the triangles of the mesh.
//!     It only works for watertight meshes, but guarantees the sign is correct.
//! - [`SignMethod::Normal`]: uses the normals of the triangles to estimate the sign by doing a dot product with the direction of the query point.
//!     It works for non-watertight meshes but might leak negative distances outside the mesh.
//!
//! For grid generation, `Raycast` is ~1% slower.
//! For query points, `Raycast` is ~10% slower.
//! Note that it depends on the query points / grid size to triangle ratio, but this gives a rough idea.
//!
//! ---
//!
//! #### Using your favorite library
//!
//! To use your favorite math library with `mesh_to_sdf`, you need to add it to `mesh_to_sdf` dependency. For example, to use `glam`:
//! ```toml
//! [dependencies]
//! mesh_to_sdf = { version = "0.2.1", features = ["glam"] }
//! ```
//!
//! Currently, the following libraries are supported:
//! - [cgmath] ([`cgmath::Vector3<f32>`])
//! - [glam] ([`glam::Vec3`])
//! - [mint] ([`mint::Vector3<f32>`] and [`mint::Point3<f32>`])
//! - [nalgebra] ([`nalgebra::Vector3<f32>`] and [`nalgebra::Point3<f32>`])
//! - `[f32; 3]`
//!
//! ---
//!
//! #### Serialization
//!
//! If you want to serialize and deserialize signed distance fields, you need to enable the `serde` feature.
//! This features also provides helpers to save and load signed distance fields to and from files via `save_to_file` and `read_from_file`.
//!
//! ---
//!
//! #### Benchmarks
//!
//! [`generate_grid_sdf`] is much faster than [`generate_sdf`] and should be used whenever possible.
//! [`generate_sdf`] does not allocate memory (except for the result array) but is slow. A faster implementation is planned for the future.
//!
//! [`SignMethod::Raycast`] is slightly slower than [`SignMethod::Normal`] but is robust and should be used whenever possible (~1% in [`generate_grid_sdf`], ~10% in [`generate_sdf`]).
use std::{boxed::Box, cmp::Ordering};

use bvh::{bounding_hierarchy::BoundingHierarchy, bvh::Bvh};
use bvh_ext::BvhDistance;
use itertools::Itertools;
use ordered_float::NotNan;
use rayon::prelude::*;

mod bvh_ext;
mod geo;
mod grid;
mod point;

#[cfg(feature = "serde")]
mod serde;

pub use grid::{Grid, SnapResult};
pub use point::Point;

#[cfg(feature = "serde")]
pub use serde::*;

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

/// Method to compute the sign of the distance.
///
/// Raycast is the default method. It is robust but requires the mesh to be watertight.
///
/// Normal is not robust and might leak negative distances outside the mesh.
///
/// For grid generation, Raycast is ~1% slower.
/// For query points, Raycast is ~10% slower.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SignMethod {
    /// A robust method to compute the sign of the distance.
    /// It counts the number of intersection between a ray starting from the query point and the mesh.
    /// If the number of intersections is odd, the point is inside the mesh.
    /// This requires the mesh to be watertight.
    #[default]
    Raycast,
    /// A faster but not robust method to compute the sign of the distance.
    /// It uses the normals of the triangles to estimate the sign.
    /// It might leak negative distances outside the mesh.
    Normal,
}

/// Acceleration structure to speed up the computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccelerationMethod {
    /// No acceleration structure.
    None,
    /// Bounding Volume Hierarchy.
    /// Recommended unless you have very few queries and triangles (less than a couple thousands)
    /// or if you're really tight on memory as it requires more memory than the default method.
    #[default]
    Bvh,
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

struct BvhNode<V: Point> {
    vertex_indices: (usize, usize, usize),
    node_index: usize,
    bounding_box: (V, V),
}

impl<V: Point> bvh::aabb::Bounded<f32, 3> for BvhNode<V> {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        let min = nalgebra::Point3::new(
            self.bounding_box.0.x(),
            self.bounding_box.0.y(),
            self.bounding_box.0.z(),
        );
        let max = nalgebra::Point3::new(
            self.bounding_box.1.x(),
            self.bounding_box.1.y(),
            self.bounding_box.1.z(),
        );
        bvh::aabb::Aabb::with_bounds(min, max)
    }
}

impl<V: Point> bvh::bounding_hierarchy::BHShape<f32, 3> for BvhNode<V> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

/// Generate a signed distance field from a mesh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
/// ```
/// use mesh_to_sdf::{generate_sdf, SignMethod, Topology, AccelerationMethod};
///
/// let vertices: Vec<[f32; 3]> = vec![[0., 1., 0.], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let query_points: Vec<[f32; 3]> = vec![[0., 0., 0.]];
///
/// // Query points are expected to be in the same space as the mesh.
/// let sdf: Vec<f32> = generate_sdf(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     &query_points,
///     AccelerationMethod::Bvh,    // Use bvh to accelerate queries.
///                                 // Recommended unless you have very few queries and triangles (less than a couple thousands)
///                                 // or if you're really tight on memory as it requires more memory than the default method.
///     SignMethod::Raycast,        // How the sign is computed.
/// );                              // Raycast is robust but requires the mesh to be watertight.
///
/// for point in query_points.iter().zip(sdf.iter()) {
///     println!("Distance to {:?}: {}", point.0, point.1);
/// }
///
/// # assert_eq!(sdf, vec![1.0]);
/// ```
pub fn generate_sdf<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
    acceleration_method: AccelerationMethod,
    sign_method: SignMethod,
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    match acceleration_method {
        AccelerationMethod::None => {
            generate_sdf_default(vertices, indices, query_points, sign_method)
        }
        AccelerationMethod::Bvh => generate_sdf_bvh(vertices, indices, query_points, sign_method),
    }
}

/// Generate a signed distance field from a mesh using a bvh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
fn generate_sdf_bvh<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
    sign_method: SignMethod,
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    let mut bvh_nodes = Topology::get_triangles(vertices, &indices)
        .map(|triangle| BvhNode {
            vertex_indices: triangle,
            node_index: 0,
            bounding_box: geo::triangle_bounding_box(
                &vertices[triangle.0],
                &vertices[triangle.1],
                &vertices[triangle.2],
            ),
        })
        .collect_vec();

    let bvh = Bvh::build_par(&mut bvh_nodes);

    query_points
        .par_iter()
        .map(|point| {
            let bvh_indices = bvh.nearest_candidates(point);

            let mut min_dist = f32::MAX;
            if sign_method == SignMethod::Normal {
                for index in &bvh_indices {
                    let triangle = &bvh_nodes[*index];
                    let a = &vertices[triangle.vertex_indices.0];
                    let b = &vertices[triangle.vertex_indices.1];
                    let c = &vertices[triangle.vertex_indices.2];
                    let distance = geo::point_triangle_signed_distance(point, a, b, c);

                    if compare_distances(min_dist, distance) == Ordering::Greater {
                        min_dist = distance;
                    }
                }
                min_dist
            } else {
                for index in &bvh_indices {
                    let triangle = &bvh_nodes[*index];
                    let a = &vertices[triangle.vertex_indices.0];
                    let b = &vertices[triangle.vertex_indices.1];
                    let c = &vertices[triangle.vertex_indices.2];
                    let distance = geo::point_triangle_distance(point, a, b, c);

                    min_dist = min_dist.min(distance);
                }

                let alignments = [
                    (geo::GridAlign::X, nalgebra::Vector3::new(1.0, 0.0, 0.0)),
                    (geo::GridAlign::Y, nalgebra::Vector3::new(0.0, 1.0, 0.0)),
                    (geo::GridAlign::Z, nalgebra::Vector3::new(0.0, 0.0, 1.0)),
                ];

                let mut insides = 0;
                for (alignment, direction) in alignments {
                    let ray = bvh::ray::Ray::new(
                        nalgebra::Point3::new(point.x(), point.y(), point.z()),
                        direction,
                    );
                    let mut intersection_count = 0;
                    let hitcast = bvh.traverse(&ray, &bvh_nodes);
                    for bvh_node in hitcast {
                        let a = &vertices[bvh_node.vertex_indices.0];
                        let b = &vertices[bvh_node.vertex_indices.1];
                        let c = &vertices[bvh_node.vertex_indices.2];
                        let intersect =
                            geo::ray_triangle_intersection_aligned(point, [a, b, c], alignment);
                        if intersect.is_some() {
                            intersection_count += 1;
                        }
                    }

                    if intersection_count % 2 == 1 {
                        insides += 1;
                    }
                }

                // Return inside if at least two are insides.
                if insides > 1 {
                    -min_dist
                } else {
                    min_dist
                }
            }
        })
        .collect()
}

/// Generate a signed distance field from a mesh.
/// Query points are expected to be in the same space as the mesh.
///
/// Returns a vector of signed distances.
/// Queries outside the mesh will have a positive distance, and queries inside the mesh will have a negative distance.
pub fn generate_sdf_default<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    query_points: &[V],
    sign_method: SignMethod,
) -> Vec<f32>
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
                .map(|(a, b, c)| match sign_method {
                    // Raycast: returns (distance, ray_intersection)
                    SignMethod::Raycast => (
                        geo::point_triangle_distance(query, a, b, c),
                        geo::ray_triangle_intersection_aligned(query, [a, b, c], geo::GridAlign::X)
                            .is_some(),
                    ),
                    // Normal: returns (signed_distance, false)
                    SignMethod::Normal => {
                        (geo::point_triangle_signed_distance(query, a, b, c), false)
                    }
                })
                .fold(
                    (f32::MAX, 0),
                    |(min_distance, intersection_count), (distance, ray_intersection)| {
                        match sign_method {
                            SignMethod::Raycast => (
                                min_distance.min(distance),
                                intersection_count + ray_intersection as u32,
                            ),
                            SignMethod::Normal => (
                                match compare_distances(distance, min_distance) {
                                    std::cmp::Ordering::Less => distance,
                                    _ => min_distance,
                                },
                                intersection_count,
                            ),
                        }
                    },
                )
        })
        .map(|(distance, intersection_count)| {
            if intersection_count % 2 == 0 {
                distance
            } else {
                // can only be odd if in raycast mode
                -distance
            }
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
/// use mesh_to_sdf::{generate_grid_sdf, SignMethod, Topology, Grid};
///
/// let vertices: Vec<[f32; 3]> = vec![[0.5, 1.5, 0.5], [1., 2., 3.], [1., 3., 4.]];
/// let indices: Vec<u32> = vec![0, 1, 2];
///
/// let bounding_box_min = [0., 0., 0.];
/// let bounding_box_max = [10., 10., 10.];
/// let cell_count = [10, 10, 10];
///
/// let grid = Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, cell_count);
///
/// let sdf: Vec<f32> = generate_grid_sdf(
///     &vertices,
///     Topology::TriangleList(Some(&indices)),
///     &grid,
///     SignMethod::Raycast, // How the sign is computed.
/// );                       // Raycast is robust but requires the mesh to be watertight.
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
pub fn generate_grid_sdf<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    grid: &Grid<V>,
    sign_method: SignMethod,
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
    //          To avoid having multiple triangles for a cell in the resulting heap, we use a pre-heap grid first.
    //
    // - while the heap is not empty:
    //    - pop the cell with the smallest distance (wrt sign)
    //    - for each neighbour cell:
    //        - compute the distance to the triangle
    //        - if the distance is smaller than the current distance in the grid, update the grid
    //          and add the cell to the heap.
    //
    // - if we're in Raycast mode, we compute the intersections with the triangles to determine the sign.
    //
    // - return the grid.
    let mut distances = vec![f32::MAX; grid.get_total_cell_count()];

    let mut preheap = vec![((0, 0, 0), f32::MAX); grid.get_total_cell_count()];

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

            let cell_pos = grid.get_cell_center(&cell);

            let distance = match sign_method {
                SignMethod::Raycast => geo::point_triangle_distance(&cell_pos, a, b, c),
                SignMethod::Normal => geo::point_triangle_signed_distance(&cell_pos, a, b, c),
            };
            if compare_distances(distance, distances[cell_idx]).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                steps += 1;

                preheap[cell_idx] = (triangle, distance);
                distances[cell_idx] = distance;
            }
        }
    });
    // First step is done: we have the closest triangle for each cell.
    // And a bit more since a triangle might erase the distance of another triangle later in the process.

    let mut heap = preheap
        .into_iter()
        .enumerate()
        .filter(|(_, (_, d))| *d < f32::MAX)
        .map(|(cell_idx, (triangle, distance))| {
            let cell = grid.get_cell_integer_coordinates(cell_idx);
            State {
                distance: NotNan::new(distance)
                    .unwrap_or(unsafe { NotNan::new_unchecked(f32::MAX) }),

                triangle,
                cell,
            }
        })
        .collect::<std::collections::BinaryHeap<_>>();

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

            let distance = match sign_method {
                SignMethod::Raycast => geo::point_triangle_distance(&neighbour_cell_pos, a, b, c),
                SignMethod::Normal => {
                    geo::point_triangle_signed_distance(&neighbour_cell_pos, a, b, c)
                }
            };

            if compare_distances(distance, distances[neighbour_cell_idx]).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                steps += 1;

                distances[neighbour_cell_idx] = distance;
                let state = State {
                    distance: NotNan::new(distance)
                        .unwrap_or(unsafe { NotNan::new_unchecked(f32::MAX) }),
                    triangle,
                    cell: neighbour_cell,
                };
                heap.push(state);
            }
        }
    }
    log::info!("[generate_grid_sdf] propagation steps: {}", steps);

    if sign_method == SignMethod::Raycast {
        // `ray_triangle_intersection` tests for direction [1.0, 0.0, 0.0]
        // The idea here is to tests for all cells (x=0, y, z) and triangle.
        // To optimize, we first iterate on triangles and for each triangle,
        // only consider cells that are in its bounding box.
        // If there is no intersection, don't consider the triangle.
        // If there is one with distance `t`,
        // each cell before `t` intersects the triangle, each cell after `t` does not.
        // Finally, count the number of intersections for each cell.
        // If the number is odd, the cell is inside the mesh.
        // If the number is even, the cell is outside the mesh.
        // Since this is so inexpensive (n^2 vs n^3), we can afford to do it in the three directions.
        let mut intersections = vec![[0, 0, 0]; grid.get_total_cell_count()];
        let mut raycasts_done = 0;
        for triangle in Topology::get_triangles(vertices, &indices) {
            let a = &vertices[triangle.0];
            let b = &vertices[triangle.1];
            let c = &vertices[triangle.2];
            let bounding_box = geo::triangle_bounding_box(a, b, c);

            // The bounding box is snapped to the grid.
            let min_cell = match grid.snap_point_to_grid(&bounding_box.0) {
                SnapResult::Inside(cell) | SnapResult::Outside(cell) => cell,
            };
            let max_cell = match grid.snap_point_to_grid(&bounding_box.1) {
                SnapResult::Inside(cell) | SnapResult::Outside(cell) => cell,
            };

            // x.
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    let cell = [0, y, z];
                    let cell_pos = grid.get_cell_center(&cell);
                    raycasts_done += 1;
                    if let Some(distance) = geo::ray_triangle_intersection_aligned(
                        &cell_pos,
                        [a, b, c],
                        geo::GridAlign::X,
                    ) {
                        let cell_count = distance / grid.get_cell_size().x();
                        let cell_count = cell_count.floor() as usize;
                        for x in 0..=cell_count {
                            let cell = [x, y, z];
                            let cell_idx = grid.get_cell_idx(&cell);
                            if cell_idx < intersections.len() {
                                intersections[cell_idx][0] += 1;
                            }
                        }
                    }
                }
            }

            // y.
            for x in min_cell[0]..=max_cell[0] {
                for z in min_cell[2]..=max_cell[2] {
                    let cell = [x, 0, z];
                    let cell_pos = grid.get_cell_center(&cell);
                    raycasts_done += 1;
                    if let Some(distance) = geo::ray_triangle_intersection_aligned(
                        &cell_pos,
                        [a, b, c],
                        geo::GridAlign::Y,
                    ) {
                        let cell_count = distance / grid.get_cell_size().y();
                        let cell_count = cell_count.floor() as usize;
                        for y in 0..=cell_count {
                            let cell = [x, y, z];
                            let cell_idx = grid.get_cell_idx(&cell);
                            if cell_idx < intersections.len() {
                                intersections[cell_idx][1] += 1;
                            }
                        }
                    }
                }
            }

            // z.
            for x in min_cell[0]..=max_cell[0] {
                for y in min_cell[1]..=max_cell[1] {
                    let cell = [x, y, 0];
                    let cell_pos = grid.get_cell_center(&cell);
                    raycasts_done += 1;
                    if let Some(distance) = geo::ray_triangle_intersection_aligned(
                        &cell_pos,
                        [a, b, c],
                        geo::GridAlign::Z,
                    ) {
                        let cell_count = distance / grid.get_cell_size().z();
                        let cell_count = cell_count.floor() as usize;
                        for z in 0..=cell_count {
                            let cell = [x, y, z];
                            let cell_idx = grid.get_cell_idx(&cell);
                            if cell_idx < intersections.len() {
                                intersections[cell_idx][2] += 1;
                            }
                        }
                    }
                }
            }
        }
        for (i, distance) in distances.iter_mut().enumerate() {
            // distance is always positive here since we didn't check the normal.
            // We decide based on the parity of the intersections.
            // And a best of 3.
            // This helps when the mesh is not watertight
            // and to compensate the discrete nature of the grid.
            let inter = intersections[i];
            match (inter[0] % 2, inter[1] % 2, inter[2] % 2) {
                // if at least two are odd, the cell is deeemed inside.
                (1, 1, _) | (1, _, 1) | (_, 1, 1) => *distance = -*distance,
                // conversely, if at least two are even, the cell is deeemed outside.
                _ => {}
            }
        }
        log::info!("[generate_grid_sdf] raycasts done: {}", raycasts_done);
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
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
            AccelerationMethod::None,
            SignMethod::Normal,
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
        let grid = Grid::from_bounding_box(&[0., 0., 0.], &[5., 5., 5.], [5, 5, 5]);
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
            AccelerationMethod::None,
            SignMethod::Raycast,
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(&indices)),
            &grid,
            SignMethod::Raycast,
        );

        // Test against generate_sdf
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            assert_eq!(sdf, grid_sdf, "i: {}", i);
        }
    }

    /// Test continuity.
    /// Only valid for watertight meshes and Raycast method.
    #[test]
    fn test_grid_continuity() {
        let model = &easy_gltf::load("assets/ferris3d.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();

        let bbox_min = vertices.iter().fold(
            cgmath::Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.min(v.x),
                y: acc.y.min(v.y),
                z: acc.z.min(v.z),
            },
        );
        let bbox_max = vertices.iter().fold(
            cgmath::Vector3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.max(v.x),
                y: acc.y.max(v.y),
                z: acc.z.max(v.z),
            },
        );

        let extend = 0.2 * (bbox_max - bbox_min);
        let bbox_min = bbox_min - extend;
        let bbox_max = bbox_max + extend;

        // Most of the time is spent reading the file, so we can afford a large grid.
        let grid = Grid::from_bounding_box(&bbox_min, &bbox_max, [32, 32, 32]);

        let sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Raycast,
        );

        let test_continuity = |distance: f32, neigh_distance: f32, size: f32| {
            // make sure the unsigned distance respects the triangle inequality.
            let valid_unsigned = (distance.abs() - neigh_distance.abs()).abs() <= size;
            // since the grid is discrete, we cannot support the triangle inequality for near the surface.
            // instead we make sure if there is a sign change, the distance is smaller than the cell size
            // meaning the surface is somewhere around (hopefully between) the two cells.
            // This test is not perfect and might fail for some cases.
            let valid_signed = match (distance * neigh_distance).signum() < 0.0 {
                true => distance.abs() <= size && neigh_distance.abs() <= size,
                false => true,
            };
            assert!(
                valid_unsigned && valid_signed,
                "({} {}) <= {}",
                distance,
                neigh_distance,
                size
            );
        };

        let cell_size = grid.get_cell_size();
        for x in 0..grid.get_cell_count()[0] - 1 {
            for y in 0..grid.get_cell_count()[1] - 1 {
                for z in 0..grid.get_cell_count()[2] - 1 {
                    let index = grid.get_cell_idx(&[x, y, z]);

                    let distance = sdf[index];

                    let neigh = grid.get_cell_idx(&[x + 1, y, z]);
                    let neigh_distance = sdf[neigh];
                    test_continuity(distance, neigh_distance, cell_size.x());

                    let neigh = grid.get_cell_idx(&[x, y + 1, z]);
                    let neigh_distance = sdf[neigh];
                    test_continuity(distance, neigh_distance, cell_size.y());

                    let neigh = grid.get_cell_idx(&[x, y, z + 1]);
                    let neigh_distance = sdf[neigh];
                    test_continuity(distance, neigh_distance, cell_size.z());
                }
            }
        }
    }

    // Make sure the raycasts on grid do not access out of bounds cells.
    #[test]
    fn test_grid_raycast() {
        let model = &easy_gltf::load("assets/ferris3d.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();

        let bbox_min = vertices.iter().fold(
            cgmath::Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.min(v.x),
                y: acc.y.min(v.y),
                z: acc.z.min(v.z),
            },
        );
        let mut bbox_max = vertices.iter().fold(
            cgmath::Vector3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.max(v.x),
                y: acc.y.max(v.y),
                z: acc.z.max(v.z),
            },
        );

        bbox_max *= 0.5;

        let grid = Grid::from_bounding_box(&bbox_min, &bbox_max, [32, 32, 32]);

        generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(&indices)),
            &grid,
            SignMethod::Raycast,
        );
    }

    #[test]
    fn test_topology() {
        let grid = Grid::from_bounding_box(&[0., 0., 0.], &[5., 5., 5.], [25, 25, 25]);

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
                SignMethod::Normal,
            )
        };

        let triangle_list_none = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v1, v2, v3, v2, v3, v0];
            generate_grid_sdf(
                &vertices,
                Topology::TriangleList::<u32>(None),
                &grid,
                SignMethod::Normal,
            )
        };

        let triangle_strip_indices = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v3];
            let indices: Vec<u32> = vec![0, 1, 2, 3, 0];
            generate_grid_sdf(
                &vertices,
                Topology::TriangleStrip(Some(&indices)),
                &grid,
                SignMethod::Normal,
            )
        };

        let triangle_strip_none = {
            let vertices: Vec<[f32; 3]> = vec![v0, v1, v2, v3, v0];
            generate_grid_sdf(
                &vertices,
                Topology::TriangleStrip::<u32>(None),
                &grid,
                SignMethod::Normal,
            )
        };

        let cell_count = grid.get_total_cell_count();
        for i in 0..cell_count {
            assert_eq!(triangle_list_indices[i], triangle_list_none[i]);
            assert_eq!(triangle_list_indices[i], triangle_strip_indices[i]);
            assert_eq!(triangle_list_indices[i], triangle_strip_none[i]);
        }
    }

    #[test]
    fn test_generate_bvh() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();
        let query_points = [
            cgmath::Vector3::new(0.01, 0.01, 0.5),
            cgmath::Vector3::new(1.0, 1.0, 1.0),
            cgmath::Vector3::new(0.1, 0.2, 0.2),
            cgmath::Vector3::new(1.1, 2.2, 5.2),
            cgmath::Vector3::new(-0.1, 0.2, -0.2),
        ];

        let bvh_sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh,
            SignMethod::Raycast,
        );

        let sdf = generate_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::None,
            SignMethod::Raycast,
        );

        for (idx, (bvh, sdf)) in bvh_sdf.iter().zip(sdf.iter()).enumerate() {
            assert!(
                (bvh - sdf).abs() < 0.01,
                "{:?}: {} != {}",
                query_points[idx],
                bvh,
                sdf
            );
        }
    }

    #[test]
    fn test_generate_bvh_big() {
        let model = &easy_gltf::load("assets/suzanne.glb").unwrap()[0].models[0];
        let vertices = model.vertices().iter().map(|v| v.position).collect_vec();
        let indices = model.indices().unwrap();

        let bbox_min = vertices.iter().fold(
            cgmath::Vector3::new(f32::MAX, f32::MAX, f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.min(v.x),
                y: acc.y.min(v.y),
                z: acc.z.min(v.z),
            },
        );
        let bbox_max = vertices.iter().fold(
            cgmath::Vector3::new(-f32::MAX, -f32::MAX, -f32::MAX),
            |acc, v| cgmath::Vector3 {
                x: acc.x.max(v.x),
                y: acc.y.max(v.y),
                z: acc.z.max(v.z),
            },
        );

        let grid = Grid::from_bounding_box(&bbox_min, &bbox_max, [32, 32, 32]);
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
            crate::Topology::TriangleList(Some(indices)),
            &query_points,
            AccelerationMethod::Bvh,
            SignMethod::Raycast,
        );
        let grid_sdf = generate_grid_sdf(
            &vertices,
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Raycast,
        );

        // Test against generate_sdf
        for (i, (sdf, grid_sdf)) in sdf.iter().zip(grid_sdf.iter()).enumerate() {
            assert!(
                (sdf - grid_sdf).abs() < 0.01,
                "i: {}: {} {}",
                i,
                sdf,
                grid_sdf
            );
        }
    }
}
