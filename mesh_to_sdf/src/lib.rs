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
//!     AccelerationMethod::RtreeBvh, // Use an r-tree and a bvh to accelerate queries.
//! );
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
//! Both methods have roughly the same performances, depending on the acceleration structure used for generic queries.
//!
//! ---
//!
//! #### Acceleration structures
//!
//! For generic queries, you can use acceleration structures to speed up the computation.
//! - [`AccelerationMethod::None`]: no acceleration structure. This is the slowest method but requires no extra memory.
//! - [`AccelerationMethod::Bvh`]: Bounding Volume Hierarchy. Accepts a `SignMethod`.
//! - [`AccelerationMethod::Rtree`]: R-tree. Only compatible with `SignMethod::Normal`. The fastest method assuming you have more than a couple thousands of queries.
//! - [`AccelerationMethod::RtreeBvh`] (default): Uses R-tree for nearest neighbor search and Bvh for raycasting.
//!
//! If your mesh is watertight and you have more than a thousand queries/triangles, you should use `AccelerationMethod::RtreeBvh` for best performances.
//! If it's not watertight, you can use `AccelerationMethod::Rtree` instead.
//!
//! `Rtree` methods are ~4x faster than `Bvh` methods for big enough data. `AccelerationMethod::None` scales really poorly and should be avoided unless for small datasets or if you're really tight on memory.
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
//! [`SignMethod::Raycast`] is slightly slower than [`SignMethod::Normal`] but is robust and should be used whenever possible (~1% in [`generate_grid_sdf`], ~10% in [`generate_sdf`]).
use std::boxed::Box;

use itertools::Itertools;

use generate::generic::{
    bvh::generate_sdf_bvh, default::generate_sdf_default, rtree::generate_sdf_rtree,
    rtree_bvh::generate_sdf_rtree_bvh,
};

mod bvh_ext;
mod generate;
mod geo;
mod grid;
mod point;

#[cfg(feature = "serde")]
mod serde;

pub use generate::grid::generate_grid_sdf;
pub use grid::{Grid, SnapResult};
pub use point::Point;

#[cfg(feature = "serde")]
pub use serde::*;

/// Mesh Topology: how indices are stored.
#[derive(Copy, Clone)]
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
        vertices: &'a [V],
        indices: Self,
    ) -> Box<dyn Iterator<Item = (usize, usize, usize)> + Send + 'a>
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
///
/// `RtreeBvh` is the fastest method but also the most memory intensive.
/// If your mesh is not watertight, you can use `Rtree` instead.
/// `Bvh` is about 4x slower than `Rtree`.
/// `None` is the slowest method and scales really poorly but requires no extra memory.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AccelerationMethod {
    /// No acceleration structure.
    None(SignMethod),
    /// Bounding Volume Hierarchy.
    /// Recommended unless you have very few queries and triangles (less than a couple thousands)
    /// or if you're really tight on memory as it requires more memory than the default method.
    Bvh(SignMethod),
    /// R-tree
    /// Only compatible with `SignMethod::Normal`
    Rtree,
    /// R-tree and Bvh
    /// Uses R-tree for nearest neighbor search and Bvh for ray intersection.
    #[default]
    RtreeBvh,
}

/// Compare two signed distances, taking into account floating point errors and signs.
fn compare_distances(a: f32, b: f32) -> core::cmp::Ordering {
    // for a point to be inside, it has to be inside all normals of nearest triangles.
    // if one distance is positive, then the point is outside.
    // this check is sensible to floating point errors though
    // so it's not perfect, but it reduces the number of false positives considerably.
    // TODO: expose ulps and epsilon?
    if float_cmp::approx_eq!(f32, a.abs(), b.abs(), ulps = 2, epsilon = 1e-6) {
        // they are equals: return the one with the smallest distance, privileging positive distances.
        match (a.is_sign_negative(), b.is_sign_negative()) {
            (true, false) => core::cmp::Ordering::Greater,
            (false, true) => core::cmp::Ordering::Less,
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
///     AccelerationMethod::RtreeBvh,   // Use an rtree and a bvh to accelerate queries.
///                                     // Recommended unless you have very few queries and triangles (less than a couple thousands)
///                                     // or if you're really tight on memory as it requires more memory than other methods.
///                                     // This uses raycasting to compute sign. This is robust but requires the mesh to be watertight.
/// );                                  // If your mesh isn't watertight, you can use AccelerationMethod::Rtree instead.
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
) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    match acceleration_method {
        AccelerationMethod::None(sign_method) => {
            generate_sdf_default(vertices, indices, query_points, sign_method)
        }
        AccelerationMethod::Bvh(sign_method) => {
            generate_sdf_bvh(vertices, indices, query_points, sign_method)
        }
        AccelerationMethod::Rtree => generate_sdf_rtree(vertices, indices, query_points),
        AccelerationMethod::RtreeBvh => generate_sdf_rtree_bvh(vertices, indices, query_points),
    }
}
