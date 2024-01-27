use std::boxed::Box;

use itertools::Itertools;
use ordered_float::NotNan;
use rayon::prelude::*;

mod geo;
mod point;

pub use point::Point;

/// Mesh Topology
pub enum Topology<'a, I>
where
    // I should be a u32 or u16
    I: Into<u32>,
{
    /// Vertex data is a list of triangles. Each set of 3 vertices composes a new triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create two triangles `0 1 2` and `3 4 5`
    TriangleList(Option<&'a [I]>),
    /// Vertex data is a triangle strip. Each set of three adjacent vertices form a triangle.
    ///
    /// Vertices `0 1 2 3 4 5` create four triangles `0 1 2`, `2 1 3`, `2 3 4`, and `4 3 5`
    TriangleStrip(Option<&'a [I]>),
}

/// Compute the triangles list
/// Returns an iterator of tuples of 3 indices representing a triangle.
fn get_triangles<'a, V, I>(
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

fn compare_distances(a: f32, b: f32) -> std::cmp::Ordering {
    // for a point to be inside, it has to be inside all normals of nearest triangles.
    // if one distance is positive, then the point is outside.
    // this check is sensible to floating point errors though
    // so it's not perfect, but it reduces the number of false positives considerably.
    if float_cmp::approx_eq!(f32, a.abs(), b.abs(), ulps = 2, epsilon = 1e-6) {
        // return the one with the smallest distance, privileging positive distances.
        match (a.is_sign_negative(), b.is_sign_negative()) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => a.abs().partial_cmp(&b.abs()).unwrap(),
        }
    } else {
        a.abs().partial_cmp(&b.abs()).unwrap()
    }
}
/// Generate a signed distance field from a mesh.
/// Compute the signed distance for each query point.
pub fn generate_sdf<V, I>(vertices: &[V], indices: Topology<I>, query_points: &[V]) -> Vec<f32>
where
    V: Point,
    I: Copy + Into<u32> + Sync + Send,
{
    query_points
        .par_iter()
        .map(|query| {
            get_triangles(vertices, &indices)
                .map(|(i, j, k)| (&vertices[i], &vertices[j], &vertices[k]))
                .map(|(a, b, c)| geo::point_triangle_signed_distance(query, a, b, c))
                // find the closest triangle
                .min_by(|a, b| compare_distances(*a, *b))
                .unwrap()
        })
        .collect()
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    cost: NotNan<f32>,
    cell: (usize, usize, usize),
    triangle: (usize, usize, usize),
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        compare_distances(other.cost.into_inner(), self.cost.into_inner())
            .then_with(|| self.cell.cmp(&other.cell))
            .then_with(|| self.triangle.cmp(&other.triangle))
    }
}

// `PartialOrd` needs to be implemented as well.
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Generate a signed distance field from a mesh.
/// Compute the signed distance for each query point.
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
    let total_cell_count = (cell_count[0] * cell_count[1] * cell_count[2]) as usize;
    let mut grid = vec![f32::MAX; total_cell_count];

    let mut heap = std::collections::BinaryHeap::new();

    let get_cell_idx = |cell: (usize, usize, usize)| {
        cell.2
            + cell.1 * cell_count[2] as usize
            + cell.0 * cell_count[1] as usize * cell_count[2] as usize
    };

    let mut steps = 0;

    // init heap.
    get_triangles(vertices, &indices).for_each(|triangle| {
        let a = &vertices[triangle.0];
        let b = &vertices[triangle.1];
        let c = &vertices[triangle.2];

        let bounding_box = geo::triangle_bounding_box(a, b, c);

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

        for x in min_cell[0]..=max_cell[0] {
            for y in min_cell[1]..=max_cell[1] {
                for z in min_cell[2]..=max_cell[2] {
                    let cell = (x, y, z);
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
                    if compare_distances(distance, grid[cell_idx]) == std::cmp::Ordering::Less {
                        steps += 1;

                        grid[cell_idx] = distance;
                        let state = State {
                            cost: NotNan::new(distance.abs()).unwrap(), // TODO: handle error
                            triangle,
                            cell,
                        };
                        heap.push(state);
                    }
                }
            }
        }
    });

    println!("init steps: {}", steps);
    steps = 0;

    while let Some(State { triangle, cell, .. }) = heap.pop() {
        let a = &vertices[triangle.0];
        let b = &vertices[triangle.1];
        let c = &vertices[triangle.2];

        // check neighbour cells
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
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

                    if compare_distances(distance, grid[neighbour_cell_idx])
                        == std::cmp::Ordering::Less
                    {
                        steps += 1;

                        grid[neighbour_cell_idx] = distance;
                        let state = State {
                            cost: NotNan::new(distance.abs()).unwrap(), // TODO: handle error
                            triangle,
                            cell: neighbour_cell,
                        };
                        heap.push(state);
                    }
                }
            }
        }
    }
    println!("steps: {}", steps);

    grid
}
