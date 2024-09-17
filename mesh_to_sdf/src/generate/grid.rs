//! Grid generation module.

use core::sync::atomic::AtomicU32;
use std::thread::ScopedJoinHandle;

use bvh::{bounding_hierarchy::BoundingHierarchy, bvh::Bvh};
use itertools::Itertools;
use ordered_float::NotNan;
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::{compare_distances, geo, Grid, Point, SignMethod, SnapResult, Topology};

use super::generic::bvh::BvhNode;

/// State for the binary heap.
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
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        compare_distances(other.distance.into_inner(), self.distance.into_inner())
            .then_with(|| self.cell.cmp(&other.cell))
            .then_with(|| self.triangle.cmp(&other.triangle))
    }
}
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Data used to generate the raycasts.
struct RaycastData {
    /// The direction of the ray.
    direction: geo::GridAlign,
    /// The starting cell of the ray.
    start_cell: [usize; 3],
}

/// A triangle is a tuple of 3 indices.
type Triangle = (usize, usize, usize);

/// Precomputation type.
/// It's a scoped join handle that can be None if the precomputation is not needed
/// or if it's already been fetched.
type Precomputation<'scope, T> = Option<ScopedJoinHandle<'scope, T>>;

/// Helper struct to generate a grid sdf.
///
/// Starts the precomputations in parallel and fetches them when needed.
/// See `generate_grid_sdf` for more details.
struct Precomputations<'scope, V>
where
    V: Point + Sync + Send + 'static,
{
    bvh: Precomputation<'scope, (Bvh<f32, 3>, Vec<BvhNode<V>>)>, // Not initialized if not using Raycast sign method.
    preheap: Precomputation<'scope, Vec<RwLock<(Triangle, f32)>>>,
    triangles: Precomputation<'scope, Vec<Triangle>>,
    distances: Precomputation<'scope, Vec<RwLock<f32>>>,
    intersections: Precomputation<'scope, Vec<[AtomicU32; 3]>>,
}

impl<'scope, V> Precomputations<'scope, V>
where
    V: Point + Sync + Send + 'static,
{
    /// Create a new Precomputations struct.
    /// Starts the precomputations in parallel.
    pub fn new<I>(
        vertices: &'scope [V],
        indices: Topology<'scope, I>,
        grid: &'scope Grid<V>,
        sign_method: SignMethod,
        scope: &'scope std::thread::Scope<'scope, '_>,
    ) -> Self
    where
        I: Copy + Into<u32> + Sync + Send,
    {
        let grid_size = grid.get_total_cell_count();

        // bvh computation. We start by this one since it's pretty slow and we'd like it to finish before
        // starting the preheap computation as they both use the same rayon thread pool.
        // this is the only precomputation using the rayon thread pool.
        // If we're not in Raycast mode, we don't compute the bvh.
        let bvh_handle = if sign_method == SignMethod::Raycast {
            Some(scope.spawn(move || {
                let mut bvh_nodes = Topology::get_triangles(vertices, indices)
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
                (bvh, bvh_nodes)
            }))
        } else {
            None
        };

        // prehead initialization. RwLock can be slow to create one by one.
        // Needed for the first step (preheap computation).
        let preheap_handle = scope.spawn(move || {
            let mut preheap = Vec::with_capacity(grid_size);
            for _ in 0..grid_size {
                preheap.push(RwLock::new(((0, 0, 0), f32::MAX)));
            }
            preheap
        });

        // Triangles vec
        // Needed for the first step (preheap computation).
        // We need to store them so `rayon` can optimize better the iteration.
        let triangles_raw_handle = {
            scope.spawn(move || {
                Topology::get_triangles(vertices, indices).collect::<Vec<Triangle>>()
            })
        };

        // distances initialization. RwLock can be slow to create one by one.
        // Needed after the preheap computation so we have time.
        let distances_handle = scope.spawn(move || {
            let mut distances = Vec::with_capacity(grid_size);
            for _ in 0..grid_size {
                distances.push(RwLock::new(f32::MAX));
            }
            distances
        });

        // collision initialization. RwLock can be slow to create one by one.
        // Needed for the raycast step only so we have time.
        let intersections_handle = scope.spawn(move || {
            if sign_method == SignMethod::Raycast {
                let mut intersections = Vec::with_capacity(grid_size);
                for _ in 0..grid_size {
                    intersections.push([AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)]);
                }
                intersections
            } else {
                Vec::new()
            }
        });

        Self {
            bvh: bvh_handle,
            preheap: Some(preheap_handle),
            triangles: Some(triangles_raw_handle),
            distances: Some(distances_handle),
            intersections: Some(intersections_handle),
        }
    }

    /// Get the distances vec.
    /// Panic if the distances were already fetched.
    fn get_distances(&mut self) -> Vec<RwLock<f32>> {
        self.distances.take().unwrap().join().unwrap()
    }

    /// Get the preheap vec.
    /// Panic if the prehead was already fetched.
    fn get_preheap(&mut self) -> Vec<RwLock<(Triangle, f32)>> {
        self.preheap.take().unwrap().join().unwrap()
    }

    /// Get the triangles vec.
    /// Panic if the triangles were already fetched.
    fn get_triangles(&mut self) -> Vec<Triangle> {
        self.triangles.take().unwrap().join().unwrap()
    }

    /// Get the intersections vec.
    /// Panic if the intersections were already fetched.
    fn get_intersections(&mut self) -> Vec<[AtomicU32; 3]> {
        self.intersections.take().unwrap().join().unwrap()
    }

    /// Get the bvh.
    /// Panic if the bvh was already fetched or if we're not using Raycast sign method.
    fn get_bvh(&mut self) -> (Bvh<f32, 3>, Vec<BvhNode<V>>) {
        self.bvh.take().unwrap().join().unwrap()
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
///
/// ## Explanations
///
/// The generation works in the following way:
/// - init a preheap grid with `f32::MAX` and no triangle
/// - for each triangle in the mesh (in parallel):
///    - compute the bounding box of the triangle
///    - for each cell in the bounding box:
///        - compute the distance to the triangle
///        - if the distance is smaller than the current distance in the grid,
///          update the grid with the new distance and triangle.
///
/// Once done, iterate the grid and add the cells with valid distances to a heap.
/// Split the heap in multiple parts and process them in parallel.
/// For each sub-heap:
/// - while the heap is not empty:
///    - pop the cell with the smallest distance (wrt sign)
///    - for each neighbour cell:
///        - compute the distance to the triangle
///        - if the distance is smaller than the current distance in the grid, update the grid
///          and add the cell to the sub-heap.
///
/// Lastly, if we're in raycast mode:
/// - Compute a bvh of the triangles.
///
/// For each cell (0, y, z), (x, 0, z) and (x, y, 0):
/// - cast a ray in the direction of the cell (+x, +y, +z)
/// - for each intersection with a triangle, increment the intersection counter of the cell.
/// - if the number of intersections is odd, the cell is inside the mesh.
/// - We do a best of three to avoid issues with non-watertight meshes, so we need two rays saying the cell is inside.
///
/// - return the grid.
pub fn generate_grid_sdf<V, I>(
    vertices: &[V],
    indices: Topology<I>,
    grid: &Grid<V>,
    sign_method: SignMethod,
) -> Vec<f32>
where
    V: Point + Sync + Send + 'static,
    I: Copy + Into<u32> + Sync + Send,
{
    // All the computation is done in parallel.
    std::thread::scope(|scope| {
        // debug step counter and time.
        let mut now = web_time::Instant::now();
        let steps = AtomicU32::new(0);

        // We have many things to instantiate that takes time.
        // We run them in parallel in the beginning to avoid being forced to single thread later.
        let mut precomputations: Precomputations<'_, V> =
            Precomputations::new(vertices, indices, grid, sign_method, scope);

        // Fetch the precomputations for the first step.
        // Triangles stored in a vec are 10x faster to iterate than an iterator.
        // And 5x faster than iterator a iterator with `par_bridge`.
        let triangles = precomputations.get_triangles();
        let preheap = precomputations.get_preheap();

        // All triangles are processed in parallel.
        // The preheap stores the nearest triangle for each cell with its distance.
        generate_preheap(vertices, &triangles, grid, &preheap, sign_method, &steps);

        // Fetch the distances precomputation.
        let distances = precomputations.get_distances();

        // Generate the heap from the preheap by extracting the cells with valid triangles.
        let heap = generate_heap(grid, &preheap, &distances);

        // First step done.
        log::info!(
            "[generate_grid_sdf] init steps: {} in {:.3}ms",
            steps.fetch_min(0, core::sync::atomic::Ordering::SeqCst),
            now.elapsed().as_secs_f64() * 1000.0
        );
        now = web_time::Instant::now();

        // The most expensive part of the propagation step is gettings the next point of the heap.
        // So the combo (triangle, cell) that has the smallest distance.
        // To improve this, we split the heap in multiple parts and process them in parallel.
        // This means we will have more nodes as it is less optimized, but running in parallel will make it more than worth it.
        // This is much faster than having a lock on a global heap or a thread dedicated to the heap sending
        // states to working threads.
        //
        // As such, each working thread will have its own heap and will process it until it is empty.
        std::thread::scope(|s| {
            let thread_count = rayon::current_num_threads();

            // Split the heap in multiple parts.
            let mut heaps: Vec<Vec<State>> = vec![Vec::with_capacity(heap.len()); thread_count];
            for (i, state) in heap.into_iter().enumerate() {
                heaps[i % thread_count].push(state);
            }

            for _ in 0..thread_count {
                // Get the heap and process it.
                let heap = heaps.pop().unwrap();
                let heap = std::collections::BinaryHeap::from(heap);

                let steps = &steps;
                let distances = &distances;

                s.spawn(move || {
                    propagate_heap(vertices, grid, heap, distances, sign_method, steps);
                });
            }
        });

        // Second step done.
        log::info!(
            "[generate_grid_sdf] propagation steps: {} in {:.3}ms",
            steps.fetch_min(0, core::sync::atomic::Ordering::SeqCst),
            now.elapsed().as_secs_f64() * 1000.0
        );
        now = web_time::Instant::now();

        // We don't need the `RwLock` anymore on distances.
        let mut distances = distances.iter().map(|d| *d.read()).collect_vec();

        if sign_method == SignMethod::Raycast {
            // Fetch the intersections and bvh precomputation.
            let intersections = precomputations.get_intersections();
            let (bvh, bvh_nodes) = precomputations.get_bvh();

            // Run raycasts from three faces of the grid and count the intersections.
            // Update the distances based on the parity of the intersections.
            let raycasts_done = compute_raycasts(
                vertices,
                grid,
                &mut distances,
                &intersections,
                &bvh,
                &bvh_nodes,
            );

            // Raycasts done.
            log::info!(
                "[generate_grid_sdf] raycasts done: {} in {:.3}ms",
                raycasts_done,
                now.elapsed().as_secs_f64() * 1000.0
            );
        }

        distances
    })
}

/// Generate the preheap by going through all the triangles and their bounding boxes.
/// The preheap is a grid storing the closest triangle and distance for each cell.
/// Triangles are processed in parallel.
fn generate_preheap<V: Point>(
    vertices: &[V],
    triangles: &[Triangle],
    grid: &Grid<V>,
    preheap: &[RwLock<(Triangle, f32)>],
    sign_method: SignMethod,
    steps: &AtomicU32,
) {
    triangles.par_iter().for_each(|&triangle| {
        let a = &vertices[triangle.0];
        let b = &vertices[triangle.1];
        let c = &vertices[triangle.2];

        // TODO: We can reduce the number of point here by following the triangle "slope" instead of the bounding box.
        // Like a bresenham algorithm but in 3D. Not sure how to do it though.
        // This would help a lot for large triangles.
        // But large triangles means not a lot of them so it should be ok without this optimisation.
        let bounding_box = geo::triangle_bounding_box(a, b, c);

        // The bounding box is snapped to the grid.
        let mut min_cell = match grid.snap_point_to_grid(&bounding_box.0) {
            SnapResult::Inside(cell) | SnapResult::Outside(cell) => cell,
        };
        let mut max_cell = match grid.snap_point_to_grid(&bounding_box.1) {
            SnapResult::Inside(cell) | SnapResult::Outside(cell) => cell,
        };

        // We add the neighbour cells if the bounding box is on the wrong side of the grid aligned bounding box.
        let min_cell_f = grid.get_cell_center(&min_cell);
        #[expect(clippy::needless_range_loop)]
        for i in 0..3 {
            if min_cell[i] > 0 && min_cell_f.get(i) > bounding_box.0.get(i) {
                min_cell[i] -= 1;
            }
        }
        let max_cell_f = grid.get_cell_center(&max_cell);
        #[expect(clippy::needless_range_loop)]
        for i in 0..3 {
            if max_cell[i] < grid.get_cell_count()[i] - 1
                && max_cell_f.get(i) < bounding_box.1.get(i)
            {
                max_cell[i] += 1;
            }
        }

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

            // We first do an inexpensive check to avoid acquiring the write lock.
            let stored_distance = preheap[cell_idx].read().1;
            if compare_distances(distance, stored_distance).is_lt() {
                // It seems the distance is smaller: acquire the lock and check again.
                let mut stored_distance = preheap[cell_idx].write();
                if compare_distances(distance, stored_distance.1).is_lt() {
                    // New smallest ditance: update the grid and add the cell to the heap.
                    steps.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                    *stored_distance = (triangle, distance);
                }
            }
        }
    });
}

/// Extract the valid cells and triangles from the preheap to generate the heap.
/// Also prepare the distances for the propagation step.
/// We return a vec as it will be split in multiple parts for the propagation step.
/// We sort it to make sure all threads will have good candidates to start with
/// to avoid having a thread with only far away cells that will get discarded.
fn generate_heap<V: Point>(
    grid: &Grid<V>,
    preheap: &[RwLock<(Triangle, f32)>],
    distances: &[RwLock<f32>],
) -> Vec<State> {
    preheap
        .iter()
        .map(|m| m.read())
        .map(|g| *g)
        .enumerate()
        .filter(|(_, (_, d))| *d < f32::MAX)
        .map(|(cell_idx, (triangle, distance))| {
            let cell = grid.get_cell_integer_coordinates(cell_idx);

            *distances[cell_idx].write() = distance;
            State {
                distance: NotNan::new(distance)
                    // SAFETY: f32::MAX is not Nan.
                    .unwrap_or(unsafe { NotNan::new_unchecked(f32::MAX) }),

                triangle,
                cell,
            }
        })
        .sorted_unstable()
        .collect::<Vec<_>>()
}

/// Propagate the distances in the grid.
/// The propagation is done in parallel by splitting the heap in multiple parts
/// and running a bfs for all threads.
fn propagate_heap<V: Point>(
    vertices: &[V],
    grid: &Grid<V>,
    mut heap: std::collections::BinaryHeap<State>,
    distances: &[RwLock<f32>],
    sign_method: SignMethod,
    steps: &AtomicU32,
) {
    while let Some(State { triangle, cell, .. }) = heap.pop() {
        steps.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let a = &vertices[triangle.0];
        let b = &vertices[triangle.1];
        let c = &vertices[triangle.2];

        // Compute neighbours around the cell in the three directions.
        // Discard neighbours that are outside the grid.
        #[expect(clippy::cast_possible_wrap)]
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

            let mut stored_distance = distances[neighbour_cell_idx].write();
            if compare_distances(distance, *stored_distance).is_lt() {
                // New smallest ditance: update the grid and add the cell to the heap.
                *stored_distance = distance;
                let state = State {
                    distance: NotNan::new(distance)
                        // SAFETY: f32::MAX is not Nan.
                        .unwrap_or(unsafe { NotNan::new_unchecked(f32::MAX) }),
                    triangle,
                    cell: neighbour_cell,
                };

                heap.push(state);
            }
        }
    }
}

/// `ray_triangle_intersection` tests for direction [1.0, 0.0, 0.0]
/// The idea here is to tests for all cells (x=0, y, z) and triangle via a bvh
/// For each triangle with an intersection at distance `t`,
/// each cell before `t` intersects the triangle and get incremented for this axis,
/// and each cell after `t` does not.
/// Finally, count the number of intersections for each cell on the three axes.
/// If the number is odd for at least two axes, the cell is inside the mesh. (best of 3)
/// Otherwise the cell is outside the mesh.
fn compute_raycasts<V: Point>(
    vertices: &[V],
    grid: &Grid<V>,
    distances: &mut [f32],
    intersections: &[[AtomicU32; 3]],
    bvh: &Bvh<f32, 3>,
    bvh_nodes: &[BvhNode<V>],
) -> u32 {
    let raycasts_to_do = generate_raycasts(grid);
    let raycasts_done = AtomicU32::new(0);

    raycasts_to_do.par_iter().for_each(|data| {
        let cell_pos = grid.get_cell_center(&data.start_cell);
        let ray_dir = match data.direction {
            geo::GridAlign::X => nalgebra::Vector3::new(1.0, 0.0, 0.0),
            geo::GridAlign::Y => nalgebra::Vector3::new(0.0, 1.0, 0.0),
            geo::GridAlign::Z => nalgebra::Vector3::new(0.0, 0.0, 1.0),
        };

        let ray = bvh::ray::Ray::new(
            nalgebra::Point3::new(cell_pos.x(), cell_pos.y(), cell_pos.z()),
            ray_dir,
        );
        let candidates = bvh.traverse(&ray, bvh_nodes);
        raycasts_done.fetch_add(
            candidates.len() as u32,
            core::sync::atomic::Ordering::Relaxed,
        );
        for candidate in candidates {
            let a = &vertices[candidate.vertex_indices.0];
            let b = &vertices[candidate.vertex_indices.1];
            let c = &vertices[candidate.vertex_indices.2];

            if let Some(distance) =
                geo::ray_triangle_intersection_aligned(&cell_pos, [a, b, c], data.direction)
            {
                let direction_index = data.direction as usize;
                let cell_count = distance / grid.get_cell_size().get(direction_index);
                let cell_count =
                    (cell_count.floor() as usize).min(grid.get_cell_count()[direction_index] - 1);
                // Note: it might seem better to just store in the last cell and then
                // propagate at the end after all raycasts have been done.
                // Sadly this lead to no visible speedup.
                let mut cell = data.start_cell;
                for index in 0..=cell_count {
                    cell[direction_index] = index;
                    let cell_idx = grid.get_cell_idx(&cell);
                    intersections[cell_idx][direction_index]
                        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    });

    for (i, distance) in distances.iter_mut().enumerate() {
        // distance is always positive here since we didn't check the normal.
        // We decide based on the parity of the intersections.
        // And a best of 3.
        // This helps when the mesh is not watertight
        // and to compensate the discrete nature of the grid.
        let inter = [
            intersections[i][0].load(core::sync::atomic::Ordering::SeqCst),
            intersections[i][1].load(core::sync::atomic::Ordering::SeqCst),
            intersections[i][2].load(core::sync::atomic::Ordering::SeqCst),
        ];
        match (inter[0] % 2, inter[1] % 2, inter[2] % 2) {
            // if at least two are odd, the cell is deeemed inside.
            (1, 1, _) | (1, _, 1) | (_, 1, 1) => *distance = -*distance,
            // conversely, if at least two are even, the cell is deeemed outside.
            _ => {}
        }
    }

    raycasts_done.load(core::sync::atomic::Ordering::SeqCst)
}

/// Generate raycasts for the sign raycast method in a grid.
///
/// It generates rays starting from the three negatives faces of the grid cube
/// and pointing inside the cube and uses a bvh for acceleration.
fn generate_raycasts<V: Point>(grid: &Grid<V>) -> Vec<RaycastData> {
    let gx = grid.get_cell_count()[0];
    let gy = grid.get_cell_count()[1];
    let gz = grid.get_cell_count()[2];

    let mut raycasts_to_do = Vec::with_capacity(gx * gy + gx * gz + gy * gz);

    // x.
    for y in 0..grid.get_cell_count()[1] {
        for z in 0..grid.get_cell_count()[2] {
            raycasts_to_do.push(RaycastData {
                direction: geo::GridAlign::X,
                start_cell: [0, y, z],
            });
        }
    }
    // y.
    for x in 0..grid.get_cell_count()[0] {
        for z in 0..grid.get_cell_count()[2] {
            raycasts_to_do.push(RaycastData {
                direction: geo::GridAlign::Y,
                start_cell: [x, 0, z],
            });
        }
    }
    // z.
    for x in 0..grid.get_cell_count()[0] {
        for y in 0..grid.get_cell_count()[1] {
            raycasts_to_do.push(RaycastData {
                direction: geo::GridAlign::Z,
                start_cell: [x, y, 0],
            });
        }
    }

    raycasts_to_do
}

#[cfg(test)]
mod tests {
    use crate::{generate_sdf, AccelerationMethod};

    use super::*;

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
            AccelerationMethod::None(SignMethod::Raycast),
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
            crate::Topology::TriangleList(Some(indices)),
            &grid,
            SignMethod::Raycast,
        );
    }

    // Make sure all topologies are properly handled.
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
            float_cmp::assert_approx_eq!(f32, triangle_list_indices[i], triangle_list_none[i]);
            float_cmp::assert_approx_eq!(f32, triangle_list_indices[i], triangle_strip_indices[i]);
            float_cmp::assert_approx_eq!(f32, triangle_list_indices[i], triangle_strip_none[i]);
        }
    }
}
