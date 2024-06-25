//! This example demonstrates how to use mesh_to_sdf to generate signed distance fields from a glTF model.
use itertools::Itertools;

fn main() {
    let path = "assets/suzanne.glb";
    let gltf = easy_gltf::load(path).unwrap();

    let model = &gltf.first().unwrap().models[0];
    let vertices = model.vertices();
    let indices = model.indices().unwrap();

    // easy_gltf use cgmath::Vector3 for vertices, which is compatible with mesh_to_sdf
    // you need to specify the cgmath feature in your Cargo.toml for mesh_to_sdf:
    // [dependencies]
    // mesh_to_sdf = { version = "0.2.1", features = ["cgmath"] }
    let vertices = vertices.iter().map(|v| v.position).collect_vec();

    // query points must be of the same type as vertices
    // and in the vertices space
    // if you want to sample from world space, apply the model transform to the query points
    let query_points = [cgmath::Vector3::new(0.0, 0.0, 0.0)];

    println!("query_points: {:?}", query_points.len());
    println!("vertices: {:?}", vertices.len());
    println!("triangles: {:?}", indices.len() / 3);

    // Get signed distance at query points
    let sdf = mesh_to_sdf::generate_sdf(
        &vertices,
        mesh_to_sdf::Topology::TriangleList(Some(indices)),
        &query_points,
        mesh_to_sdf::AccelerationMethod::Bvh,
        mesh_to_sdf::SignMethod::Raycast,
    );

    for point in query_points.iter().zip(sdf.iter()) {
        println!("Distance to {:?}: {}", point.0, point.1);
    }

    // Get grid sdf
    // if you can, use generate_grid_sdf instead of generate_sdf.
    let bounding_box_min = cgmath::Vector3::new(0., 0., 0.);
    let bounding_box_max = cgmath::Vector3::new(2., 2., 2.);
    let cell_count = [3_usize, 3, 3];

    let grid =
        mesh_to_sdf::Grid::from_bounding_box(&bounding_box_min, &bounding_box_max, cell_count);

    let sdf: Vec<f32> = mesh_to_sdf::generate_grid_sdf(
        &vertices,
        mesh_to_sdf::Topology::TriangleList(Some(indices)),
        &grid,
        mesh_to_sdf::SignMethod::Raycast,
    );

    for x in 0..cell_count[0] {
        for y in 0..cell_count[1] {
            for z in 0..cell_count[2] {
                let index = grid.get_cell_idx(&[x, y, z]);
                println!("Distance to cell [{}, {}, {}]: {}", x, y, z, sdf[index]);
            }
        }
    }
}
