use itertools::Itertools;

fn main() {
    let path = "assets/suzanne.glb";
    let gltf = easy_gltf::load(path).unwrap();

    let model = &gltf.first().unwrap().models[0];
    let vertices = model.vertices();
    let indices = model.indices().unwrap();

    // easy_gltf use cgmath::Vector3 for vertices, which is compatible with mesh_to_sdf
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
    );

    for point in query_points.iter().zip(sdf.iter()) {
        println!("Distance to {:?}: {}", point.0, point.1);
    }

    // Get grid sdf
    // if you can, use generate_grid_sdf instead of generate_sdf.
    let bounding_box_min = cgmath::Vector3::new(0., 0., 0.);
    let cell_radius = cgmath::Vector3::new(1., 1., 1.);
    let cell_count = [3, 3, 3];
    // sample from [0, 0, 0] to [2, 2, 2]

    let sdf: Vec<f32> = mesh_to_sdf::generate_grid_sdf(
        &vertices,
        mesh_to_sdf::Topology::TriangleList(Some(&indices)),
        &bounding_box_min,
        &cell_radius,
        &cell_count,
    );

    for x in 0..cell_count[0] {
        for y in 0..cell_count[1] {
            for z in 0..cell_count[2] {
                let index = z + y * cell_count[2] + x * cell_count[1] * cell_count[2];
                println!(
                    "Distance to cell [{}, {}, {}]: {}",
                    x, y, z, sdf[index as usize]
                );
            }
        }
    }
}
