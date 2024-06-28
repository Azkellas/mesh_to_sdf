//! Benchmark for the `generate_grid_sdf` function
use itertools::{Itertools, MinMaxResult};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
fn criterion_benchmark(c: &mut Criterion) {
    // env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let path = "assets/knight.glb";
    let gltf = easy_gltf::load(path).unwrap();

    let model = &gltf.first().unwrap().models[0];
    let vertices = model.vertices();
    let indices = model.indices().unwrap();

    let xbounds = vertices.iter().map(|v| v.position.x).minmax();
    let ybounds = vertices.iter().map(|v| v.position.y).minmax();
    let zbounds = vertices.iter().map(|v| v.position.z).minmax();

    let MinMaxResult::MinMax(xmin, xmax) = xbounds else {
        panic!("No vertices");
    };
    let MinMaxResult::MinMax(ymin, ymax) = ybounds else {
        panic!("No vertices");
    };
    let MinMaxResult::MinMax(zmin, zmax) = zbounds else {
        panic!("No vertices");
    };

    let vertices = vertices
        .iter()
        .map(|v| [v.position.x, v.position.y, v.position.z])
        .collect_vec();

    let cell_count = [16, 16, 16];
    let grid =
        mesh_to_sdf::Grid::from_bounding_box(&[xmin, ymin, zmin], &[xmax, ymax, zmax], cell_count);

    println!("vertices: {:?}", vertices.len());
    println!("triangles: {:?}", indices.len() / 3);
    println!("grid size: {cell_count:?}");

    c.bench_function("generate_grid_sdf_normal", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_grid_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(indices))),
                black_box(&grid),
                black_box(mesh_to_sdf::SignMethod::Normal),
            );
        });
    });

    c.bench_function("generate_grid_sdf_raycast", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_grid_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(indices))),
                black_box(&grid),
                black_box(mesh_to_sdf::SignMethod::Raycast),
            );
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
