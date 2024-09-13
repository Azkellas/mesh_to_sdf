//! Benchmark for the `generate_sdf` function
use easy_gltf::model::Vertex;
use itertools::{Itertools, MinMaxResult};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use mesh_to_sdf::{AccelerationMethod, SignMethod, Topology};

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

    // generate points in x,y,z bounds with a cell radius of cell_radius
    let cell_radius = 0.01;
    let mut query_points = vec![];
    let xsize = ((xmax - xmin) / cell_radius).ceil();
    let ysize = ((ymax - ymin) / cell_radius).ceil();
    let zsize = ((zmax - zmin) / cell_radius).ceil();

    for xi in 0..xsize as u32 {
        for yi in 0..ysize as u32 {
            for zi in 0..zsize as u32 {
                let x = xmin + xi as f32 * cell_radius;
                let y = ymin + yi as f32 * cell_radius;
                let z = zmin + zi as f32 * cell_radius;
                query_points.push([x, y, z]);
            }
        }
    }

    let vertices = vertices
        .iter()
        .map(|v| [v.position.x, v.position.y, v.position.y])
        .collect_vec();

    println!("query_points: {:?}", query_points.len());
    println!("vertices: {:?}", vertices.len());
    println!("triangles: {:?}", indices.len() / 3);

    c.bench_function("generate_sdf_normal", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::None(SignMethod::Normal)),
            );
        });
    });

    c.bench_function("generate_sdf_raycast", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::None(SignMethod::Raycast)),
            );
        });
    });

    c.bench_function("generate_sdf_bvh", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::Bvh(SignMethod::Normal)),
            );
        });
    });

    c.bench_function("generate_sdf_rtree", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::Rtree),
            );
        });
    });

    c.bench_function("generate_sdf_rtree_bvh", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::RtreeBvh),
            );
        });
    });

    // big query count
    for _ in 0..19 {
        for xi in 0..xsize as u32 {
            for yi in 0..ysize as u32 {
                for zi in 0..zsize as u32 {
                    let x = xmin + xi as f32 * cell_radius;
                    let y = ymin + yi as f32 * cell_radius;
                    let z = zmin + zi as f32 * cell_radius;
                    query_points.push([x, y, z]);
                }
            }
        }
    }

    c.bench_function("generate_sdf_bvh_big", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::Bvh(SignMethod::Normal)),
            );
        });
    });

    c.bench_function("generate_sdf_rtree_big", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::Rtree),
            );
        });
    });

    c.bench_function("generate_sdf_rtree_bvh_big", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::RtreeBvh),
            );
        });
    });
}

fn criterion_benchmark_big(c: &mut Criterion) {
    let path = "assets/FlightHelmet.glb";
    let gltf = easy_gltf::load(path).unwrap();

    let mut vertices = vec![];
    let mut indices = vec![];
    for model in gltf.first().unwrap().models.iter() {
        let len = vertices.len();
        vertices.extend(
            model
                .vertices()
                .iter()
                .map(|v| [v.position.x, v.position.y, v.position.y]),
        );
        indices.extend(model.indices().unwrap().iter().map(|i| i + len as u32));
    }

    let xbounds = vertices.iter().map(|v| v[0]).minmax();
    let ybounds = vertices.iter().map(|v| v[1]).minmax();
    let zbounds = vertices.iter().map(|v| v[2]).minmax();

    let MinMaxResult::MinMax(xmin, xmax) = xbounds else {
        panic!("No vertices");
    };
    let MinMaxResult::MinMax(ymin, ymax) = ybounds else {
        panic!("No vertices");
    };
    let MinMaxResult::MinMax(zmin, zmax) = zbounds else {
        panic!("No vertices");
    };

    // generate points in x,y,z bounds with a cell radius of cell_radius
    let cell_radius = 0.01;
    let mut query_points = vec![];
    let xsize = ((xmax - xmin) / cell_radius).ceil();
    let ysize = ((ymax - ymin) / cell_radius).ceil();
    let zsize = ((zmax - zmin) / cell_radius).ceil();

    for xi in 0..xsize as u32 {
        for yi in 0..ysize as u32 {
            for zi in 0..zsize as u32 {
                let x = xmin + xi as f32 * cell_radius;
                let y = ymin + yi as f32 * cell_radius;
                let z = zmin + zi as f32 * cell_radius;
                query_points.push([x, y, z]);
            }
        }
    }

    println!("query_points: {:?}", query_points.len());
    println!("vertices: {:?}", vertices.len());
    println!("triangles: {:?}", indices.len() / 3);

    c.bench_function("generate_sdf_bvh_big_big", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(&indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::Bvh(SignMethod::Normal)),
            );
        });
    });

    c.bench_function("generate_sdf_rtree_big_big", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(&indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::Rtree),
            );
        });
    });

    c.bench_function("generate_sdf_rtree_bvh_big_big", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(Topology::TriangleList(Some(&indices))),
                black_box(&query_points),
                black_box(AccelerationMethod::RtreeBvh),
            );
        });
    });
}

criterion_group!(benches, criterion_benchmark);

criterion_group! {
    name = benches_big;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark_big
}

criterion_main!(benches, benches_big);
