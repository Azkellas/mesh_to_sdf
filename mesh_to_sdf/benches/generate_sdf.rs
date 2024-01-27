use itertools::{Itertools, MinMaxResult};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
fn criterion_benchmark(c: &mut Criterion) {
    let path = "assets/knight.glb";
    let gltf = easy_gltf::load(path).unwrap();

    let model = &gltf.first().unwrap().models[0];
    let vertices = model.vertices();
    let indices = model.indices().unwrap();

    let xbounds = vertices.iter().map(|v| v.position.x).minmax();
    let ybounds = vertices.iter().map(|v| v.position.y).minmax();
    let zbounds = vertices.iter().map(|v| v.position.z).minmax();

    println!("x bounds: {:?}", xbounds);
    println!("y bounds: {:?}", ybounds);
    println!("z bounds: {:?}", zbounds);

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
    let cell_radius = 0.02;
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

    c.bench_function("generate_sdf", |b| {
        b.iter(|| {
            mesh_to_sdf::generate_sdf(
                black_box(&vertices),
                black_box(mesh_to_sdf::Topology::TriangleList(Some(indices))),
                black_box(&query_points),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
