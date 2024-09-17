use itertools::Itertools;

use super::{Mesh, MeshVertex};

pub fn create_box(device: &wgpu::Device) -> Mesh {
    // Draw box.
    let positions = [
        // front
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // back
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        // left
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        // right
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        // top
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        // bottom
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
    ];
    let normals = [
        [0.0, 0.0, 1.0],  // front
        [0.0, 0.0, -1.0], // back
        [-1.0, 0.0, 0.0], // left
        [1.0, 0.0, 0.0],  // right
        [0.0, 1.0, 0.0],  // top
        [0.0, -1.0, 0.0], // bottom
    ];

    let mut indices: Vec<u32> = vec![];
    #[allow(clippy::identity_op)]
    for face in 0..6 {
        indices.append(&mut vec![face * 4 + 2, face * 4 + 1, face * 4 + 3]);
        indices.append(&mut vec![face * 4 + 0, face * 4 + 1, face * 4 + 2]);
    }

    let vertices = positions
        .into_iter()
        .zip(normals.iter().flat_map(|&n| core::iter::repeat(n).take(4)))
        .map(|(position, normal)| MeshVertex {
            position,
            normal,
            tex_coords: [0.0, 0.0],
        })
        .collect_vec();

    Mesh::new(device, vertices, indices)
}
