#![deny(missing_docs)]

//! This folder is a modified version of `easy-gltf`
//! It provides
//! - Mesh loading parallelization
//! - Texture loading parallelization
//! - Scene hierarchy for models
//!
//! This crate is intended to load [glTF 2.0](https://www.khronos.org/gltf), a
//! file format designed for the efficient transmission of 3D assets.
//!
//! It's base on [gltf](https://github.com/gltf-rs/gltf) crate but has an easy to use output.
//!
//! # Installation
//!
//! ```toml
//! [dependencies]
//! easy-gltf="1.1.1"
//! ```
//!
//! # Example
//!
//! ```
//! let (scenes, data) = easy_gltf::load("tests/cube.glb").expect("Failed to load glTF");
//! for scene in scenes {
//!     println!(
//!         "Cameras: #{}  Lights: #{}  Models: #{}",
//!         scene.cameras.len(),
//!         scene.lights.len(),
//!         scene.models.len()
//!     )
//! }
//! ```

use anyhow::Result;
use hashbrown::HashMap;
use itertools::Itertools;
use rayon::prelude::*;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

mod scene;
mod utils;

pub use scene::*;
pub use utils::GltfData;

use utils::load_all_images;

use crate::pbr::{model::Model, model_instance::ModelInstance};

/// Load scene via a modified version of `easy_gltf` to support parallelization and scene hierarchy..
/// `easy_gltf` is still a bottleneck here since we need to convert their model to our model (they don't support bytemucks).
pub fn load_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    path: &str,
) -> Result<(HashMap<usize, Model>, Vec<ModelInstance>)> {
    let (gltf, gltf_data) = load(path)?;

    // Convert `GltfModel`s to pbr `Model`s
    let models: HashMap<usize, Model> = gltf_data
        .models
        .into_iter()
        .map(|(id, model)| {
            let material = gltf_data.materials.get(&model.material_index()).cloned();
            (
                id,
                Model::from_gtlf(device, queue, &model, material.as_deref()),
            )
        })
        .collect();

    // Flatten the hierarchy with `ModelInstance`s
    let mut instances = vec![];
    for scene in gltf {
        instances.extend(flatten_hierarchy(
            device,
            scene.root_node,
            glam::Mat4::IDENTITY,
        ));
    }

    Ok((models, instances))
}

fn flatten_hierarchy(
    device: &wgpu::Device,
    node: ModelNode,
    transform: glam::Mat4,
) -> Vec<ModelInstance> {
    let mut instances = vec![];
    let transform = transform * node.transform;
    for child in node.children {
        instances.extend(flatten_hierarchy(device, child, transform));
    }
    if let Some(model_id) = node.model_id {
        instances.push(ModelInstance::new(device, model_id, transform));
    }
    instances
}

/// Load scenes from path to a glTF 2.0.
///
/// Note: You can use this function with either a `Gltf` (standard `glTF`) or `Glb` (binary glTF).
///
/// # Example
///
/// ```
/// let (scenes, data) = easy_gltf::load("tests/cube.glb").expect("Failed to load glTF");
/// println!("Scenes: #{}", scenes.len()); // Output "Scenes: #1"
/// let scene = &scenes[0]; // Retrieve the first and only scene
/// println!("Cameras: #{}", scene.cameras.len());
/// println!("Lights: #{}", scene.lights.len());
/// println!("Models: #{}", scene.models.len());
/// ```
pub fn load<P>(path: P) -> Result<(Vec<Scene>, GltfData), gltf::Error>
where
    P: AsRef<Path>,
{
    // Run gltf
    let (doc, buffers, _) = gltf::import(&path)?;

    // Init data and collection useful for conversion
    let data = Arc::new(RwLock::new(GltfData::new(buffers, &path)));

    let meshes = doc
        .meshes()
        .flat_map(|mesh| {
            mesh.primitives()
                .map(move |primitive| (mesh.clone(), primitive))
        })
        .collect_vec();

    meshes.par_iter().for_each(|(mesh, primitive)| {
        let (model_id, model) = {
            let data = data.read().unwrap();
            let model = scene::GltfModel::load(mesh, primitive, &data);
            let model_id = mesh.index();
            (model_id, model)
        };

        let mut data = data.write().unwrap();
        data.models.insert(model_id, Arc::new(model));
    });

    load_all_images(&doc, &data);

    let materials = doc.materials().collect_vec();
    materials.par_iter().for_each(|mat| {
        let material = {
            let data = data.read().unwrap();
            Material::load(mat, &data)
        };

        data.write()
            .unwrap()
            .materials
            .insert(mat.index(), material);
    });

    // Convert gltf -> easy_gltf
    let mut res = vec![];
    for scene in doc.scenes() {
        let scene_name = path
            .as_ref()
            .file_stem()
            .and_then(|f| f.to_str())
            .unwrap_or("Gltf");
        res.push(Scene::load(scene_name, &scene));
    }

    let data = data.read().unwrap().clone();

    Ok((res, data))
}

#[cfg(test)]
mod tests {
    use gltf::mesh::Mode;

    use super::*;

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if !($x - $y < $d || $y - $x < $d) {
                panic!();
            }
        };
    }

    fn load_test_file(path: &str) -> Result<(Vec<Scene>, GltfData), gltf::Error> {
        let folder = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::Path::new(&folder).join(path);
        load(path)
    }

    #[test]
    fn check_cube_glb() {
        let (scenes, data) = load_test_file("tests/cube.glb").unwrap();
        assert_eq!(scenes.len(), 1);
        let scene = &scenes[0];
        assert_eq!(scene.cameras.len(), 1);
        assert_eq!(scene.lights.len(), 3);
        assert_eq!(data.models.len(), 1);
    }

    #[test]
    fn check_different_meshes() {
        let (scenes, data) = load_test_file("tests/complete.glb").unwrap();
        assert_eq!(scenes.len(), 1);
        let scene = &scenes[0];
        for model_id in scene.models.iter() {
            let model = data.models.get(model_id).unwrap();
            match model.mode() {
                Mode::Triangles | Mode::TriangleFan | Mode::TriangleStrip => {
                    assert!(model.triangles().is_ok());
                }
                Mode::Lines | Mode::LineLoop | Mode::LineStrip => {
                    assert!(model.lines().is_ok());
                }
                Mode::Points => {
                    assert!(model.points().is_ok());
                }
            }
        }
    }

    #[test]
    fn check_cube_gltf() {
        let _ = load_test_file("tests/cube_classic.gltf").unwrap();
    }

    #[test]
    fn check_default_texture() {
        let _ = load_test_file("tests/box_sparse.glb").unwrap();
    }

    #[test]
    fn check_camera() {
        // TODO: re-enable direction/position checks by computing global transform.
        // let (scenes, _data) = load("tests/cube.glb").unwrap();
        // let scene = &scenes[0];
        // let cam = &scene.cameras[0];
        // assert!((cam.position() - glam::Vec3::new(7.3589, 4.9583, 6.9258)).length() < 0.1);
    }

    #[test]
    fn check_lights() {
        let (scenes, _data) = load_test_file("tests/cube.glb").unwrap();
        let scene = &scenes[0];
        // TODO: re-enable direction/position checks by computing global transform.
        for light in scene.lights.iter() {
            match light {
                Light::Directional {
                    direction: _,
                    color: _,
                    intensity,
                    ..
                } => {
                    // assert!(
                    //     (*direction - glam::Vec3::new(0.6068, -0.7568, -0.2427)).length() < 0.1
                    // );
                    assert_delta!(intensity, 542., 0.01);
                }
                Light::Point {
                    position: _,
                    color: _,
                    intensity,
                    ..
                } => {
                    // assert!((*position - glam::Vec3::new(4.0762, 5.9039, -1.0055)).length() < 0.1);
                    assert_delta!(intensity, 1000., 0.01);
                }
                Light::Spot {
                    position: _,
                    direction: _,
                    color: _,
                    intensity,
                    inner_cone_angle: _,
                    outer_cone_angle,
                    ..
                } => {
                    // assert!((*position - glam::Vec3::new(4.337, 15.541, -8.106)).length() < 0.1);
                    // assert!(
                    //     (*direction - glam::Vec3::new(-0.0959, -0.98623, 0.1346)).length() < 0.1
                    // );
                    assert_delta!(intensity, 42., 0.01);
                    assert_delta!(outer_cone_angle, 40., 0.01);
                }
            }
        }
    }

    #[test]
    fn check_model() {
        let (_scenes, data) = load_test_file("tests/cube.glb").unwrap();
        for model in data.models.values() {
            assert!(model.has_normals());
            assert!(model.has_tex_coords());
            assert!(model.has_tangents());
            for t in model.triangles().unwrap().iter().flatten() {
                // Check that the tangent w component is 1 or -1
                assert_eq!(t.tangent.w.abs(), 1.);
            }
        }
    }

    #[test]
    fn check_material() {
        let (_scenes, data) = load_test_file("tests/head.glb").unwrap();
        for mat in data.materials.values() {
            assert!(mat.pbr.base_color_texture.is_some());
            assert_eq!(mat.pbr.metallic_factor, 0.);
        }
    }

    #[test]
    fn check_invalid_path() {
        assert!(load_test_file("tests/invalid.glb").is_err());
    }

    #[test]
    fn check_model_no_material() {
        let res = load_test_file("tests/suzanne.glb");
        assert!(res.is_ok());
        let (scenes, data) = res.unwrap();
        assert!(scenes.len() == 1);
        let scene = &scenes[0];
        assert!(scene.models.len() == 1);
        let model = data.models.get(&0).unwrap();
        let material = data.materials.get(&model.material_index());
        assert!(material.is_none());
    }

    #[test]
    fn check_dragon() {
        let res = load_test_file("tests/dragon.glb");
        assert!(res.is_err());
    }
}
