mod camera;
mod light;
/// Contains model and material
/// # Usage
/// Check [Model](struct.Model.html) for more information about how to use this module.
pub mod model;

pub use camera::Camera;
use gltf::scene::Node;
pub use light::Light;
pub use model::{GltfModel, Material};

use super::utils::transform_to_matrix;

/// Contains cameras, models and lights of a scene.
#[derive(Default, Clone, Debug)]
pub struct Scene {
    /// Scene name. Requires the `names` feature.
    pub name: Option<String>,
    /// Scene extra data. Requires the `extras` feature.
    pub extras: gltf::json::extras::Extras,
    /// List of models in the scene
    pub models: Vec<usize>,
    /// List of cameras in the scene
    pub cameras: Vec<Camera>,
    /// List of lights in the scene
    pub lights: Vec<Light>,
    /// Scene root not
    pub root_node: ModelNode,
}

/// A model in the scene graph
#[derive(Debug, Clone, Default)]
pub struct ModelNode {
    /// Model.
    pub model_id: Option<usize>,
    /// Node transform.
    pub transform: glam::Mat4,
    /// Node children.
    pub children: Vec<ModelNode>,
    /// Name
    pub name: Option<String>,
}

impl ModelNode {
    /// Debug print the node.
    pub fn debug(&self, level: usize) {
        for _ in 0..level {
            print!("  ");
        }
        log::debug!("{:?}", self.model_id);
        for child in &self.children {
            child.debug(level + 1);
        }
    }

    /// Simplify tree by removing nodes that only contain a single child.
    pub fn simplify_tree(&mut self) {
        // First simplify children
        for node in &mut self.children {
            node.simplify_tree();
        }

        // self is only a transform node. No need to keep it.
        if self.model_id.is_none() && self.children.len() == 1 {
            let mut new_node = self.children.remove(0);
            new_node.transform = self.transform * new_node.transform;
            // Keep the parent name if present since the model name is ofter generic as it can be shared.
            if new_node.model_id.is_some() && self.name.is_some() {
                new_node.name = self.name.take();
            }
            *self = new_node;
        }
    }
}

impl Scene {
    fn new(name: Option<&str>, extras: gltf::json::Extras) -> Self {
        Self {
            name: name.map(String::from),
            extras,
            ..Default::default()
        }
    }

    pub(crate) fn load(file_name: &str, gltf_scene: &gltf::Scene) -> Self {
        let mut scene = Self::new(gltf_scene.name(), gltf_scene.extras().clone());

        for node in gltf_scene.nodes() {
            let mut new_root = ModelNode::default();
            scene.read_node(&mut new_root, &node);
            scene.root_node.children.push(new_root);
        }

        // Simplify tree by removing nodes that only contain a single child and no model.
        scene.root_node.simplify_tree();

        // Rename root node if it has no name with the file name.
        if scene.root_node.name.is_none() {
            scene.root_node.name = Some(file_name.to_string());
        }

        scene
    }

    fn read_node(&mut self, parent_node: &mut ModelNode, node: &Node) {
        // Compute transform of the current node
        let mut new_node = ModelNode {
            name: node.name().map(String::from),
            ..Default::default()
        };

        // Rename parent node if it has no name. Useful sinces meshes are often their own nodes.
        if parent_node.name.is_none() {
            parent_node.name.clone_from(&new_node.name);
        }

        let transform = transform_to_matrix(node.transform());

        new_node.transform = transform;

        // Recurse on children
        for child in node.children() {
            self.read_node(&mut new_node, &child);
        }

        // Load camera
        if let Some(camera) = node.camera() {
            self.cameras.push(Camera::load(&camera, &transform));
        }

        // Load light
        if let Some(light) = node.light() {
            self.lights.push(Light::load(&light, &transform));
        }

        // Load model
        if let Some(mesh) = node.mesh() {
            let model_id = mesh.index();
            self.models.push(model_id);
            new_node.model_id = Some(model_id);
            new_node.name = mesh.name().map(String::from).or(new_node.name);

            // Rename parent node if it has no name. Useful sinces meshes are often their own nodes.
            if parent_node.name.is_none() {
                parent_node.name.clone_from(&new_node.name);
            }
        }
        parent_node.children.push(new_node);
    }
}
