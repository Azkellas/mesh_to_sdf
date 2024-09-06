use glam::*;
use gltf::khr_lights_punctual::{Kind, Light as GltfLight};

/// Represents a light.
#[derive(Clone, Debug)]
pub enum Light {
    /// Directional lights are light sources that act as though they are
    /// infinitely far away and emit light in the `direction`. Because it is at
    /// an infinite distance, the light is not attenuated. Its intensity is
    /// defined in lumens per metre squared, or lux (lm/m2).
    Directional {
        /// Light name. Requires the `names` feature.
        name: Option<String>,
        /// Light extra data. Requires the `extras` feature
        extras: gltf::json::extras::Extras,
        /// Direction of the directional light
        direction: Vec3,
        /// Color of the directional light
        color: Vec3,
        /// Intensity of the directional light
        intensity: f32,
    },

    /// Point lights emit light in all directions from their `position` in space;
    /// The brightness of the light attenuates in a physically correct manner as
    /// distance increases from the light's position (i.e.  brightness goes like
    /// the inverse square of the distance). Point light intensity is defined in
    /// candela, which is lumens per square radian (lm/sr).
    Point {
        /// Light name. Requires the `names` feature.
        name: Option<String>,
        /// Light extra data. Requires the `extras` feature
        extras: gltf::json::extras::Extras,
        /// Position of the point light
        position: Vec3,
        /// Color of the point light
        color: Vec3,
        /// Intensity of the point light
        intensity: f32,
    },

    /// Spot lights emit light in a cone in `direction`. The angle and falloff
    /// of the cone is defined using two numbers, the `inner_cone_angle` and
    /// `outer_cone_angle`. As with point lights, the brightness also attenuates
    /// in a physically correct manner as distance increases from the light's
    /// position (i.e. brightness goes like the inverse square of the distance).
    /// Spot light intensity refers to the brightness inside the
    /// `inner_cone_angle` (and at the location of the light) and is defined in
    /// candela, which is lumens per square radian (lm/sr). Engines that don't
    /// support two angles for spotlights should use `outer_cone_angle` as the
    /// spotlight angle (leaving `inner_cone_angle` to implicitly be `0`).
    Spot {
        /// Light name. Requires the `names` feature.
        name: Option<String>,
        /// Light extra data. Requires the `extras` feature
        extras: gltf::json::extras::Extras,
        /// Position of the spot light
        position: Vec3,
        /// Direction of the spot light
        direction: Vec3,
        /// Color of the spot light
        color: Vec3,
        /// Intensity of the spot light
        intensity: f32,
        /// Inner cone angle of the spot light
        inner_cone_angle: f32,
        /// Outer cone angle of the spot light
        outer_cone_angle: f32,
    },
}

impl Light {
    pub(crate) fn load(gltf_light: GltfLight, transform: &Mat4) -> Self {
        match gltf_light.kind() {
            Kind::Directional => Light::Directional {
                name: gltf_light.name().map(String::from),
                extras: gltf_light.extras().clone(),
                direction: -1. * transform.col(2).xyz().normalize(),
                intensity: gltf_light.intensity(),
                color: Vec3::from(gltf_light.color()),
            },
            Kind::Point => Light::Point {
                name: gltf_light.name().map(String::from),
                extras: gltf_light.extras().clone(),
                position: transform.col(3).xyz(),
                intensity: gltf_light.intensity(),
                color: Vec3::from(gltf_light.color()),
            },
            Kind::Spot {
                inner_cone_angle,
                outer_cone_angle,
            } => Light::Spot {
                name: gltf_light.name().map(String::from),
                extras: gltf_light.extras().clone(),
                position: transform.col(3).xyz(),
                direction: -1. * transform.col(2).xyz().normalize(),
                intensity: gltf_light.intensity(),
                color: Vec3::from(gltf_light.color()),
                inner_cone_angle,
                outer_cone_angle,
            },
        }
    }
}
