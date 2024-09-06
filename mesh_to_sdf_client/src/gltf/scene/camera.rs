use glam::*;
use gltf::camera::Projection as GltfProjection;

/// Contains camera properties.
#[derive(Clone, Debug)]
pub struct Camera {
    /// Camera name. Requires the `names` feature.
    pub name: Option<String>,

    /// Scene extra data. Requires the `extras` feature.
    pub extras: gltf::json::extras::Extras,

    /// Transform matrix (also called world to camera matrix)
    pub transform: Mat4,

    /// Projection type and specific parameters
    pub projection: Projection,

    /// The distance to the far clipping plane.
    ///
    /// For perspective projection, this may be infinite.
    pub zfar: f32,

    /// The distance to the near clipping plane.
    pub znear: f32,
}

/// Camera projections
#[derive(Debug, Clone)]
pub enum Projection {
    /// Perspective projection
    Perspective {
        /// Y-axis FOV, in radians
        yfov: f32,
        /// Aspect ratio, if specified
        aspect_ratio: Option<f32>,
    },
    /// Orthographic projection
    Orthographic {
        /// Projection scale
        scale: Vec2,
    },
}
impl Default for Projection {
    fn default() -> Self {
        Self::Perspective {
            yfov: 0.399,
            aspect_ratio: None,
        }
    }
}

impl Camera {
    /// Position of the camera.
    pub fn position(&self) -> Vec3 {
        self.transform.col(3).truncate()
    }

    /// Right vector of the camera.
    pub fn right(&self) -> Vec3 {
        self.transform.col(0).truncate().normalize()
    }

    /// Up vector of the camera.
    pub fn up(&self) -> Vec3 {
        self.transform.col(1).truncate().normalize()
    }

    /// Forward vector of the camera (backside direction).
    pub fn forward(&self) -> Vec3 {
        self.transform.col(2).truncate().normalize()
    }

    /// Apply the transformation matrix on a vector.
    ///
    /// # Example
    /// ```
    /// # use easy_gltf::Camera;
    /// # use glam::*;
    /// # let cam = Camera::default();
    /// let ray_dir = Vec3::new(1., 0., 0.);
    /// let ray_dir = cam.apply_transform_vector(&ray_dir);
    /// ```
    pub fn apply_transform_vector(&self, pos: &Vec3) -> Vec3 {
        let res = self.transform * pos.extend(1.0);
        res.truncate() / res.w
    }

    pub(crate) fn load(gltf_cam: gltf::Camera, transform: &Mat4) -> Self {
        let mut cam = Self {
            transform: *transform,
            ..Default::default()
        };

        cam.name = gltf_cam.name().map(String::from);
        cam.extras.clone_from(gltf_cam.extras());

        match gltf_cam.projection() {
            GltfProjection::Orthographic(ortho) => {
                cam.projection = Projection::Orthographic {
                    scale: Vec2::new(ortho.xmag(), ortho.ymag()),
                };
                cam.zfar = ortho.zfar();
                cam.znear = ortho.znear();
            }
            GltfProjection::Perspective(pers) => {
                cam.projection = Projection::Perspective {
                    yfov: pers.yfov(),
                    aspect_ratio: pers.aspect_ratio(),
                };
                cam.zfar = pers.zfar().unwrap_or(f32::INFINITY);
                cam.znear = pers.znear();
            }
        };
        cam
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            name: None,
            extras: None,
            transform: Mat4::ZERO,
            projection: Projection::default(),
            zfar: f32::INFINITY,
            znear: 0.1,
        }
    }
}
