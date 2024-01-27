use crate::camera_control::CameraLookAt;

#[derive(Debug)]
pub struct Camera {
    pub look_at: CameraLookAt,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
}

impl Camera {
    pub fn get_view_matrix(&self) -> glam::Mat4 {
        self.look_at.get_view_matrix()
    }

    pub fn get_projection_matrix(&self) -> glam::Mat4 {
        // Note: we use reverse z.
        glam::Mat4::perspective_infinite_reverse_rh(self.fovy.to_radians(), self.aspect, self.znear)
    }

    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = self.get_view_matrix();
        let proj = self.get_projection_matrix();
        proj * view
    }

    pub fn update_resolution(&mut self, resolution: [u32; 2]) {
        self.aspect = resolution[0] as f32 / resolution[1] as f32;
    }
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: glam::Mat4,
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
    pub view_inv: glam::Mat4,
    pub proj_inv: glam::Mat4,
    pub eye: glam::Vec4,
    pub resolution: [u32; 2],
    // TODO: already in CameraData, redondant
    pub znear: f32,
    _padding: f32,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY,
            view: glam::Mat4::IDENTITY,
            proj: glam::Mat4::IDENTITY,
            view_inv: glam::Mat4::IDENTITY,
            proj_inv: glam::Mat4::IDENTITY,

            eye: glam::Vec4::ZERO,
            resolution: [800, 600],
            znear: 0.1,
            _padding: 0.0,
        }
    }

    pub fn from_camera(camera: &Camera) -> Self {
        let mut res = Self::new();
        res.update_view_proj(camera);
        res
    }

    /// Update the view and projection matrices.
    pub fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix();
        self.view = camera.get_view_matrix();
        self.proj = camera.get_projection_matrix();
        self.view_inv = self.view.inverse();
        self.proj_inv = self.proj.inverse();
        self.eye = camera.look_at.get_eye();
        self.znear = camera.znear;
    }

    /// Unproject a pixel coordinate to a ray in world space.
    pub fn unproject(&self, pixel: [f32; 2]) -> glam::Vec3 {
        let x = pixel[0] / self.resolution[0] as f32;
        let y = pixel[1] / self.resolution[1] as f32;

        let x = x * 2.0 - 1.0;
        let y = 1.0 - y * 2.0;

        let dir_eye = self.proj_inv.transform_point3(glam::Vec3::new(x, y, 0.0));
        let dir_world = self.view_inv.transform_vector3(dir_eye);
        dir_world.normalize()
    }
}

#[derive(Debug)]
pub struct CameraData {
    pub camera: Camera,
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CameraData {
    pub fn update_resolution(&mut self, resolution: [u32; 2]) {
        self.camera.update_resolution(resolution);
        self.uniform.resolution = resolution;
    }
}
