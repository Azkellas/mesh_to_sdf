use wgpu::util::DeviceExt;
use wgpu::Extent3d;

use crate::camera::{Camera, CameraData, CameraUniform};
use crate::camera_control::CameraLookAt;
use crate::texture::Texture;

pub struct ShadowMap {
    pub light: CameraData,
    pub texture: Texture,
}

impl ShadowMap {
    pub fn new(device: &wgpu::Device) -> Self {
        let camera = Camera {
            look_at: CameraLookAt {
                distance: 24.0,
                latitude: 0.85,
                longitude: 6.10,
                ..Default::default()
            },
            aspect: 800.0 / 600.0,
            fovy: 45.0,
            znear: 0.01,
        };

        let camera_uniform = CameraUniform::from_camera(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Map Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(camera_buffer.size()),
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let camera_data = CameraData {
            camera,
            uniform: camera_uniform,
            buffer: camera_buffer,
            bind_group: camera_bind_group,
            bind_group_layout: camera_bind_group_layout,
        };

        Self {
            light: camera_data,
            texture: Texture::create_depth_texture(
                device,
                Extent3d {
                    width: 1024,
                    height: 1024,
                    depth_or_array_layers: 1,
                },
                "shadow_map",
            ),
        }
    }
}
