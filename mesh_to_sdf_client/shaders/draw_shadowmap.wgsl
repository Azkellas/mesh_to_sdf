// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    eye: vec4<f32>,
    resolution: vec2<u32>,
    znear: f32,
    _padding: f32,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;
 
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn main_vs(
    in: VertexInput,
) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4(in.position, 1.0);
}
