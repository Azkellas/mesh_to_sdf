struct ModelUniforms {
    transform: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> model: ModelUniforms;

// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    eye: vec4<f32>,
    resolution: vec2<u32>,
    z_near: f32,
    _padding: f32,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;
 
@group(2) @binding(0) var albedo_map: texture_2d<f32>;
@group(2) @binding(1) var albedo_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) svposition: vec4<f32>,
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn main_vs(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.position = model.transform * vec4(in.position, 1.0);
    out.svposition = camera.view_proj * out.position;
    out.normal = normalize(model.transform * vec4(in.normal, 0.0));
    out.uv = in.uv;

    // this way we don't clear the depth buffer to 1.0.
    out.svposition.z = 1.0 - out.svposition.z;

    return out;
}

const EPSILON: f32 = 0.0001;
@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    var color_rgba = textureSample(albedo_map, albedo_sampler, in.uv);

    if color_rgba.a < 0.5 {
        discard;
    }
    var color = color_rgba.rgb;

    return vec4(color, 1.0);
}
