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
 
@group(2) @binding(0) var<uniform> shadow_camera: CameraUniform;
@group(2) @binding(1) var shadow_map: texture_2d<f32>;
@group(2) @binding(2) var shadow_sampler: sampler;


@group(3) @binding(0) var albedo_map: texture_2d<f32>;
@group(3) @binding(1) var albedo_sampler: sampler;

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

    let light = shadow_camera.eye.xyz;
    let light_dir = normalize(light - in.position.xyz);
    let ambiant = 0.2;
    let diffuse = max(0.0, dot(in.normal.xyz, light_dir));

    var shadow_uv = shadow_camera.view_proj * vec4<f32>(in.position.xyz, 1.0);
    shadow_uv /= shadow_uv.w;
    shadow_uv.x = shadow_uv.x * 0.5 + 0.5;
    shadow_uv.y = shadow_uv.y * -0.5 + 0.5;
    var depth = shadow_uv.z;

    let threshold = depth * (1.05);

    var diffuse_strength = 1.0;

    let PCF: bool = true;

    if PCF {
        var light_depth = 0.0;
        let inv_res = vec2<f32>(1.0 / f32(camera.resolution.x), 1.0 / f32(camera.resolution.y));
        for (var y = -1.; y <= 1.; y += 1.0) {
            for (var x = -1.; x <= 1.; x += 1.0) {
                let pdepth = textureSample(shadow_map, shadow_sampler, shadow_uv.xy + vec2(x, y) * inv_res).x;
                light_depth += f32(pdepth < threshold);
            }
        }
        light_depth /= 9.0;

        diffuse_strength *= light_depth;
    } else {
        let pdepth = textureSample(shadow_map, shadow_sampler, shadow_uv.xy).x;
        diffuse_strength *= f32(pdepth < threshold);
    }


    let view_dir = normalize(camera.eye.xyz - in.position.xyz);
    let half_dir = normalize(view_dir + light_dir);
    let specular = max(0.0, dot(in.normal.xyz, half_dir));

    let brightness = ambiant + (diffuse + specular) * diffuse_strength;

    // arbitrary attenuation
    color.r *= exp(-1.8 * (1.0 - brightness));
    color.g *= exp(-1.9 * (1.0 - brightness));
    color.b *= exp(-1.9 * (1.0 - brightness));

    return vec4(color, 1.0);
}
