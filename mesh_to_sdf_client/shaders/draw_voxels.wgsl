struct SdfUniforms {
    // acts as bounding box
    start: vec4<f32>,
    end: vec4<f32>,
    cell_size: vec4<f32>,
    cell_count: vec4<u32>,
}
@group(0) @binding(0) var<uniform> uniforms : SdfUniforms;
@group(0) @binding(1) var<storage, read> sdf : array<f32>;
@group(0) @binding(2) var<storage, read> ordered_indices : array<u32>;

// Vertex shader
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    eye: vec4<f32>,
    resolution: vec2<u32>,
    _padding: vec2<u32>,
};
@group(1) @binding(0) var<uniform> camera: CameraUniform;

struct VisUniforms {
    positive_color: vec4<f32>,
    negative_color: vec4<f32>,
    surface_color: vec4<f32>,
    positive_power: f32,
    negative_power: f32,
    surface_power: f32,
    surface_width: f32,
    point_size: f32,
    raymarch_mode: u32,
};
@group(2) @binding(0) var<uniform> vis_uniforms: VisUniforms;

@group(3) @binding(0) var<uniform> shadow_camera: CameraUniform;
@group(3) @binding(1) var shadow_map: texture_2d<f32>;
@group(3) @binding(2) var shadow_sampler: sampler;

struct VertexInput {
    @location(0) vertex_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) svposition: vec4<f32>,
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) distance: f32,
};


fn phong_lighting(k_d: f32, k_s: f32, alpha: f32, position: vec3<f32>, eye: vec3<f32>, light_pos: vec3<f32>, light_intensity: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let N = normal;
    let L = normalize(light_pos - position);
    let V = normalize(eye - position);
    let R = normalize(reflect(-L, N));

    let dotLN = dot(L, N);
    let dotRV = dot(R, V);

    if dotLN < 0.0 {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    }

    if dotRV < 0.0 {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return light_intensity * (k_d * dotLN);
    }
    return light_intensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}


@vertex
fn main_vs(
    in: VertexInput,
) -> VertexOutput {
    let cell_count = uniforms.cell_count.xyz;
    let cell_size = uniforms.cell_size.xyz;

    // compute cell world position
    // index was generated via for x in 0..cell_x { for y in 0..cell_y { for z in 0..cell_z { ... } } }
    // so index is x*cell_y*cell_z + y*cell_z + z
    var index = ordered_indices[in.instance_index];
    let distance = sdf[index];

    let cell_z = index % cell_count.z;
    index /= cell_count.z;
    let cell_y = index % cell_count.y;
    index /= cell_count.y;
    let cell_x = index;
    let cell_idx = vec3<f32>(f32(cell_x), f32(cell_y), f32(cell_z));
    let cell = uniforms.start.xyz + cell_idx * cell_size;

    let world_pos = cell + in.vertex_position * uniforms.cell_size.xyz;

    // cell view position 
    let svposition = camera.view_proj * vec4<f32>(world_pos, 1.0);


    // output
    var out: VertexOutput;
    out.svposition = svposition;
    out.position = vec4(world_pos, 1.0);
    out.normal = vec4(in.normal, 0.0);
    out.distance = distance;
    return out;
}


@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = vec3(0.5, 0.5, 0.5);

    let light = shadow_camera.eye.xyz;
    let light_dir = normalize(light - in.position.xyz);
    let ambiant = 0.2;
    let diffuse = max(0.0, dot(in.normal.xyz, light_dir));

    var diffuse_strength = 0.5;

    let view_dir = normalize(camera.eye.xyz - in.position.xyz);
    let half_dir = normalize(view_dir + light_dir);
    let specular = max(0.0, dot(in.normal.xyz, half_dir));

    let brightness = ambiant + (diffuse + 0.5 * specular) * diffuse_strength;

    // arbitrary attenuation
    color.r *= exp(-1.8 * (1.0 - brightness));
    color.g *= exp(-1.9 * (1.0 - brightness));
    color.b *= exp(-1.9 * (1.0 - brightness));

    return vec4(color, 1.0);
}
