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
    surface_iso: f32,
    surface_power: f32,
    surface_width: f32,
    point_size: f32,
    raymarch_mode: u32,
    bounding_box_extension: f32,
    mesh_bounding_box_min: vec4<f32>,
    mesh_bounding_box_max: vec4<f32>,
    map_material: u32,
};
@group(2) @binding(0) var<uniform> vis_uniforms: VisUniforms;

@group(3) @binding(0) var<uniform> shadow_camera: CameraUniform;
@group(3) @binding(1) var shadow_map: texture_2d<f32>;
@group(3) @binding(2) var shadow_sampler: sampler;

@group(4) @binding(0) var cubemap: texture_2d_array<f32>;
@group(4) @binding(1) var cubemap_sampler: sampler;
@group(4) @binding(2) var cubemap_depth: texture_2d_array<f32>;
@group(4) @binding(3) var cubemap_depth_sampler: sampler;
@group(4) @binding(4) var<uniform> cubemap_viewprojs: array<mat4x4<f32>, 6>;

struct VertexInput {
    @location(0) vertex_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) svposition: vec4<f32>,
    @location(0) position: vec4<f32>,
    @location(1) cell_center: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) distance: f32,
    @location(4) cell_idx: u32,
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
    let distance = sdf[index] - vis_uniforms.surface_iso;

    let cell_z = index % cell_count.z;
    index /= cell_count.z;
    let cell_y = index % cell_count.y;
    index /= cell_count.y;
    let cell_x = index;
    let cell_idx = vec3<f32>(f32(cell_x), f32(cell_y), f32(cell_z));
    let cell = uniforms.start.xyz + cell_idx * cell_size;

    let world_pos = cell + in.vertex_position * uniforms.cell_size.xyz * 0.5;

    // cell view position 
    let svposition = camera.view_proj * vec4<f32>(world_pos, 1.0);


    // output
    var out: VertexOutput;
    out.svposition = svposition;
    out.position = vec4(world_pos, 1.0);
    out.cell_center = vec4(cell, 1.0);
    out.normal = vec4(in.normal, 0.0);
    out.distance = distance;
    out.cell_idx = ordered_indices[in.instance_index];
    return out;
}

fn get_albedo(p: vec3<f32>) -> vec3<f32> {
    let bbox_center = (vis_uniforms.mesh_bounding_box_min.xyz + vis_uniforms.mesh_bounding_box_max.xyz) * 0.5;
    let bbox_size = vis_uniforms.mesh_bounding_box_max.xyz - vis_uniforms.mesh_bounding_box_min.xyz;
    let bbox_min = bbox_center - bbox_size * 0.5;
    let bbox_max = bbox_center + bbox_size * 0.5;

    let pmin = p - bbox_min;
    let pmax = bbox_max - p;
    let mini = min(pmin, pmax);
    let min = min(mini.x, min(mini.y, mini.z));

    var fars = array(bbox_size.x, bbox_size.x, bbox_size.z, bbox_size.z, bbox_size.y, bbox_size.y);
    var dists = array(pmin.x, pmax.x, pmin.z, pmax.z, pmin.y, pmax.y);

    var layer = -1;

    var min_dist = 1e10;
    for (var i = 0; i < 6; i = i + 1) {
        var projected = cubemap_viewprojs[i] * vec4(p, 1.0);
        projected /= projected.w;
        var uv = projected.xy * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;
        let depth = textureSample(cubemap_depth, cubemap_depth_sampler, uv, i).x;
        let depth_lin = (1.0 - depth) * fars[i];
        let delta = abs(depth_lin - projected.z);
        if delta < min_dist && depth > 0.0 {
            layer = i;
            min_dist = delta;
        }
    }


    var color = vec3(1.0, 0.0, 1.0);
    if layer >= 0 {
        var projected = cubemap_viewprojs[layer] * vec4(p, 1.0);
        projected /= projected.w;
        var uv = projected.xy * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;
        color = textureSample(cubemap, cubemap_sampler, uv, layer).xyz;
    }

    return color;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    // We send the cell center because we want a single color per cell.
    var color = mix(vec3(0.5, 0.5, 0.5), get_albedo(in.cell_center.xyz), f32(vis_uniforms.map_material > 0));

    let light = shadow_camera.eye.xyz;
    let light_dir = normalize(light - in.position.xyz);
    let ambiant = 0.2;
    let diffuse = max(0.0, dot(in.normal.xyz, light_dir));

    var diffuse_strength = 1.0;

    var shadow_uv = shadow_camera.view_proj * vec4<f32>(in.position.xyz, 1.0);
    shadow_uv /= shadow_uv.w;
    shadow_uv.x = shadow_uv.x * 0.5 + 0.5;
    shadow_uv.y = shadow_uv.y * -0.5 + 0.5;
    var depth = shadow_uv.z;

    let threshold = depth * 1.05;

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

    let brightness = ambiant + (diffuse + 0.5 * specular) * diffuse_strength;

    // arbitrary attenuation
    color.r *= exp(-1.8 * (1.0 - brightness));
    color.g *= exp(-1.9 * (1.0 - brightness));
    color.b *= exp(-1.9 * (1.0 - brightness));

    return vec4(color, 1.0);
}
