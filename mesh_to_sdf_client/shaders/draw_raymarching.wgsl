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

const MODE_SNAP: u32 = 0u;
const MODE_TRILINEAR: u32 = 1u;
const MODE_TETRAHEDRAL: u32 = 2u;
const MODE_SNAP_STYLIZED: u32 = 3u;

@group(2) @binding(0) var<uniform> vis_uniforms: VisUniforms;

@group(3) @binding(0) var<uniform> shadow_camera: CameraUniform;
@group(3) @binding(1) var shadow_map: texture_2d<f32>;
@group(3) @binding(2) var shadow_sampler: sampler;


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn main_vs(
    @builtin(vertex_index) vertex_id: u32,
) -> VertexOutput {
    var out: VertexOutput;
    // Create a triangle that covers the screen
    if vertex_id == 0u {
        out.clip_position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
    } else if vertex_id == 1u {
        out.clip_position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
    } else if vertex_id == 2u {
        out.clip_position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
    }
    return out;
}

const EPSILON = 0.001;

fn sdf_cube(p: vec3<f32>, origin: vec3<f32>, half_size: vec3<f32>) -> f32 {
    let d = abs(p - origin) - half_size;
    let insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    let outsideDistance = length(max(d, vec3(0.0)));
    return insideDistance + outsideDistance;
}

fn sdf_bounding_box(position: vec3<f32>) -> f32 {
    let box_center = (uniforms.end.xyz + uniforms.start.xyz) * 0.5;
    let box_half_size = (uniforms.end.xyz - uniforms.start.xyz) * 0.5;
    let dist = sdf_cube(position, box_center, box_half_size);

    return dist;
}

fn get_distance(in_cell: vec3<i32>) -> f32 {
    var cell = max(in_cell, vec3<i32>(0, 0, 0));
    cell = min(cell, vec3<i32>(uniforms.cell_count.xyz - vec3(1u)));
    let ucell = vec3<u32>(cell);
    let idx = ucell.z + ucell.y * uniforms.cell_count.z + ucell.x * uniforms.cell_count.z * uniforms.cell_count.y;
    return sdf[idx];
}

fn sdf_grid(position: vec3<f32>) -> f32 {
    // origin is inside the box, so we are inside the box.
    // snap origin to the sdf grid.
    if position.x < uniforms.start.x || position.y < uniforms.start.y || position.z < uniforms.start.z {
        return 100.0;
    }
    if position.x > uniforms.end.x || position.y > uniforms.end.y || position.z > uniforms.end.z {
        return 100.0;
    }

    var distance = 100.0;

    // uniforms.start is the first cell, the center of the first cell.
    let start_grid = uniforms.start.xyz - uniforms.cell_size.xyz * 0.5;

    switch vis_uniforms.raymarch_mode {
        case MODE_SNAP, MODE_SNAP_STYLIZED: {
            // snap the position on the grid
            let cell_size = uniforms.cell_size.xyz;
            let cell_count = uniforms.cell_count.xyz;
            let cell_index = vec3<i32>(floor((position - start_grid) / cell_size));

            distance = get_distance(cell_index);
        }
        case MODE_TRILINEAR: {
            // 0       0'       1      1'      2
            // ================================== 2
            // ||      |       ||      |       ||
            // ||      |       ||      |       ||
            // ||------+---------------+--------- 1'
            // ||      |       ||      |       ||
            // ||      |       ||      |       ||
            // ================================== 1
            // ||      |   +p  ||      |       ||
            // ||      |       ||      |       ||
            // ||------+---------------+--------- 0'
            // ||      |cell   ||      |       ||
            // ||      |center ||      |       ||
            // ================================== 0


            // instead of snapping to the =|| sdf grid (0, 1 2) (cell centered)
            // we snap to the -| dual grid. (0', 1', 2') (vertex centered)
            // so the cell centers are the boundaries of the new cells.
            // then we can use interpolation on this grid to get the distance.
            // snap the position on the grid
            let cell_size = uniforms.cell_size.xyz;
            let cell_count = uniforms.cell_count.xyz;
            let cell_index = (position - uniforms.start.xyz) / cell_size;
            let idx = vec3<i32>(floor(cell_index));

            let c00 = get_distance(idx + vec3<i32>(0, 0, 0)) * (1.0 - fract(cell_index.x)) + get_distance(idx + vec3<i32>(1, 0, 0)) * fract(cell_index.x);
            let c01 = get_distance(idx + vec3<i32>(0, 0, 1)) * (1.0 - fract(cell_index.x)) + get_distance(idx + vec3<i32>(1, 0, 1)) * fract(cell_index.x);
            let c10 = get_distance(idx + vec3<i32>(0, 1, 0)) * (1.0 - fract(cell_index.x)) + get_distance(idx + vec3<i32>(1, 1, 0)) * fract(cell_index.x);
            let c11 = get_distance(idx + vec3<i32>(0, 1, 1)) * (1.0 - fract(cell_index.x)) + get_distance(idx + vec3<i32>(1, 1, 1)) * fract(cell_index.x);

            let c0 = c00 * (1.0 - fract(cell_index.y)) + c10 * fract(cell_index.y);
            let c1 = c01 * (1.0 - fract(cell_index.y)) + c11 * fract(cell_index.y);

            let c = c0 * (1.0 - fract(cell_index.z)) + c1 * fract(cell_index.z);

            distance = c;
        }
        case MODE_TETRAHEDRAL: {
            let cell_size = uniforms.cell_size.xyz;
            let cell_count = uniforms.cell_count.xyz;
            let cell_index = (position - uniforms.start.xyz) / cell_size;
            let idx = vec3<i32>(floor(cell_index));
            let r = fract(cell_index);

            let c = r.xyz > r.yzx;
            let c_xy = c.x;
            let c_yz = c.y;
            let c_zx = c.z;
            let c_yx = !c.x;
            let c_zy = !c.y;
            let c_xz = !c.z;

            var s = vec3(0.0, 0.0, 0.0);
            var vert0 = vec3(0, 0, 0);
            var vert1 = vec3(1, 1, 1);
            var vert2 = vec3(0, 0, 0);
            var vert3 = vec3(1, 1, 1);

            // xyz
            if c_xy && c_yz {
                s = r.xyz;
                vert2.x = 1;
                vert3.z = 0;
            }
            // xzy
            if c_xz && c_zy {
                s = r.xzy;
                vert2.x = 1;
                vert3.y = 0;
            }
            // zxy
            if c_zx && c_xy {
                s = r.zxy;
                vert2.z = 1;
                vert3.y = 0;
            }
            // zyx
            if c_zy && c_yx {
                s = r.zyx;
                vert2.z = 1;
                vert3.x = 0;
            }
            // yzx
            if c_yz && c_zx {
                s = r.yzx;
                vert2.y = 1;
                vert3.x = 0;
            }
            // yxz
            if c_yx && c_xz {
                s = r.yxz;
                vert2.y = 1;
                vert3.z = 0;
            }

            let bary = vec4(1.0 - s.x, s.z, s.x - s.y, s.y - s.z);

            let samples = vec4(
                get_distance(idx + vert0),
                get_distance(idx + vert1),
                get_distance(idx + vert2),
                get_distance(idx + vert3)
            );

            distance = dot(bary, samples);
        }
        default: {}
    }

    return distance;
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    return normalize(vec3(
        sdf_grid(vec3(p.x + EPSILON, p.y, p.z)) - sdf_grid(vec3(p.x - EPSILON, p.y, p.z)),
        sdf_grid(vec3(p.x, p.y + EPSILON, p.z)) - sdf_grid(vec3(p.x, p.y - EPSILON, p.z)),
        sdf_grid(vec3(p.x, p.y, p.z + EPSILON)) - sdf_grid(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

fn phong_lighting(k_d: f32, k_s: f32, alpha: f32, position: vec3<f32>, eye: vec3<f32>, light_pos: vec3<f32>, light_intensity: vec3<f32>) -> vec3<f32> {
    let N = estimate_normal(position);
    let L = normalize(light_pos - position);
    let V = normalize(eye - position);
    let R = normalize(reflect(-L, N));

    let dotLN = dot(L, N);
    let dotRV = dot(R, V);

    if dotLN < 0.0 {
        // Light not visible from this point on the surface
        return light_intensity * 0.02;
    }

    if dotRV < 0.0 {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return light_intensity * (k_d * dotLN);
    }
    return light_intensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

fn unproject(pixel: vec2<f32>) -> vec3<f32> {
    var x = pixel.x / f32(camera.resolution[0]);
    var y = pixel.y / f32(camera.resolution[1]);

    x = x * 2.0 - 1.0;
    y = 1.0 - y * 2.0;

    let dir_eye = camera.proj_inv * vec4(x, y, 0.0, 1.0);
    let dir_world = camera.view_inv * vec4(dir_eye.xyz, 0.0);
    return normalize(dir_world.xyz);
}

// entry point of the 3d raymarching.
fn sdf_3d(p: vec2<f32>) -> vec4<f32> {

    let eye = camera.eye.xyz;
    let ray = unproject(p);

    var position = eye;
    var dist = 0.0;

    // go inside of box if possible
    let MAX_STEPS = 100;
    for (var i = 0; i < MAX_STEPS; i++) {
        dist = sdf_bounding_box(position);
        if dist < EPSILON {
                break;
        }
        if dist == 100.0 {
            break;
        }
        position += ray * dist;
    }

    // put the position inside the box.
    // position += ray * uniforms.cell_size.xyz * 0.5;

    // snap the position on the grid
    position = max(position, uniforms.start.xyz + vec3(EPSILON));
    position = min(position, uniforms.end.xyz - vec3(EPSILON));

    // actual ray marching.
    dist = 0.0;
    for (var i = 0; i < MAX_STEPS; i++) {
        dist = sdf_grid(position);
        if dist < EPSILON {
                break;
        }
        position += ray * dist;
    }

    var color = vec3(0.4, 0.4, 0.4);
    if dist < EPSILON {
        if vis_uniforms.raymarch_mode == MODE_SNAP_STYLIZED {
            color = phong_lighting(0.8, 0.5, 50.0, position, eye, vec3(-5.0, 5.0, 5.0), vec3(0.4, 1.0, 0.4));
        }
        else {
            // add lighting only if we hit something.
            // color = phong_lighting(0.8, 0.5, 50.0, position, eye, shadow_camera.eye.xyz, color);
            let light = shadow_camera.eye.xyz;
            let light_dir = normalize(light - position);
            let ambiant = 0.2;
            let normal = estimate_normal(position);
            let diffuse = max(0.0, dot(normal, light_dir));
            var diffuse_strength = 0.5;

            let view_dir = normalize(camera.eye.xyz - position);
            let half_dir = normalize(view_dir + light_dir);
            let specular = max(0.0, dot(normal, half_dir));

            let brightness = ambiant + (diffuse + specular) * diffuse_strength;

            // arbitrary attenuation
            color.r *= exp(-1.8 * (1.0 - brightness));
            color.g *= exp(-1.9 * (1.0 - brightness));
            color.b *= exp(-1.9 * (1.0 - brightness));
        }

    } else {
        color = vec3(0.0, 0.0, 0.0);
    }

    return vec4<f32>(color, 1.0);
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let xy = in.clip_position.xy / vec2<f32>(camera.resolution);
    let color = sdf_3d(in.clip_position.xy);

    return color;
}

 