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

const MODE_SNAP: u32 = 0u;
const MODE_TRILINEAR: u32 = 1u;
const MODE_TETRAHEDRAL: u32 = 2u;
const MODE_SNAP_STYLIZED: u32 = 3u;

@group(2) @binding(0) var<uniform> vis_uniforms: VisUniforms;

@group(3) @binding(0) var<uniform> shadow_camera: CameraUniform;
@group(3) @binding(1) var shadow_map: texture_2d<f32>;
@group(3) @binding(2) var shadow_sampler: sampler;

@group(4) @binding(0) var cubemap: texture_2d_array<f32>;
@group(4) @binding(1) var cubemap_sampler: sampler;
@group(4) @binding(2) var cubemap_depth: texture_2d_array<f32>;
@group(4) @binding(3) var cubemap_depth_sampler: sampler;
@group(4) @binding(4) var<uniform> cubemap_viewprojs: array<mat4x4<f32>, 6>;

// struct CubemapUniforms {
//     left: amat4x4<f32>,
//     right: mat4x4<f32>,
//     front: mat4x4<f32>,
//     rear: mat4x4<f32>,
//     bottom: mat4x4<f32>,
//     top: mat4x4<f32>,
// }

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

// Relative to cell_radius.
const EPSILON = 0.01;

fn get_distance(in_cell: vec3<i32>) -> f32 {
    var cell = max(in_cell, vec3<i32>(0, 0, 0));
    cell = min(cell, vec3<i32>(uniforms.cell_count.xyz - vec3(1u)));
    let ucell = vec3<u32>(cell);
    let idx = ucell.z + ucell.y * uniforms.cell_count.z + ucell.x * uniforms.cell_count.z * uniforms.cell_count.y;
    return sdf[idx] - vis_uniforms.surface_iso;
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
            let cell_fract = fract(cell_index);
            let idx = vec3<i32>(floor(cell_index));

            let c00 = get_distance(idx + vec3(0, 0, 0)) * (1.0 - cell_fract.x) + get_distance(idx + vec3(1, 0, 0)) * cell_fract.x;
            let c01 = get_distance(idx + vec3(0, 0, 1)) * (1.0 - cell_fract.x) + get_distance(idx + vec3(1, 0, 1)) * cell_fract.x;
            let c10 = get_distance(idx + vec3(0, 1, 0)) * (1.0 - cell_fract.x) + get_distance(idx + vec3(1, 1, 0)) * cell_fract.x;
            let c11 = get_distance(idx + vec3(0, 1, 1)) * (1.0 - cell_fract.x) + get_distance(idx + vec3(1, 1, 1)) * cell_fract.x;

            let c0 = c00 * (1.0 - cell_fract.y) + c10 * cell_fract.y;
            let c1 = c01 * (1.0 - cell_fract.y) + c11 * cell_fract.y;

            let c = c0 * (1.0 - cell_fract.z) + c1 * cell_fract.z;

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
    let epsilon = EPSILON * max(uniforms.cell_size.x, max(uniforms.cell_size.y, uniforms.cell_size.z));
    return normalize(vec3(
        sdf_grid(vec3(p.x + epsilon, p.y, p.z)) - sdf_grid(vec3(p.x - epsilon, p.y, p.z)),
        sdf_grid(vec3(p.x, p.y + epsilon, p.z)) - sdf_grid(vec3(p.x, p.y - epsilon, p.z)),
        sdf_grid(vec3(p.x, p.y, p.z + epsilon)) - sdf_grid(vec3(p.x, p.y, p.z - epsilon))
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

fn intersectAABB(rayOrigin: vec3<f32>, rayDir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
    let tMin = (boxMin - rayOrigin) / rayDir;
    let tMax = (boxMax - rayOrigin) / rayDir;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2<f32>(tNear, tFar);
} 

fn get_grid_epsilon() -> f32 {
    return EPSILON * max(uniforms.cell_size.x, max(uniforms.cell_size.y, uniforms.cell_size.z));
}

// entry point of the 3d raymarching.
fn sdf_3d(eye: vec3<f32>, ray: vec3<f32>) -> vec4<f32> {
    let epsilon = get_grid_epsilon();
    var position = eye;

    if eye.x < uniforms.start.x || eye.y < uniforms.start.y || eye.z < uniforms.start.z || eye.x > uniforms.end.x || eye.y > uniforms.end.y || eye.z > uniforms.end.z {
        let box_hit = intersectAABB(eye, ray, uniforms.start.xyz, uniforms.end.xyz);
        if box_hit.x > box_hit.y {
            // outside the box
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    
        position = eye + (box_hit.x + epsilon) * ray;
    }


    // actual ray marching.
    var dist = 0.0;
    let MAX_STEPS = 100;
    for (var i = 0; i < MAX_STEPS; i++) {
        dist = sdf_grid(position);
        if dist < epsilon {
                break;
        }
        position += ray * dist;
    }

    return vec4(position, dist);
}

fn sdf_scene(p: vec2<f32>) -> vec4<f32> {
    let eye = camera.eye.xyz;
    let ray = unproject(p);

    let result = sdf_3d(eye, ray);
    let position = result.xyz;
    let dist = result.w;


    var color = vec3(0.0, 0.0, 0.0);
    if dist < get_grid_epsilon() {
        let eye = camera.eye.xyz;

        if vis_uniforms.raymarch_mode == MODE_SNAP_STYLIZED {
            // Stylized shading
            // It's due to the degenerated normals in the snap grid.
            // Since the gradient is stepped, the normals are 0 most of the time.
            color = phong_lighting(0.8, 0.5, 50.0, position, eye, vec3(-5.0, 5.0, 5.0), vec3(0.4, 1.0, 0.4));
        } else {
            // add lighting only if we hit something.
            color = mix(vec3(0.5, 0.5, 0.5), get_albedo(position).rgb, f32(vis_uniforms.map_material > 0));

            let light = shadow_camera.eye.xyz;
            let light_dir = normalize(light - position);
            let ambiant = 0.2;
            let normal = estimate_normal(position);
            let diffuse = max(0.0, dot(normal, light_dir));
            // var diffuse_strength = 0.5;

            var shadow_uv = shadow_camera.view_proj * vec4<f32>(position.xyz, 1.0);
            shadow_uv /= shadow_uv.w;
            shadow_uv.x = shadow_uv.x * 0.5 + 0.5;
            shadow_uv.y = shadow_uv.y * -0.5 + 0.5;
            var depth = shadow_uv.z;

            let threshold = depth * 1.05;

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

            let view_dir = normalize(camera.eye.xyz - position);
            let half_dir = normalize(view_dir + light_dir);
            let specular = max(0.0, dot(normal, half_dir));

            let brightness = ambiant + (diffuse + specular) * 1.0;

            // arbitrary attenuation
            color.r *= exp(-1.8 * (1.0 - brightness));
            color.g *= exp(-1.9 * (1.0 - brightness));
            color.b *= exp(-1.9 * (1.0 - brightness));
        }
    }

    return vec4<f32>(color, 1.0);
}

fn get_albedo(p: vec3<f32>) -> vec4<f32> {
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

    let normal = estimate_normal(p);

    let epsilon = get_grid_epsilon();
    var layer = -1;
    let offset = epsilon * 10.0;

    var directions = array(
        vec3(-1.0, 0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 1.0, 0.0),
    );
    var best_dot = 0.0;

    // SDF to check visibility.
    for (var i = 0i; i < 6; i = i + 1) {
        let dir = directions[i];
        let dist = sdf_3d(p + offset * dir, dir).w;
        // we check if > epsilon to see if the ray managed to escape the object.
        // when that's the case we keep the best dot product to make it as parallel to the camera look direction as possible.
        // this way we can avoid stretching the cubemap too much.
        // TODO: interpolation between the valid cubemaps.
        if dist > epsilon && dot(dir, normal) > best_dot {
            best_dot = dot(dir, normal);
            layer = i;
        }
    }


    if layer < 0 {
        // If the raymarch gave no result, we find the least worst projection via the cubemap depthmaps.
        // This avoids not being able to render mesh interiors but gives a less accurate result.
        var min_dist = 1e10;
        for (var i = 0; i < 6; i = i + 1) {
            var projected = cubemap_viewprojs[i] * vec4(p, 1.0);
            projected /= projected.w;
            var uv = projected.xy * 0.5 + 0.5;
            uv.y = 1.0 - uv.y;
            let depth = textureSample(cubemap_depth, cubemap_depth_sampler, uv, i).x;
            let depth_lin = (1.0 - depth) * fars[i];
            let delta = abs(depth_lin - projected.z);
            if delta < min_dist {
                layer = i;
                min_dist = delta;
            }
        }
    }

    var color = vec4(1.0, 0.0, 1.0, 1.0);
    if layer >= 0 {
        var projected = cubemap_viewprojs[layer] * vec4(p, 1.0);
        projected /= projected.w;
        var uv = projected.xy * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;
        color = textureSample(cubemap, cubemap_sampler, uv, layer);
    }

    return color;
}

fn draw_lumen_cards(p: vec2<f32>) -> vec4<f32> {
    let eye = camera.eye.xyz;
    let ray = unproject(p);

    let bbox_center = (vis_uniforms.mesh_bounding_box_min.xyz + vis_uniforms.mesh_bounding_box_max.xyz) * 0.5;
    let bbox_size = vis_uniforms.mesh_bounding_box_max.xyz - vis_uniforms.mesh_bounding_box_min.xyz;
    let bbox_min = bbox_center - bbox_size * 0.5;
    let bbox_max = bbox_center + bbox_size * 0.5;

    var dist = 1e10;
    var color = vec4(0.0, 0.0, 0.0, 1.0);

    // left plane
    if ray.x != 0.0 {
        let off = bbox_min.x;
        let t = (off - eye.x) / ray.x;
        let zy = eye.zy + t * ray.zy;
        var uv = (zy - bbox_min.zy) / (bbox_max.zy - bbox_min.zy);
        uv.y = 1.0 - uv.y;
        if t < dist && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            dist = t;
            color = textureSample(cubemap, cubemap_sampler, uv, 0);
        }
    }
    // right plane
    if ray.x != 0.0 {
        let off = bbox_max.x;
        let t = (off - eye.x) / ray.x;
        let zy = eye.zy + t * ray.zy;
        var uv = (zy - bbox_min.zy) / (bbox_max.zy - bbox_min.zy);
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
        if t < dist && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            dist = t;
            color = textureSample(cubemap, cubemap_sampler, uv, 1);
        }
    }
    // front plane
    if ray.z != 0.0 {
        let off = bbox_min.z;
        let t = (off - eye.z) / ray.z;
        let xy = eye.xy + t * ray.xy;
        var uv = (xy - bbox_min.xy) / (bbox_max.xy - bbox_min.xy);
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
        if t < dist && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            dist = t;
            color = textureSample(cubemap, cubemap_sampler, uv, 3);
        }
    }
    // rear plane
    if ray.z != 0.0 {
        let off = bbox_max.z;
        let t = (off - eye.z) / ray.z;
        let xy = eye.xy + t * ray.xy;
        var uv = (xy - bbox_min.xy) / (bbox_max.xy - bbox_min.xy);
        uv.y = 1.0 - uv.y;
        if t < dist && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            dist = t;
            color = textureSample(cubemap, cubemap_sampler, uv, 2);
        }
    }
    // bottom plane
    if ray.y != 0.0 {
        let off = bbox_min.y;
        let t = (off - eye.y) / ray.y;
        let xz = eye.xz + t * ray.xz;
        var uv = (xz - bbox_min.xz) / (bbox_max.xz - bbox_min.xz);
        uv.y = 1.0 - uv.y;
        if t < dist && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            dist = t;
            color = textureSample(cubemap, cubemap_sampler, uv, 4);
        }
    }
    // top plane
    if ray.y != 0.0 {
        let off = bbox_max.y;
        let t = (off - eye.y) / ray.y;
        let x = eye.x + t * ray.x;
        let z = eye.z + t * ray.z;
        var uv = (vec2(x, z) - vec2(bbox_min.x, bbox_min.z)) / vec2(bbox_max.x - bbox_min.x, bbox_max.z - bbox_min.z);
        uv.x = 1.0 - uv.x;
        uv.y = 1.0 - uv.y;
        if t < dist && uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 {
            dist = t;
            color = textureSample(cubemap, cubemap_sampler, uv, 5);
        }
    }

    return color;
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let xy = in.clip_position.xy / vec2<f32>(camera.resolution);

    let color = sdf_scene(in.clip_position.xy);
    // let color = draw_lumen_cards(in.clip_position.xy);

    return color;
}

 