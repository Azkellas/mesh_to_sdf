struct SdfUniforms {
    // acts as bounding box
    start: vec4<f32>,
    end: vec4<f32>,
    cell_size: vec4<f32>,
    cell_count: vec4<u32>,
}
@group(0) @binding(0) var<uniform> uniforms : SdfUniforms;
@group(0) @binding(1) var<storage, read> sdf : array<f32>;

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

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    // @location(0) vertex_position: vec2<f32>,
    @builtin(instance_index) instance_index: u32,
};

struct VertexOutput {
    @builtin(position) svposition: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) center_pos: vec2<f32>,
    @location(2) radius: f32,
};


@vertex
fn main_vs(
    in: VertexInput,
) -> VertexOutput {
    let cell_count = uniforms.cell_count.xyz;
    let cell_size = uniforms.cell_size.xyz;

    // compute cell world position
    // index was generated via for x in 0..cell_x { for y in 0..cell_y { for z in 0..cell_z { ... } } }
    // so index is x*cell_y*cell_z + y*cell_z + z
    var index = in.instance_index;
    let cell_z = index % cell_count.z;
    index /= cell_count.z;
    let cell_y = index % cell_count.y;
    index /= cell_count.y;
    let cell_x = index;
    let cell_idx = vec3<f32>(f32(cell_x), f32(cell_y), f32(cell_z));
    let cell = uniforms.start.xyz + cell_idx * cell_size;

    // cell view position
    let viewpos = camera.view * vec4<f32>(cell, 1.0);

    // we want the color to be homogeneous in the sphere
    // so we compute it here directly.
    let distance = sdf[in.instance_index];

    // affect a score for a point in the sphere
    var positive_strength = 0.;
    var negative_strength = 0.;
    var surface_strength = 0.;

    let cell_radius = min(cell_size.x, min(cell_size.y, cell_size.z)) * 0.5;

    // positive strength is how "positive" is the point
    if distance > vis_uniforms.surface_width {
        positive_strength = saturate(vis_uniforms.positive_power * distance / cell_radius);
    }
    // negative strength is how "negative" is the point
    // we boost it arbitrarily to make it more visible since inside points are more likely to be small
    // and less visible since behind the surface
    if distance < - vis_uniforms.surface_width {
        negative_strength = saturate(- vis_uniforms.negative_power * distance / cell_radius);
    }
    // surface strength is how close to the surface is the point
    if abs(distance) < vis_uniforms.surface_width {
        surface_strength = saturate(vis_uniforms.surface_power * (1.0 - abs(distance) / vis_uniforms.surface_width));
    }

    // compute vertex position
    // the equilateral triangle inscribing the unit circle has its vertices on the circle of radius 2.
    // the vertex position is 2d (in view space), but before the projection to keep perspective.
    // this allow us to draw unlit spheres cheaply.
    // see https://mastodon.gamedev.place/@Az/111824726660048628
    let PI_2_3 = 2.094395; // 2 * pi / 3
    let angle = PI_2_3 * f32(in.vertex_index);
    var vertex_position = 2.0 * vec4<f32>(cos(angle), sin(angle), 0.0, 0.0);

    // scale the vertex position to the cell size
    // we use the minimum dimension to keep the triangle equilateral and thus the sphere inscribed in the cell
    let min_dim = min(cell_size.x, min(cell_size.y, cell_size.z));
    vertex_position *= min_dim;

    // color is a mix of the 3 colors
    // since we expect only one strength to be strong, we can just add them
    let positive_color = vis_uniforms.positive_color * positive_strength;
    let negative_color = vis_uniforms.negative_color * negative_strength;
    let surface_color = vis_uniforms.surface_color * surface_strength;
    let color = positive_color + negative_color + surface_color;

    // alpha is the sum of the strengths
    // only one should be "big"
    // this let us dim the point if the "big" strength was reduced by its power
    // note that while we call it alpha, it is actually a size factor
    // we don't have alpha blending in this shader since we have depth testing
    // so reducing the size acts as a dimming
    let alpha = positive_strength + negative_strength + surface_strength;
    let point_size = vis_uniforms.point_size * alpha * 0.5;
    // svposition is the position in screen space, normalized
    let svposition = camera.proj * (viewpos + vertex_position * point_size);

    // we need to know the distance between the drawn pixel in the fragment shader and the center pixel
    // so we can discard pixels outside of the inscribed circle
    // since the vertex returns a triangle.

    // center is the exact pixel position of the center of the sphere
    var center_svpos = camera.proj * viewpos;
    center_svpos = center_svpos / center_svpos.w;
    var center = (center_svpos.xy + 1.0) / 2.0 * vec2<f32>(camera.resolution);
    center.y = f32(camera.resolution.y) - center.y; // invert y

    // svpos is the exact pixel position of the drawn vertice
    var svpos = svposition.xy / svposition.w;
    svpos = (svpos + 1.0) / 2.0 * vec2<f32>(camera.resolution);
    svpos.y = f32(camera.resolution.y) - svpos.y; // invert y

    // delta between the two pixels
    let offset = svpos - center;
    // distance_squared / 2^2 since we go from circle 2 to circle 1.
    let radius = dot(offset, offset) * 0.25;

    // output
    var out: VertexOutput;
    out.svposition = svposition;
    out.color = color.xyz;
    out.center_pos = center;
    out.radius = radius;
    return out;
}


@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4<f32> {
    let offset = in.svposition.xy - in.center_pos;
    // discard pixels outside of the inscribed circle
    if dot(offset, offset) > in.radius {
        discard;
    }
    return vec4(in.color, 1.0);
}
