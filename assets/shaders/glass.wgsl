// Glass: an alpha-blended translucent surface (M26). Samples the framed, semi-transparent glass
// tile, applies the same sun + sky/block light as opaque chunks, and outputs with the tile's alpha
// so you can see through it. No GI hemisphere rays (cheap) and no UV animation. rtx_common.wgsl is
// prepended at build time and provides Camera/Volume, VsOut, sample_tile(), and sun_visibility().

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let texel = sample_tile(in.tile, in.uv);
    let albedo = texel.rgb;

    let sun = normalize(camera.sun_dir.xyz);
    let ambient = camera.params.z;
    let sun_intensity = camera.params.w;
    let ndl = max(dot(n, sun), 0.0);

    var shadow = 1.0;
    if (volume.params.y >= 1u) {
        // Primary sun-shadow ray; range is a tier knob (P21, paramsg.x; maxed tier = the old 96.0).
        shadow = sun_visibility(in.world_pos, n, volume.paramsg.x);
    }
    let direct = sun_intensity * ndl * shadow;

    let sky = in.light.x;
    let blockl = in.light.y;
    let warm = blockl * vec3<f32>(1.15, 0.92, 0.55);
    var rgb = albedo * (vec3<f32>(ambient) * sky + vec3<f32>(direct) + warm);

    let dist = length(in.world_pos - camera.cam_pos.xyz);
    let fog = smoothstep(camera.params.x, camera.params.y, dist);
    rgb = mix(rgb, camera.fog_color.rgb, fog);

    return vec4<f32>(rgb, texel.a);
}
