// GI compute pass (M33-G6): trace the cosine-weighted hemisphere GI from the deferred G-buffer and
// write a noisy, albedo-demodulated irradiance buffer — the input the denoiser / DLSS Ray
// Reconstruction (G7/G8) resolves. rtx_common.wgsl (Camera/Volume bindings + gather_gi() + the DDA
// tracer) and the active trace()/sun_visibility() are prepended by the renderer; when hardware ray
// query is active, the world TLAS `rt_acc` is bound at group 3 (declared in the injected prefix).
//
// One thread per G-buffer texel. The irradiance is left un-multiplied by albedo and skylight — the
// composite (gi_composite.wgsl) demodulates — so the buffer is a clean indirect-light signal.

@group(2) @binding(0) var g_pos: texture_2d<f32>;     // world-space surface position .xyz (flash .w)
@group(2) @binding(1) var g_normal: texture_2d<f32>;  // world normal .xyz (emission .w)
@group(2) @binding(2) var gi_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn gi_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(gi_out);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let nrm = textureLoad(g_normal, coord, 0).xyz;
    // No opaque surface here (the G-buffer was cleared to a zero normal): sky / background. Write 0
    // so the additive composite is a no-op and no NaN leaks from the undefined storage contents.
    if (dot(nrm, nrm) < 0.5) {
        textureStore(gi_out, coord, vec4<f32>(0.0));
        return;
    }
    let world_pos = textureLoad(g_pos, coord, 0).xyz;
    let n = normalize(nrm);
    // Match the in-fragment GI's per-pixel RNG seed EXACTLY: there `@builtin(position).xy` is the
    // pixel center (integer + 0.5), so seeding gather_gi with `gid + 0.5` reproduces the same sample
    // sequence. Ray directions then match the oracle exactly wherever the fp16-stored normal equals
    // the fragment's (axis-aligned terrain); for rotated normals (mobs) they differ only by the fp16
    // normal rounding — sub-noise, inside the GI buffer's tolerance.
    let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

    // Ambient baseline when GI is off (rtx_mode < 2); the in-fragment path uses the same baseline,
    // so the two agree across every rtx mode (off / shadows / shadows+GI).
    var irr = vec3<f32>(camera.params.z);
    if (volume.params.y >= 2u) {
        irr = gather_gi(world_pos, n, px);
    }
    textureStore(gi_out, coord, vec4<f32>(irr, 1.0));
}
