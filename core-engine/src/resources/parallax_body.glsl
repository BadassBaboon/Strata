// Strata depth-parallax shader body — original implementation (Parallax Studio).
//
// Gives a flat photo a 3D feel by shifting each pixel along the camera offset in
// proportion to its depth, pivoting around a mid-scene focal plane so near and far
// move in opposite directions (like looking around a real scene). Uses a smooth
// fixed-point refinement instead of stepped parallax-occlusion marching, which
// avoids the "stacked ripple" banding that POM produces on photo depth maps.
// Written from scratch (DepthFlow referenced for the focal-plane idea only).
//
//   iChannel0 = source image (RGB)
//   iChannel1 = depth map (height in .r; brighter = nearer the camera)
//
// The camera offset comes from the cursor when mouse interactivity is on; with no
// mouse the offset is zero, so the wallpaper sits still (no auto-animation). The
// Studio's live preview injects a synthetic orbiting cursor so the effect is
// visible while authoring. HEIGHT / ZOOM / STEPS are injected above by
// `parallax::parallax_shader` and baked per creation by the Studio.

const float PIVOT = 0.5;   // depth that stays fixed (focal plane), 0=far .. 1=near

float depthAt(vec2 uv) {
    return texture(iChannel1, uv).r;            // near = 1.0
}

// Smooth depth-parallax: shift `uv` by the camera offset scaled by (depth - pivot),
// refined with a few fixed-point iterations so the sampled depth matches the shift.
vec2 parallax(vec2 uv, vec2 v) {
    vec2 disp = v * HEIGHT;
    vec2 p = uv;
    for (int i = 0; i < STEPS; i++) {
        float d = depthAt(p);
        p = uv - disp * (d - PIVOT);
    }
    return p;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // 0..1 across the screen. The engine pre-flips Y for Shadertoy's origin, so
    // flip back to sample the (top-left-origin) photo/depth textures upright.
    vec2 uv = fragCoord / iResolution.xy;
    vec2 st = vec2(uv.x, 1.0 - uv.y);

    // Zoom slightly around the center so displaced samples stay inside the image.
    st = (st - 0.5) / ZOOM + 0.5;

    // Camera offset follows the cursor when interactive; otherwise the scene is
    // still (parallax is a viewpoint effect — no movement, no parallax).
    vec2 v = vec2(0.0);
    if (iMouse.z > 0.5) {
        v = (iMouse.xy / iResolution.xy - 0.5) * 2.0;   // -1..1 from screen center
        v.y = -v.y;                                     // screen y is top-down
    }

    vec2 p = parallax(st, v);

    // Displaced outside the image → black border (the sampler otherwise repeats).
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    fragColor = vec4(texture(iChannel0, p).rgb, 1.0);
}
