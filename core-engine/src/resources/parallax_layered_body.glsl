// Strata layered (LDI) depth-parallax body — foreground over an inpainted
// background, each with its OWN depth so distant and near background elements
// parallax independently (a mountain barely moves; a nearby tree moves more).
// Where a near subject's silhouette parallaxes away, the real (LaMa-inpainted)
// background shows through instead of stretched edge pixels.
//
//   iChannel0 = foreground image (the original photo)
//   iChannel1 = depth: .r = scene depth (subject), .g = background-filled depth
//   iChannel2 = inpainted background image (subject painted out — hole-free)
//   iChannel3 = subject alpha matte (.r, white = foreground)
//
// The matted subject is displaced by a depth DAMPED toward its anchor (SUBJECT_DEPTH):
//     effective_depth = SUBJECT_DEPTH + clamp(sceneDepth - SUBJECT_DEPTH, ±MAX_DELTA) * d
// where d is the *local* damping. Three refinements over a flat global damp:
//   1. Anchor-weighted damp — d = DAMP * smoothstep(FEET_Y, HEAD_Y, y): the contact band
//      (feet/hooves) is a rigid monolith (d→0, perfectly grounded, immune to depth-map
//      noise), scaling up to DAMP at the head so the upper body shows depth.
//   2. Delta clamp — caps |sceneDepth − anchor| at MAX_DELTA so a part pointing at the
//      camera (a raised rifle/arm) can't smear to taffy.
//   3. Edge crispness — the packed R (foreground depth) has the subject's interior depth
//      morphologically extended to its silhouette (done on the CPU), so edge pixels don't
//      sample the antialiased subject/sky average and warp into a halo.
// DAMP = 0 → rigid plane at SUBJECT_DEPTH (Billboard, a clean free-floating graphic).
// Coherent 3-D anchors SUBJECT_DEPTH at the GROUND-CONTACT depth and bakes BG_FACTOR =
// 1.0 so the feet and the trail beneath move at the same velocity (locked/planted);
// Billboard anchors at MEAN depth with BG_FACTOR < 1. HEIGHT/ZOOM/STEPS/SUBJECT_DEPTH/
// BG_FACTOR/DAMP/MAX_DELTA/FEET_Y/HEAD_Y injected by parallax_shader_layered.

const float PIVOT = 0.5;        // focal plane (depth that stays fixed)

float depthBg(vec2 uv) { return texture(iChannel1, uv).g; } // subjects filled/flattened
float depthFg(vec2 uv) {
    float delta = clamp(texture(iChannel1, uv).r - SUBJECT_DEPTH, -MAX_DELTA, MAX_DELTA);
    // Anchor-weighted ramp: 0 at the contact band (FEET_Y), 1 at the top (HEAD_Y). The
    // division handles either Y order, then smooth it. Rigid at the feet, freer up top.
    float t = clamp((uv.y - FEET_Y) / (HEAD_Y - FEET_Y), 0.0, 1.0);
    t = t * t * (3.0 - 2.0 * t);
    return SUBJECT_DEPTH + delta * (DAMP * t);
}

// Per-pixel depth parallax (fixed-point refinement) against a chosen depth channel.
vec2 parallaxBg(vec2 uv, vec2 v) {
    vec2 disp = v * HEIGHT * BG_FACTOR;
    vec2 p = uv;
    for (int i = 0; i < STEPS; i++) { p = uv - disp * (depthBg(p) - PIVOT); }
    return p;
}
vec2 parallaxFg(vec2 uv, vec2 v) {
    vec2 disp = v * HEIGHT;
    vec2 p = uv;
    for (int i = 0; i < STEPS; i++) { p = uv - disp * (depthFg(p) - PIVOT); }
    return p;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 st = vec2(uv.x, 1.0 - uv.y);
    st = (st - 0.5) / ZOOM + 0.5;

    // Camera offset. When the cursor drives this layer (mouse interactivity on for
    // it) follow it; otherwise drift on a slow Lissajous path so the layers keep
    // gently, smoothly moving on their own instead of sitting as a dead still image.
    vec2 v;
    if (iMouse.z > 0.5) {
        // Gentle cursor follow. A LOW gain (0.6, was 2.0) keeps the parallax subtle:
        // the displacement is `v * HEIGHT`, and a large displacement is what smears
        // near/inpainted layers and the background as the cursor moves. This restores
        // the calm feel of the original automatic-mode wallpapers.
        v = (iMouse.xy / iResolution.xy - 0.5) * 0.6;
        v.y = -v.y;
    } else {
        // Idle drift (no cursor driving this layer): slow Lissajous, kept subtle
        // (amplitude 0.4, trimmed ~20% from 0.5).
        v = vec2(sin(iTime * 0.23), sin(iTime * 0.31 + 1.7)) * 0.4;
    }

    // Background: per-pixel depth parallax against the smooth background depth.
    vec2 pb = clamp(parallaxBg(st, v), 0.0, 1.0);
    vec3 bg = texture(iChannel2, pb).rgb;

    // Foreground: damped per-pixel — rigid-dominant, with a fraction of true depth.
    vec2 pf = parallaxFg(st, v);
    float m = texture(iChannel3, pf).r;
    if (pf.x < 0.0 || pf.x > 1.0 || pf.y < 0.0 || pf.y > 1.0) {
        m = 0.0;
    }
    vec3 fg = texture(iChannel0, clamp(pf, 0.0, 1.0)).rgb;

    // Composite subject over the inpainted background by the (feathered) alpha.
    fragColor = vec4(mix(bg, fg, smoothstep(0.35, 0.65, m)), 1.0);
}
