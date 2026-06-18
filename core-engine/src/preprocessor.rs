use wgpu::naga::ShaderStage;
use wgpu::naga::front::glsl::{Frontend, Options};

/// Maps a line in the *preprocessed* shader back to where it came from, so a
/// compile error can name the user's original `image.glsl`/`common.glsl` line
/// instead of a line in the engine-injected prelude/wrapper.
#[derive(Clone, Debug, Default)]
pub struct SourceMap {
    header_newlines: usize,  // engine uniform/binding header
    prelude_newlines: usize, // header + injected common.glsl block (start of user body)
    body_lines: usize,       // user shader lines
    has_common: bool,
}

impl SourceMap {
    /// Human-readable origin of a 1-based line in the preprocessed source.
    pub fn describe(&self, line: usize) -> String {
        if line > self.prelude_newlines && line <= self.prelude_newlines + self.body_lines {
            format!("line {}", line - self.prelude_newlines)
        } else if self.has_common && line > self.header_newlines && line <= self.prelude_newlines {
            // -1 for the "// --- COMMON GLSL INJECTION ---" marker line.
            format!("common.glsl line {}", line.saturating_sub(self.header_newlines + 1))
        } else {
            "engine-generated code".to_string()
        }
    }
}

pub fn preprocess_shader(
    shader_body: &str,
    common_glsl: Option<&str>,
    stage: ShaderStage,
    // True for the final on-screen "image" pass, false for offscreen buffer
    // passes. Offscreen passes must NOT Y-flip fragCoord so that a pass which
    // samples its own previous frame (Shadertoy-style BufferA feedback) stays
    // coordinate-consistent with how the texture is stored; the image pass keeps
    // the flip for correct on-screen orientation. (Single-pass shaders are
    // image-only, so they're unaffected.)
    final_pass: bool,
    // Per-channel cubemap flag. A `true` channel is declared `textureCube` /
    // `samplerCube` (Shadertoy cubemap input) instead of the default 2D sampler,
    // so `texture(iChannelN, vec3 dir)` works. The pipeline's bind-group layout
    // for the pass must match (Cube vs D2 view dimension).
    cube_channels: [bool; 4],
) -> (String, SourceMap) {
    // Strip #version but keep a blank line in its place so line numbers in the
    // user's source still line up exactly with the preprocessed source.
    let mut stripped_body = String::new();
    for line in shader_body.lines() {
        if !line.trim().starts_with("#version") {
            stripped_body.push_str(line);
        }
        stripped_body.push('\n');
    }

    let mut stripped_common = String::new();
    if let Some(common) = common_glsl {
        for line in common.lines() {
            if !line.trim().starts_with("#version") {
                stripped_common.push_str(line);
            }
            stripped_common.push('\n');
        }
    }

    // Standard header defining the global uniforms and binding layouts. The
    // per-channel sampler types depend on `cube_channels` so a cubemap input is
    // declared `textureCube`/`samplerCube` (Shadertoy `texture(iChannelN, vec3)`).
    let mut header = String::from(
        r#"#version 450

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    vec3 iResolution;
    float iTime;
    float iTimeDelta;
    float iFrameRate;
    int iFrame;
    float iSampleRate;
    vec4 iMouse;
    vec4 iDate;
    vec4 iChannelTime;
    vec4 iChannelResolution[4];
    vec3 iGlobalResolution;
    float _pad0;
    vec2 iMonitorOffset;
    float iOpacity;
    int iBlendMode;
};

"#,
    );
    for ch in 0..4 {
        let tex_binding = ch * 2;
        let smp_binding = ch * 2 + 1;
        let tex_type = if cube_channels[ch] { "textureCube" } else { "texture2D" };
        header.push_str(&format!(
            "layout(set = 1, binding = {tex_binding}) uniform {tex_type} iChannel{ch}_tex;\n\
             layout(set = 1, binding = {smp_binding}) uniform sampler iChannel{ch}_sampler;\n"
        ));
    }
    header.push('\n');
    for ch in 0..4 {
        let smp_type = if cube_channels[ch] { "samplerCube" } else { "sampler2D" };
        header.push_str(&format!(
            "#define iChannel{ch} {smp_type}(iChannel{ch}_tex, iChannel{ch}_sampler)\n"
        ));
    }

    // naga's GLSL frontend miscompiles the single-argument matrix constructors
    // `mat2(vec4)` / `mat2(float)` (the rotation idiom `mat2(cos(a+vec4(...)))`
    // common in Shadertoy shaders) into a malformed Compose -> "Function is
    // invalid". The Shadertoy importer rewrites such single-arg `mat2(...)` calls
    // to `_stm2(...)`; these overloads give the spelled-out, naga-safe form.
    header.push_str(
        "\nmat2 _stm2(float s) { return mat2(s, 0.0, 0.0, s); }\n\
         mat2 _stm2(vec4 v) { return mat2(v.x, v.y, v.z, v.w); }\n",
    );

    let mut preprocessed = String::new();
    preprocessed.push_str(&header);
    let header_newlines = preprocessed.matches('\n').count();

    let has_common = !stripped_common.is_empty();
    if has_common {
        preprocessed.push_str("// --- COMMON GLSL INJECTION ---\n");
        preprocessed.push_str(&stripped_common);
        preprocessed.push_str("// --- END COMMON GLSL INJECTION ---\n");
    }
    let prelude_newlines = preprocessed.matches('\n').count();

    let body_lines = stripped_body.matches('\n').count();
    preprocessed.push_str(&stripped_body);

    if stage == ShaderStage::Fragment && !final_pass {
        // Offscreen buffer pass: render in the texture's native (top-left) space
        // with NO Y-flip and NO per-layer compositing, so self-feedback sampling
        // (e.g. `texture(iChannel0, fragCoord/iResolution)`) hits the matching
        // texel. The final image pass re-orients the composited result.
        preprocessed.push_str(r#"
layout(location = 0) out vec4 outColor;
void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(color, fragCoord);
    outColor = color;
}
"#);
    } else if stage == ShaderStage::Fragment {
        preprocessed.push_str(r#"
layout(location = 0) out vec4 outColor;
void main() {
    // Map this monitor's pixel into the global canvas.  In the normal
    // independent case iMonitorOffset is (0,0) and iResolution equals the
    // monitor's own size, so this reduces to the usual full-screen mapping.
    // In "Span Monitors" mode iMonitorOffset is this monitor's top-left within
    // the unified desktop canvas and iResolution is the whole canvas, so each
    // screen renders just its slice of one continuous shader.
    vec2 globalFrag = gl_FragCoord.xy + iMonitorOffset;
    // Flip Y to match Shadertoy's bottom-left origin (within the canvas space).
    vec2 fragCoord = vec2(globalFrag.x, iResolution.y - globalFrag.y);
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    mainImage(color, fragCoord);

    // Per-layer compositing for the screen 'image' pass.  Offscreen passes keep
    // iOpacity = 1 / iBlendMode = 0 (engine default), so this is a no-op there.
    if (iBlendMode == 2) {
        // Multiply: fade toward the identity (white) as opacity drops, so the
        // hardware multiply blend (src*dst) leaves the backdrop untouched at 0.
        outColor = vec4(mix(vec3(1.0), color.rgb, iOpacity), 1.0);
    } else {
        // Normal / additive: premultiplied output (rgb*a). Shadertoy's image pass
        // treats fragColor as OPAQUE (its alpha is ignored), so we use the user's
        // layer opacity as alpha — NOT color.a. Many shaders leave a garbage or
        // <1 alpha (e.g. a final tanh()) which would otherwise dim the wallpaper.
        float a = iOpacity;
        outColor = vec4(color.rgb * a, a);
    }
}
"#);
    }

    let map = SourceMap { header_newlines, prelude_newlines, body_lines, has_common };
    (preprocessed, map)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_compile_simple_shader() {
        let source = "void mainImage(out vec4 fragColor, vec2 fragCoord) { fragColor = vec4(1.0); }";
        let (preprocessed, _map) = preprocess_shader(source, None, ShaderStage::Fragment, true, [false;4]);
        let result = compile_shader(&preprocessed, ShaderStage::Fragment);
        if let Err(ref e) = result {
            println!("{}", e);
        }
        assert!(result.is_ok());
    }

    #[test]
    #[ignore] // run explicitly: cargo test -p core-engine audit_all_wallpapers -- --ignored --nocapture
    fn audit_all_wallpapers() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().join("wallpapers");
        let mut pass = 0;
        let mut fail = 0;
        let mut entries: Vec<_> = std::fs::read_dir(&root).unwrap()
            .filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        for dir in entries {
            let image = dir.join("image.glsl");
            if !image.exists() { continue; }
            let body = std::fs::read_to_string(&image).unwrap();
            let common = std::fs::read_to_string(dir.join("common.glsl")).ok();
            let (pre, _map) = preprocess_shader(&body, common.as_deref(), ShaderStage::Fragment, true, [false;4]);
            let name = dir.file_name().unwrap().to_string_lossy().to_string();
            match compile_shader(&pre, ShaderStage::Fragment) {
                Ok(_) => { pass += 1; println!("  PASS  {}", name); }
                Err(e) => { fail += 1; println!("  FAIL  {}  -> {}", name, e.lines().next().unwrap_or("")); }
            }
        }
        println!("\n=== {} passed, {} failed ===", pass, fail);
    }

    #[test]
    fn test_compile_shader_with_channel() {
        let source = "void mainImage(out vec4 fragColor, vec2 fragCoord) { fragColor = texture(iChannel0, fragCoord/iResolution.xy); }";
        let (preprocessed, _map) = preprocess_shader(source, None, ShaderStage::Fragment, true, [false;4]);
        let result = compile_shader(&preprocessed, ShaderStage::Fragment);
        if let Err(ref e) = result {
            println!("{}", e);
        }
        assert!(result.is_ok());
    }
}

pub fn compile_shader(
    source: &str,
    stage: ShaderStage,
) -> Result<wgpu::naga::Module, String> {
    compile_shader_mapped(source, stage, None)
}

/// Like `compile_shader` but, given the `SourceMap` from `preprocess_shader`,
/// reports compile/validation errors at the user's *original* shader line
/// instead of a line in the engine-injected prelude.
pub fn compile_shader_mapped(
    source: &str,
    stage: ShaderStage,
    map: Option<&SourceMap>,
) -> Result<wgpu::naga::Module, String> {
    let loc = |line: usize| -> String {
        map.map(|m| m.describe(line)).unwrap_or_else(|| format!("line {}", line))
    };

    // naga's GLSL frontend can *panic* on some valid-but-unusual shaders (e.g.
    // large const arrays with const-indexed initializers). Catch the unwind so a
    // single bad shader can never crash the render thread / freeze the desktop —
    // it just fails to load with an error instead.
    let parse_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut frontend = Frontend::default();
        let options = Options { stage, defines: Default::default() };
        frontend.parse(&options, source).map_err(|errors| {
            let mut formatted = String::from("GLSL compile error(s):\n");
            for err in errors.errors {
                let line = err.meta.location(source).line_number as usize;
                formatted.push_str(&format!("  - {}: {:?}\n", loc(line), err.kind));
            }
            formatted
        })
    }));
    let module = match parse_result {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("GLSL frontend panicked (unsupported construct)".to_string()),
    };

    // Validate (also panic-guarded for the same reason).
    let validate_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut validator = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        );
        validator.validate(&module).map(|_| ()).map_err(|e| {
            match e.location(source) {
                Some(l) => format!("Shader validation error at {}: {}", loc(l.line_number as usize), e.as_inner()),
                None => format!("Shader validation error: {}", e.as_inner()),
            }
        })
    }));
    match validate_result {
        Ok(Ok(())) => Ok(module),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("Shader validator panicked".to_string()),
    }
}
