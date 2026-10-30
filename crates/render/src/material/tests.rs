//! Tests of the MatCap images and their half-float texels.

use super::matcap::{LEVELS, SIDE, half_bits, mips};
use super::*;

/// The half-float value of `bits`, decoded independently of [`half_bits`].
fn half_value(bits: u16) -> f64 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f64::from(bits & 0x3FF);
    if exponent == 0 {
        return sign * mantissa * 2_f64.powi(-24);
    }
    sign * (1.0 + mantissa / 1024.0) * 2_f64.powi(exponent - 15)
}

#[test]
fn half_floats_round_to_the_nearest_ties_to_even() {
    let known = [
        (0.0, 0x0000),
        (-0.0, 0x8000),
        (1.0, 0x3C00),
        (0.5, 0x3800),
        (-2.0, 0xC000),
        (65504.0, 0x7BFF),
        (65520.0, 0x7C00),
        (1e9, 0x7C00),
        (2_f32.powi(-24), 0x0001),
        (2_f32.powi(-25), 0x0000),
        (2_f32.powi(-14), 0x0400),
        (0.1, 0x2E66),
        (1.0 / 3.0, 0x3555),
        // Ties: 1 + 2^-11 lies halfway between 1 and the next half, and goes to even (1).
        (1.0 + 2_f32.powi(-11), 0x3C00),
        (1.0 + 3.0 * 2_f32.powi(-11), 0x3C02),
        (f32::INFINITY, 0x7C00),
    ];
    for (value, bits) in known {
        assert_eq!(half_bits(value), bits, "{value}");
    }
    assert_eq!(
        half_bits(f32::NAN) & 0x7C00,
        0x7C00,
        "NaN keeps its exponent"
    );
    assert_ne!(half_bits(f32::NAN) & 0x3FF, 0, "and a mantissa");
}

#[test]
fn every_half_float_in_range_is_the_nearest_to_its_value() {
    // A sweep over the range the images use, subnormals included: the chosen half is never
    // farther than another from the value.
    // From 1e-8 (a subnormal half) to about 49,000, short of the largest half (65,504).
    for step in 0..79_000 {
        let value = 1e-8_f32 * 1.000_37_f32.powi(step);
        let bits = half_bits(value);
        let error = (half_value(bits) - f64::from(value)).abs();
        for other in [bits.saturating_sub(1), bits + 1] {
            let other_error = (half_value(other) - f64::from(value)).abs();
            assert!(
                error <= other_error,
                "{value}: {bits:#x} against {other:#x}"
            );
        }
    }
}

/// The texel of `image` (a level of `side`) at `(x, y)` in `-1..1`, y up.
fn at(texels: &[[f32; 4]], side: u32, x: f32, y: f32) -> [f32; 4] {
    let side_f = f32::from(u16::try_from(side).expect("small"));
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a texel index of a small image"
    )]
    let (s, t) = (
        (f32::midpoint(x, 1.0) * side_f) as usize,
        ((1.0 - y) * 0.5 * side_f) as usize,
    );
    texels[t * side as usize + s]
}

/// The luminance of `texel`.
fn luminance(texel: [f32; 4]) -> f32 {
    0.2126 * texel[0] + 0.7152 * texel[1] + 0.0722 * texel[2]
}

#[test]
fn the_studio_lights_each_sphere_from_the_upper_left_and_the_glossy_one_past_white() {
    for style in MatCapStyle::ALL {
        let levels = mips(style);
        let top = &levels[0];
        assert_eq!(top.side(), SIDE);
        let texels = top.texels();
        assert!(
            texels.iter().flatten().all(|c| c.is_finite() && *c >= 0.0),
            "{style:?}: finite, not negative"
        );
        let key = luminance(at(texels, SIDE, -0.35, 0.45));
        let away = luminance(at(texels, SIDE, 0.35, -0.6));
        assert!(key > 1.5 * away, "{style:?}: lit {key} against {away}");
    }
    let glossy = mips(MatCapStyle::Glossy);
    let brightest = glossy[0]
        .texels()
        .iter()
        .map(|t| t[0].max(t[1]).max(t[2]))
        .fold(0.0_f32, f32::max);
    assert!(brightest > 1.5, "an HDR highlight: {brightest}");
    let clay = mips(MatCapStyle::Clay);
    let clay_brightest = clay[0]
        .texels()
        .iter()
        .map(|t| t[0].max(t[1]).max(t[2]))
        .fold(0.0_f32, f32::max);
    assert!(clay_brightest < 1.2, "clay stays soft: {clay_brightest}");
}

#[test]
fn each_mip_level_halves_the_one_before_and_keeps_its_mean() {
    let levels = mips(MatCapStyle::Clay);
    assert_eq!(u32::try_from(levels.len()).expect("small"), LEVELS);
    let mean = |image: &[[f32; 4]]| {
        let n = f64::from(u32::try_from(image.len()).expect("small"));
        image.iter().map(|t| f64::from(t[0])).sum::<f64>() / n
    };
    let first = mean(levels[0].texels());
    let pairs = levels.windows(2).enumerate();
    for (k, pair) in pairs {
        assert_eq!(pair[1].side(), pair[0].side() / 2, "level {k}");
        let next = mean(pair[1].texels());
        assert!(
            (next - first).abs() < 1e-4 * first,
            "level {k}: {next} against {first}"
        );
    }
    assert_eq!(levels.last().expect("levels").side(), 1);
}

/// The probe, after the interface and the MatCap's source: `material` for each case's normal
/// (view space, the eye down -z) and roughness term.
const PROBE_WGSL: &str = r"
// Per case: the normal (xyz) and the roughness term (w).
@group(3) @binding(0) var<storage, read> cases: array<vec4<f32>>;
// What the material returns for each.
@group(3) @binding(1) var<storage, read_write> colours: array<vec4<f32>>;

// Runs the material on each case, seen head on from +z.
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&cases) {
        return;
    }
    let c = cases[id.x];
    let point = Surface(vec3<f32>(0.0), normalize(c.xyz), c.w, vec4<f32>(0.0));
    let eye = Eye(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0));
    colours[id.x] = vec4<f32>(material(point, eye), 1.0);
}
";

/// The MatCap of `style`'s colour for each `(normal, roughness)` of `cases`, on the GPU.
fn shade_cases(gpu: &Gpu, style: MatCapStyle, cases: &[[f32; 4]]) -> Vec<[f32; 4]> {
    use wgpu::util::DeviceExt as _;
    let matcap = MatCap::new(gpu, style).expect("the matcap");
    let device = gpu.device();
    let source = format!("{SURFACE_WGSL}{}{PROBE_WGSL}", matcap.source());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("material probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let entry = |binding, read_only| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let probe_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("material probe"),
        entries: &[entry(0, true), entry(1, false)],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("material probe"),
        bind_group_layouts: &[None, None, Some(matcap.layout()), Some(&probe_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("material probe"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cases"),
        contents: bytemuck::cast_slice(cases),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output =
        crate::cull::storage_buffer(gpu, "colours", cases.len() as u64 * 16).expect("a buffer");
    let group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("material probe"),
        layout: &probe_layout,
        entries: &[
            crate::cull::entry(0, &input),
            crate::cull::entry(1, &output),
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(2, matcap.bind_group(), &[]);
        pass.set_bind_group(3, &group, &[]);
        let groups = u32::try_from(cases.len()).expect("small").div_ceil(64);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    gpu.queue().submit([encoder.finish()]);
    gpu.read_buffer(&output).expect("read back")
}

#[test]
fn the_roughness_term_widens_the_lookup_so_filtered_detail_still_shades() {
    crate::gpu::GpuTest::run(|gpu| {
        // The glossy MatCap's key highlight (the key light's half vector with the eye's), and a
        // dark place beside it.
        let half = [-0.474_f32, 0.685, 0.553 + 1.0];
        let length = half.iter().map(|c| c * c).sum::<f32>().sqrt();
        let highlight = half.map(|c| c / length);
        let beside = [0.2, -0.3, 0.93];
        let roughness = [1.0, 0.99, 0.9, 0.6];
        let cases: Vec<[f32; 4]> = [highlight, beside]
            .iter()
            .flat_map(|n| roughness.map(|r| [n[0], n[1], n[2], r]))
            .collect();
        let colours = shade_cases(gpu, MatCapStyle::Glossy, &cases);
        let peak: Vec<f32> = colours[..4].iter().map(|c| luminance(*c)).collect();
        let dark: Vec<f32> = colours[4..].iter().map(|c| luminance(*c)).collect();
        assert!(peak[0] > 1.5, "the highlight, sharp: {peak:?}");
        assert!(
            peak.windows(2).all(|w| w[1] < w[0]),
            "rougher, the highlight spreads and falls: {peak:?}"
        );
        assert!(peak[3] < 0.5 * peak[0], "spread out: {peak:?}");
        assert!(dark[3] > dark[0], "and lights what lay beside it: {dark:?}");
        // At roughness 1 the lookup is the image itself, at its finest level.
        let top = mips(MatCapStyle::Glossy);
        let texel = at(top[0].texels(), SIDE, highlight[0], highlight[1]);
        assert!(
            (luminance(colours[0]) - luminance(texel)).abs() < 0.05 * luminance(texel),
            "{:?} against {texel:?}",
            colours[0]
        );
    });
}
