//! The material interface (spec, "Shading"): surface attributes in, linear colour out, and
//! MatCap, its first implementation.
//!
//! # The interface
//!
//! The shading pass ([`crate::shade`]) rebuilds, at each pixel, the surface the raster drew
//! there and hands it to the material as WGSL's `Surface` ([`SURFACE_WGSL`]):
//!
//! - `position`: the point, world units;
//! - `normal`: the unit normal, interpolated across the triangle from its vertices' normals;
//! - `roughness`: the roughness term, the length of the mean normal the display layer averaged for
//!   the vertex (Toksvig): 1 where the surface under the vertex is flat at the drawn scale (a
//!   finest tile is always 1), shorter where finer detail than the tile draws turns the normals
//!   apart, so a material can widen its response and that detail still shades rather than
//!   vanishing;
//! - `channels`: the painted material channels: none until painting (zero), the hook PBR needs.
//!
//! with the view as WGSL's `Eye` (the unit direction to the eye, and the view's up and right).
//! A material is WGSL defining `fn material(surface: Surface, eye: Eye) -> vec3<f32>`, which
//! returns **linear** colour (HDR: values past 1 are kept by the 16-bit float target, the tone map
//! rolls them off), plus its own resources at bind group [`MATERIAL_GROUP`]: [`Material`]. The pass
//! is built once from it, so a change to the material's settings is a uniform write and one
//! shading pass, with no raster: independent of the mesh (spec target).
//!
//! PBR (image-based lighting, material channels in the display layer, depth-based ambient
//! occlusion) comes with painting, as another [`Material`]: the interface already carries what it
//! needs (position, the normal, the roughness term, channels).
//!
//! # MatCap
//!
//! A MatCap is a sphere's image, lit, seen head on: the colour a surface shows is the sphere's
//! where its normal points the same way. [`MatCap`] looks the normal up in the view's frame turned
//! towards the pixel's line of sight (so a surface at the image's edge isn't stretched), and
//! widens the lookup by the roughness term (Toksvig): it samples a mip level of the image as
//! blurred as the normals under the vertex are spread, so detail filtered out of the geometry
//! still shades, as a softer response rather than nothing. The images are ours, generated
//! procedurally ([`MatCapStyle`]), never copied from anywhere.

mod matcap;

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBindingType, Extent3d, FilterMode, MipmapFilterMode, Origin3d, SamplerBindingType,
    SamplerDescriptor, ShaderStages, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

pub use self::matcap::MatCapStyle;
use crate::cull::uniform_buffer;
use crate::gpu::{Gpu, GpuError};

/// The bind group a material's resources sit at in the shading pass (groups 0 and 1 are the
/// pass's own: the frame's buffers and the slot buffers).
pub const MATERIAL_GROUP: u32 = 2;

/// The WGSL of the material interface: `Surface` and `Eye`.
///
/// `Surface` is what the shading pass rebuilt at a pixel, `Eye` the view. A material defines
/// `fn material(surface: Surface, eye: Eye) -> vec3<f32>` after it, returning linear colour.
pub const SURFACE_WGSL: &str = r"
// A point of the drawn surface, as the shading pass rebuilt it (render::material, the material
// interface).
struct Surface {
    // The point, world units.
    position: vec3<f32>,
    // The unit normal, interpolated from the triangle's vertices.
    normal: vec3<f32>,
    // The roughness term: the length of the mean normal under the point, 0..1 (Toksvig); 1 where
    // nothing finer than the drawn level turns the normals apart.
    roughness: f32,
    // The painted material channels: none until painting, zero.
    channels: vec4<f32>,
}

// The view, for a material that needs it.
struct Eye {
    // Unit length, from the point towards the eye.
    to_eye: vec3<f32>,
    // The view's up, unit length.
    up: vec3<f32>,
    // The view's right, unit length.
    right: vec3<f32>,
}
";

/// A material: WGSL defining `fn material(surface: Surface, eye: Eye) -> vec3<f32>` (after
/// [`SURFACE_WGSL`]), linear colour out, with its resources at bind group [`MATERIAL_GROUP`].
///
/// The shading pass builds its pipelines once from [`Material::source`] and
/// [`Material::layout`], and binds [`Material::bind_group`] each time it runs: a material's
/// settings ([`Material::Settings`]) change through its own buffers, with no new pipeline. The
/// buffers its group binds count against the shading pass's per-stage limits (on Metal storage
/// and uniform buffers share one table), so it declares them: the pass refuses a slot layout
/// they leave no room for.
pub trait Material: std::fmt::Debug {
    /// What can change about it without a new pipeline.
    type Settings: Copy + PartialEq + std::fmt::Debug;

    /// Uniform buffers its bind group holds.
    const UNIFORM_BUFFERS: u32;

    /// Storage buffers its bind group holds.
    const STORAGE_BUFFERS: u32;

    /// The WGSL: `material` and the bindings it reads, at group [`MATERIAL_GROUP`], seen by
    /// the compute and fragment stages (the shading pass runs as either).
    fn source(&self) -> &str;

    /// The layout of its bind group.
    fn layout(&self) -> &BindGroupLayout;

    /// Its bind group.
    fn bind_group(&self) -> &BindGroup;

    /// Its settings now.
    fn settings(&self) -> Self::Settings;

    /// Sets its settings, from the next shading pass on.
    fn set_settings(&mut self, gpu: &Gpu, settings: Self::Settings);
}

/// The MatCap's settings as the GPU reads them: WGSL's `MatCapParams`, 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
struct GpuMatCapParams {
    /// The style's layer in the image array.
    layer: u32,
    /// The images' side, texels.
    size: f32,
    /// The images' mip levels.
    levels: f32,
    /// Zero.
    reserved: u32,
}

/// The MatCap's WGSL: its bindings and `material`.
///
/// The Toksvig mip choice assumes the image's scale at its centre (a turn of `d` radians is `d /
/// 2` of the width), so near silhouettes, where the image's scale differs, the blur is not the
/// lobe's: see the comment on `material`.
const MATCAP_WGSL: &str = r"
// The MatCap images, one layer a style (render::material::MatCapStyle), linear, mipmapped.
@group(2) @binding(0) var matcaps: texture_2d_array<f32>;
// Trilinear, clamped.
@group(2) @binding(1) var matcap_sampler: sampler;
// Which style, and the images' size.
@group(2) @binding(2) var<uniform> matcap: MatCapParams;

// The MatCap's settings (render::material::GpuMatCapParams).
struct MatCapParams {
    // The style's layer.
    layer: u32,
    // The images' side, texels.
    size: f32,
    // Their mip levels.
    levels: f32,
    // Zero.
    reserved: u32,
}

// The MatCap: the style's lit sphere where it faces the way the normal does, in the view's frame
// turned towards this point's line of sight; blurred by the roughness term (Toksvig).
//
// A mean normal of length r over a spread of normals is that of a von Mises-Fisher lobe of
// concentration k = r (3 - r^2) / (1 - r^2), whose angular spread is about 1 / sqrt(k) radians.
// The image maps a normal's turn of d radians near its centre to d / 2 of its width, so the lobe
// covers spread * size / 2 texels: the mip level whose texels are that wide. r = 1 (a finest
// tile, or a flat one) is level 0. The scale is the image's centre's: a turn of d radians at
// angle a from the view covers d cos(a) / 2 of the width, less towards the rim, so there the
// chosen level is wider than the lobe and silhouettes over-blur.
fn material(surface: Surface, eye: Eye) -> vec3<f32> {
    let towards = eye.to_eye;
    var across = cross(eye.up, towards);
    if dot(across, across) < 1e-12 {
        across = eye.right;
    }
    let right = normalize(across);
    let up = cross(towards, right);
    let n = vec2<f32>(dot(surface.normal, right), dot(surface.normal, up));
    let uv = vec2<f32>(0.5 + 0.5 * n.x, 0.5 - 0.5 * n.y);
    let r = clamp(surface.roughness, 1e-3, 1.0);
    let k = r * (3.0 - r * r) / max(1.0 - r * r, 1e-6);
    let texels = inverseSqrt(k) * 0.5 * matcap.size;
    let lod = clamp(log2(max(texels, 1.0)), 0.0, matcap.levels - 1.0);
    return textureSampleLevel(matcaps, matcap_sampler, uv, matcap.layer, lod).rgb;
}
";

/// The MatCap material: every [`MatCapStyle`]'s image, generated at startup, and which one is
/// shown.
#[derive(Debug)]
pub struct MatCap {
    /// The images, one layer a style.
    images: Texture,
    /// Its settings.
    params: Buffer,
    /// The style shown.
    style: MatCapStyle,
    /// The bind group's layout.
    layout: BindGroupLayout,
    /// The bind group.
    group: BindGroup,
}

impl MatCap {
    /// The MatCap's images on `gpu` (every style, generated: a few milliseconds), showing
    /// `style`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the image, its sampler or its bind group is refused.
    pub fn new(gpu: &Gpu, style: MatCapStyle) -> Result<Self, GpuError> {
        let device = gpu.device();
        let images = images(gpu)?;
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("matcap"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..SamplerDescriptor::default()
        });
        let params = uniform_buffer(gpu, "matcap params", size_of::<GpuMatCapParams>() as u64)?;
        let layout = gpu.scoped("matcap layout", || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("matcap"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: SHADING_STAGES,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: SHADING_STAGES,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: SHADING_STAGES,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
        })?;
        let group = gpu.scoped("matcap bind group", || {
            let view = images.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..TextureViewDescriptor::default()
            });
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("matcap"),
                layout: &layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&sampler),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            })
        })?;
        let mut material = Self {
            images,
            params,
            style,
            layout,
            group,
        };
        material.set_style(gpu, style);
        Ok(material)
    }

    /// Shows `style` from the next shading pass on: a uniform write, no new pipeline.
    pub fn set_style(&mut self, gpu: &Gpu, style: MatCapStyle) {
        self.style = style;
        let params = GpuMatCapParams {
            layer: style.layer(),
            size: f32::from(u16::try_from(matcap::SIDE).unwrap_or(u16::MAX)),
            levels: f32::from(u16::try_from(matcap::LEVELS).unwrap_or(u16::MAX)),
            reserved: 0,
        };
        gpu.queue()
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    /// The style shown.
    #[must_use]
    pub const fn style(&self) -> MatCapStyle {
        self.style
    }

    /// The images, one layer a style: for tests and tools.
    #[must_use]
    pub const fn images(&self) -> &Texture {
        &self.images
    }
}

impl Material for MatCap {
    /// The style shown.
    type Settings = MatCapStyle;

    /// None: the images are a texture.
    const STORAGE_BUFFERS: u32 = 0;
    /// Its settings' uniform: the style's layer and the images' size.
    const UNIFORM_BUFFERS: u32 = 1;

    fn source(&self) -> &str {
        MATCAP_WGSL
    }

    fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    fn bind_group(&self) -> &BindGroup {
        &self.group
    }

    fn settings(&self) -> MatCapStyle {
        self.style
    }

    fn set_settings(&mut self, gpu: &Gpu, settings: MatCapStyle) {
        if settings != self.style {
            self.set_style(gpu, settings);
        }
    }
}

/// Every [`MatCapStyle`]'s image, generated, with its mip levels, one layer a style, on `gpu`.
///
/// # Errors
///
/// As [`Gpu::scoped`], if the texture is refused.
fn images(gpu: &Gpu) -> Result<Texture, GpuError> {
    let layers = u32::try_from(MatCapStyle::ALL.len()).unwrap_or(u32::MAX);
    let images = gpu.scoped("matcap images", || {
        gpu.device().create_texture(&TextureDescriptor {
            label: Some("matcap images"),
            size: Extent3d {
                width: matcap::SIDE,
                height: matcap::SIDE,
                depth_or_array_layers: layers,
            },
            mip_level_count: matcap::LEVELS,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        })
    })?;
    let styles = (0..).zip(MatCapStyle::ALL);
    for (layer, each) in styles {
        let levels = (0..).zip(matcap::mips(each));
        for (level, image) in levels {
            let side = image.side();
            gpu.queue().write_texture(
                TexelCopyTextureInfo {
                    texture: &images,
                    mip_level: level,
                    origin: Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: TextureAspect::All,
                },
                bytemuck::cast_slice(&image.half_texels()),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 8),
                    rows_per_image: Some(side),
                },
                Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    Ok(images)
}

/// The stages a shading pass runs in: compute on desktop, fragment on tile-based GPUs.
pub(crate) const SHADING_STAGES: ShaderStages = ShaderStages::COMPUTE.union(ShaderStages::FRAGMENT);
