//! Where frames go: a window's swapchain ([`WindowSurface`]), or an offscreen texture
//! ([`Offscreen`]) for headless runs and tests; [`Target`] is either, so the code that draws a
//! frame doesn't care which.
//!
//! A window presents in mailbox mode where the surface offers it (the newest frame replaces a
//! queued one, so a frame is never held back behind an older one), else in FIFO mode, which
//! every surface offers ([`Presentation::for_modes`]). Which one is in use is reported
//! ([`Target::presentation`]), so the app can show it.
//!
//! The adapter is chosen for the window: [`WindowSurface::open`] creates the surface first and
//! opens a [`Gpu`] on an adapter that can present to it, from the same `wgpu` instance.

use std::fmt;
use std::num::NonZeroU32;
use std::sync::Arc;

use thiserror::Error;
use wgpu::rwh::HasDisplayHandle;
use wgpu::{
    Adapter, BufferDescriptor, BufferUsages, COPY_BYTES_PER_ROW_ALIGNMENT,
    CommandEncoderDescriptor, CompositeAlphaMode, CreateSurfaceError, CurrentSurfaceTexture,
    DownlevelFlags, Extent3d, Instance, PresentMode, SurfaceColorSpace, SurfaceConfiguration,
    SurfaceTarget, SurfaceTexture, TexelCopyBufferInfo, TexelCopyBufferLayout, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, WindowHandle,
};

use crate::gpu::{self, AdapterRequest, Gpu, GpuError};

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests;

/// How a [`Target`]'s frames reach the screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Presentation {
    /// A window's swapchain in mailbox mode: a frame waiting to be shown is replaced by a newer
    /// one, so the newest is shown at the next refresh, and drawing never waits for the display.
    Mailbox,
    /// A window's swapchain in FIFO mode: frames are shown in order, one per refresh, and
    /// drawing waits when the queue is full. Every surface offers it.
    Fifo,
    /// No window: frames are drawn into a texture and never shown.
    Offscreen,
}

impl Presentation {
    /// The mode a window presents in, given the modes its surface offers: mailbox if offered,
    /// else FIFO (which every surface must offer, so it is assumed even if `offered` leaves it
    /// out).
    #[must_use]
    pub fn for_modes(offered: &[PresentMode]) -> Self {
        if offered.contains(&PresentMode::Mailbox) {
            Self::Mailbox
        } else {
            Self::Fifo
        }
    }

    /// The `wgpu` present mode a window surface is configured with; `None` offscreen.
    #[must_use]
    pub const fn present_mode(self) -> Option<PresentMode> {
        match self {
            Self::Mailbox => Some(PresentMode::Mailbox),
            Self::Fifo => Some(PresentMode::Fifo),
            Self::Offscreen => None,
        }
    }
}

impl fmt::Display for Presentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Mailbox => "mailbox",
            Self::Fifo => "fifo",
            Self::Offscreen => "offscreen",
        })
    }
}

/// A frame's size in physical pixels, never zero in either direction (a minimized window has
/// no frames).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Size {
    /// Pixels across.
    width: NonZeroU32,
    /// Pixels down.
    height: NonZeroU32,
}

impl Size {
    /// `width × height` pixels; `None` if either is zero.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Option<Self> {
        match (NonZeroU32::new(width), NonZeroU32::new(height)) {
            (Some(width), Some(height)) => Some(Self { width, height }),
            _ => None,
        }
    }

    /// Pixels across.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width.get()
    }

    /// Pixels down.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height.get()
    }

    /// The size as a single-layer texture extent.
    const fn extent(self) -> Extent3d {
        Extent3d {
            width: self.width(),
            height: self.height(),
            depth_or_array_layers: 1,
        }
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {}", self.width, self.height)
    }
}

/// Why a window surface couldn't be made, configured or drawn to.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SurfaceError {
    /// The window's handles couldn't make a surface.
    #[error("creating the window surface failed")]
    Create(#[from] CreateSurfaceError),
    /// The chosen adapter offers no texture format for the surface: it can't present to it
    /// after all.
    #[error("the adapter offers no format to present to the window with")]
    NoFormat,
    /// Opening the GPU, or configuring the surface on it, failed.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Acquiring a frame raised a validation error (reported by the device, which
    /// [`Gpu::check`] then returns).
    #[error("acquiring the window's next frame failed validation")]
    Acquire,
}

/// Makes the surface target for the window again, when the surface is lost and must be
/// recreated.
type Remake = Box<dyn Fn() -> SurfaceTarget<'static> + Send + Sync>;

/// A window's swapchain: the surface, how it is configured, and how to make it again.
pub struct WindowSurface {
    /// The instance the surface and the adapter came from; a lost surface is recreated on it.
    instance: Instance,
    /// The adapter the [`Gpu`] was opened on: the surface's capabilities are its.
    adapter: Adapter,
    /// The surface, holding the window it presents to.
    surface: wgpu::Surface<'static>,
    /// The window again, for a surface that is lost.
    remake: Remake,
    /// The configuration last applied: format, size, present mode.
    config: SurfaceConfiguration,
    /// The size it was configured at, as `config` holds it.
    size: Size,
    /// The present mode chosen, as reported.
    presentation: Presentation,
    /// The format of a view of its images that isn't sRGB, if it has one.
    display_format: Option<TextureFormat>,
    /// Whether the last image acquired was suboptimal: drawn and presented as it was, the
    /// surface is configured again before the next is acquired.
    suboptimal: bool,
}

impl fmt::Debug for WindowSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowSurface")
            .field("adapter", &self.adapter.get_info().name)
            .field("config", &self.config)
            .field("presentation", &self.presentation)
            .finish_non_exhaustive()
    }
}

/// What acquiring the next frame gave.
#[derive(Debug)]
pub enum Acquired<'target> {
    /// A frame to draw into, then present.
    Frame(Frame<'target>),
    /// No frame this time (the surface changed size or was lost and has been set up again, or
    /// the wait for a frame timed out): try again with the next frame.
    Retry,
    /// The window can't be seen (minimized, or covered on a compositor that says so): no frame
    /// until the window reports it is visible again.
    Hidden,
}

impl WindowSurface {
    /// Makes a surface for `window`, opens a [`Gpu`] on the adapter `request` names among those
    /// that can present to it, and configures the surface at `size`.
    ///
    /// `window` is cloned whenever the surface has to be recreated (an `Arc` of the window, as
    /// winit gives).
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Create`] if the window's handles make no surface;
    /// [`SurfaceError::Gpu`] with [`GpuError::NoAdapter`] if no matching adapter can present
    /// to it, or another [`GpuError`] if the device can't be created or the configuration is
    /// refused; [`SurfaceError::NoFormat`] if the adapter offers no format for it.
    pub fn open<W>(
        window: W,
        size: Size,
        request: &AdapterRequest,
    ) -> Result<(Arc<Gpu>, Self), SurfaceError>
    where
        W: WindowHandle + HasDisplayHandle + Clone + 'static,
    {
        let instance = gpu::instance();
        let remake: Remake = Box::new(move || SurfaceTarget::from(window.clone()));
        let surface = instance.create_surface(remake())?;
        let adapter = pollster::block_on(gpu::choose(&instance, request, Some(&surface)))?;
        let gpu = Arc::new(Gpu::on_adapter(&adapter)?);
        let caps = surface.get_capabilities(&adapter);
        let format = preferred_format(&caps.formats).ok_or(SurfaceError::NoFormat)?;
        let presentation = Presentation::for_modes(&caps.present_modes);
        let alpha_mode = if caps.alpha_modes.contains(&CompositeAlphaMode::Opaque) {
            CompositeAlphaMode::Opaque
        } else {
            CompositeAlphaMode::Auto
        };
        let reinterprets = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(DownlevelFlags::SURFACE_VIEW_FORMATS);
        let display_format = display_format(format, reinterprets);
        let view_formats = display_format.filter(|&display| display != format);
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: SurfaceColorSpace::Auto,
            width: size.width(),
            height: size.height(),
            present_mode: presentation.present_mode().unwrap_or(PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: view_formats.into_iter().collect(),
        };
        let this = Self {
            instance,
            adapter,
            surface,
            remake,
            config,
            size,
            presentation,
            display_format,
            suboptimal: false,
        };
        this.configure(&gpu)?;
        Ok((gpu, this))
    }

    /// The present mode in use.
    #[must_use]
    pub const fn presentation(&self) -> Presentation {
        self.presentation
    }

    /// The format frames are drawn in.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        self.config.format
    }

    /// The size the swapchain is configured at.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// Configures the swapchain at `size`, if it isn't already.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the configuration is refused.
    pub fn resize(&mut self, gpu: &Gpu, size: Size) -> Result<(), SurfaceError> {
        if size == self.size {
            return Ok(());
        }
        self.size = size;
        self.config.width = size.width();
        self.config.height = size.height();
        self.configure(gpu)
    }

    /// The next frame to draw into, waiting for one as the present mode does (FIFO waits for
    /// a free image).
    ///
    /// A surface that has changed is configured again, and a lost one recreated; both give
    /// [`Acquired::Retry`]. A suboptimal image (usable, but no longer matching the surface) is
    /// given as a frame, to draw and present, and the surface is configured again before the
    /// next: dropping it instead could leave the window with nothing shown.
    ///
    /// # Errors
    ///
    /// [`SurfaceError::Acquire`] if acquiring fails validation; [`SurfaceError::Create`] or a
    /// [`GpuError`] if a lost surface can't be recreated and configured.
    pub fn acquire(&mut self, gpu: &Gpu) -> Result<Acquired<'_>, SurfaceError> {
        if std::mem::take(&mut self.suboptimal) {
            self.configure(gpu)?;
        }
        match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(texture) => Ok(Acquired::Frame(Frame::swapchain(
                texture,
                self.display_format.is_some(),
            ))),
            CurrentSurfaceTexture::Suboptimal(texture) => {
                // Configuring while the image is held is refused: it is done before the next.
                self.suboptimal = true;
                Ok(Acquired::Frame(Frame::swapchain(
                    texture,
                    self.display_format.is_some(),
                )))
            }
            CurrentSurfaceTexture::Outdated => {
                self.configure(gpu)?;
                Ok(Acquired::Retry)
            }
            CurrentSurfaceTexture::Lost => {
                self.surface = self.instance.create_surface((self.remake)())?;
                self.configure(gpu)?;
                Ok(Acquired::Retry)
            }
            CurrentSurfaceTexture::Timeout => Ok(Acquired::Retry),
            CurrentSurfaceTexture::Occluded => Ok(Acquired::Hidden),
            CurrentSurfaceTexture::Validation => Err(SurfaceError::Acquire),
        }
    }

    /// Applies the configuration to the surface.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the configuration is refused.
    fn configure(&self, gpu: &Gpu) -> Result<(), SurfaceError> {
        Ok(gpu.scoped("configuring the window surface", || {
            self.surface.configure(gpu.device(), &self.config);
        })?)
    }
}

/// The format a window draws in, of those its surface offers: the first sRGB one (so shading
/// writes linear values and the hardware encodes them), else the first offered; `None` if none
/// is.
fn preferred_format(offered: &[TextureFormat]) -> Option<TextureFormat> {
    offered
        .iter()
        .find(|format| format.is_srgb())
        .or_else(|| offered.first())
        .copied()
}

/// The format of a view that blends in display values, of a target of `format`: the target's own if
/// it isn't sRGB, otherwise the same without the encoding if the target can be `reinterpreted` so.
fn display_format(format: TextureFormat, reinterpreted: bool) -> Option<TextureFormat> {
    (!format.is_srgb() || reinterpreted).then(|| format.remove_srgb_suffix())
}

/// One frame of a [`Target`], acquired: draw into [`Frame::view`], then [`Frame::present`] it.
/// A window's frame dropped unpresented is discarded.
#[derive(Debug)]
pub struct Frame<'target> {
    /// A window's swapchain image; `None` offscreen.
    image: Option<SurfaceTexture>,
    /// The offscreen texture drawn into; `None` for a window's.
    offscreen: Option<&'target Offscreen>,
    /// A view of the whole image, to draw into.
    view: TextureView,
    /// Whether it can be viewed without sRGB encoding.
    has_display_view: bool,
    /// Ties the frame to the target it came from, which must not be configured or resized
    /// while the frame is held.
    target: std::marker::PhantomData<&'target mut Target>,
}

impl<'target> Frame<'target> {
    /// A frame of an acquired swapchain image, with a view of it.
    fn swapchain(image: SurfaceTexture, has_display_view: bool) -> Self {
        let view = image.texture.create_view(&TextureViewDescriptor::default());
        Self {
            image: Some(image),
            offscreen: None,
            view,
            has_display_view,
            target: std::marker::PhantomData,
        }
    }

    /// A frame of an offscreen texture.
    fn offscreen(offscreen: &'target Offscreen) -> Self {
        Self {
            image: None,
            offscreen: Some(offscreen),
            view: offscreen.view.clone(),
            has_display_view: offscreen.display_format.is_some(),
            target: std::marker::PhantomData,
        }
    }

    /// The image to draw into.
    #[must_use]
    pub const fn view(&self) -> &TextureView {
        &self.view
    }

    /// A view of the image that blends in display values rather than in linear light; `None` if
    /// the target can't offer one. Made on asking, so a frame with no UI makes none.
    #[must_use]
    pub fn display_view(&self) -> Option<TextureView> {
        if !self.has_display_view {
            return None;
        }
        let window = self.image.as_ref().map(|image| &image.texture);
        let texture = window.or_else(|| self.offscreen.map(|offscreen| &offscreen.texture))?;
        Some(texture.create_view(&TextureViewDescriptor {
            format: Some(texture.format().remove_srgb_suffix()),
            ..TextureViewDescriptor::default()
        }))
    }

    /// Shows the frame: queues a window's image for the display on `gpu`'s queue, once the
    /// work submitted for it is done (submit it first); nothing to do offscreen.
    pub fn present(self, gpu: &Gpu) {
        if let Some(image) = self.image {
            gpu.queue().present(image);
        }
    }
}

/// A frame-sized texture that frames are drawn into instead of a window: for runs with no
/// display, and for tests that read the pixels back.
#[derive(Debug)]
pub struct Offscreen {
    /// The texture drawn into.
    texture: Texture,
    /// A view of the whole texture.
    view: TextureView,
    /// The format of a view of it that isn't sRGB, if it can have one.
    display_format: Option<TextureFormat>,
    /// Its size.
    size: Size,
}

impl Offscreen {
    /// The format offscreen frames are drawn in: 8-bit sRGB RGBA, as a window's usually is.
    pub const FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

    /// A texture of `size` to draw frames into, which can be read back.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the texture can't be created (larger than the device allows).
    pub fn new(gpu: &Gpu, size: Size) -> Result<Self, GpuError> {
        Self::with_display_format(gpu, size, display_format(Self::FORMAT, true))
    }

    /// As [`Offscreen::new`], with no view that isn't sRGB, as some surfaces have none.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the texture can't be created.
    #[cfg(test)]
    pub(crate) fn without_display_view(gpu: &Gpu, size: Size) -> Result<Self, GpuError> {
        Self::with_display_format(gpu, size, None)
    }

    /// A texture of `size` that can be viewed as `display_format`.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], if the texture can't be created.
    fn with_display_format(
        gpu: &Gpu,
        size: Size,
        display_format: Option<TextureFormat>,
    ) -> Result<Self, GpuError> {
        let view_formats: Vec<TextureFormat> = display_format.into_iter().collect();
        let texture = gpu.scoped("offscreen target", || {
            gpu.device().create_texture(&TextureDescriptor {
                label: Some("offscreen target"),
                size: size.extent(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: Self::FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                view_formats: &view_formats,
            })
        })?;
        let view = texture.create_view(&TextureViewDescriptor::default());
        Ok(Self {
            texture,
            view,
            display_format,
            size,
        })
    }

    /// The texture's size.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }

    /// The texture to draw into.
    #[must_use]
    pub const fn view(&self) -> &TextureView {
        &self.view
    }

    /// The texture's pixels as stored (sRGB-encoded RGBA), row by row from the top, waiting for
    /// everything submitted so far.
    ///
    /// It blocks, so it is for tests and tools.
    ///
    /// # Errors
    ///
    /// As [`Gpu::read_buffer`].
    pub fn read_pixels(&self, gpu: &Gpu) -> Result<Vec<[u8; 4]>, GpuError> {
        let width = self.size.width();
        let height = self.size.height();
        let row_bytes = width * 4;
        let padded = row_bytes.next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.scoped("offscreen readback", || {
            let buffer = gpu.device().create_buffer(&BufferDescriptor {
                label: Some("offscreen readback"),
                size: u64::from(padded) * u64::from(height),
                usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = gpu
                .device()
                .create_command_encoder(&CommandEncoderDescriptor::default());
            encoder.copy_texture_to_buffer(
                self.texture.as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(height),
                    },
                },
                self.size.extent(),
            );
            gpu.queue().submit([encoder.finish()]);
            buffer
        })?;
        let texels: Vec<[u8; 4]> = gpu.read_buffer(&buffer)?;
        let padded_texels = (padded / 4) as usize;
        let rows = texels.chunks_exact(padded_texels);
        let pixels: Vec<[u8; 4]> = rows
            .flat_map(|row| row.iter().take(width as usize).copied())
            .collect();
        Ok(pixels)
    }
}

/// Where a frame is drawn: a window or an offscreen texture.
#[derive(Debug)]
pub enum Target {
    /// A window's swapchain.
    Window(WindowSurface),
    /// An offscreen texture.
    Offscreen(Offscreen),
}

impl Target {
    /// How frames reach the screen.
    #[must_use]
    pub const fn presentation(&self) -> Presentation {
        match self {
            Self::Window(surface) => surface.presentation(),
            Self::Offscreen(_) => Presentation::Offscreen,
        }
    }

    /// The format frames are drawn in.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        match self {
            Self::Window(surface) => surface.format(),
            Self::Offscreen(_) => Offscreen::FORMAT,
        }
    }

    /// The size frames are drawn at.
    #[must_use]
    pub fn size(&self) -> Size {
        match self {
            Self::Window(surface) => surface.size(),
            Self::Offscreen(offscreen) => offscreen.size(),
        }
    }

    /// Sets the size frames are drawn at: a window's swapchain is configured again; an
    /// offscreen texture is replaced by one of the new size.
    ///
    /// # Errors
    ///
    /// As [`WindowSurface::resize`] and [`Offscreen::new`].
    pub fn resize(&mut self, gpu: &Gpu, size: Size) -> Result<(), SurfaceError> {
        match self {
            Self::Window(surface) => surface.resize(gpu, size),
            Self::Offscreen(offscreen) => {
                if offscreen.size() != size {
                    *offscreen = Offscreen::new(gpu, size)?;
                }
                Ok(())
            }
        }
    }

    /// Whether its frames have a view that blends in display values ([`Frame::display_view`]).
    #[must_use]
    pub const fn has_display_view(&self) -> bool {
        match self {
            Self::Window(surface) => surface.display_format.is_some(),
            Self::Offscreen(offscreen) => offscreen.display_format.is_some(),
        }
    }

    /// The next frame to draw into: a window's as [`WindowSurface::acquire`] gives it, the
    /// offscreen texture every time.
    ///
    /// # Errors
    ///
    /// As [`WindowSurface::acquire`].
    pub fn acquire(&mut self, gpu: &Gpu) -> Result<Acquired<'_>, SurfaceError> {
        match self {
            Self::Window(surface) => surface.acquire(gpu),
            Self::Offscreen(offscreen) => Ok(Acquired::Frame(Frame::offscreen(offscreen))),
        }
    }
}
