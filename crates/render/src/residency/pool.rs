//! Slot pools: per size class, a set of storage buffers cut into equal slots, created as the
//! budget allows and kept for the residency's life, so a slot stays where it is.

use geom::Level;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use super::{CLASSES, SizeClass, Slot};
use crate::gpu::{Gpu, GpuError};

/// The most buffers a class may have: [`Slot::packed`] holds the buffer in 8 bits.
const MAX_BUFFERS: usize = 1 << 8;

/// The most slots a buffer may have: [`Slot::packed`] holds the index in 21 bits.
const MAX_SLOTS: u64 = 1 << 21;

/// One size class's slots.
#[derive(Debug)]
struct ClassPool {
    /// The class.
    class: SizeClass,
    /// Slots per buffer: as many as the buffer size allows, but no more than the project has
    /// tiles of this class (at least 1).
    slots_per_buffer: u32,
    /// Its buffers, in creation order: a slot's buffer is its index here.
    buffers: Vec<Buffer>,
    /// Slots handed out and given back, to hand out again first.
    free: Vec<Slot>,
    /// The first slot of the last buffer never handed out.
    fresh: u32,
    /// How many tiles of this class the project has: no buffer is made once there are slots
    /// for all of them.
    tiles: u64,
}

impl ClassPool {
    /// Bytes per buffer.
    fn buffer_size(&self) -> u64 {
        u64::from(self.slots_per_buffer) * self.class.slot_bytes()
    }

    /// Slots in every buffer made so far.
    fn slots(&self) -> u64 {
        self.buffers.len() as u64 * u64::from(self.slots_per_buffer)
    }
}

/// The slot pools of every class, and the budget they share.
#[derive(Debug)]
pub(crate) struct Pools {
    /// Per class, its pool.
    classes: Vec<ClassPool>,
    /// The most bytes the buffers may take together.
    budget: u64,
    /// Bytes the buffers take.
    allocated: u64,
}

impl Pools {
    /// Empty pools for a project of `faces` faces whose finest level is `finest`: buffers of
    /// up to `buffer_bytes` each, at most `budget` bytes of them.
    pub(crate) fn new(faces: u32, finest: Level, buffer_bytes: u64, budget: u64) -> Self {
        let classes = (0..CLASSES)
            .filter_map(|c| u8::try_from(c).ok().and_then(SizeClass::new))
            .map(|class| {
                let tiles = class_tiles(class, faces, finest);
                let by_size = (buffer_bytes / class.slot_bytes()).clamp(1, MAX_SLOTS);
                let slots_per_buffer = u32::try_from(by_size.min(tiles.max(1))).unwrap_or(1);
                ClassPool {
                    class,
                    slots_per_buffer,
                    buffers: Vec::new(),
                    free: Vec::new(),
                    fresh: slots_per_buffer,
                    tiles,
                }
            })
            .collect();
        Self {
            classes,
            budget,
            allocated: 0,
        }
    }

    /// A free slot of `class`: one given back, else the next of its newest buffer, else the
    /// first of a new buffer if the budget allows one; `None` if not.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], creating a buffer.
    pub(crate) fn acquire(
        &mut self,
        gpu: &Gpu,
        class: SizeClass,
    ) -> Result<Option<Slot>, GpuError> {
        let Some(pool) = self.classes.get_mut(usize::from(class.get())) else {
            return Ok(None);
        };
        if let Some(slot) = pool.free.pop() {
            return Ok(Some(slot));
        }
        if pool.fresh < pool.slots_per_buffer {
            let slot = Slot {
                class,
                buffer: u16::try_from(pool.buffers.len() - 1).unwrap_or(u16::MAX),
                index: pool.fresh,
            };
            pool.fresh += 1;
            return Ok(Some(slot));
        }
        let size = pool.buffer_size();
        if self.allocated + size > self.budget
            || pool.buffers.len() >= MAX_BUFFERS
            || pool.slots() >= pool.tiles
        {
            return Ok(None);
        }
        let label = format!("tile slots, side {}", class.side());
        let buffer = gpu.scoped(&label, || {
            gpu.device().create_buffer(&BufferDescriptor {
                label: Some(&label),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })?;
        self.allocated += size;
        pool.buffers.push(buffer);
        pool.fresh = 1;
        Ok(Some(Slot {
            class,
            buffer: u16::try_from(pool.buffers.len() - 1).unwrap_or(u16::MAX),
            index: 0,
        }))
    }

    /// Gives `slot` back, to be handed out again.
    pub(crate) fn release(&mut self, slot: Slot) {
        if let Some(pool) = self.classes.get_mut(usize::from(slot.class.get())) {
            pool.free.push(slot);
        }
    }

    /// `slot`'s buffer and the byte offset of its first vertex there.
    pub(crate) fn locate(&self, slot: Slot) -> Option<(&Buffer, u64)> {
        let pool = self.classes.get(usize::from(slot.class.get()))?;
        let buffer = pool.buffers.get(usize::from(slot.buffer))?;
        Some((buffer, u64::from(slot.index) * slot.class.slot_bytes()))
    }

    /// `class`'s buffers, in slot order: a slot's buffer is its index here.
    pub(crate) fn buffers(&self, class: SizeClass) -> &[Buffer] {
        self.classes
            .get(usize::from(class.get()))
            .map_or(&[], |pool| pool.buffers.as_slice())
    }

    /// Bytes the buffers take.
    pub(crate) const fn allocated(&self) -> u64 {
        self.allocated
    }

    /// The most slots the pools could ever hand out together, within the budget: a bound on
    /// the tiles resident at once.
    pub(crate) fn max_slots(&self) -> u64 {
        let per_class = self.classes.iter().map(|pool| {
            let buffers = (self.budget / pool.buffer_size()).min(MAX_BUFFERS as u64);
            (buffers * u64::from(pool.slots_per_buffer)).min(pool.tiles)
        });
        per_class.sum()
    }
}

/// How many tiles of `class` a project of `faces` faces, finest level `finest`, has: at every
/// level of the class up to the finest.
fn class_tiles(class: SizeClass, faces: u32, finest: Level) -> u64 {
    let levels = (0..=finest.get())
        .filter_map(|l| Level::new(l).ok())
        .filter(|&level| SizeClass::of(level) == class);
    let per_face: u64 = levels
        .map(|level| u64::from(level.tiles_per_side()).pow(2))
        .sum();
    per_face * u64::from(faces)
}
