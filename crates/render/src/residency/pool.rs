//! Slot pools: per size class, a few storage buffers cut into equal slots, created as the
//! budget allows and kept for the residency's life.
//!
//! A buffer is as large as the device binds ([`Pools::new`]'s `buffer_bytes`), so a class is a
//! few buffers on desktop. A class may hold twice its project's tiles: the second copy is
//! headroom for shadow slots, so an epoch over every tile of a class (a resync of the floor)
//! still has somewhere to go. The floor's classes are tiny, and get both copies up front, so
//! the finer classes never take their budget.

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
    /// Slots in a full buffer: as many as the buffer size allows (at least 1).
    slots_per_buffer: u64,
    /// Its buffers in creation order: a slot's buffer is its index here.
    buffers: Vec<Buffer>,
    /// The slots of each buffer, in the same order.
    counts: Vec<u32>,
    /// Slots handed out and given back, to hand out again first.
    free: Vec<Slot>,
    /// The first slot of the last buffer never handed out.
    fresh: u32,
    /// How many tiles of this class the project has.
    tiles: u64,
}

impl ClassPool {
    /// Slots in every buffer made so far.
    fn slots(&self) -> u64 {
        self.counts.iter().map(|&slots| u64::from(slots)).sum()
    }

    /// The most slots the class may have: its tiles twice, the second time as headroom for
    /// shadow slots.
    const fn most_slots(&self) -> u64 {
        self.tiles.saturating_mul(2)
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
            .map(|class| ClassPool {
                class,
                slots_per_buffer: (buffer_bytes / class.slot_bytes()).clamp(1, MAX_SLOTS),
                buffers: Vec::new(),
                counts: Vec::new(),
                free: Vec::new(),
                fresh: 0,
                tiles: class_tiles(class, faces, finest),
            })
            .collect();
        Self {
            classes,
            budget,
            allocated: 0,
        }
    }

    /// Makes `class`'s buffers for all its slots now (its tiles and their headroom), as far as
    /// the budget allows: for the floor's classes, so no finer class takes their budget first.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`], creating a buffer.
    pub(crate) fn reserve(&mut self, gpu: &Gpu, class: SizeClass) -> Result<(), GpuError> {
        while self.grow(gpu, class)? {}
        Ok(())
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
        if let Some(slot) = self.take(class) {
            return Ok(Some(slot));
        }
        if !self.grow(gpu, class)? {
            return Ok(None);
        }
        Ok(self.take(class))
    }

    /// A free slot of `class` among the buffers made, if there is one.
    fn take(&mut self, class: SizeClass) -> Option<Slot> {
        let pool = self.classes.get_mut(usize::from(class.get()))?;
        if let Some(slot) = pool.free.pop() {
            return Some(slot);
        }
        let last = pool.counts.len().checked_sub(1)?;
        if pool.fresh >= *pool.counts.get(last)? {
            return None;
        }
        let slot = Slot {
            class,
            buffer: u16::try_from(last).ok()?,
            index: pool.fresh,
        };
        pool.fresh += 1;
        Some(slot)
    }

    /// Makes a new buffer for `class`: a full one, or what the budget and the class's most
    /// slots leave; whether one was made.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn grow(&mut self, gpu: &Gpu, class: SizeClass) -> Result<bool, GpuError> {
        let room = self.budget.saturating_sub(self.allocated) / class.slot_bytes();
        let Some(pool) = self.classes.get_mut(usize::from(class.get())) else {
            return Ok(false);
        };
        let left = pool.most_slots().saturating_sub(pool.slots());
        let slots = pool.slots_per_buffer.min(left).min(room);
        let Ok(slots) = u32::try_from(slots) else {
            return Ok(false);
        };
        if slots == 0 || pool.buffers.len() >= MAX_BUFFERS {
            return Ok(false);
        }
        let size = u64::from(slots) * class.slot_bytes();
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
        // The last buffer's slots never handed out stay to be had, from the free list.
        if let Some(last) = pool.counts.len().checked_sub(1)
            && let Ok(index) = u16::try_from(last)
            && let Some(&count) = pool.counts.get(last)
        {
            let unused = (pool.fresh..count).map(|i| Slot {
                class,
                buffer: index,
                index: i,
            });
            pool.free.extend(unused);
        }
        pool.buffers.push(buffer);
        pool.counts.push(slots);
        pool.fresh = 0;
        Ok(true)
    }

    /// The most tiles `class` could ever hold at once: the slots made, and those the budget
    /// still allows (within the class's most).
    pub(crate) fn capacity(&self, class: SizeClass) -> u64 {
        let room = self.budget.saturating_sub(self.allocated) / class.slot_bytes();
        self.classes
            .get(usize::from(class.get()))
            .map_or(0, |pool| {
                let made = pool.slots();
                made + room.min(pool.most_slots().saturating_sub(made))
            })
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

    /// Slots made, and how many of them are free (given back, or never handed out): for the
    /// tests' accounting of every slot.
    #[cfg(test)]
    pub(crate) fn census(&self) -> (u64, u64) {
        let made = self.classes.iter().map(ClassPool::slots).sum();
        let free = self.classes.iter().map(|pool| {
            let unused = pool
                .counts
                .last()
                .map_or(0, |&count| count.saturating_sub(pool.fresh));
            pool.free.len() as u64 + u64::from(unused)
        });
        (made, free.sum())
    }

    /// Bytes the buffers take.
    pub(crate) const fn allocated(&self) -> u64 {
        self.allocated
    }

    /// The most tiles that can be resident at once: per class, its tiles, or the slots the
    /// whole budget would buy, whichever is fewer.
    pub(crate) fn max_resident(&self) -> u64 {
        let per_class = self
            .classes
            .iter()
            .map(|pool| pool.tiles.min(self.budget / pool.class.slot_bytes()));
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
