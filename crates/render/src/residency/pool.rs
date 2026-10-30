//! Slot pools: per size class, a few storage buffers cut into equal slots, created as the
//! budget allows and kept for the residency's life.
//!
//! **Buffer sizes.** A class's first buffer is small ([`FIRST_BUFFER_BYTES`], or
//! [`FIRST_BUFFER_SLOTS`] slots if that is more), and each next one doubles, up to as large as the
//! device binds ([`Pools::new`]'s `buffer_bytes`): so the first finer tile doesn't stall the
//! owner thread creating a 2 GiB buffer (34.6 ms on the 4090, measured in the app), and a class
//! still reaches the budget in a few buffers (seven on the 4090 and the M4 at the desktop
//! budget). Creating a buffer costs about its size (the device zeroes it), so doubling spreads
//! the cost over the frames the class grows in, each step as long as all the steps before it.
//!
//! A class may hold twice its project's tiles: the second copy is headroom for shadow slots, so
//! an epoch over every tile of a class (a resync of the floor) still has somewhere to go. The
//! floor's classes are tiny, and get both copies up front, so the finer classes never take
//! their budget: that is twice the floor's slots reserved for the session (219 MiB on the
//! benchmark project, for a resync of the whole floor), which the plan revisits.

use geom::Level;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use super::{CLASSES, SizeClass, Slot};
use crate::gpu::{Gpu, GpuError};

/// The most buffers a class may have: [`Slot::packed`] holds the buffer in 8 bits.
const MAX_BUFFERS: usize = 1 << 8;

/// The most slots a buffer may have: [`Slot::packed`] holds the index in 21 bits.
const MAX_SLOTS: u64 = 1 << 21;

/// A class's first buffer's bytes, at least ([`FIRST_BUFFER_SLOTS`] slots if those are more).
///
/// Small enough to create within a frame's uploads (a few milliseconds where the binding-sized
/// buffers took 34.6 ms on the 4090), large enough that a class growing to the desktop budget
/// does it in seven buffers, not dozens past the raster's binding limit.
pub(crate) const FIRST_BUFFER_BYTES: u64 = 64 << 20;

/// A class's first buffer's slots, at least: a few hundred tiles, so a class of large slots
/// still gets a useful first buffer whatever [`FIRST_BUFFER_BYTES`] buys.
pub(crate) const FIRST_BUFFER_SLOTS: u64 = 256;

/// One size class's slots.
#[derive(Debug)]
struct ClassPool {
    /// The class.
    class: SizeClass,
    /// Slots in a full buffer: as many as the buffer size allows (at least 1).
    slots_per_buffer: u64,
    /// Slots in its first buffer: the doubling's start, at most a full buffer.
    first_slots: u64,
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

    /// The slots of its buffer `index` (counting from 0) when the budget and its most slots
    /// don't cut it short: the first buffer's, doubling, up to a full buffer.
    fn scheduled(&self, index: usize) -> u64 {
        let doublings = u32::try_from(index).unwrap_or(u32::MAX).min(63);
        self.first_slots
            .saturating_mul(1 << doublings)
            .min(self.slots_per_buffer)
    }

    /// The slots its next buffer gets, with `room` slots left in the budget: as scheduled, or
    /// what the budget and its most slots leave, whichever is fewer. A buffer cut short takes
    /// the last of one or the other, so it is the class's last.
    fn next_slots(&self, room: u64) -> u64 {
        let left = self.most_slots().saturating_sub(self.slots());
        self.scheduled(self.counts.len()).min(left).min(room)
    }

    /// The most buffers it can ever have, its slots limited to `slots` (its most, or what the
    /// whole budget would buy): the scheduled buffers it takes to reach them. Every buffer but
    /// the last is as scheduled, so the class never has more.
    fn most_buffers(&self, slots: u64) -> u64 {
        let mut covered = 0_u64;
        let mut buffers = 0;
        while covered < slots && buffers < MAX_BUFFERS {
            covered = covered.saturating_add(self.scheduled(buffers));
            buffers += 1;
        }
        buffers as u64
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
    /// Bytes of the slots handed out and not given back.
    used: u64,
}

impl Pools {
    /// Empty pools for a project of `faces` faces whose finest level is `finest`: buffers of
    /// up to `buffer_bytes` each, at most `budget` bytes of them.
    pub(crate) fn new(faces: u32, finest: Level, buffer_bytes: u64, budget: u64) -> Self {
        let classes = (0..CLASSES)
            .filter_map(|c| u8::try_from(c).ok().and_then(SizeClass::new))
            .map(|class| {
                let slots_per_buffer = (buffer_bytes / class.slot_bytes()).clamp(1, MAX_SLOTS);
                let first = (FIRST_BUFFER_BYTES / class.slot_bytes()).max(FIRST_BUFFER_SLOTS);
                (class, slots_per_buffer, first.min(slots_per_buffer))
            })
            .map(|(class, slots_per_buffer, first_slots)| ClassPool {
                class,
                slots_per_buffer,
                first_slots,
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
            used: 0,
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
            self.used += class.slot_bytes();
            return Ok(Some(slot));
        }
        if !self.grow(gpu, class)? {
            return Ok(None);
        }
        let slot = self.take(class);
        if slot.is_some() {
            self.used += class.slot_bytes();
        }
        Ok(slot)
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

    /// Makes a new buffer for `class`: as scheduled (the first small, each next doubling, up to
    /// a full one), or what the budget and the class's most slots leave; whether one was made.
    ///
    /// # Errors
    ///
    /// As [`Gpu::scoped`].
    fn grow(&mut self, gpu: &Gpu, class: SizeClass) -> Result<bool, GpuError> {
        let room = self.budget.saturating_sub(self.allocated) / class.slot_bytes();
        let Some(pool) = self.classes.get_mut(usize::from(class.get())) else {
            return Ok(false);
        };
        let Ok(slots) = u32::try_from(pool.next_slots(room)) else {
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
            self.used = self.used.saturating_sub(slot.class.slot_bytes());
        }
    }

    /// `slot`'s buffer and the byte offset of its first vertex there.
    pub(crate) fn locate(&self, slot: Slot) -> Option<(&Buffer, u64)> {
        let pool = self.classes.get(usize::from(slot.class.get()))?;
        let buffer = pool.buffers.get(usize::from(slot.buffer))?;
        Some((buffer, u64::from(slot.index) * slot.class.slot_bytes()))
    }

    /// The most buffers `class` can ever have: those its buffer schedule takes to reach its
    /// most slots (its tiles and their headroom) or those the whole budget would buy, whichever
    /// is fewer. Exact as a bound: every buffer but a class's last is as scheduled, and the last
    /// is cut short only by its most slots or the budget, after which it gets no more.
    pub(crate) fn most_buffers(&self, class: SizeClass) -> u16 {
        self.classes
            .get(usize::from(class.get()))
            .map_or(0, |pool| {
                let slots = pool.most_slots().min(self.budget / class.slot_bytes());
                u16::try_from(pool.most_buffers(slots)).unwrap_or(u16::MAX)
            })
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

    /// Bytes the buffers take: every slot made, handed out or not.
    pub(crate) const fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Bytes of the slots handed out: what resident tiles (and pending epochs' shadow copies)
    /// take of [`Pools::allocated`].
    pub(crate) const fn used(&self) -> u64 {
        self.used
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

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;

    /// A MiB.
    const MIB: u64 = 1 << 20;

    /// Level `l`.
    fn level(l: u8) -> Level {
        Level::new(l).expect("a level")
    }

    /// Grows `class`'s pool in `pools` as [`Pools::grow`] does, without a GPU, until the budget
    /// or its most slots stop it; returns the slots of each buffer made.
    fn grow_dry(pools: &mut Pools, class: SizeClass) -> Vec<u64> {
        let mut made = Vec::new();
        loop {
            let room = pools.budget.saturating_sub(pools.allocated) / class.slot_bytes();
            let pool = &mut pools.classes[usize::from(class.get())];
            let slots = pool.next_slots(room);
            if slots == 0 || pool.counts.len() >= MAX_BUFFERS {
                return made;
            }
            pool.counts
                .push(u32::try_from(slots).expect("a buffer's slots"));
            pools.allocated += slots * class.slot_bytes();
            made.push(slots);
        }
    }

    #[test]
    fn a_classs_buffers_start_small_and_double_up_to_the_binding_size() {
        // The benchmark project's shape on the 4090: 2400 faces to level 9, buffers of up to
        // 2047 MiB, the desktop budget.
        let mut pools = Pools::new(2400, level(9), 2047 * MIB, 4096 * MIB);
        let class = SizeClass::of(level(9));
        let bytes: Vec<u64> = grow_dry(&mut pools, class)
            .into_iter()
            .map(|slots| slots * class.slot_bytes() / MIB)
            .collect();
        assert!(
            bytes
                .first()
                .is_some_and(|&first| (60..=64).contains(&first)),
            "the first is about 64 MiB: {bytes:?}"
        );
        assert!(
            bytes.windows(2).all(|pair| pair[1] <= 2 * pair[0] + 1),
            "each next at most doubles: {bytes:?}"
        );
        assert!(
            bytes.iter().all(|&b| b <= 2047),
            "none past the binding: {bytes:?}"
        );
        assert_eq!(bytes.len(), 7, "{bytes:?}");
        assert!(pools.allocated <= 4096 * MIB);
    }

    #[test]
    fn the_benchmark_projects_layout_fits_every_tier_measured() {
        // Every class's most buffers on the benchmark's shape at the desktop budget: what the
        // raster binds, against each adapter's storage buffers a stage less its own (residency's
        // tier table).
        let total = |binding: u64| -> u32 {
            let pools = Pools::new(2400, level(9), binding, 4096 * MIB);
            (0..7)
                .filter_map(SizeClass::new)
                .map(|class| u32::from(pools.most_buffers(class)))
                .sum()
        };
        assert_eq!(total(2047 * MIB), 14, "the 4090");
        assert_eq!(total(4095 * MIB), 14, "the M4");
        assert_eq!(total(128 * MIB), 40, "lavapipe, under its 44");
    }

    #[test]
    fn the_layout_bounds_every_buffer_a_class_can_make_as_it_grows_to_its_budget() {
        // Buffer sizes from one slot to past the first buffer's, budgets from a few slots to
        // more than every tile, and the floor's classes and the finest one.
        let projects = [6, 24, 2400];
        let buffers = [128 * 1024, 4 * MIB, 64 * MIB, 128 * MIB, 2047 * MIB];
        let budgets = [MIB, 50 * MIB, 700 * MIB, 4096 * MIB];
        let cases = projects
            .iter()
            .flat_map(|f| buffers.iter().map(move |b| (f, b)))
            .flat_map(|(f, b)| budgets.iter().map(move |g| (f, b, g)));
        for (&faces, &buffer_bytes, &budget) in cases {
            for class in (0..7).filter_map(SizeClass::new) {
                let mut pools = Pools::new(faces, level(9), buffer_bytes, budget);
                let most = pools.most_buffers(class);
                let made = grow_dry(&mut pools, class);
                assert!(
                    made.len() <= usize::from(most),
                    "{faces} faces, {buffer_bytes} B buffers, {budget} B: {class:?} made {} \
                     of at most {most}",
                    made.len()
                );
                // Exact: growing alone to the budget makes as many as the bound.
                assert_eq!(
                    made.len(),
                    usize::from(most),
                    "{faces} faces, {buffer_bytes} B, {budget} B: {class:?} {made:?}"
                );
            }
        }
    }
}
