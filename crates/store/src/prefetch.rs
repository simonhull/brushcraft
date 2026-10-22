//! Background prefetch: worker threads that read and decode correction tiles ahead of use.
//!
//! The tile store's owner thread looks up where each wanted tile's record lives (in the working
//! file, or in the base under it) and queues `(key, source, location)` jobs. A few worker
//! threads, started on the first request, take jobs in priority order, read each record through
//! a [`FileReader`] of its source into a boxed [`CorrectionSlot`] with their own reusable
//! buffers, and send the result back over a channel. The owner installs
//! finished slots at its next [`crate::TileStore::poll`], checking first that each one is still
//! wanted; the workers touch neither the pools nor the file's index, so the hot path takes no
//! locks.
//!
//! # Shutdown
//!
//! Dropping the [`Prefetcher`] closes the queue and joins the workers. A worker checks the closed
//! flag before every job and while it waits for one, so it stops after at most the read in hand,
//! however much work is still queued. A read whose file has been closed under it fails with
//! [`StoreError::Closed`], which is sent back like any other error: the job is answered, and the
//! worker goes on to the next.
//!
//! A prefetcher reads the files it was started with. The store makes a new one when it swaps its
//! files ([`crate::TileStore::rebase`]), so no job or result outlives the files it was for.

use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZero;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};

use geom::TileKey;

use crate::OpId;
use crate::file::{BaseFile, FileReader, Loc, ProjectFile, ReadScratch, StoreError};
use crate::slot::CorrectionSlot;

/// Most worker threads a prefetcher starts, however many cores there are.
///
/// Reads mostly wait on the disk and decoding is quick, so more threads would only compete with
/// the owner for memory bandwidth.
const MAX_WORKERS: usize = 4;

/// Most spare slot boxes kept for workers to reuse. A few per worker is enough to keep them
/// from allocating while the owner polls regularly.
const MAX_SPARE: usize = 4 * MAX_WORKERS;

/// Which of the store's files a record is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Source {
    /// The working file, read first.
    Working,
    /// The base under it: the project file as of its last save.
    Base,
}

/// One record to read: the tile, and which file its record was in and where when the job was
/// queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Job {
    /// The correction tile.
    pub(crate) key: TileKey,
    /// The file holding the record.
    pub(crate) source: Source,
    /// Its latest record when the owner queued the job.
    pub(crate) loc: Loc,
}

/// A worker's result: the job, and the decoded tile with the op its record carries, or why it
/// couldn't be read.
#[derive(Debug)]
pub(crate) struct Fetched {
    /// The job this answers.
    pub(crate) job: Job,
    /// The op of the record read and the decoded tile, boxed so the channel moves a pointer,
    /// not the slot.
    pub(crate) slot: Result<(OpId, Box<CorrectionSlot>), StoreError>,
}

/// The queue the owner fills and the workers drain, behind the [`Shared`] mutex.
#[derive(Debug, Default)]
struct Queue {
    /// Jobs not taken yet, the most wanted first.
    jobs: VecDeque<Job>,
    /// Slot boxes the owner has finished with, for workers to fill again.
    spare: Vec<Box<CorrectionSlot>>,
    /// Set once by the owner's drop: workers take no more jobs and return.
    closed: bool,
    /// Holds the workers back while set, so tests can arrange what happens between a request
    /// and its result.
    #[cfg(test)]
    paused: bool,
}

impl Queue {
    /// Whether a worker should keep waiting: nothing to take (or held back by a test) and not
    /// closed.
    fn idle(&self) -> bool {
        #[cfg(test)]
        if self.paused && !self.closed {
            return true;
        }
        self.jobs.is_empty() && !self.closed
    }
}

/// The state the owner and the workers share: the queue and the condition variable workers
/// sleep on until there is a job or the queue closes.
#[derive(Debug, Default)]
struct Shared {
    /// The queue.
    queue: Mutex<Queue>,
    /// Signalled when jobs are queued, when the queue closes, and (in tests) on unpausing.
    ready: Condvar,
}

impl Shared {
    /// The queue, locked. A poisoned lock is taken over as it is: the queue is only ever
    /// changed by whole pushes and pops, so a panic elsewhere can't leave it half-updated.
    fn lock(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Waits for the next job and a box to read it into (a spare one if there is one); `None`
    /// once the queue is closed, even if jobs are left.
    fn next_job(&self) -> Option<(Job, Option<Box<CorrectionSlot>>)> {
        let mut queue = self.lock();
        while queue.idle() {
            queue = self
                .ready
                .wait(queue)
                .unwrap_or_else(PoisonError::into_inner);
        }
        if queue.closed {
            return None;
        }
        let job = queue.jobs.pop_front()?;
        Some((job, queue.spare.pop()))
    }
}

/// The owner's side of the prefetch workers: the shared queue, the workers' handles, the
/// channel their results come back on and the jobs not answered yet.
///
/// Belongs to the thread that owns the tile store. Workers start on the first
/// [`Prefetcher::request`] with work to do, and stop and are joined when the prefetcher is
/// dropped.
#[derive(Debug)]
pub(crate) struct Prefetcher {
    /// The queue, shared with the workers.
    shared: Arc<Shared>,
    /// The running workers; empty until the first request.
    workers: Vec<JoinHandle<()>>,
    /// Given to each worker as it starts; kept so workers can be started later.
    send: Sender<Fetched>,
    /// Where finished jobs arrive.
    results: Receiver<Fetched>,
    /// Every job queued or being read, by key, at the file and location it was asked for: a key
    /// is queued again only if its record has moved since.
    pending: HashMap<TileKey, (Source, Loc)>,
}

impl Prefetcher {
    /// A prefetcher with no workers yet.
    pub(crate) fn new() -> Self {
        let (send, results) = mpsc::channel();
        Self {
            shared: Arc::default(),
            workers: Vec::new(),
            send,
            results,
            pending: HashMap::new(),
        }
    }

    /// Jobs queued or being read whose results haven't been taken by [`Prefetcher::next`].
    pub(crate) fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Queues `jobs` (each key once), in order, ahead of everything queued before; returns how
    /// many weren't pending already.
    ///
    /// A job for a key already pending at the same location isn't queued twice: if it is still
    /// waiting it moves up with this request, and if a worker has it already it is left to
    /// finish. A job at a different location (the record moved since the earlier request) is
    /// queued anew; the earlier result will be found stale. Starts the workers if none run yet,
    /// reading through readers of `working` and `base`.
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`] if no worker could be started or a worker's read buffers can't be
    /// made. Nothing is queued then.
    pub(crate) fn request(
        &mut self,
        working: &ProjectFile,
        base: Option<&BaseFile>,
        jobs: &[Job],
    ) -> Result<usize, StoreError> {
        if jobs.is_empty() {
            return Ok(0);
        }
        if self.workers.is_empty() {
            self.start(working, base)?;
        }
        let wanted: HashSet<TileKey> = jobs.iter().map(|job| job.key).collect();
        let mut new = 0;
        {
            let mut queue = self.shared.lock();
            // Jobs of these keys still waiting are taken out, to go back in at the front.
            let mut waiting = HashSet::new();
            queue.jobs.retain(|job| {
                let asked_again = wanted.contains(&job.key);
                if asked_again {
                    waiting.insert(job.key);
                }
                !asked_again
            });
            // Pushed to the front in reverse, so the request's own order is kept.
            for &job in jobs.iter().rev() {
                let at = (job.source, job.loc);
                let asked = self.pending.insert(job.key, at);
                if asked == Some(at) && !waiting.contains(&job.key) {
                    // A worker is reading it at this location already.
                    continue;
                }
                if asked != Some(at) {
                    new += 1;
                }
                queue.jobs.push_front(job);
            }
        }
        self.shared.ready.notify_all();
        Ok(new)
    }

    /// The next finished job, if one has arrived, and forgets it as pending (unless the key has
    /// been asked for again at another location since).
    pub(crate) fn next(&mut self) -> Option<Fetched> {
        let fetched = self.results.try_recv().ok()?;
        if self.pending.get(&fetched.job.key) == Some(&(fetched.job.source, fetched.job.loc)) {
            self.pending.remove(&fetched.job.key);
        }
        Some(fetched)
    }

    /// Hands a slot box back for a worker to fill again, unless enough are spare already.
    pub(crate) fn recycle(&self, slot: Box<CorrectionSlot>) {
        let mut queue = self.shared.lock();
        if queue.spare.len() < MAX_SPARE {
            queue.spare.push(slot);
        }
    }

    /// Starts the workers: one per available core, at most [`MAX_WORKERS`], each with its own
    /// read buffers and readers of `working` and `base`. A worker that fails to start is
    /// skipped.
    ///
    /// # Errors
    ///
    /// [`StoreError::Io`] if read buffers can't be made, or no worker started.
    fn start(&mut self, working: &ProjectFile, base: Option<&BaseFile>) -> Result<(), StoreError> {
        let count = thread::available_parallelism()
            .map_or(1, NonZero::get)
            .min(MAX_WORKERS);
        let mut failed = None;
        for n in 0..count {
            let worker = Worker {
                shared: Arc::clone(&self.shared),
                working: working.reader(),
                base: base.map(BaseFile::reader),
                scratch: ReadScratch::new()?,
                send: self.send.clone(),
            };
            match thread::Builder::new()
                .name(format!("prefetch-{n}"))
                .spawn(move || worker.run())
            {
                Ok(handle) => self.workers.push(handle),
                Err(e) => failed = Some(e),
            }
        }
        match failed {
            Some(e) if self.workers.is_empty() => Err(e.into()),
            _ => Ok(()),
        }
    }

    /// Holds the workers back, or lets them go: tests use it to make a result arrive after
    /// something they do in between.
    #[cfg(test)]
    pub(crate) fn pause(&self, paused: bool) {
        self.shared.lock().paused = paused;
        self.shared.ready.notify_all();
    }

    /// A handle that upgrades only while a worker (or the prefetcher) is alive: once the
    /// prefetcher is dropped and its workers joined, it can't.
    #[cfg(test)]
    pub(crate) fn liveness(&self) -> std::sync::Weak<impl Sized + use<>> {
        Arc::downgrade(&self.shared)
    }
}

impl Drop for Prefetcher {
    /// Closes the queue, wakes every worker and waits for each to return: at most the read in
    /// hand, since a worker checks the flag before taking another job.
    fn drop(&mut self) {
        self.shared.lock().closed = true;
        self.shared.ready.notify_all();
        for worker in self.workers.drain(..) {
            // A worker that panicked has nothing left to clean up; its panic was reported on
            // its own thread.
            drop(worker.join());
        }
    }
}

/// One worker thread's state: the queue, its own reader and buffers, and the way back.
struct Worker {
    /// The queue it takes jobs from.
    shared: Arc<Shared>,
    /// Reads the owner's working file while it stays open.
    working: FileReader,
    /// Reads the base under it while its project file stays open; `None` without a base.
    base: Option<FileReader>,
    /// Record and decompression buffers, reused for every job.
    scratch: ReadScratch,
    /// Where results go.
    send: Sender<Fetched>,
}

impl Worker {
    /// Takes jobs until the queue closes or the owner stops listening.
    fn run(mut self) {
        while let Some((job, spare)) = self.shared.next_job() {
            let mut slot = spare.unwrap_or_else(bytemuck::zeroed_box);
            let reader = match job.source {
                Source::Working => Some(&self.working),
                Source::Base => self.base.as_ref(),
            };
            // A base job without a base reader can't be queued: the store queues base jobs
            // only when it has a base, and starts its workers with it.
            let read = reader.map_or(Err(StoreError::Closed), |reader| {
                reader.read_displacement_into(&job.key, job.loc, &mut slot, &mut self.scratch)
            });
            let slot = read.map(|op| (op, slot));
            if self.send.send(Fetched { job, slot }).is_err() {
                return;
            }
        }
    }
}
