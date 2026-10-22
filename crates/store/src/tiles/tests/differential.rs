//! The store against the dense evaluation, checked at every read.
//!
//! A random mix of every store operation runs at the minimum budget and above it, and each
//! evaluation, normals computation and correction read is checked bit for bit against `geom`'s
//! dense evaluation of the same edits. The default suite runs one seed per budget on the cube;
//! the ignored test runs many seeds on the cube and on a prism with a valence-7 vertex.

use std::collections::HashMap;

use super::*;

/// How often each kind of check ran, so a run that never reaches one fails.
#[derive(Debug, Default, Clone, Copy)]
struct Stats {
    /// Evaluated tiles checked against the oracle.
    positions: usize,
    /// Normal tiles checked against the oracle.
    normals: usize,
    /// Correction tiles sampled against the oracle's map.
    corrections: usize,
    /// Pin sets held.
    pins: usize,
    /// Pin requests refused up front as over the budget.
    over_budget: usize,
    /// Operations refused because the pins held left no room.
    full: usize,
    /// Saves followed by a drop and a reopen of the working file.
    reopens: usize,
}

impl std::ops::Add for Stats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            positions: self.positions + rhs.positions,
            normals: self.normals + rhs.normals,
            corrections: self.corrections + rhs.corrections,
            pins: self.pins + rhs.pins,
            over_budget: self.over_budget + rhs.over_budget,
            full: self.full + rhs.full,
            reopens: self.reopens + rhs.reopens,
        }
    }
}

/// A closed quad mesh with a valence-`n` vertex at the centre of each cap: an `n`-sided prism
/// with each polygon split into quads around its centroid.
fn split_prism(n: usize) -> BaseMesh {
    let angle = |k: usize| {
        let k = f32::from(u16::try_from(k).unwrap());
        let n = f32::from(u16::try_from(n).unwrap());
        2.0 * std::f32::consts::PI * k / n
    };
    let mut corners: Vec<Vec3> = Vec::new();
    for z in [1.0, -1.0] {
        corners.extend((0..n).map(|k| Vec3::new(angle(k).cos(), angle(k).sin(), z)));
    }
    let mut polygons: Vec<Vec<usize>> =
        vec![(0..n).collect(), (0..n).rev().map(|k| n + k).collect()];
    polygons.extend((0..n).map(|k| vec![k, n + k, n + (k + 1) % n, (k + 1) % n]));
    // Each polygon becomes one quad per corner: the corner, its two edge midpoints and the
    // centroid. Midpoints are shared between the two polygons of an edge.
    let mut positions = corners.clone();
    let mut midpoints: HashMap<(usize, usize), usize> = HashMap::new();
    let mut quads: Vec<[u32; 4]> = Vec::new();
    for polygon in &polygons {
        let sides = polygon.len();
        let sum = polygon.iter().fold(Vec3::ZERO, |sum, &v| sum + corners[v]);
        let centroid = positions.len();
        positions.push(sum * (1.0 / f32::from(u16::try_from(sides).unwrap())));
        let mids: Vec<usize> = (0..sides)
            .map(|a| {
                let (u, v) = (polygon[a], polygon[(a + 1) % sides]);
                *midpoints.entry((u.min(v), u.max(v))).or_insert_with(|| {
                    positions.push((corners[u] + corners[v]) * 0.5);
                    positions.len() - 1
                })
            })
            .collect();
        for a in 0..sides {
            let quad = [polygon[a], mids[a], centroid, mids[(a + sides - 1) % sides]];
            quads.push(quad.map(|v| u32::try_from(v).unwrap()));
        }
    }
    BaseMesh::from_polygons(positions, &quads).unwrap()
}

/// The budgets a run is tried at: the mesh's minimum and a few chunks more, in chunks of four
/// evaluated slots.
fn budgets(mesh: &BaseMesh, extra_chunks: &[usize]) -> Vec<StoreConfig> {
    let chunk = 4 * EVAL;
    let base = StoreConfig {
        budget_bytes: 0,
        chunk_bytes: chunk,
        site: SITE,
    };
    let minimum = base.minimum_budget(mesh);
    extra_chunks
        .iter()
        .map(|extra| StoreConfig {
            budget_bytes: minimum + extra * chunk,
            ..base
        })
        .collect()
}

/// One random run's state, besides the store itself (which a reopen replaces).
struct Run<'a> {
    /// The mesh.
    mesh: &'a Arc<BaseMesh>,
    /// The finest level evaluated.
    top: u8,
    /// The store's configuration.
    config: StoreConfig,
    /// The seed, for messages.
    seed: u64,
    /// Every tile of levels 1 to `top`.
    all: Vec<TileKey>,
    /// Picks operations and tiles.
    rng: Rng,
    /// The single-vertex edits to make, in order.
    edits: std::vec::IntoIter<Edit>,
    /// The oracle's corrections: every edit the store accepted.
    map: CorrectionMap,
    /// The oracle's dense evaluation of `map`, made when first needed after an edit.
    oracle: Option<DenseEvaluation>,
    /// Pins held.
    guards: Vec<PinGuard>,
    /// The op of the latest edit, which the next edit may reuse (a stroke's dabs).
    last_op: Option<OpId>,
    /// What was checked.
    stats: Stats,
}

impl Run<'_> {
    /// A tile of levels 1 to `top`.
    fn pick(&mut self) -> TileKey {
        self.all[self.rng.below(u32::try_from(self.all.len()).unwrap()) as usize]
    }

    /// The dense evaluation of every edit so far.
    fn oracle(&mut self) -> &DenseEvaluation {
        let (mesh, map, top) = (self.mesh, &self.map, self.top);
        self.oracle
            .get_or_insert_with(|| DenseEvaluation::new(mesh, level(top), map).unwrap())
    }

    /// The op for the next edit: the last one again, half the time.
    fn op(&mut self, store: &mut TileStore) -> OpId {
        let op = match self.last_op {
            Some(op) if self.rng.below(2) == 0 => op,
            _ => store.next_op().unwrap(),
        };
        self.last_op = Some(op);
        op
    }

    /// Counts `result` as refused for lack of room if pins are held, and fails the test on any
    /// other error; the value, if there is one.
    fn allow_full<T>(&mut self, result: Result<T, CacheError>, what: &str) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(CacheError::Full) if !self.guards.is_empty() => {
                self.stats.full += 1;
                None
            }
            Err(e) => panic!("seed {} {what}: {e:?}", self.seed),
        }
    }

    /// Evaluates a tile (a pinned one, a third of the time) and checks it.
    fn evaluate(&mut self, store: &mut TileStore) {
        let key = if !self.guards.is_empty() && self.rng.below(3) == 0 {
            let n = self.rng.below(u32::try_from(self.guards.len()).unwrap()) as usize;
            self.guards[n].keys()[0]
        } else {
            self.pick()
        };
        let got = store.evaluate(key).map(<[Vec3]>::to_vec);
        if let Some(got) = self.allow_full(got, "evaluate") {
            let grid = self.oracle().grid(key.face(), key.level()).unwrap();
            assert_tile(&got, |i, j| grid.get(i, j).unwrap(), &key, "evaluate");
            self.stats.positions += 1;
        }
    }

    /// Computes a tile's normals and checks them.
    fn normals(&mut self, store: &mut TileStore) {
        let key = self.pick();
        let got = store.normals(key).map(<[Vec3]>::to_vec);
        if let Some(got) = self.allow_full(got, "normals") {
            let region = Region::of_tile(&key);
            let want = normals(self.mesh, self.oracle(), region).unwrap();
            assert_tile(
                &got,
                |i, j| want.get(i - region.x0(), j - region.y0()).unwrap(),
                &key,
                "normals",
            );
            self.stats.normals += 1;
        }
    }

    /// Sets one vertex's correction.
    fn set_correction(&mut self, store: &mut TileStore) {
        let e = self.edits.next().unwrap();
        let op = self.op(store);
        let set = store.set_correction(e.face, e.level, e.i, e.j, e.c, op);
        if self.allow_full(set, "set_correction").is_some() {
            self.map
                .set(self.mesh.topology(), e.face, e.level, e.i, e.j, e.c)
                .unwrap();
            self.oracle = None;
        }
    }

    /// Edits up to forty vertices of one tile in one batch; the oracle follows the writes that
    /// were made, up to the first one refused for lack of room.
    fn edit_tile(&mut self, store: &mut TileStore) {
        let count = 1 + self.rng.below(40);
        let (key, writes) = tile_batches(self.mesh, 1, count, self.rng.next_u64()).remove(0);
        let op = self.op(store);
        let seed = self.seed;
        let edited = store.edit_tile(key, op, |edit| {
            let mut done = Vec::new();
            for &(i, j, c) in &writes {
                match edit.set(i, j, c) {
                    Ok(()) => done.push((i, j, c)),
                    Err(CacheError::Full) => return (done, true),
                    Err(e) => panic!("seed {seed} edit_tile set: {e:?}"),
                }
            }
            (done, false)
        });
        let Some((done, refused)) = self.allow_full(edited, "edit_tile") else {
            return;
        };
        if refused {
            assert!(
                !self.guards.is_empty(),
                "seed {seed}: refused with no pins held"
            );
            self.stats.full += 1;
        }
        let region = Region::of_tile(&key);
        for (i, j, c) in done {
            let (face, level) = (key.face(), key.level());
            self.map
                .set(
                    self.mesh.topology(),
                    face,
                    level,
                    region.x0() + i,
                    region.y0() + j,
                    c,
                )
                .unwrap();
            self.oracle = None;
        }
    }

    /// Samples a tile's corrections as stored against the oracle's map.
    fn correction(&mut self, store: &mut TileStore) {
        let key = self.pick();
        let samples: Vec<(u32, u32)> = (0..16)
            .map(|_| (self.rng.below(key.side()), self.rng.below(key.side())))
            .collect();
        let got = store.correction(key).map(|view| {
            samples
                .iter()
                .map(|&(i, j)| view.get(i, j).unwrap())
                .collect::<Vec<_>>()
        });
        if let Some(got) = self.allow_full(got, "correction") {
            let region = Region::of_tile(&key);
            for (&(i, j), have) in samples.iter().zip(got) {
                let want = self
                    .map
                    .get(key.face(), key.level(), region.x0() + i, region.y0() + j);
                assert_eq!(
                    have.to_bits(),
                    want.to_bits(),
                    "seed {} correction {key:?} ({i}, {j})",
                    self.seed
                );
            }
            self.stats.corrections += 1;
        }
    }

    /// Pins one to three tiles, keeping at most four pinned.
    fn pin(&mut self, store: &mut TileStore) {
        let pinned: usize = self.guards.iter().map(|g| g.keys().len()).sum();
        let count = 1 + self.rng.below(2) as usize;
        let keys: Vec<TileKey> = (0..count).map(|_| self.pick()).collect();
        if pinned + keys.len() > 4 {
            return;
        }
        match store.pin(&keys) {
            Ok(guard) => {
                self.guards.push(guard);
                self.stats.pins += 1;
            }
            Err(CacheError::OverBudget { .. }) => self.stats.over_budget += 1,
            Err(CacheError::Full) => self.stats.full += 1,
            Err(e) => panic!("seed {} pin {keys:?}: {e:?}", self.seed),
        }
    }

    /// Drops one pin set, if any is held.
    fn unpin(&mut self) {
        if !self.guards.is_empty() {
            let n = self.rng.below(u32::try_from(self.guards.len()).unwrap()) as usize;
            drop(self.guards.swap_remove(n));
        }
    }

    /// Prefetches up to nine tiles, and polls half the time.
    fn prefetch(&mut self, store: &mut TileStore) {
        let count = 1 + self.rng.below(8);
        let keys: Vec<TileKey> = (0..count).map(|_| self.pick()).collect();
        store.prefetch(&keys).unwrap();
        if self.rng.below(2) == 0 {
            store.poll().unwrap();
        }
    }

    /// Evaluates four tiles unchecked: fills the cache, and evicts.
    fn burst(&mut self, store: &mut TileStore) {
        for _ in 0..4 {
            let key = self.pick();
            let evaluated = store.evaluate(key).map(drop);
            self.allow_full(evaluated, "burst");
        }
    }

    /// Runs one random operation; its name, for messages.
    fn step(&mut self, store: &mut TileStore) -> &'static str {
        match self.rng.below(100) {
            0..25 => {
                self.evaluate(store);
                "evaluate"
            }
            25..35 => {
                self.normals(store);
                "normals"
            }
            35..47 => {
                self.set_correction(store);
                "set_correction"
            }
            47..59 => {
                self.edit_tile(store);
                "edit_tile"
            }
            59..64 => {
                self.correction(store);
                "correction"
            }
            64..71 => {
                self.pin(store);
                "pin"
            }
            71..76 => {
                self.unpin();
                "unpin"
            }
            76..84 => {
                self.prefetch(store);
                "prefetch"
            }
            84..88 => {
                store.save().unwrap();
                "save"
            }
            88..91 => {
                store.evict_all().unwrap();
                "evict_all"
            }
            91..95 => {
                let budget = u32::try_from(self.config.budget_bytes).unwrap();
                store
                    .shrink_to(self.rng.below(budget + 1) as usize)
                    .unwrap();
                "shrink_to"
            }
            95..97 => "reopen",
            _ => {
                self.burst(store);
                "burst"
            }
        }
    }
}

/// Runs `steps` random operations, chosen with `seed`, on a new store over `mesh` with `config`;
/// returns what was checked.
///
/// Every read is checked against the dense evaluation to level `top`, the budget and the
/// dependency graph after every step, and every tile at the end.
fn run(mesh: &Arc<BaseMesh>, top: u8, config: StoreConfig, seed: u64, steps: u32) -> Stats {
    let dir = TestDir::new();
    let path = dir.path("work.bcp");
    let open = |file: ProjectFile| TileStore::open(file, config).unwrap();
    let mut store = open(ProjectFile::create(&path, meta_for(mesh), Arc::clone(mesh)).unwrap());
    let mut run = Run {
        mesh,
        top,
        config,
        seed,
        all: tiles(mesh, 1, top),
        rng: Rng(seed),
        edits: random_edits(mesh, top, steps, seed ^ 0xABCD).into_iter(),
        map: CorrectionMap::new(),
        oracle: None,
        guards: Vec::new(),
        last_op: None,
        stats: Stats::default(),
    };
    for step in 0..steps {
        let what = run.step(&mut store);
        if what == "reopen" {
            // Save, drop, reopen: the file holds every edit.
            store.save().unwrap();
            run.guards.clear();
            drop(store);
            store = open(ProjectFile::open(&path).unwrap());
            run.last_op = None;
            run.stats.reopens += 1;
        }
        let at = format!("seed {seed} step {step} ({what})");
        assert!(
            store.bytes() <= config.budget_bytes,
            "{at}: over the budget"
        );
        if let Err(broken) = store.deps.check_invariants() {
            panic!("{at}: dependency graph: {broken}");
        }
    }
    run.guards.clear();
    let dense = DenseEvaluation::new(mesh, level(top), &run.map).unwrap();
    assert_positions(
        &mut store,
        &dense,
        &run.all,
        &format!("seed {seed} at the end"),
    );
    run.stats
}

/// Asserts that the runs together reached every kind of check.
fn assert_covered(stats: Stats, what: &str) {
    assert!(
        stats.positions > 0 && stats.normals > 0 && stats.corrections > 0,
        "{what}: reads checked: {stats:?}"
    );
    assert!(
        stats.pins > 0 && stats.reopens > 0,
        "{what}: pins and reopens: {stats:?}"
    );
}

#[test]
fn the_store_reads_as_the_dense_evaluation_at_every_step() {
    let mesh = mesh();
    // Level 7, where faces are already split into tiles, keeps the oracle cheap: it is evaluated
    // again after every edit.
    thread::scope(|scope| {
        for (seed, config) in (0x5EED..).zip(budgets(&mesh, &[0, 3])) {
            let mesh = &mesh;
            scope.spawn(move || {
                let stats = run(mesh, 7, config, seed, 400);
                assert_covered(stats, &format!("{} bytes", config.budget_bytes));
            });
        }
    });
}

#[test]
#[ignore = "slow: many seeds of the_store_reads_as_the_dense_evaluation_at_every_step"]
fn the_store_reads_as_the_dense_evaluation_for_many_seeds() {
    let cube = mesh();
    let prism = Arc::new(split_prism(7));
    let cases = [(&cube, TOP, 8, 1500), (&prism, 7, 4, 800)];
    for (mesh, top, seeds, steps) in cases {
        // At the prism's minimum every pin is over the budget, so coverage is counted per mesh.
        let mut total = Stats::default();
        for config in budgets(mesh, &[0, 1, 3, 10]) {
            let stats = thread::scope(|scope| {
                #[expect(
                    clippy::needless_collect,
                    reason = "every run is spawned before the first is joined"
                )]
                let runs: Vec<_> = (1..=seeds)
                    .map(|seed| {
                        scope.spawn(move || run(mesh, top, config, seed * 7919 + 13, steps))
                    })
                    .collect();
                runs.into_iter()
                    .map(|run| run.join().unwrap())
                    .fold(Stats::default(), std::ops::Add::add)
            });
            println!(
                "{} faces, {} bytes: {stats:?}",
                mesh.face_count(),
                config.budget_bytes
            );
            total = total + stats;
        }
        assert_covered(total, &format!("{} faces", mesh.face_count()));
    }
}
