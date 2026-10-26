//! Random workloads of every store operation, checked against the budget at every step.

use super::*;

/// Most tiles the budget workload holds pinned at once.
const WORKLOAD_PINS: usize = 3;

/// Asserts what holds after every step of [`random_workload`]: the pools are within the budget,
/// which counts exactly their chunks, and the dependency graph is whole.
fn check_workload_step(store: &TileStore, config: StoreConfig, what: &str) {
    assert!(
        store.bytes() <= pools(&config),
        "{what}: {} bytes in a budget of {}",
        store.bytes(),
        pools(&config)
    );
    assert_eq!(
        store.bytes(),
        store.corrections.bytes() + store.evaluated.bytes(),
        "{what}: the budget counts the pools' chunks"
    );
    if let Err(broken) = store.deps.check_invariants() {
        panic!("{what}: dependency graph: {broken}");
    }
}

/// Runs `steps` random store operations, chosen with `seed`, under a budget of a few chunks.
///
/// The operations: evaluations, normals, edits (single vertices and tile batches), pins and unpins
/// (at most [`WORKLOAD_PINS`] tiles at once), prefetches and polls, saves, evictions and shrinks.
/// After every operation the pools are within the budget, the budget counts exactly their chunks
/// and the dependency graph is whole; at the end every tile of levels 1–8 evaluates as the dense
/// evaluation of the same edits does, bit for bit.
fn random_workload(seed: u64, steps: u32) {
    let config = pinning();
    let dir = TestDir::new();
    let mesh = mesh();
    let mut store = new_store(&dir, "work.bcp", &mesh, config);
    let mut map = CorrectionMap::new();
    let all = tiles(&mesh, 1, TOP);
    let mut edits = random_edits(&mesh, TOP, steps, seed.wrapping_add(1)).into_iter();
    let mut rng = Rng(seed);
    let mut guards: Vec<PinGuard> = Vec::new();
    // How often the rarer paths ran, so a workload that never reaches them fails.
    let mut pin_sets = 0;
    let pick = |from: &mut Rng| all[from.below(u32::try_from(all.len()).unwrap()) as usize];
    for step in 0..steps {
        let what = match rng.below(100) {
            0..30 => {
                store.evaluate(pick(&mut rng)).unwrap();
                "evaluate"
            }
            30..40 => {
                store.normals(pick(&mut rng)).unwrap();
                "normals"
            }
            40..52 => {
                apply(&mut store, &mut map, &[edits.next().unwrap()]);
                "set_correction"
            }
            52..62 => {
                let batch = tile_batches(&mesh, 1, 1 + rng.below(8), rng.next_u64());
                apply_batches(&mut store, &mut map, &batch);
                "edit_tile"
            }
            62..70 => {
                let pinned: usize = guards.iter().map(|g| g.keys().len()).sum();
                let keys: Vec<TileKey> = (0..=rng.below(2)).map(|_| pick(&mut rng)).collect();
                if pinned + keys.len() <= WORKLOAD_PINS {
                    match store.pin(&keys) {
                        Ok(guard) => {
                            guards.push(guard);
                            pin_sets += 1;
                        }
                        Err(CacheError::OverBudget { .. }) => {}
                        Err(e) => panic!("step {step}: pin {keys:?}: {e:?}"),
                    }
                }
                "pin"
            }
            70..76 => {
                if !guards.is_empty() {
                    let n = rng.below(u32::try_from(guards.len()).unwrap()) as usize;
                    drop(guards.swap_remove(n));
                }
                "unpin"
            }
            76..88 => {
                let keys: Vec<TileKey> = (0..=rng.below(8)).map(|_| pick(&mut rng)).collect();
                store.prefetch(&keys).unwrap();
                store.poll().unwrap();
                "prefetch"
            }
            88..93 => {
                store.save().unwrap();
                "save"
            }
            93..96 => {
                store.evict_all().unwrap();
                "evict_all"
            }
            _ => {
                let budget = u32::try_from(pools(&config)).unwrap();
                store.shrink_to(rng.below(budget + 1) as usize).unwrap();
                "shrink_to"
            }
        };
        check_workload_step(&store, config, &format!("seed {seed} step {step} ({what})"));
    }
    drop(guards);
    drain_prefetch(&mut store);
    assert!(pin_sets > 10, "seed {seed}: only {pin_sets} pin sets held");
    // Installs happen at whichever call polls first, so they are counted by the store.
    assert!(
        store.log.prefetched > 10,
        "seed {seed}: only {} prefetched tiles installed",
        store.log.prefetched
    );
    assert!(
        store.log.positions.len() > 2 * all.len(),
        "seed {seed}: evicting, so tiles were evaluated again"
    );
    let dense = dense(&mesh, &map);
    assert_positions(&mut store, &dense, &all, &format!("seed {seed}"));
    let some: Vec<TileKey> = all.iter().step_by(7).copied().collect();
    assert_normals(&mut store, &mesh, &dense, &some, &format!("seed {seed}"));
}

#[test]
fn random_workloads_respect_the_budget() {
    // Seed 1 evicts a parent between its child's prediction and the child's slot: the window
    // where recording the child after unpinning its parents lost the parent's ancestry.
    for seed in [0x00B0_D6E7, 1] {
        random_workload(seed, 2_000);
    }
}

#[test]
#[ignore = "slow: more seeds of random_workloads_respect_the_budget"]
fn random_workloads_respect_the_budget_for_many_seeds() {
    for seed in 1..=16 {
        random_workload(seed, 2_000);
    }
}
