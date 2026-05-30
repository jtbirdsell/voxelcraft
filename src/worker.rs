//! Worker thread pool for the CPU-heavy stages (generation, meshing). Workers run pure
//! functions on owned/immutable data and return owned results; the main thread owns the world
//! map and performs GPU uploads. crossbeam's multi-consumer channel load-balances jobs across
//! the workers (one shared receiver), which suits the wildly varying per-chunk cost.

use std::thread::JoinHandle;

use crossbeam_channel::{unbounded, Receiver, Sender, TryIter};
use glam::IVec3;

use crate::mesher::{self, MeshData};
use crate::world::{generate_chunk, Chunk, Neighborhood};

pub enum Job {
    Generate {
        pos: IVec3,
    },
    Mesh {
        pos: IVec3,
        neigh: Neighborhood,
        origin: [i32; 3],
    },
}

pub enum JobResult {
    Generated { pos: IVec3, chunk: Chunk },
    Meshed { pos: IVec3, mesh: MeshData },
}

pub struct WorkerPool {
    job_tx: Sender<Job>,
    result_rx: Receiver<JobResult>,
    workers: usize,
    _handles: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    pub fn new(seed: u64, num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let (job_tx, job_rx) = unbounded::<Job>();
        let (result_tx, result_rx) = unbounded::<JobResult>();

        let mut handles = Vec::with_capacity(num_workers);
        for i in 0..num_workers {
            let job_rx = job_rx.clone();
            let result_tx = result_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("voxel-worker-{i}"))
                .spawn(move || worker_loop(seed, job_rx, result_tx))
                .expect("failed to spawn worker thread");
            handles.push(handle);
        }

        Self {
            job_tx,
            result_rx,
            workers: num_workers,
            _handles: handles,
        }
    }

    pub fn worker_count(&self) -> usize {
        self.workers
    }

    pub fn submit(&self, job: Job) {
        let _ = self.job_tx.send(job);
    }

    /// Non-blocking drain of completed results.
    pub fn drain(&self) -> TryIter<'_, JobResult> {
        self.result_rx.try_iter()
    }
}

fn worker_loop(seed: u64, job_rx: Receiver<Job>, result_tx: Sender<JobResult>) {
    while let Ok(job) = job_rx.recv() {
        let result = match job {
            Job::Generate { pos } => JobResult::Generated {
                pos,
                chunk: generate_chunk(pos, seed),
            },
            Job::Mesh { pos, neigh, origin } => JobResult::Meshed {
                pos,
                mesh: mesher::build_mesh(&neigh, origin),
            },
        };
        if result_tx.send(result).is_err() {
            break; // main thread is gone
        }
    }
}
