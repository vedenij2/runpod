import time
import queue
import threading
from typing import List, Dict, Any
from concurrent.futures import Future

from pow.data import ProofBatch
from pow.compute.compute import Compute
from pow.models.utils import Params
from pow.random import get_target
from common.logger import create_logger

logger = create_logger(__name__)


class NonceIterator:
    """
    Deterministic nonce iterator that distributes nonces across workers.
    Each worker gets non-overlapping nonce ranges.
    """
    def __init__(self, worker_id: int, n_workers: int, start_nonce: int = 0):
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.start_nonce = start_nonce
        self._current_x = 0
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        with self._lock:
            # Nonce = start + worker_id + current_x * n_workers
            # This ensures non-overlapping nonces across workers
            value = self.start_nonce + self.worker_id + self._current_x * self.n_workers
            self._current_x += 1
            return value


def worker_thread(
    worker_id: int,
    params: Params,
    block_hash: str,
    block_height: int,
    public_key: str,
    r_target: float,
    batch_size: int,
    devices: List[str],
    start_nonce: int,
    n_workers: int,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    ready_event: threading.Event,
    max_duration: float = 420.0,  # 7 minutes default
):
    """
    Worker thread that runs on a specific GPU group.
    Generates nonces and puts results into result_queue.
    """
    import torch

    try:
        logger.info(f"[Worker {worker_id}] Starting on devices {devices}, batch_size={batch_size}")

        # Initialize compute for this worker's GPU(s)
        compute = Compute(
            params=params,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            r_target=r_target,
            devices=devices,
            node_id=worker_id,
        )

        target = get_target(block_hash, params.vocab_size)

        # Signal that worker is ready
        ready_event.set()
        logger.info(f"[Worker {worker_id}] Model loaded, ready to generate")

        # Create nonce iterator for this worker
        nonce_iter = NonceIterator(
            worker_id=worker_id,
            n_workers=n_workers,
            start_nonce=start_nonce
        )

        start_time = time.time()
        batch_count = 0
        total_computed = 0
        total_valid = 0

        # Pre-generate first batch of nonces
        next_nonces = [next(nonce_iter) for _ in range(batch_size)]

        while not stop_event.is_set():
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_duration:
                logger.info(f"[Worker {worker_id}] Timeout after {elapsed:.0f}s")
                break

            # Current batch
            nonces = next_nonces
            # Pre-fetch next batch
            next_nonces = [next(nonce_iter) for _ in range(batch_size)]

            # Process batch with pre-fetching
            future: Future = compute(
                nonces=nonces,
                public_key=public_key,
                target=target,
                next_nonces=next_nonces,
            )

            # Get result
            proof_batch = future.result()
            filtered_batch = proof_batch.sub_batch(r_target)

            batch_count += 1
            total_computed += len(proof_batch)
            total_valid += len(filtered_batch)

            # Put result in queue
            if filtered_batch.nonces:
                # Convert dist to list if it's a numpy array
                dist_list = filtered_batch.dist
                if hasattr(dist_list, 'tolist'):
                    dist_list = dist_list.tolist()
                elif not isinstance(dist_list, list):
                    dist_list = list(dist_list)

                result = {
                    "worker_id": worker_id,
                    "public_key": filtered_batch.public_key,
                    "block_hash": filtered_batch.block_hash,
                    "block_height": filtered_batch.block_height,
                    "nonces": filtered_batch.nonces,
                    "dist": dist_list,
                    "node_id": filtered_batch.node_id,
                    "batch_number": batch_count,
                    "batch_computed": len(proof_batch),
                    "batch_valid": len(filtered_batch),
                    "total_computed": total_computed,
                    "total_valid": total_valid,
                    "elapsed_seconds": int(elapsed),
                    "next_nonce": max(nonces) + n_workers if nonces else 0,
                }
                try:
                    result_queue.put(result, timeout=5)
                except queue.Full:
                    logger.warning(f"[Worker {worker_id}] Queue full, skipping batch")

            if batch_count % 10 == 0:
                logger.info(f"[Worker {worker_id}] Batch #{batch_count}: {len(filtered_batch)} valid, elapsed={int(elapsed)}s")

        logger.info(f"[Worker {worker_id}] Stopped: {batch_count} batches, {total_computed} computed, {total_valid} valid")

        # Cleanup
        compute.shutdown()

    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error: {e}", exc_info=True)
        try:
            result_queue.put({
                "worker_id": worker_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }, timeout=5)
        except:
            pass
    finally:
        ready_event.set()  # Ensure we signal even on error


class ParallelWorkerManager:
    """
    Manages multiple worker threads for parallel nonce generation.
    Uses threading instead of multiprocessing to avoid spawn issues.
    """
    def __init__(
        self,
        params: Params,
        block_hash: str,
        block_height: int,
        public_key: str,
        r_target: float,
        batch_size_per_worker: int,
        gpu_groups: List[List[str]],  # List of device lists per worker
        start_nonce: int = 0,
        max_duration: float = 420.0,
    ):
        self.params = params
        self.block_hash = block_hash
        self.block_height = block_height
        self.public_key = public_key
        self.r_target = r_target
        self.batch_size_per_worker = batch_size_per_worker
        self.gpu_groups = gpu_groups
        self.start_nonce = start_nonce
        self.max_duration = max_duration
        self.n_workers = len(gpu_groups)

        self.result_queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.workers: List[threading.Thread] = []
        self.ready_events: List[threading.Event] = []

    def start(self):
        """Start all worker threads"""
        for worker_id, devices in enumerate(self.gpu_groups):
            ready_event = threading.Event()
            self.ready_events.append(ready_event)

            worker = threading.Thread(
                target=worker_thread,
                args=(
                    worker_id,
                    self.params,
                    self.block_hash,
                    self.block_height,
                    self.public_key,
                    self.r_target,
                    self.batch_size_per_worker,
                    devices,
                    self.start_nonce,
                    self.n_workers,
                    self.result_queue,
                    self.stop_event,
                    ready_event,
                    self.max_duration,
                ),
                daemon=True,
                name=f"Worker-{worker_id}",
            )
            self.workers.append(worker)
            worker.start()
            logger.info(f"Started worker {worker_id} on devices {devices}")

    def wait_for_ready(self, timeout: float = 300.0) -> bool:
        """Wait for all workers to be ready (models loaded)"""
        start = time.time()
        for i, event in enumerate(self.ready_events):
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                logger.warning(f"Timeout waiting for worker {i}")
                return False
            if not event.wait(timeout=remaining):
                logger.warning(f"Worker {i} not ready within timeout")
                return False
        logger.info(f"All {self.n_workers} workers ready in {time.time() - start:.1f}s")
        return True

    def get_results(self, timeout: float = 0.1) -> List[Dict[str, Any]]:
        """Get all available results from workers"""
        results = []
        while True:
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results

    def stop(self):
        """Stop all workers gracefully"""
        logger.info("Stopping all workers...")
        self.stop_event.set()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not stop in time")

        logger.info("All workers stopped")

    def is_alive(self) -> bool:
        """Check if any worker is still running"""
        return any(w.is_alive() for w in self.workers)
