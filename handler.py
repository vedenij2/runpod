import os
import sys
import time
import logging
from typing import Dict, Any

import runpod

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pow.compute.compute import Compute
from pow.models.utils import Params
from pow.random import get_target

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global compute instance (loaded once, reused across requests)
COMPUTE = None
CURRENT_BLOCK_HASH = None

# Maximum job duration: 7 minutes
MAX_JOB_DURATION = 7 * 60


def initialize_compute(
    block_hash: str,
    block_height: int,
    params_dict: Dict[str, Any],
    devices: list,
) -> Compute:
    """Initialize the compute model for a specific block hash"""
    global COMPUTE, CURRENT_BLOCK_HASH

    # If compute already initialized for this block_hash, reuse it
    if COMPUTE is not None and CURRENT_BLOCK_HASH == block_hash:
        logger.info(f"Reusing existing compute for block_hash={block_hash}")
        return COMPUTE

    logger.info(f"Initializing compute for block_hash={block_hash}")

    params = Params(**params_dict)

    # Create compute instance
    COMPUTE = Compute(
        params=params,
        block_hash=block_hash,
        block_height=block_height,
        public_key="",
        r_target=0.0,
        devices=devices,
        node_id=0,
    )

    CURRENT_BLOCK_HASH = block_hash
    logger.info("Compute initialized successfully")

    return COMPUTE


def handler(event: Dict[str, Any]):
    """
    Streaming nonce generator.

    Stops when:
    1. Client calls POST /cancel/{job_id}
    2. Timeout after 7 minutes (MAX_JOB_DURATION)

    Input from client (ALL REQUIRED):
    {
        "block_hash": str,
        "block_height": int,
        "public_key": str,
        "r_target": float,
        "batch_size": int,
        "start_nonce": int,
        "params": dict,
        "devices": list
    }

    Yields:
    {
        "nonces": [...],
        "dist": [...],
        "batch_number": int,
        "next_nonce": int,
        "elapsed_seconds": int,
        ...
    }
    """
    batch_count = 0
    total_computed = 0
    total_valid = 0

    try:
        input_data = event.get("input", {})

        # Get ALL parameters from client - NO DEFAULTS
        block_hash = input_data["block_hash"]
        block_height = input_data["block_height"]
        public_key = input_data["public_key"]
        r_target = input_data["r_target"]
        batch_size = input_data["batch_size"]
        start_nonce = input_data["start_nonce"]
        params_dict = input_data["params"]

        # Auto-detect all available GPUs (ignore client's devices parameter)
        # Import torch here (not at module level) to avoid CUDA init issues in Runpod
        import torch
        gpu_count = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(gpu_count)]

        logger.info(f"START: block={block_height}, batch_size={batch_size}, start={start_nonce}, gpus={gpu_count}")

        # Initialize compute
        compute = initialize_compute(
            block_hash=block_hash,
            block_height=block_height,
            params_dict=params_dict,
            devices=devices,
        )

        target = get_target(block_hash, compute.params.vocab_size)

        current_nonce = start_nonce
        start_time = time.time()

        # Streaming until timeout or cancel
        while True:
            # Check 7-minute timeout
            elapsed = time.time() - start_time
            if elapsed > MAX_JOB_DURATION:
                logger.info(f"TIMEOUT: {elapsed:.0f}s exceeded {MAX_JOB_DURATION}s limit")
                logger.info(f"STOPPED: {batch_count} batches, {total_computed} computed, {total_valid} valid")
                return

            nonces = list(range(current_nonce, current_nonce + batch_size))

            proof_batch = compute(
                nonces=nonces,
                public_key=public_key,
                target=target,
            )

            filtered_batch = proof_batch.sub_batch(r_target)

            batch_count += 1
            total_computed += len(proof_batch)
            total_valid += len(filtered_batch)

            logger.info(f"Batch #{batch_count}: {len(filtered_batch)} valid, elapsed={int(elapsed)}s")

            yield {
                "public_key": filtered_batch.public_key,
                "block_hash": filtered_batch.block_hash,
                "block_height": filtered_batch.block_height,
                "nonces": filtered_batch.nonces,
                "dist": filtered_batch.dist,
                "node_id": filtered_batch.node_id,
                "batch_number": batch_count,
                "batch_computed": len(proof_batch),
                "batch_valid": len(filtered_batch),
                "total_computed": total_computed,
                "total_valid": total_valid,
                "next_nonce": current_nonce + batch_size,
                "elapsed_seconds": int(elapsed),
            }

            current_nonce += batch_size

    except GeneratorExit:
        logger.info(f"CANCELLED: {batch_count} batches, {total_computed} computed, {total_valid} valid")
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        yield {
            "error": str(e),
            "error_type": type(e).__name__,
        }


# Start serverless handler with streaming support
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
