# Gonka Nonce Calculator - Runpod Serverless

[![Runpod](https://api.runpod.io/badge/vedenij/runpod)](https://console.runpod.io/hub/vedenij/runpod)

Serverless nonce calculator for Gonka decentralized network using LLaMA model inference. This service calculates proof-of-compute nonces for small nodes during Random Confirmation PoC phases.

## Overview

This is a Runpod serverless implementation that:
- Accepts nonce calculation requests from small nodes
- Uses deterministic LLaMA model inference to compute distances
- Filters results by target threshold (r_target)
- Returns valid nonces that pass the threshold

## Architecture

```
handler.py          # Runpod serverless entry point
src/
  models/           # LLaMA model architecture
    llama31.py      # Transformer implementation
    utils.py        # Model utilities
  compute/          # Compute engine
    compute.py      # Simplified synchronous compute
    model_init.py   # Model initialization with multi-GPU
    utils.py        # Stats and utilities
  random.py         # Deterministic random generation
  random_pool_optimized.py  # Fast weight initialization
  data.py           # ProofBatch data structures
  common/           # Common utilities
    logger.py       # Logging setup
```

## API Contract

### Input Format

```json
{
  "input": {
    "block_hash": "string (hex)",
    "block_height": 123,
    "public_key": "string",
    "nonces": [1, 2, 3, ...],
    "r_target": 2.0,
    "params": {  // optional, defaults provided
      "dim": 4096,
      "n_layers": 32,
      "n_heads": 32,
      "n_kv_heads": 8,
      "vocab_size": 128256,
      "seq_len": 1024,
      "ffn_dim_multiplier": 1.3,
      "multiple_of": 1024,
      "norm_eps": 1e-05,
      "rope_theta": 500000.0
    },
    "devices": ["cuda:0"]  // optional
  }
}
```

### Output Format

```json
{
  "public_key": "string",
  "block_hash": "string",
  "block_height": 123,
  "nonces": [1, 3, 5],  // only valid nonces
  "dist": [1.2, 1.5, 1.8],  // corresponding distances
  "node_id": 0,
  "total_computed": 10,
  "total_valid": 3
}
```

### Error Format

```json
{
  "error": "error message",
  "error_type": "ExceptionType"
}
```

## Key Features

### Model Caching
- Model is loaded once per block_hash and reused across requests
- Significant performance improvement for warm requests
- Different block_hash triggers model reinitialization

### Synchronous Processing
- Simplified from original multiprocessing design
- Suitable for serverless single-request pattern
- Clean, predictable execution flow

### GPU Support
- Multi-GPU distribution via Accelerate
- Automatic device mapping and load balancing
- Configurable device list

## Deployment

### Build Docker Image

```bash
docker build -t gonka-nonce-calculator .
```

### Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run handler with test input
python -c "
import json
from handler import handler

with open('test_input.json') as f:
    event = json.load(f)

result = handler(event)
print(json.dumps(result, indent=2))
"
```

### Deploy to Runpod

1. Build and push Docker image to registry:
```bash
docker tag gonka-nonce-calculator:latest your-registry/gonka-nonce-calculator:latest
docker push your-registry/gonka-nonce-calculator:latest
```

2. Create serverless endpoint on Runpod:
- Go to Runpod Serverless
- Create new endpoint
- Use your Docker image
- Configure GPU requirements (24GB+ VRAM recommended)
- Set timeout to 600s

3. Test the endpoint:
```bash
curl -X POST https://your-endpoint.runpod.ai/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @test_input.json
```

## Performance Considerations

### Model Initialization
- First request (cold start): ~30-60 seconds for model initialization
- Subsequent requests (warm): ~1-5 seconds depending on batch size
- Different block_hash requires reinitialization

### Batch Size
- Recommended: 16-256 nonces per request
- Larger batches improve GPU utilization
- Balance between latency and throughput

### Memory Requirements
- Base model: ~18B parameters
- float16: ~36GB
- Recommended: GPUs with 24GB+ VRAM (A6000, A100, etc.)

## Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (default: INFO)
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocator settings

### Model Parameters

Default parameters match LLaMA 3.1 8B architecture:
- dim: 4096
- n_layers: 32
- n_heads: 32
- vocab_size: 128256

Can be overridden per request via `params` field.

## Differences from Original

This serverless version differs from the original implementation:

1. **Synchronous Processing**: No ThreadPoolExecutor, no async batch preparation
2. **No Caching Between Requests**: Removed next_batch_future prefetching
3. **Stateless**: No persistent state between different requests
4. **Simplified API**: Direct input/output, no FastAPI/HTTP layers
5. **Model Reuse**: Global model instance reused for same block_hash

## Troubleshooting

### Out of Memory
- Reduce batch size in nonces array
- Use fewer/smaller GPUs in devices list
- Check PYTORCH_CUDA_ALLOC_CONF settings

### Slow Performance
- Check if model is being reinitialized (different block_hash)
- Increase batch size for better GPU utilization
- Verify GPU is being used (check logs)

### Model Initialization Errors
- Verify GPU has sufficient VRAM (24GB+ recommended)
- Check CUDA compatibility
- Review device list configuration

## License

Same as parent Gonka project.
