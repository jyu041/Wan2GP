# MMGP Phase 2B — threaded next-block prefetch

> **2026-09-22 correction:** commit `6122f089` is invalid for benchmarking.
> It prefetched CUDA allocations on MMGP's transfer stream and consumed them on
> the main stream without `Tensor.record_stream()`. The first four validation
> runs exposed silent numerical corruption: First Block Cache reported
> `Skipped Steps:7/8`, the model emitted an invalid-value cast warning, and
> transformer blocks 1-49 executed only once. The apparent ~45 s warm runtime
> was therefore not a valid speedup.
>
> Phase 2B v2 records all prefetched block and LoRA CUDA storage on the consumer
> stream before it can be unloaded/recycled. An experiment-only finite-signature
> guard now aborts if First Block Cache sees non-finite values, so the failure
> mode cannot silently turn into extra cache skipping.

Phase 2A established that MMGP 3.8.1's built-in async shuttle is stable on the
RTX 3070 profile-4.5 workload, but slower than the synchronous baseline.

Warm Phase 1 baseline (profile 4.5, asyncTransfers=False, host pin cap 0.30):

- 143.083 s
- 139.907 s
- 142.190 s
- median: **142.190 s**

Warm Phase 2A built-in async:

- 151.336 s
- 148.600 s
- 146.938 s
- median: **148.600 s**

The built-in async path therefore regressed median end-to-end time by about
4.5%. Root-block GPU compute remained ~105 s, while MMGP load-path wall time
increased.

## Hypothesis

MMGP's built-in async path prepares/prefetches the next block inside
`gpu_load_blocks()` before the current block's Python forward is entered.

For this H3 workload, preparing a ~389 MB block consumes substantial host-side
time (parameter iteration, CUDA allocation, tensor/Parameter construction and
copy enqueue). Although the DMA can overlap with GPU compute, much of that
host-side preparation happens on the critical path before current-block forward
submission.

Phase 2B moves the *next-block preparation* to a background host thread started
immediately after MMGP has loaded the current block and immediately before the
current root block executes. The background worker uses MMGP's existing transfer
stream. At the next block boundary, the main thread adopts the prefetched block
and inserts a stream event dependency rather than a device-wide synchronization.

This is still an experiment. It is intentionally limited to the main
`transformer/blocks.N` tower and does not prefetch token-refiner, text encoder,
VAE or other model towers.

## Enable it

Use this branch with Phase 1 profiling enabled:

```powershell
$env:WAN2GP_MMGP_PROFILE = "1"
$env:WAN2GP_MMGP_EXPERIMENT_THREADED_PREFETCH = "1"
$env:WAN2GP_MMGP_PROFILE_DIR = "$PWD\outputs\mmgp_profiles"

python wgp.py --perc-reserved-mem-max 0.3
```

Profile 4.5 must remain selected. Do **not** enable the Phase 2A flag
`WAN2GP_MMGP_EXPERIMENT_ASYNC45`.

Startup must show:

```text
[MMGP profiler] Phase 1 enabled.
[MMGP experiment] Phase 2B threaded next-block prefetch enabled
```

The JSON manager plan should report:

```json
"async_transfers": false,
"experimental_threaded_prefetch": true,
"experimental_prefetch_stream_lifetime_tracking": true
```

## What is measured

The normal `gpu_load_blocks` records remain present. When a block is satisfied
by the background prefetcher, its load-path wall time represents the exposed
adoption/join path at the block boundary instead of the whole background
preparation time.

The JSON additionally contains `prefetch_records`, including:

- source block;
- prefetched block;
- bytes;
- background host preparation/enqueue time;
- host time spent waiting for the background worker at adoption;
- memory samples before preparation, after preparation and after adoption.

The summary includes:

- `experimental_prefetch_calls`;
- `experimental_prefetch_host_prepare_ms`;
- `experimental_prefetch_join_wait_ms`.

The host preparation time can overlap current-block work and therefore must not
be added directly to end-to-end generation time.

## Safety

The worker modifies only the next transformer's module weights while the current
block executes. The next block is not executed until its worker has completed
host-side setup and the current CUDA stream waits on the transfer-stream ready
event.

If generation changes path or the model is unloaded while a prefetch is
pending, the profiler's unload wrapper joins the worker and restores the
prefetched module to its CPU weights.

This experiment still requires approximately one extra streamed block to exist
in VRAM while current-block execution is active. Stop after the first failure if
you see OOM, corrupted output, CUDA illegal access, or an exception from the
prefetch worker.

To roll back:

```powershell
Remove-Item Env:WAN2GP_MMGP_EXPERIMENT_THREADED_PREFETCH -ErrorAction SilentlyContinue
```

and restart Wan2GP.

## Benchmark

Use the exact fixed H3 workload:

- MiniMax H3 FL2VA Pruned PDD;
- INT8 ConvRot;
- profile 4.5;
- `--perc-reserved-mem-max 0.3`;
- 832x480;
- 124 frames;
- 8 Euler steps;
- seed 42;
- identical prompt/input/settings.

First run one cold generation. If it completes and output is visually normal,
run three identical warm generations.

Decision gate:

- keep the approach if warm median is reproducibly faster than 142.190 s;
- consider it promising at >=5% improvement (<135.1 s median);
- reject it if it is slower, unstable, corrupts output or produces OOM.
