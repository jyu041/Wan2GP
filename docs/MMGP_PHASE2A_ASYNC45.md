# MMGP Phase 2A — profile 4.5 async shuttle experiment

This branch tests one narrow hypothesis from the Phase 1 RTX 3070 profiles:

> Can MMGP's existing one-block-ahead async shuttle reduce steady-state H3
> generation time on profile 4.5 without exceeding the 8 GB VRAM budget?

No MMGP algorithm is rewritten in this phase. Wan2GP simply permits the existing
MMGP `asyncTransfers` path to be enabled for profile 4.5 behind an environment
flag.

## Baseline

The fixed Phase 1 workload on the RTX 3070 8 GB / 48 GB RAM machine is:

- MiniMax H3 FL2VA Pruned PDD;
- INT8 ConvRot transformer;
- profile 4.5;
- `--perc-reserved-mem-max 0.3`;
- 832x480;
- 124 frames;
- 8 Euler steps;
- seed 42;
- identical prompt/input/configuration.

Warm steady-state generation times:

- 143.083 s
- 139.907 s
- 142.190 s

Median baseline: **142.190 s**.

The corresponding warm load-path totals were approximately 40.96–42.99 s while
root-block GPU compute stayed around 105 s.

## Enable Phase 2A

Keep Phase 1 profiling enabled and launch Wan2GP with the same 0.30 host-memory
cap:

```powershell
$env:WAN2GP_MMGP_PROFILE = "1"
$env:WAN2GP_MMGP_EXPERIMENT_ASYNC45 = "1"

python wgp.py --perc-reserved-mem-max 0.3
```

At model setup the console must show:

```text
[MMGP experiment] Phase 2A: enabling MMGP asyncTransfers for profile 4.5
```

The generated profile JSON should report:

```json
"async_transfers": true
```

## Safety / rollback

This is experimental. MMGP's existing async path keeps the current transformer
block resident while prefetching the next block, so peak VRAM can increase by
roughly one streamed block plus allocator overhead.

If the first generation OOMs or becomes unstable:

1. Stop Wan2GP.
2. Remove the experiment variable:
   ```powershell
   Remove-Item Env:WAN2GP_MMGP_EXPERIMENT_ASYNC45 -ErrorAction SilentlyContinue
   ```
3. Restart with the same `--perc-reserved-mem-max 0.3`.

With the environment variable absent, profile 4.5 retains its original
`asyncTransfers=False` behavior.

## Benchmark protocol

Do not change any generation setting relative to the Phase 1 baseline.

Run:

1. one cold generation;
2. three identical warm generations.

Preserve all four JSON profiles.

Primary decision metrics:

- median warm end-to-end generation time;
- median warm root-block GPU compute time;
- MMGP load-path wall time;
- observed/recorded VRAM peaks;
- OOM/stability behavior;
- block/load call counts and addressed bytes.

Decision gate:

- **Keep** the direction if the warm median improves reproducibly by at least
  ~5% without OOM or material VRAM instability.
- **Reject** the built-in two-block shuttle for profile 4.5 if it OOMs or gives
  negligible improvement.
- If rejected because of memory pressure, the next experiment should be a
  lower-memory custom prefetch design rather than forcing the existing shuttle.
