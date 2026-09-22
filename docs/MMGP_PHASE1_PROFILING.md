# MMGP Phase 1 profiling

This experimental branch adds **measurement only** around MMGP 3.8.1. It does
not change MMGP budgets, residency selection, prefetch order, quantization,
pinning policy, or transfer synchronization behavior.

The profiler is disabled by default. When disabled, Wan2GP still uses the normal
MMGP path and the profiling helpers return immediately.

## Enable it

Set the environment variable before launching Wan2GP normally.

### PowerShell

```powershell
$env:WAN2GP_MMGP_PROFILE = "1"
# Optional; this is the default:
$env:WAN2GP_MMGP_PROFILE_DIR = "outputs\mmgp_profiles"
```

Then start Wan2GP using your normal launcher.

To return to a clean baseline process:

```powershell
Remove-Item Env:WAN2GP_MMGP_PROFILE -ErrorAction SilentlyContinue
```

Restart Wan2GP after changing the flag because the observation hooks are
installed at process startup.

Optional variables:

| Variable | Default | Purpose |
|---|---:|---|
| `WAN2GP_MMGP_PROFILE` | off | Enables Phase 1 instrumentation. |
| `WAN2GP_MMGP_PROFILE_DIR` | `outputs/mmgp_profiles` | Output directory. |
| `WAN2GP_MMGP_PROFILE_PRINT` | `1` | Print the per-generation summary. |
| `WAN2GP_MMGP_PROFILE_MAX_RECORDS` | `200000` | Safety cap for detailed in-memory records. |

## Output

Each `wan_model.generate(...)` call gets a unique profile ID and, after its
CUDA event timings have resolved, produces:

- `mmgp-profile-<id>.json` — full machine-readable data;
- `mmgp-profile-<id>-blocks.csv` — block-level aggregate metrics;
- `mmgp-profile-<id>-summary.txt` — concise human-readable summary.

Prompt text is **not** stored. Only a SHA-256 hash is recorded for run matching.

The JSON contains:

- model/generation metadata: model/config, resolution, frame count, steps, seed,
  sampler, attention mode, memory profile and window/repeat number;
- MMGP's detected residency plan and per-block weight sizes;
- per-root-block CPU-forward wall time;
- per-root-block CUDA-event compute timing where available;
- every observed `gpu_load_blocks` call, addressed bytes, current/next block,
  async state, residency state and load-path wall time;
- duration/count of `torch.cuda.synchronize()` calls while inside an MMGP
  load/unload path;
- sampled allocated/reserved VRAM and PyTorch process peak counters;
- MMGP's global pinned-RAM byte count at export.

## Measurement limitations

Phase 1 intentionally does not rewrite MMGP's nested `cpu_to_gpu()` transfer
loop.

1. `gpu_load_blocks` wall time is **not pure PCIe DMA time**. In async mode it
   includes synchronization/bookkeeping and may consume a block prefetched by
   the previous call.
2. Hidden H2D transfer duration cannot yet be isolated exactly. Phase 1 records
   bytes and synchronization waits as overlap proxies. A later trace/CUPTI lane
   can provide direct copy-engine timing.
3. CUDA root-block timing measures current-stream work. Model-specific auxiliary
   streams may not be fully represented.
4. VRAM peaks are sampled at profiler observation points, so the true
   instantaneous peak may be somewhat higher.
5. Profiling has overhead. Compare candidate implementations with the same
   profiler state, then confirm any final speedup again with profiling disabled.

These limitations are also embedded in every JSON profile.

## Recommended RTX 3070 benchmark protocol

Pick one stable workload, preferably a large MMGP-streamed model such as MiniMax
H3 or Viggle Animate, and freeze all settings.

Keep identical across runs:

- Wan2GP branch/commit;
- MMGP version (3.8.1 for this baseline);
- checkpoint and quantization;
- prompt/input media;
- seed;
- width/height and frame count/duration;
- sampler and denoising steps;
- attention backend;
- Wan2GP memory profile;
- LoRAs, accelerators and cache settings;
- background GPU workloads.

Suggested sequence:

1. Restart Wan2GP.
2. Run one warm-up generation. Keep the profile, but do not use it as the main
   steady-state result.
3. Run the exact same generation three more times.
4. Preserve all generated JSON/CSV files.
5. Repeat on an optimization branch.
6. Compare medians, not the single fastest run.

Cold model loading and steady-state generation are kept separate: the generation
wall clock begins immediately before `wan_model.generate(...)`; manager
residency/pinning information is captured when `offload.profile(...)` creates
the MMGP manager.

## Phase 1 decision gate

Use the first RTX 3070 profiles to choose the next phase:

- high MMGP synchronization/load-path time -> event-driven scheduling candidate;
- repeated stalls with poor overlap proxy -> prefetch/residency candidate;
- high addressed weight bytes with little residency -> residency or lower-bit
  transport candidate;
- low MMGP load/sync time but long generation -> compute/model/VAE work is the
  dominant bottleneck and MMGP scheduling should not be optimized first.

Do not select Phase 2 from theory alone; select it from the measured profile.
