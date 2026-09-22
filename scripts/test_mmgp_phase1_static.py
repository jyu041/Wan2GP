"""Static checks for the experimental MMGP Phase 1 instrumentation.

These checks intentionally avoid importing Wan2GP or MMGP, so they can run in
GitHub Actions without installing the GPU stack.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "shared" / "utils" / "mmgp_profiler.py"
WGP = ROOT / "wgp.py"


def parse(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    return source


def main() -> None:
    profiler = parse(PROFILER)
    wgp = parse(WGP)

    profiler_markers = (
        'WAN2GP_MMGP_PROFILE',
        "def active()",
        "def install(",
        "def start_generation(",
        "def finish_generation(",
        "def wrap_callback(",
        "torch.cuda.Event(enable_timing=True)",
        "def synchronize_wrapper",
        "gpu_load_blocks_wrapper",
        "step_records",
        "profile_setup_ms",
        "-blocks.csv",
        "-summary.txt",
    )
    for marker in profiler_markers:
        assert marker in profiler, f"missing profiler marker: {marker}"

    wgp_markers = (
        "mmgp_profiler.install(offload)",
        "if mmgp_profiler.active():",
        "mmgp_profiler.start_generation(",
        "mmgp_profiler.wrap_callback(",
        'mmgp_profiler.finish_generation(_mmgp_profile_token, status="ok")',
        'mmgp_profiler.finish_generation(_mmgp_profile_token, status="error", error=e)',
    )
    for marker in wgp_markers:
        assert marker in wgp, f"missing Wan2GP integration marker: {marker}"

    phase2b_markers = (
        'WAN2GP_MMGP_EXPERIMENT_THREADED_PREFETCH',
        'def start_threaded_prefetch(',
        'def consume_threaded_prefetch(',
        '[MMGP experiment] Phase 2B threaded next-block prefetch enabled',
        '"prefetch_records": _json_safe(prefetches)',
        'experimental_prefetch_stream_lifetime_tracking',
        'record_stream(stream)',
    )
    for marker in phase2b_markers:
        assert marker in profiler, f"missing Phase 2B marker: {marker}"

    first_block_cache = (root / "models" / "minimax_h3" / "first_block_cache.py").read_text(encoding="utf-8")
    assert "Phase 2B detected a non-finite First Block Cache signature" in first_block_cache

    # The experimental integration must keep the upstream MMGP pin intact.
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "mmgp==3.8.1" in requirements

    print("MMGP Phase 1/2B static checks passed.")


if __name__ == "__main__":
    main()
