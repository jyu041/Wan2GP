"""Opt-in Phase 1 profiler for MMGP offloading in Wan2GP.

This module is measurement-only: it does not change MMGP budgets, residency,
prefetch order, pinning, quantization, or synchronization policy. When
WAN2GP_MMGP_PROFILE is disabled, all public helpers are no-ops.
"""

from __future__ import annotations

import atexit
import csv
import functools
import hashlib
import json
import os
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_ENV_ENABLE = "WAN2GP_MMGP_PROFILE"
_ENV_DIR = "WAN2GP_MMGP_PROFILE_DIR"
_ENV_PRINT = "WAN2GP_MMGP_PROFILE_PRINT"
_ENV_MAX_RECORDS = "WAN2GP_MMGP_PROFILE_MAX_RECORDS"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def enabled() -> bool:
    return _env_bool(_ENV_ENABLE, False)


def hash_text(value: Any) -> str | None:
    """Hash reproducibility text without writing the original prompt."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


class _RuntimeProfiler:
    def __init__(self, offload_module):
        self.offload_module = offload_module
        self.output_dir = Path(os.getenv(_ENV_DIR, "outputs/mmgp_profiles"))
        self.print_summary = _env_bool(_ENV_PRINT, True)
        try:
            self.max_records = max(1000, int(os.getenv(_ENV_MAX_RECORDS, "200000")))
        except ValueError:
            self.max_records = 200000
        self.lock = threading.RLock()
        self.local = threading.local()
        self.managers: dict[int, dict[str, Any]] = {}
        self.generations: dict[str, dict[str, Any]] = {}
        self.pending_cuda: list[dict[str, Any]] = []
        self.block_records: list[dict[str, Any]] = []
        self.load_records: list[dict[str, Any]] = []
        self.sync_records: list[dict[str, Any]] = []
        self._exported_sessions: set[str] = set()
        self._closed = False

    def current_generation_id(self) -> str | None:
        return getattr(self.local, "generation_id", None)

    def _set_generation_id(self, generation_id: str | None) -> None:
        self.local.generation_id = generation_id

    def _set_mmgp_context(self, context: dict[str, Any] | None):
        previous = getattr(self.local, "mmgp_context", None)
        self.local.mmgp_context = context
        return previous

    def _cuda_available(self) -> bool:
        return bool(torch is not None and torch.cuda.is_available())

    def _memory_sample(self) -> dict[str, int | None]:
        empty = {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "process_peak_allocated_bytes": None,
            "process_peak_reserved_bytes": None,
        }
        if not self._cuda_available():
            return empty
        try:
            return {
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "process_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "process_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        except Exception:
            return empty

    @staticmethod
    def _block_entry(model_id: str, blocks_name: str | None) -> str:
        return model_id if blocks_name is None else f"{model_id}/{blocks_name}"

    def _manager_plan(self, manager) -> dict[str, Any]:
        models: dict[str, Any] = {}
        all_sizes = getattr(manager, "blocks_of_modules_sizes", {}) or {}
        preloaded = getattr(manager, "preloaded_blocks_per_model", {}) or {}
        for model_id in (getattr(manager, "models", {}) or {}):
            resident_blocks = list(preloaded.get(model_id, []) or [])
            base_bytes = int(all_sizes.get(model_id, 0) or 0)
            resident_block_bytes = 0
            block_sizes: dict[str, int] = {}
            prefix = f"{model_id}/"
            for entry_name, size in all_sizes.items():
                if entry_name.startswith(prefix):
                    short = entry_name[len(prefix):]
                    block_sizes[short] = int(size or 0)
                    if short in resident_blocks:
                        resident_block_bytes += int(size or 0)
            models[str(model_id)] = {
                "base_resident_bytes": base_bytes,
                "preloaded_blocks": resident_blocks,
                "preloaded_block_bytes": resident_block_bytes,
                "estimated_persistent_weight_bytes": base_bytes + resident_block_bytes,
                "block_sizes": block_sizes,
            }
        return {
            "async_transfers": bool(getattr(manager, "async_transfers", False)),
            "device_mem_capacity": int(getattr(manager, "device_mem_capacity", 0) or 0),
            "models": models,
        }

    def attach_manager(self, manager) -> None:
        with self.lock:
            self.managers[id(manager)] = {
                "manager": manager,
                "attached_at": _utc_now(),
                "plan": self._manager_plan(manager),
            }

    def start_generation(self, manager, metadata: dict[str, Any] | None) -> str:
        generation_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        now_ns = time.perf_counter_ns()
        manager_id = id(manager) if manager is not None else None
        with self.lock:
            self._resolve_cuda_events()
            if manager is not None and manager_id not in self.managers:
                self.attach_manager(manager)
            self.generations[generation_id] = {
                "id": generation_id,
                "started_at": _utc_now(),
                "start_ns": now_ns,
                "end_ns": None,
                "duration_ms": None,
                "status": "running",
                "error_type": None,
                "metadata": _json_safe(metadata or {}),
                "manager_id": manager_id,
                "memory_start": self._memory_sample(),
                "memory_end": None,
                "observed_peak_allocated_bytes": 0,
                "observed_peak_reserved_bytes": 0,
            }
        self._set_generation_id(generation_id)
        return generation_id

    def finish_generation(self, generation_id: str | None, status="ok", error=None) -> None:
        if not generation_id:
            return
        end_ns = time.perf_counter_ns()
        with self.lock:
            gen = self.generations.get(generation_id)
            if gen is None or gen.get("end_ns") is not None:
                return
            gen["end_ns"] = end_ns
            gen["duration_ms"] = (end_ns - gen["start_ns"]) / 1_000_000.0
            gen["status"] = status
            gen["error_type"] = None if error is None else type(error).__name__
            gen["memory_end"] = self._memory_sample()
            self._resolve_cuda_events()
        if self.current_generation_id() == generation_id:
            self._set_generation_id(None)

    def block_forward_start(self, manager, model_id: str, blocks_name: str):
        generation_id = self.current_generation_id()
        if generation_id is None:
            return None
        token = {
            "generation_id": generation_id,
            "manager_id": id(manager),
            "model_id": str(model_id),
            "block_name": str(blocks_name),
            "cpu_start_ns": time.perf_counter_ns(),
            "start_event": None,
            "end_event": None,
            "memory_start": self._memory_sample(),
        }
        if self._cuda_available():
            try:
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream())
                token["start_event"] = event
            except Exception:
                pass
        return token

    def block_forward_end(self, token) -> None:
        if token is None:
            return
        token["cpu_end_ns"] = time.perf_counter_ns()
        token["cpu_forward_ms"] = (token["cpu_end_ns"] - token["cpu_start_ns"]) / 1_000_000.0
        token["memory_end"] = self._memory_sample()
        if token.get("start_event") is not None and self._cuda_available():
            try:
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream())
                token["end_event"] = event
            except Exception:
                token["end_event"] = None
        with self.lock:
            if len(self.block_records) + len(self.pending_cuda) >= self.max_records:
                return
            if token.get("start_event") is not None and token.get("end_event") is not None:
                self.pending_cuda.append(token)
            else:
                token["gpu_compute_ms"] = None
                token["gpu_timing_pending"] = False
                self.block_records.append(self._strip_events(token))
            self._update_generation_peak(token["generation_id"], token.get("memory_end"))
            self._resolve_cuda_events()

    @staticmethod
    def _strip_events(record: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in record.items() if k not in {"start_event", "end_event"}}

    def _resolve_cuda_events(self) -> None:
        if not self.pending_cuda:
            return
        remaining = []
        for record in self.pending_cuda:
            end_event = record.get("end_event")
            try:
                ready = bool(end_event is not None and end_event.query())
            except Exception:
                ready = False
            if not ready:
                remaining.append(record)
                continue
            try:
                record["gpu_compute_ms"] = float(record["start_event"].elapsed_time(end_event))
            except Exception:
                record["gpu_compute_ms"] = None
            record["gpu_timing_pending"] = False
            self.block_records.append(self._strip_events(record))
        self.pending_cuda = remaining

    def before_gpu_load(self, manager, model_id, blocks_name, preload) -> dict[str, Any]:
        model_key = str(model_id)
        block_key = None if blocks_name is None else str(blocks_name)
        entry = self._block_entry(model_key, block_key)
        sizes = getattr(manager, "blocks_of_modules_sizes", {}) or {}
        next_map = getattr(manager, "next_blocks_names", {}) or {}
        context = {
            "kind": "gpu_load_blocks",
            "generation_id": self.current_generation_id(),
            "manager_id": id(manager),
            "model_id": model_key,
            "block_name": block_key,
            "entry_name": entry,
            "preload": bool(preload),
            "bytes": int(sizes.get(entry, 0) or 0),
            "loaded_block_before": _json_safe((getattr(manager, "loaded_blocks", {}) or {}).get(model_id)),
            "next_entry": _json_safe(next_map.get(entry)),
            "async_transfers": bool(getattr(manager, "async_transfers", False)),
            "started_ns": time.perf_counter_ns(),
            "memory_before": self._memory_sample(),
        }
        context["previous_context"] = self._set_mmgp_context(context)
        return context

    def after_gpu_load(self, context: dict[str, Any], manager) -> None:
        ended_ns = time.perf_counter_ns()
        self._set_mmgp_context(context.pop("previous_context", None))
        context["ended_ns"] = ended_ns
        context["wall_ms"] = (ended_ns - context["started_ns"]) / 1_000_000.0
        context["memory_after"] = self._memory_sample()
        original_model_id = context["model_id"]
        context["loaded_block_after"] = _json_safe(
            (getattr(manager, "loaded_blocks", {}) or {}).get(original_model_id)
        )
        if context["wall_ms"] > 0 and context["bytes"] > 0:
            context["load_path_effective_gbps"] = (
                (context["bytes"] / 1e9) / (context["wall_ms"] / 1000.0)
            )
        else:
            context["load_path_effective_gbps"] = None
        with self.lock:
            if len(self.load_records) < self.max_records:
                self.load_records.append(_json_safe(context))
            self._update_generation_peak(context.get("generation_id"), context.get("memory_after"))
            self._resolve_cuda_events()

    def on_synchronize(self, duration_ms: float) -> None:
        context = getattr(self.local, "mmgp_context", None)
        if context is None:
            return
        record = {
            "generation_id": context.get("generation_id"),
            "manager_id": context.get("manager_id"),
            "kind": context.get("kind"),
            "model_id": context.get("model_id"),
            "block_name": context.get("block_name"),
            "entry_name": context.get("entry_name"),
            "preload": context.get("preload"),
            "duration_ms": duration_ms,
            "timestamp": _utc_now(),
        }
        with self.lock:
            if len(self.sync_records) < self.max_records:
                self.sync_records.append(record)

    def _update_generation_peak(self, generation_id, memory) -> None:
        if generation_id is None or memory is None:
            return
        gen = self.generations.get(generation_id)
        if gen is None:
            return
        allocated = memory.get("allocated_bytes") or 0
        reserved = memory.get("reserved_bytes") or 0
        gen["observed_peak_allocated_bytes"] = max(gen["observed_peak_allocated_bytes"], int(allocated))
        gen["observed_peak_reserved_bytes"] = max(gen["observed_peak_reserved_bytes"], int(reserved))

    def on_manager_unload(self, manager) -> None:
        with self.lock:
            self._resolve_cuda_events()
            finished = [gid for gid, gen in self.generations.items() if gen.get("end_ns") is not None]
        for generation_id in finished:
            self.export_generation(generation_id)

    @staticmethod
    def _records_for(records, generation_id):
        return [r for r in records if r.get("generation_id") == generation_id]

    def _aggregate_blocks(self, generation_id: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        blocks = self._records_for(self.block_records, generation_id)
        loads = self._records_for(self.load_records, generation_id)
        syncs = self._records_for(self.sync_records, generation_id)

        def group(model_id, block_name):
            key = (model_id or "", block_name or "<base>")
            if key not in grouped:
                grouped[key] = {
                    "model_id": key[0], "block_name": key[1],
                    "compute_calls": 0, "gpu_compute_ms": [], "cpu_forward_ms": [],
                    "load_calls": 0, "load_wall_ms": [], "load_bytes": 0,
                    "sync_calls": 0, "sync_ms": [],
                }
            return grouped[key]

        for r in blocks:
            g = group(r.get("model_id"), r.get("block_name"))
            g["compute_calls"] += 1
            if r.get("gpu_compute_ms") is not None:
                g["gpu_compute_ms"].append(float(r["gpu_compute_ms"]))
            if r.get("cpu_forward_ms") is not None:
                g["cpu_forward_ms"].append(float(r["cpu_forward_ms"]))
        for r in loads:
            g = group(r.get("model_id"), r.get("block_name"))
            g["load_calls"] += 1
            if r.get("wall_ms") is not None:
                g["load_wall_ms"].append(float(r["wall_ms"]))
            g["load_bytes"] += int(r.get("bytes") or 0)
        for r in syncs:
            g = group(r.get("model_id"), r.get("block_name"))
            g["sync_calls"] += 1
            if r.get("duration_ms") is not None:
                g["sync_ms"].append(float(r["duration_ms"]))

        rows = []
        for g in grouped.values():
            gpu, cpu = g.pop("gpu_compute_ms"), g.pop("cpu_forward_ms")
            load, sync = g.pop("load_wall_ms"), g.pop("sync_ms")
            g.update({
                "gpu_compute_total_ms": sum(gpu),
                "gpu_compute_median_ms": statistics.median(gpu) if gpu else None,
                "gpu_compute_p95_ms": _percentile(gpu, 0.95),
                "cpu_forward_total_ms": sum(cpu),
                "load_wall_total_ms": sum(load),
                "load_wall_median_ms": statistics.median(load) if load else None,
                "sync_total_ms": sum(sync),
                "sync_median_ms": statistics.median(sync) if sync else None,
            })
            rows.append(g)
        return sorted(rows, key=lambda row: (row["model_id"], row["block_name"]))

    def export_generation(self, generation_id: str, force: bool = False) -> None:
        with self.lock:
            gen = self.generations.get(generation_id)
            if gen is None or gen.get("end_ns") is None:
                return
            self._resolve_cuda_events()
            unresolved = sum(1 for r in self.pending_cuda if r.get("generation_id") == generation_id)
            if generation_id in self._exported_sessions and not force:
                return
            if unresolved and not force:
                return
            blocks = self._records_for(self.block_records, generation_id)
            loads = self._records_for(self.load_records, generation_id)
            syncs = self._records_for(self.sync_records, generation_id)
            aggregate = self._aggregate_blocks(generation_id)
            manager_info = self.managers.get(gen.get("manager_id"), {})
            pinned_bytes = int(getattr(self.offload_module, "total_pinned_bytes", 0) or 0)
            summary = {
                "generation_id": generation_id,
                "status": gen["status"],
                "duration_ms": gen["duration_ms"],
                "metadata": gen["metadata"],
                "block_compute_calls": len(blocks),
                "resolved_gpu_compute_ms": sum(float(r.get("gpu_compute_ms") or 0) for r in blocks),
                "pending_gpu_compute_events": unresolved,
                "gpu_load_calls": len(loads),
                "gpu_load_path_wall_ms": sum(float(r.get("wall_ms") or 0) for r in loads),
                "streamed_or_loaded_bytes": sum(int(r.get("bytes") or 0) for r in loads),
                "mmgp_sync_calls": len(syncs),
                "mmgp_sync_wall_ms": sum(float(r.get("duration_ms") or 0) for r in syncs),
                "observed_peak_allocated_bytes": gen["observed_peak_allocated_bytes"],
                "observed_peak_reserved_bytes": gen["observed_peak_reserved_bytes"],
                "mmgp_total_pinned_bytes_at_export": pinned_bytes,
                "manager_plan": manager_info.get("plan"),
                "limitations": [
                    "gpu_load_path_wall_ms is MMGP load-path wall time, not pure H2D DMA time.",
                    "MMGP async hidden H2D duration cannot be isolated without tracing inside cpu_to_gpu/CUPTI; Phase 1 records bytes, synchronization waits and overlap proxies.",
                    "CUDA block timing covers the current stream; auxiliary-stream work may not be fully represented.",
                    "Observed VRAM peaks are sampled and can be below the true instantaneous peak.",
                ],
            }
            payload = {
                "schema_version": 1,
                "profiler": "Wan2GP MMGP Phase 1",
                "created_at": _utc_now(),
                "generation": _json_safe(gen),
                "summary": _json_safe(summary),
                "block_aggregate": _json_safe(aggregate),
                "block_records": _json_safe(blocks),
                "load_records": _json_safe(loads),
                "sync_records": _json_safe(syncs),
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"mmgp-profile-{generation_id}"
        json_path = self.output_dir / f"{stem}.json"
        csv_path = self.output_dir / f"{stem}-blocks.csv"
        txt_path = self.output_dir / f"{stem}-summary.txt"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if aggregate:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
                writer.writeheader()
                writer.writerows(aggregate)
        txt = self._human_summary(summary, json_path, csv_path if aggregate else None)
        txt_path.write_text(txt, encoding="utf-8")
        with self.lock:
            self._exported_sessions.add(generation_id)
        if self.print_summary:
            print(txt)

    @staticmethod
    def _human_summary(summary, json_path, csv_path) -> str:
        duration_s = (summary.get("duration_ms") or 0.0) / 1000.0
        loaded_gb = (summary.get("streamed_or_loaded_bytes") or 0) / 1e9
        sync_s = (summary.get("mmgp_sync_wall_ms") or 0.0) / 1000.0
        compute_s = (summary.get("resolved_gpu_compute_ms") or 0.0) / 1000.0
        peak_vram = (summary.get("observed_peak_reserved_bytes") or 0) / (1024 ** 3)
        pinned = (summary.get("mmgp_total_pinned_bytes_at_export") or 0) / (1024 ** 3)
        lines = [
            "[MMGP profiler] Phase 1 generation summary",
            f"  id: {summary['generation_id']}",
            f"  status: {summary['status']}",
            f"  model.generate wall time: {duration_s:.3f} s",
            f"  resolved root-block GPU compute: {compute_s:.3f} s across {summary['block_compute_calls']} calls",
            f"  MMGP gpu_load_blocks calls: {summary['gpu_load_calls']} ({loaded_gb:.3f} GB addressed)",
            f"  MMGP load-path wall time: {(summary['gpu_load_path_wall_ms'] or 0)/1000.0:.3f} s",
            f"  MMGP-attributed cuda.synchronize: {summary['mmgp_sync_calls']} calls / {sync_s:.3f} s",
            f"  observed reserved-VRAM peak: {peak_vram:.3f} GiB",
            f"  MMGP pinned RAM at export: {pinned:.3f} GiB",
            f"  pending CUDA event timings: {summary['pending_gpu_compute_events']}",
            f"  JSON: {json_path}",
        ]
        if csv_path is not None:
            lines.append(f"  CSV: {csv_path}")
        return "\n".join(lines)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self.lock:
            self._resolve_cuda_events()
            generation_ids = [gid for gid, gen in self.generations.items() if gen.get("end_ns") is not None]
        for generation_id in generation_ids:
            try:
                self.export_generation(generation_id, force=True)
            except Exception:
                pass


_RUNTIME: _RuntimeProfiler | None = None
_INSTALL_LOCK = threading.Lock()
_ORIGINALS: dict[str, Any] = {}


def _wrap_root_block_forward(manager, model_id: str, blocks_name: str, previous_method):
    if getattr(previous_method, "_wan2gp_mmgp_profile_wrapped", False):
        return previous_method

    @functools.wraps(previous_method)
    def wrapped(*args, **kwargs):
        runtime = _RUNTIME
        if runtime is None:
            return previous_method(*args, **kwargs)
        token = runtime.block_forward_start(manager, model_id, blocks_name)
        try:
            return previous_method(*args, **kwargs)
        finally:
            runtime.block_forward_end(token)

    wrapped._wan2gp_mmgp_profile_wrapped = True
    return wrapped


def install(offload_module) -> bool:
    """Install measurement hooks around the imported mmgp.offload module."""
    global _RUNTIME
    if not enabled():
        return False
    if torch is None or not hasattr(offload_module, "offload"):
        print("[MMGP profiler] Unable to install: MMGP offload class not found.")
        return False

    with _INSTALL_LOCK:
        if _RUNTIME is not None:
            return True
        runtime = _RuntimeProfiler(offload_module)
        manager_cls = offload_module.offload

        original_profile = offload_module.profile
        original_gpu_load_blocks = manager_cls.gpu_load_blocks
        original_unload_all = manager_cls.unload_all
        original_hook_default = manager_cls.hook_check_load_into_GPU_if_needed_default
        original_hook_compiled = manager_cls.hook_check_load_into_GPU_if_needed
        original_synchronize = torch.cuda.synchronize
        _ORIGINALS.update({
            "profile": original_profile,
            "gpu_load_blocks": original_gpu_load_blocks,
            "unload_all": original_unload_all,
            "hook_default": original_hook_default,
            "hook_compiled": original_hook_compiled,
            "cuda_synchronize": original_synchronize,
        })

        @functools.wraps(original_profile)
        def profile_wrapper(*args, **kwargs):
            manager = original_profile(*args, **kwargs)
            runtime.attach_manager(manager)
            return manager

        @functools.wraps(original_gpu_load_blocks)
        def gpu_load_blocks_wrapper(manager, model_id, blocks_name, preload=False):
            context = runtime.before_gpu_load(manager, model_id, blocks_name, preload)
            try:
                return original_gpu_load_blocks(manager, model_id, blocks_name, preload)
            finally:
                runtime.after_gpu_load(context, manager)

        @functools.wraps(original_unload_all)
        def unload_all_wrapper(manager, *args, **kwargs):
            previous = runtime._set_mmgp_context({
                "kind": "unload_all",
                "generation_id": runtime.current_generation_id(),
                "manager_id": id(manager),
                "model_id": None, "block_name": None, "entry_name": None, "preload": False,
            })
            try:
                return original_unload_all(manager, *args, **kwargs)
            finally:
                runtime._set_mmgp_context(previous)
                runtime.on_manager_unload(manager)

        @functools.wraps(original_hook_default)
        def hook_default_wrapper(manager, target_module, model, model_id, blocks_name, previous_method, context):
            if blocks_name is not None and context == blocks_name:
                previous_method = _wrap_root_block_forward(manager, model_id, blocks_name, previous_method)
            return original_hook_default(manager, target_module, model, model_id, blocks_name, previous_method, context)

        @functools.wraps(original_hook_compiled)
        def hook_compiled_wrapper(manager, target_module, model, model_id, blocks_name, previous_method, context):
            if blocks_name is not None and context == blocks_name:
                previous_method = _wrap_root_block_forward(manager, model_id, blocks_name, previous_method)
            return original_hook_compiled(manager, target_module, model, model_id, blocks_name, previous_method, context)

        @functools.wraps(original_synchronize)
        def synchronize_wrapper(*args, **kwargs):
            start_ns = time.perf_counter_ns()
            try:
                return original_synchronize(*args, **kwargs)
            finally:
                runtime.on_synchronize((time.perf_counter_ns() - start_ns) / 1_000_000.0)

        offload_module.profile = profile_wrapper
        manager_cls.gpu_load_blocks = gpu_load_blocks_wrapper
        manager_cls.unload_all = unload_all_wrapper
        manager_cls.hook_check_load_into_GPU_if_needed_default = hook_default_wrapper
        manager_cls.hook_check_load_into_GPU_if_needed = hook_compiled_wrapper
        torch.cuda.synchronize = synchronize_wrapper

        _RUNTIME = runtime
        atexit.register(runtime.close)
        print(f"[MMGP profiler] Phase 1 enabled. Output directory: {runtime.output_dir}")
        return True


def start_generation(manager, **metadata) -> str | None:
    runtime = _RUNTIME
    if runtime is None:
        return None
    try:
        return runtime.start_generation(manager, metadata)
    except Exception as exc:
        print(f"[MMGP profiler] Failed to start generation profile: {exc}")
        return None


def finish_generation(token: str | None, *, status: str = "ok", error=None) -> None:
    runtime = _RUNTIME
    if runtime is None or token is None:
        return
    try:
        runtime.finish_generation(token, status=status, error=error)
    except Exception as exc:
        print(f"[MMGP profiler] Failed to finish generation profile: {exc}")


__all__ = ["enabled", "finish_generation", "hash_text", "install", "start_generation"]
