
from dataclasses import dataclass
import torch
import gc
from pathlib import Path

@dataclass
class ProfileResult:
    name: str
    avg_step_time_s: float
    peak_memory_mb: float | None
    self_cuda_time_per_step_ms: float | None = None


@torch.no_grad()
def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _start_cuda_memory_history(max_entries: int) -> bool:
    if not torch.cuda.is_available():
        return False
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    return True


def _stop_cuda_memory_history() -> None:
    if torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(enabled=None)


def _dump_cuda_memory_snapshot(snapshot_path: Path) -> bool:
    if not torch.cuda.is_available():
        return False
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        return True
    except Exception as exc:
        print(f"Failed to dump CUDA memory snapshot to {snapshot_path}: {exc}")
        return False


def _cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()