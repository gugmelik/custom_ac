import argparse
import time
from pathlib import Path
from typing import Iterable, Literal
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from src.transformer import NNTransformerLM, TransformerLM
from src.utilsf import _cleanup_cuda, ProfileResult, _measure_peak_memory_mb, _start_cuda_memory_history, _stop_cuda_memory_history, _dump_cuda_memory_snapshot, _sync_if_cuda

ScenarioName = Literal[
    "all",
    "baseline",
    "custom_checkpoint",
    "torch_checkpoint",
    "nn_transformer",
]

SCENARIO_CHOICES = ["all", "baseline", "custom_checkpoint", "torch_checkpoint", "nn_transformer"]
BENCHMARK_SCENARIOS = ["baseline", "custom_checkpoint", "torch_checkpoint", "nn_transformer"]


def _make_unique_run_dir(root_dir: str) -> Path:
    root = Path(root_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _get_self_cuda_time_us(event: object) -> float:
    # PyTorch profiler changed event field names across versions.
    for attr in ("self_cuda_time_total", "self_device_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _get_profiler_sort_key(events: list[object], device: torch.device) -> str:
    if device.type != "cuda":
        return "self_cpu_time_total"
    if any(hasattr(event, "self_cuda_time_total") for event in events):
        return "self_cuda_time_total"
    if any(hasattr(event, "self_device_time_total") for event in events):
        return "self_device_time_total"
    return "self_cpu_time_total"


def _run_train_steps(
    model: nn.Module,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    steps: int,
    use_checkpoint: bool,
    checkpoint_impl: str = "custom",
    model_kwargs: dict[str, object] | None = None,
    snapshot_path: Path | None = None,
    snapshot_max_entries: int = 100_000,
    profiler_trace_dir: str | None = None,
    label: str = "",
) -> tuple[float, float | None, Path | None, float | None]:
    device = tokens.device
    total_time = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history_started = False
    dumped_snapshot: Path | None = None
    self_cuda_time_per_step_ms: float | None = None

    #activities = [torch.profiler.ProfilerActivity.CPU]
    activities = []
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        if snapshot_path is not None and device.type == "cuda":
            history_started = _start_cuda_memory_history(max_entries=snapshot_max_entries)

        model.train()
        # warmup=1 silences the "skew" warning; acc_events=True keeps events
        # across schedule cycles so key_averages() sees all measured steps.
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=steps),
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
            # One extra iteration consumed by the profiler warmup phase.
            for step_idx in range(steps + 1):
                start = time.perf_counter()
                # Only pass checkpoint kwargs when caller doesn't override model args.
                forward_kwargs = model_kwargs if model_kwargs is not None else {
                    "use_checkpoint": use_checkpoint,
                    "checkpoint_impl": checkpoint_impl,
                }
                logits = model(
                    tokens,
                    **forward_kwargs,
                )
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                _sync_if_cuda(device)
                # Don't count the profiler-warmup step in wall-clock timing.
                if step_idx > 0:
                    total_time += time.perf_counter() - start
                prof.step()

        if label:
            events_for_table = list(prof.key_averages(group_by_input_shape=True))
            sort_key = _get_profiler_sort_key(events_for_table, device)
            print(f"\n=== torch.profiler [{label}] ===")
            print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=10))

        if device.type == "cuda":
            total_self_cuda_us = sum(_get_self_cuda_time_us(event) for event in prof.key_averages())
            self_cuda_time_per_step_ms = (total_self_cuda_us / 1000.0) / steps

        if profiler_trace_dir and label:
            trace_path = Path(profiler_trace_dir) / f"{label}.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace_path))
            print(f"Chrome trace saved: {trace_path}")

        if history_started and snapshot_path is not None:
            dumped_snapshot = snapshot_path if _dump_cuda_memory_snapshot(snapshot_path) else None
    finally:
        if history_started:
            _stop_cuda_memory_history()

    avg_step_time = total_time / steps
    peak_mem = _measure_peak_memory_mb(device)
    return avg_step_time, peak_mem, dumped_snapshot, self_cuda_time_per_step_ms


def profile_scenarios(
    device: torch.device,
    batch_size: int = 1,
    seq_len: int = 2048,
    vocab_size: int = 50257,
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 12,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    warmup_steps: int = 1,
    measured_steps: int = 3,
    cuda_snapshot_dir: str | None = None,
    cuda_snapshot_max_entries: int = 100_000,
    profiler_trace_dir: str | None = None,
    scenario: ScenarioName | list[ScenarioName] = "all",
    run_output_root: str = "benchmark_runs",
) -> Iterable[ProfileResult]:
    requested_scenarios = [scenario] if isinstance(scenario, str) else list(scenario)
    if not requested_scenarios or "all" in requested_scenarios:
        selected_scenarios: list[str] = list(BENCHMARK_SCENARIOS)
    else:
        selected_scenarios = []
        for name in requested_scenarios:
            if name not in BENCHMARK_SCENARIOS:
                raise ValueError(f"Unknown scenario '{name}'. Valid: {', '.join(SCENARIO_CHOICES)}")
            if name not in selected_scenarios:
                selected_scenarios.append(name)

    run_dir = _make_unique_run_dir(run_output_root)
    print(f"Run output directory: {run_dir}")

    effective_snapshot_dir: Path | None = None
    if cuda_snapshot_dir is not None:
        snapshot_leaf = Path(cuda_snapshot_dir).name
        effective_snapshot_dir = run_dir / snapshot_leaf
        effective_snapshot_dir.mkdir(parents=True, exist_ok=True)

    effective_profiler_trace_dir: Path | None = None
    if profiler_trace_dir is not None:
        trace_leaf = Path(profiler_trace_dir).name
        effective_profiler_trace_dir = run_dir / trace_leaf
        effective_profiler_trace_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "selected_scenarios": selected_scenarios,
                "cuda_snapshot_dir": str(effective_snapshot_dir) if effective_snapshot_dir else None,
                "profiler_trace_dir": str(effective_profiler_trace_dir) if effective_profiler_trace_dir else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def build_custom_model() -> TransformerLM:
        return TransformerLM(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def build_nn_transformer_model() -> NNTransformerLM:
        return NNTransformerLM(
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    torch.manual_seed(0)

    run_custom_scenarios = any(
        s in {"baseline", "custom_checkpoint", "torch_checkpoint"}
        for s in selected_scenarios
    )
    custom_template_state: dict[str, torch.Tensor] | None = None
    if run_custom_scenarios:
        template = build_custom_model()
        custom_template_state = template.state_dict()
        del template

    _cleanup_cuda(device)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    scenario_configs: dict[str, dict[str, object]] = {
        "baseline": {
            "builder": build_custom_model,
            "use_checkpoint": False,
            "checkpoint_impl": "custom",
            "model_kwargs": None,
            "load_custom_template": True,
        },
        "custom_checkpoint": {
            "builder": build_custom_model,
            "use_checkpoint": True,
            "checkpoint_impl": "custom",
            "model_kwargs": None,
            "load_custom_template": True,
        },
        "torch_checkpoint": {
            "builder": build_custom_model,
            "use_checkpoint": True,
            "checkpoint_impl": "torch",
            "model_kwargs": None,
            "load_custom_template": True,
        },
        "nn_transformer": {
            "builder": build_nn_transformer_model,
            "use_checkpoint": False,
            "checkpoint_impl": "custom",
            "model_kwargs": {},
            "load_custom_template": False,
        },
    }

    results: list[ProfileResult] = []

    for name in selected_scenarios:
        if name == "all":
            continue
        cfg = scenario_configs[name]

        builder = cfg["builder"]
        if not callable(builder):
            raise RuntimeError(f"builder is invalid for scenario '{name}'")
        model = builder().to(device)

        if bool(cfg["load_custom_template"]):
            if custom_template_state is None:
                raise RuntimeError("custom_template_state is not initialized")
            model.load_state_dict(custom_template_state)

        use_checkpoint_flag = bool(cfg["use_checkpoint"])
        checkpoint_impl_name = str(cfg["checkpoint_impl"])
        model_kwargs = cfg["model_kwargs"]
        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise RuntimeError(f"model_kwargs is invalid for scenario '{name}'")

        warmup_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        _run_train_steps(
            model=model,
            tokens=tokens,
            targets=targets,
            optimizer=warmup_optim,
            steps=warmup_steps,
            use_checkpoint=use_checkpoint_flag,
            checkpoint_impl=checkpoint_impl_name,
            model_kwargs=model_kwargs,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        snapshot_path = None
        if effective_snapshot_dir is not None and device.type == "cuda":
            snapshot_path = effective_snapshot_dir / f"{name}_snapshot.pickle"

        avg_time, peak_mem, dumped_snapshot, self_cuda_time_per_step_ms = _run_train_steps(
            model=model,
            tokens=tokens,
            targets=targets,
            optimizer=optimizer,
            steps=measured_steps,
            use_checkpoint=use_checkpoint_flag,
            checkpoint_impl=checkpoint_impl_name,
            model_kwargs=model_kwargs,
            snapshot_path=snapshot_path,
            snapshot_max_entries=cuda_snapshot_max_entries,
            profiler_trace_dir=str(effective_profiler_trace_dir) if effective_profiler_trace_dir else None,
            label=name,
        )

        results.append(
            ProfileResult(
                name=name,
                avg_step_time_s=avg_time,
                peak_memory_mb=peak_mem,
                self_cuda_time_per_step_ms=self_cuda_time_per_step_ms,
            )
        )
        if dumped_snapshot is not None:
            print(f"Saved {name} CUDA memory snapshot: {dumped_snapshot}")

        del model
        del warmup_optim
        del optimizer
        _cleanup_cuda(device)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual activation checkpointing for a Transformer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measured_steps", type=int, default=3)
    parser.add_argument(
        "--cuda_snapshot_dir",
        type=str,
        default="cuda_memory_snapshots",
        help="Directory to save PyTorch CUDA memory .pickle snapshots.",
    )
    parser.add_argument(
        "--cuda_snapshot_max_entries",
        type=int,
        default=100_000,
        help="Max number of CUDA memory events recorded before dumping a snapshot.",
    )
    parser.add_argument(
        "--profiler_trace_dir",
        type=str,
        default=None,
        help="Directory to save Chrome trace JSON files (open in chrome://tracing or Perfetto).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        nargs="+",
        default=["all"],
        choices=SCENARIO_CHOICES,
        help="Benchmark scenarios to run (multi-select). Use 'all' or one/more names.",
    )
    parser.add_argument(
        "--run_output_root",
        type=str,
        default="benchmark_runs",
        help="Root directory where each profile_scenarios run creates a unique output folder.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    results = profile_scenarios(
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        cuda_snapshot_dir=args.cuda_snapshot_dir,
        cuda_snapshot_max_entries=args.cuda_snapshot_max_entries,
        profiler_trace_dir=args.profiler_trace_dir,
        scenario=args.scenario,
        run_output_root=args.run_output_root,
    )

    print("Manual activation checkpointing profile")
    for r in results:
        mem_str = "N/A (CPU)" if r.peak_memory_mb is None else f"{r.peak_memory_mb:.2f} MB"
        self_cuda_str = (
            "N/A"
            if r.self_cuda_time_per_step_ms is None
            else f"{r.self_cuda_time_per_step_ms:.3f} ms"
        )
        print(
            f"{r.name:>10} | avg_step_time={r.avg_step_time_s:.4f}s "
            f"| peak_memory={mem_str} | self_cuda_step={self_cuda_str}"
        )

    result_by_name = {r.name: r for r in results}
    baseline = result_by_name.get("baseline")
    custom_checkpoint = result_by_name.get("custom_checkpoint")
    torch_checkpoint = result_by_name.get("torch_checkpoint")
    nn_transformer = result_by_name.get("nn_transformer")

    if baseline is not None and custom_checkpoint is not None:
        print(
            "speed_ratio(custom_checkpoint/baseline): "
            f"{custom_checkpoint.avg_step_time_s / baseline.avg_step_time_s:.3f}x"
        )
    if baseline is not None and torch_checkpoint is not None:
        print(
            "speed_ratio(torch_checkpoint/baseline): "
            f"{torch_checkpoint.avg_step_time_s / baseline.avg_step_time_s:.3f}x"
        )
    if baseline is not None and nn_transformer is not None:
        print(
            "speed_ratio(nn_transformer/baseline): "
            f"{nn_transformer.avg_step_time_s / baseline.avg_step_time_s:.3f}x"
        )
    if custom_checkpoint is not None and torch_checkpoint is not None:
        print(
            "speed_ratio(custom_checkpoint/torch_checkpoint): "
            f"{custom_checkpoint.avg_step_time_s / torch_checkpoint.avg_step_time_s:.3f}x"
        )

    if baseline is not None and custom_checkpoint is not None:
        if baseline.peak_memory_mb is not None and custom_checkpoint.peak_memory_mb is not None:
            print(
                "memory_ratio(custom_checkpoint/baseline): "
                f"{custom_checkpoint.peak_memory_mb / baseline.peak_memory_mb:.3f}x"
            )
    if baseline is not None and torch_checkpoint is not None:
        if baseline.peak_memory_mb is not None and torch_checkpoint.peak_memory_mb is not None:
            print(
                "memory_ratio(torch_checkpoint/baseline): "
                f"{torch_checkpoint.peak_memory_mb / baseline.peak_memory_mb:.3f}x"
            )
    if baseline is not None and nn_transformer is not None:
        if baseline.peak_memory_mb is not None and nn_transformer.peak_memory_mb is not None:
            print(
                "memory_ratio(nn_transformer/baseline): "
                f"{nn_transformer.peak_memory_mb / baseline.peak_memory_mb:.3f}x"
            )
    if custom_checkpoint is not None and torch_checkpoint is not None:
        if custom_checkpoint.peak_memory_mb is not None and torch_checkpoint.peak_memory_mb is not None:
            print(
                "memory_ratio(custom_checkpoint/torch_checkpoint): "
                f"{custom_checkpoint.peak_memory_mb / torch_checkpoint.peak_memory_mb:.3f}x"
            )

    if device.type == "cuda":
        print("Open .pickle snapshots at https://pytorch.org/memory_viz for timeline visualization.")


if __name__ == "__main__":
    main()
