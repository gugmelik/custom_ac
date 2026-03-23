import argparse
import csv
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from run_config import ScenarioName, profile_scenarios

SCENARIO_CHOICES = ["all", "baseline", "custom_checkpoint", "torch_checkpoint", "nn_transformer"]
BENCHMARK_SCENARIOS = ["baseline", "custom_checkpoint", "torch_checkpoint", "nn_transformer"]


def _normalize_scenarios(scenario: ScenarioName | list[ScenarioName]) -> list[str]:
    requested = [scenario] if isinstance(scenario, str) else list(scenario)
    if not requested or "all" in requested:
        return list(BENCHMARK_SCENARIOS)

    selected: list[str] = []
    for name in requested:
        if name not in BENCHMARK_SCENARIOS:
            raise ValueError(f"Unknown scenario '{name}'. Valid: {', '.join(SCENARIO_CHOICES)}")
        if name not in selected:
            selected.append(name)
    return selected


def _make_unique_run_dir(root_dir: str) -> Path:
    root = Path(root_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / f"evaluate_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_seq_lengths(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--seq_step must be > 0")
    if start > end:
        raise ValueError("--seq_start must be <= --seq_end")
    return list(range(start, end + 1, step))


def run_sweep(
    device: torch.device,
    seq_lengths: list[int],
    batch_size: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    mlp_ratio: float,
    dropout: float,
    warmup_steps: int,
    measured_steps: int,
    scenario: ScenarioName | list[ScenarioName],
    run_output_root: str,
) -> dict[str, list[tuple[int, float, float | None, float | None]]]:
    data: dict[str, list[tuple[int, float, float | None, float | None]]] = {}

    for seq_len in seq_lengths:
        print(f"\nRunning seq_len={seq_len} ...")
        results = profile_scenarios(
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            cuda_snapshot_dir=None,
            profiler_trace_dir=None,
            scenario=scenario,
            run_output_root=run_output_root,
        )

        for result in results:
            data.setdefault(result.name, []).append(
                (
                    seq_len,
                    result.avg_step_time_s,
                    result.peak_memory_mb,
                    result.self_cuda_time_per_step_ms,
                )
            )

    return data


def write_csv(data: dict[str, list[tuple[int, float, float | None, float | None]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "seq_len",
                "avg_step_time_s",
                "peak_memory_mb",
                "self_cuda_time_per_step_ms",
            ]
        )
        for scenario_name, rows in sorted(data.items()):
            for seq_len, avg_time, peak_mem, self_cuda_step in rows:
                writer.writerow(
                    [
                        scenario_name,
                        seq_len,
                        avg_time,
                        "" if peak_mem is None else peak_mem,
                        "" if self_cuda_step is None else self_cuda_step,
                    ]
                )


def _scenario_seq_to_metric(
    data: dict[str, list[tuple[int, float, float | None, float | None]]],
    scenario_name: str,
    metric: str,
) -> dict[int, float]:
    rows = data.get(scenario_name, [])
    seq_to_metric: dict[int, float] = {}
    for seq_len, _avg_time_s, peak_mem_mb, self_cuda_ms in rows:
        if metric == "memory":
            if peak_mem_mb is not None:
                seq_to_metric[seq_len] = peak_mem_mb
        elif metric == "cuda_exec":
            if self_cuda_ms is not None:
                seq_to_metric[seq_len] = self_cuda_ms
        else:
            raise ValueError(f"Unknown metric '{metric}'")
    return seq_to_metric


def _plot_pairwise_metric(
    data: dict[str, list[tuple[int, float, float | None, float | None]]],
    scenario_a: str,
    scenario_b: str,
    metric: str,
    out_png: Path,
) -> bool:
    a_map = _scenario_seq_to_metric(data, scenario_a, metric)
    b_map = _scenario_seq_to_metric(data, scenario_b, metric)
    common_seq = sorted(set(a_map).intersection(b_map))
    if not common_seq:
        return False

    x_vals = [a_map[s] for s in common_seq]
    y_vals = [b_map[s] for s in common_seq]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    scatter = ax.scatter(x_vals, y_vals, c=common_seq, cmap="viridis", s=70)

    for seq_len, x_val, y_val in zip(common_seq, x_vals, y_vals):
        ax.annotate(str(seq_len), (x_val, y_val), textcoords="offset points", xytext=(5, 5), fontsize=8)

    min_v = min(min(x_vals), min(y_vals))
    max_v = max(max(x_vals), max(y_vals))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1, color="gray")

    if metric == "cuda_exec":
        axis_label = "Self CUDA execution time per step (ms)"
        title_prefix = "CUDA Execution Time"
    else:
        axis_label = "Peak memory (MB)"
        title_prefix = "Memory Consumption"

    ax.set_title(f"{title_prefix}: {scenario_a} vs {scenario_b}")
    ax.set_xlabel(f"{scenario_a} ({axis_label})")
    ax.set_ylabel(f"{scenario_b} ({axis_label})")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("seq_len")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return True


def plot_pairwise_comparisons(
    data: dict[str, list[tuple[int, float, float | None, float | None]]],
    selected_scenarios: list[str],
    out_dir: Path,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_count = 0
    for scenario_a, scenario_b in combinations(selected_scenarios, 2):
        pair_count += 1
        cuda_out = out_dir / f"pair_cuda_exec_{scenario_a}_vs_{scenario_b}.png"
        mem_out = out_dir / f"pair_memory_{scenario_a}_vs_{scenario_b}.png"

        cuda_plotted = _plot_pairwise_metric(
            data=data,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            metric="cuda_exec",
            out_png=cuda_out,
        ) if device.type == "cuda" else False
        mem_plotted = _plot_pairwise_metric(
            data=data,
            scenario_a=scenario_a,
            scenario_b=scenario_b,
            metric="memory",
            out_png=mem_out,
        ) if device.type == "cuda" else False

        if cuda_plotted:
            print(f"Saved pairwise CUDA execution plot: {cuda_out}")
        elif device.type == "cuda":
            print(f"Skipped CUDA execution plot for {scenario_a} vs {scenario_b}: no common seq_len data.")

        if mem_plotted:
            print(f"Saved pairwise memory plot: {mem_out}")
        elif device.type == "cuda":
            print(f"Skipped memory plot for {scenario_a} vs {scenario_b}: no common seq_len data.")

    if pair_count == 0:
        print("No scenario pairs to plot. Select at least two scenarios.")
    if device.type != "cuda":
        print("Pairwise CUDA execution and memory plots require CUDA metrics; skipping on CPU.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep sequence length and plot peak memory/time for checkpointing scenarios"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_start", type=int, default=7000)
    parser.add_argument("--seq_end", type=int, default=10000)
    parser.add_argument("--seq_step", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measured_steps", type=int, default=30)
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
        help="Root directory where each evaluate run creates a unique evaluate_* folder.",
    )
    parser.add_argument("--pairwise_out_dir", type=str, default="plots/pairwise")
    parser.add_argument("--out_csv", type=str, default="plots/seq_len_sweep.csv")
    args = parser.parse_args()

    seq_lengths = _build_seq_lengths(args.seq_start, args.seq_end, args.seq_step)
    device = torch.device(args.device)

    print(
        f"Sweeping seq_len from {args.seq_start} to {args.seq_end} "
        f"(step={args.seq_step}) on device={device} with scenarios={args.scenario}"
    )

    run_dir = _make_unique_run_dir(args.run_output_root)
    print(f"Evaluate run output directory: {run_dir}")

    selected_scenarios = _normalize_scenarios(args.scenario)

    data = run_sweep(
        device=device,
        seq_lengths=seq_lengths,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        scenario=selected_scenarios,
        run_output_root=str(run_dir / "scenario_runs"),
    )

    out_csv = run_dir / Path(args.out_csv)
    pairwise_out_dir = run_dir / Path(args.pairwise_out_dir)
    write_csv(data, out_csv)
    plot_pairwise_comparisons(data, selected_scenarios, pairwise_out_dir, device)

    print(f"Saved CSV: {out_csv}")
    print(f"Pairwise plot directory: {pairwise_out_dir}")


if __name__ == "__main__":
    main()
