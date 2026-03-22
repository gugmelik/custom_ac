import argparse
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from transformer import TransformerLM
from utilsf import _cleanup_cuda, ProfileResult, _measure_peak_memory_mb, _start_cuda_memory_history, _stop_cuda_memory_history, _dump_cuda_memory_snapshot, _sync_if_cuda


def _run_train_steps(
    model: TransformerLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    steps: int,
    use_checkpoint: bool,
    checkpoint_impl: str = "custom",
    snapshot_path: Path | None = None,
    snapshot_max_entries: int = 100_000,
    profiler_trace_dir: str | None = None,
    label: str = "",
) -> tuple[float, float | None, Path | None]:
    device = tokens.device
    total_time = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history_started = False
    dumped_snapshot: Path | None = None

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
                logits = model(
                    tokens,
                    use_checkpoint=use_checkpoint,
                    checkpoint_impl=checkpoint_impl,
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
            sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            print(f"\n=== torch.profiler [{label}] ===")
            print(prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=10))

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
    return avg_step_time, peak_mem, dumped_snapshot


def profile_baseline_vs_checkpoint(
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
) -> Iterable[ProfileResult]:
    
    torch.manual_seed(0)
    template = TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )
    template_state = template.state_dict()
    del template
    _cleanup_cuda(device)

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    model = TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(template_state)

    # Warm up kernels and memory allocator.
    warmup_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    _run_train_steps(
        model=model,
        tokens=tokens,
        targets=targets,
        optimizer=warmup_optim,
        steps=warmup_steps,
        use_checkpoint=False,
        checkpoint_impl="custom",
    )

    # Baseline.
    optimizer_base = torch.optim.AdamW(model.parameters(), lr=1e-4)
    base_snapshot_path = None
    if cuda_snapshot_dir and device.type == "cuda":
        base_snapshot_path = Path(cuda_snapshot_dir) / "baseline_snapshot.pickle"

    base_time, base_mem, dumped_base_snapshot = _run_train_steps(
        model=model,
        tokens=tokens,
        targets=targets,
        optimizer=optimizer_base,
        steps=measured_steps,
        use_checkpoint=False,
        checkpoint_impl="custom",
        snapshot_path=base_snapshot_path,
        snapshot_max_entries=cuda_snapshot_max_entries,
        profiler_trace_dir=profiler_trace_dir,
        label="baseline",
    )

    del model
    del warmup_optim
    del optimizer_base
    _cleanup_cuda(device)

    # Re-init model so each mode starts from the same parameter shape/state behavior.
    model_custom_ckpt = TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)
    model_custom_ckpt.load_state_dict(template_state)

    warmup_optim_ckpt = torch.optim.AdamW(model_custom_ckpt.parameters(), lr=1e-4)
    _run_train_steps(
        model=model_custom_ckpt,
        tokens=tokens,
        targets=targets,
        optimizer=warmup_optim_ckpt,
        steps=warmup_steps,
        use_checkpoint=True,
        checkpoint_impl="custom",
    )

    optimizer_custom_ckpt = torch.optim.AdamW(model_custom_ckpt.parameters(), lr=1e-4)
    custom_ckpt_snapshot_path = None
    if cuda_snapshot_dir and device.type == "cuda":
        custom_ckpt_snapshot_path = Path(cuda_snapshot_dir) / "custom_checkpoint_snapshot.pickle"

    custom_ckpt_time, custom_ckpt_mem, dumped_custom_ckpt_snapshot = _run_train_steps(
        model=model_custom_ckpt,
        tokens=tokens,
        targets=targets,
        optimizer=optimizer_custom_ckpt,
        steps=measured_steps,
        use_checkpoint=True,
        checkpoint_impl="custom",
        snapshot_path=custom_ckpt_snapshot_path,
        snapshot_max_entries=cuda_snapshot_max_entries,
        profiler_trace_dir=profiler_trace_dir,
        label="custom_checkpoint",
    )

    del model_custom_ckpt
    del warmup_optim_ckpt
    del optimizer_custom_ckpt
    _cleanup_cuda(device)

    model_torch_ckpt = TransformerLM(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)
    model_torch_ckpt.load_state_dict(template_state)

    warmup_optim_torch_ckpt = torch.optim.AdamW(model_torch_ckpt.parameters(), lr=1e-4)
    _run_train_steps(
        model=model_torch_ckpt,
        tokens=tokens,
        targets=targets,
        optimizer=warmup_optim_torch_ckpt,
        steps=warmup_steps,
        use_checkpoint=True,
        checkpoint_impl="torch",
    )

    optimizer_torch_ckpt = torch.optim.AdamW(model_torch_ckpt.parameters(), lr=1e-4)
    torch_ckpt_snapshot_path = None
    if cuda_snapshot_dir and device.type == "cuda":
        torch_ckpt_snapshot_path = Path(cuda_snapshot_dir) / "torch_checkpoint_snapshot.pickle"

    torch_ckpt_time, torch_ckpt_mem, dumped_torch_ckpt_snapshot = _run_train_steps(
        model=model_torch_ckpt,
        tokens=tokens,
        targets=targets,
        optimizer=optimizer_torch_ckpt,
        steps=measured_steps,
        use_checkpoint=True,
        checkpoint_impl="torch",
        snapshot_path=torch_ckpt_snapshot_path,
        snapshot_max_entries=cuda_snapshot_max_entries,
        profiler_trace_dir=profiler_trace_dir,
        label="torch_checkpoint",
    )

    if dumped_base_snapshot is not None:
        print(f"Saved baseline CUDA memory snapshot: {dumped_base_snapshot}")
    if dumped_custom_ckpt_snapshot is not None:
        print(f"Saved custom checkpoint CUDA memory snapshot: {dumped_custom_ckpt_snapshot}")
    if dumped_torch_ckpt_snapshot is not None:
        print(f"Saved torch checkpoint CUDA memory snapshot: {dumped_torch_ckpt_snapshot}")

    return [
        ProfileResult(
            name="baseline",
            avg_step_time_s=base_time,
            peak_memory_mb=base_mem,
        ),
        ProfileResult(
            name="custom_checkpoint",
            avg_step_time_s=custom_ckpt_time,
            peak_memory_mb=custom_ckpt_mem,
        ),
        ProfileResult(
            name="torch_checkpoint",
            avg_step_time_s=torch_ckpt_time,
            peak_memory_mb=torch_ckpt_mem,
        ),
    ]


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
    args = parser.parse_args()

    device = torch.device(args.device)
    results = profile_baseline_vs_checkpoint(
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
    )

    print("Manual activation checkpointing profile")
    for r in results:
        mem_str = "N/A (CPU)" if r.peak_memory_mb is None else f"{r.peak_memory_mb:.2f} MB"
        print(f"{r.name:>10} | avg_step_time={r.avg_step_time_s:.4f}s | peak_memory={mem_str}")

    baseline = next(r for r in results if r.name == "baseline")
    custom_checkpoint = next(r for r in results if r.name == "custom_checkpoint")
    torch_checkpoint = next(r for r in results if r.name == "torch_checkpoint")

    custom_time_ratio = custom_checkpoint.avg_step_time_s / baseline.avg_step_time_s
    torch_time_ratio = torch_checkpoint.avg_step_time_s / baseline.avg_step_time_s
    print(f"speed_ratio(custom_checkpoint/baseline): {custom_time_ratio:.3f}x")
    print(f"speed_ratio(torch_checkpoint/baseline): {torch_time_ratio:.3f}x")
    print(
        "speed_ratio(custom_checkpoint/torch_checkpoint): "
        f"{custom_checkpoint.avg_step_time_s / torch_checkpoint.avg_step_time_s:.3f}x"
    )

    if (
        baseline.peak_memory_mb is not None
        and custom_checkpoint.peak_memory_mb is not None
        and torch_checkpoint.peak_memory_mb is not None
    ):
        custom_mem_ratio = custom_checkpoint.peak_memory_mb / baseline.peak_memory_mb
        torch_mem_ratio = torch_checkpoint.peak_memory_mb / baseline.peak_memory_mb
        print(f"memory_ratio(custom_checkpoint/baseline): {custom_mem_ratio:.3f}x")
        print(f"memory_ratio(torch_checkpoint/baseline): {torch_mem_ratio:.3f}x")
        print(
            "memory_ratio(custom_checkpoint/torch_checkpoint): "
            f"{custom_checkpoint.peak_memory_mb / torch_checkpoint.peak_memory_mb:.3f}x"
        )

    if device.type == "cuda":
        print("Open .pickle snapshots at https://pytorch.org/memory_viz for timeline visualization.")


if __name__ == "__main__":
    main()
