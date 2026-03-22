# Activation Checkpointing From Scratch

A minimal PyTorch project that compares three training modes on a Transformer language model:

- `baseline` (no checkpointing)
- `custom_checkpoint` (manual checkpointing implemented with a custom `autograd.Function`)
- `torch_checkpoint` (PyTorch built-in checkpointing via `torch.utils.checkpoint`)

The benchmark reports step time, peak CUDA memory, and optional profiler traces.

## Project Layout

- `ac_scr.py`: Main benchmark script.
- `transformer.py`: Transformer model and checkpoint mode switch.
- `ckpt.py`: Custom activation checkpointing implementation.
- `utilsf.py`: CUDA sync, cleanup, and memory snapshot helpers.
- `cuda_memory_snapshots/`: Optional CUDA memory snapshots (`.pickle`).
- `profiler_traces/`: Optional Chrome trace JSON files.
- `notebooks/`: Jupyter notebooks for experiments.

## Requirements

- Python 3.10+
- PyTorch (CUDA build recommended for memory comparison)

Install (example):

```bash
pip install torch
```

If you use Conda, install the CUDA-enabled PyTorch build that matches your driver/CUDA runtime.

## Run Benchmark

From the project root:

```bash
python ac_scr.py
```

Useful options:

```bash
python ac_scr.py \
  --device cuda \
  --batch_size 1 \
  --seq_len 2048 \
  --n_layers 12 \
  --warmup_steps 1 \
  --measured_steps 3 \
  --cuda_snapshot_dir cuda_memory_snapshots \
  --profiler_trace_dir profiler_traces
```

## CLI Arguments

`ac_scr.py` supports:

- `--device`: `cuda` or `cpu` (default: `cuda` if available, else `cpu`)
- `--batch_size`: batch size
- `--seq_len`: sequence length
- `--vocab_size`: vocabulary size
- `--d_model`: hidden size
- `--n_heads`: attention heads
- `--n_layers`: number of transformer blocks
- `--mlp_ratio`: FFN expansion ratio
- `--dropout`: dropout probability
- `--warmup_steps`: pre-measurement warmup iterations
- `--measured_steps`: timed iterations
- `--cuda_snapshot_dir`: directory for CUDA memory snapshots
- `--cuda_snapshot_max_entries`: max CUDA memory events tracked before dump
- `--profiler_trace_dir`: directory for profiler Chrome traces

## What The Script Prints

For each mode (`baseline`, `custom_checkpoint`, `torch_checkpoint`):

- `avg_step_time` (seconds)
- `peak_memory` (MB on CUDA)

It also prints ratios:

- `speed_ratio(custom_checkpoint/baseline)`
- `speed_ratio(torch_checkpoint/baseline)`
- `speed_ratio(custom_checkpoint/torch_checkpoint)`
- Memory ratios for the same pairs (on CUDA)

Expected pattern:

- Checkpointing usually reduces peak memory.
- Checkpointing usually increases step time (recompute overhead).

## Profiler Output

When enabled, the script prints `torch.profiler` tables per mode and can export traces:

- `profiler_traces/baseline.json`
- `profiler_traces/custom_checkpoint.json`
- `profiler_traces/torch_checkpoint.json`

Open traces with:

- `chrome://tracing`
- https://ui.perfetto.dev

## CUDA Memory Snapshot Output

When CUDA snapshots are enabled, the script can emit:

- `baseline_snapshot.pickle`
- `custom_checkpoint_snapshot.pickle`
- `torch_checkpoint_snapshot.pickle`

Open at:

- https://pytorch.org/memory_viz

## How Custom Checkpointing Works

`ckpt.py` defines `_CheckpointBlockFn`, a custom autograd function that:

1. Runs the module forward under `torch.no_grad()`.
2. Saves only input `x` (plus RNG state if requested).
3. Recomputes the module in backward with gradients enabled.
4. Uses `torch.autograd.grad` to produce gradients for `x` and module parameters.

This trades compute for memory.

## Notes and Limitations

- The current custom wrapper is shaped for `module(x)` style calls and single-tensor output.
- Built-in checkpointing (`torch.utils.checkpoint`) is generally safer for broad model patterns and distributed settings.
- On CPU, CUDA memory stats are reported as `N/A`.
