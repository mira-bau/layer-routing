"""Benchmark Baseline and SAAB computational overhead under controlled shapes.

This benchmark uses freshly initialized models and fixed synthetic tensors. It
does not train a paper model or require prepared data or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be comma-separated positive integers")
    return lengths


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _synchronize(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def _time_iterations(torch_module, device, fn: Callable[[], None], iterations: int) -> float:
    _synchronize(torch_module, device)
    if device.type == "cuda":
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / iterations

    start_time = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - start_time) * 1000.0 / iterations


def _memory_start(torch_module, device) -> int | None:
    if device.type != "cuda":
        return None
    _synchronize(torch_module, device)
    torch_module.cuda.reset_peak_memory_stats(device)
    return int(torch_module.cuda.memory_allocated(device))


def _memory_end(torch_module, device, base_bytes: int | None) -> tuple[int | None, int | None]:
    if device.type != "cuda" or base_bytes is None:
        return None, None
    _synchronize(torch_module, device)
    peak = int(torch_module.cuda.max_memory_allocated(device))
    return peak, max(peak - base_bytes, 0)


def _profile_forward_flops(torch_module, model, inputs: tuple[Any, ...], device) -> int:
    activities = [torch_module.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch_module.profiler.ProfilerActivity.CUDA)
    input_ids, field_ids, attention_mask = inputs
    with torch_module.inference_mode():
        with torch_module.profiler.profile(activities=activities, with_flops=True) as profile:
            model(input_ids, field_ids, attention_mask=attention_mask)
    _synchronize(torch_module, device)
    return int(sum(event.flops or 0 for event in profile.key_averages()))


def _make_model(torch_module, args, variant: str, max_length: int, device):
    from structformer.models import StructuredTransformerModel, TransformerConfig

    torch_module.manual_seed(args.seed)
    if device.type == "cuda":
        torch_module.cuda.manual_seed_all(args.seed)
    config = TransformerConfig(
        vocab_size=args.vocab_size,
        field_vocab_size=args.field_vocab_size,
        max_length=max_length,
        variant=variant,
        head_type="token",
        num_labels=args.num_labels,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pad_token_id=0,
        casa_rank=8,
        scale_embeddings=True,
        saab_field_weight=1.0,
    )
    return StructuredTransformerModel(config).to(device)


def _make_inputs(torch_module, args, length: int, device):
    generator = torch_module.Generator().manual_seed(args.seed + length)
    input_ids = torch_module.randint(
        2, args.vocab_size, (args.batch_size, length), generator=generator
    ).to(device)
    field_ids = torch_module.full((args.batch_size, length), 3, dtype=torch_module.long)
    field_ids[:, : length // 2] = 4
    field_ids = field_ids.to(device)
    attention_mask = torch_module.ones(
        args.batch_size, length, dtype=torch_module.bool, device=device
    )
    labels = torch_module.randint(
        0, args.num_labels, (args.batch_size, length), generator=generator
    ).to(device)
    return input_ids, field_ids, attention_mask, labels


def _run_inference_blocks(torch_module, model, inputs, args, device, variant: str, length: int):
    input_ids, field_ids, attention_mask, _ = inputs
    model.eval()

    def step() -> None:
        with torch_module.inference_mode():
            model(input_ids, field_ids, attention_mask=attention_mask)

    for _ in range(args.inference_warmup):
        step()
    records = []
    for block in range(1, args.repeats + 1):
        base = _memory_start(torch_module, device)
        latency = _time_iterations(torch_module, device, step, args.inference_iterations)
        peak, incremental = _memory_end(torch_module, device, base)
        records.append(
            _timing_record(
                variant, "inference", length, block, args.batch_size,
                args.inference_iterations, latency, peak, incremental,
            )
        )
        print(
            f"benchmark variant={variant} mode=inference length={length} "
            f"block={block}/{args.repeats} latency_ms={latency:.3f} "
            f"examples_per_second={args.batch_size * 1000.0 / latency:.2f} device={device}"
        )
    return records


def _run_training_blocks(torch_module, model, inputs, args, device, variant: str, length: int):
    input_ids, field_ids, attention_mask, labels = inputs
    model.train()
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=1.0e-4)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, field_ids, attention_mask=attention_mask)
        loss = torch_module.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()

    for _ in range(args.training_warmup):
        step()
    records = []
    for block in range(1, args.repeats + 1):
        base = _memory_start(torch_module, device)
        latency = _time_iterations(torch_module, device, step, args.training_iterations)
        peak, incremental = _memory_end(torch_module, device, base)
        records.append(
            _timing_record(
                variant, "training", length, block, args.batch_size,
                args.training_iterations, latency, peak, incremental,
            )
        )
        print(
            f"benchmark variant={variant} mode=training length={length} "
            f"block={block}/{args.repeats} step_ms={latency:.3f} "
            f"examples_per_second={args.batch_size * 1000.0 / latency:.2f} device={device}"
        )
    return records


def _timing_record(
    variant: str,
    mode: str,
    length: int,
    block: int,
    batch_size: int,
    iterations: int,
    latency_ms: float,
    peak_bytes: int | None,
    incremental_bytes: int | None,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "mode": mode,
        "sequence_length": length,
        "block": block,
        "batch_size": batch_size,
        "iterations": iterations,
        "latency_ms": latency_ms,
        "examples_per_second": batch_size * 1000.0 / latency_ms,
        "peak_allocated_bytes": peak_bytes,
        "incremental_peak_bytes": incremental_bytes,
    }


def _summarize(
    raw_records: list[dict[str, Any]],
    model_info: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in raw_records:
        key = (record["variant"], record["mode"], record["sequence_length"])
        groups.setdefault(key, []).append(record)

    summaries = []
    for (variant, mode, length), records in sorted(groups.items(), key=lambda item: item[0]):
        latencies = [float(record["latency_ms"]) for record in records]
        mean_latency = statistics.mean(latencies)
        info = model_info[(variant, length)]
        peaks = [record["peak_allocated_bytes"] for record in records if record["peak_allocated_bytes"] is not None]
        incremental = [record["incremental_peak_bytes"] for record in records if record["incremental_peak_bytes"] is not None]
        summaries.append(
            {
                "variant": variant,
                "mode": mode,
                "sequence_length": length,
                "batch_size": records[0]["batch_size"],
                "timed_blocks": len(records),
                "iterations_per_block": records[0]["iterations"],
                "latency_mean_ms": mean_latency,
                "latency_std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "latency_median_ms": statistics.median(latencies),
                "examples_per_second": records[0]["batch_size"] * 1000.0 / mean_latency,
                "peak_allocated_mb": max(peaks) / 1.0e6 if peaks else None,
                "incremental_peak_mb": max(incremental) / 1.0e6 if incremental else None,
                "total_parameters": info["total_parameters"],
                "trainable_parameters": info["trainable_parameters"],
                "forward_flops_pytorch_profiler": info["forward_flops_pytorch_profiler"],
                "forward_gflops_pytorch_profiler": info["forward_flops_pytorch_profiler"] / 1.0e9,
                "saab_explicit_elementwise_operations": info["saab_explicit_elementwise_operations"],
            }
        )
    return summaries


def _pairwise_overhead(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["variant"], row["mode"], row["sequence_length"]): row for row in summaries
    }
    rows = []
    for mode in ("inference", "training"):
        lengths = sorted(
            length for variant, row_mode, length in indexed
            if variant == "baseline" and row_mode == mode
        )
        for length in lengths:
            baseline = indexed[("baseline", mode, length)]
            saab = indexed[("saab", mode, length)]
            rows.append(
                {
                    "mode": mode,
                    "sequence_length": length,
                    "baseline_latency_ms": baseline["latency_mean_ms"],
                    "saab_latency_ms": saab["latency_mean_ms"],
                    "latency_overhead_percent": _percent_change(
                        baseline["latency_mean_ms"], saab["latency_mean_ms"]
                    ),
                    "baseline_examples_per_second": baseline["examples_per_second"],
                    "saab_examples_per_second": saab["examples_per_second"],
                    "throughput_change_percent": _percent_change(
                        baseline["examples_per_second"], saab["examples_per_second"]
                    ),
                    "baseline_peak_allocated_mb": baseline["peak_allocated_mb"],
                    "saab_peak_allocated_mb": saab["peak_allocated_mb"],
                    "peak_memory_overhead_percent": _optional_percent_change(
                        baseline["peak_allocated_mb"], saab["peak_allocated_mb"]
                    ),
                    "trainable_parameter_difference": (
                        saab["trainable_parameters"] - baseline["trainable_parameters"]
                    ),
                    "profiler_forward_flops_difference": (
                        saab["forward_flops_pytorch_profiler"]
                        - baseline["forward_flops_pytorch_profiler"]
                    ),
                    "saab_explicit_elementwise_operations": saab[
                        "saab_explicit_elementwise_operations"
                    ],
                }
            )
    return rows


def _percent_change(baseline: float, value: float) -> float:
    return (value - baseline) / baseline * 100.0


def _optional_percent_change(baseline: float | None, value: float | None) -> float | None:
    if baseline is None or value is None:
        return None
    return _percent_change(baseline, value)


def _environment(torch_module, device, args) -> dict[str, Any]:
    cuda = None
    if device.type == "cuda":
        properties = torch_module.cuda.get_device_properties(device)
        cuda = {
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch_module.version.cuda,
            "cudnn_version": torch_module.backends.cudnn.version(),
            "tf32_matmul_enabled": torch_module.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_enabled": torch_module.backends.cudnn.allow_tf32,
        }
    return {
        "code_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "device": str(device),
        "cuda": cuda,
        "numerical_precision": "float32",
        "automatic_mixed_precision": False,
        "gradient_scaling": False,
        "tf32_requested": bool(args.allow_tf32),
        "timing_method": "CUDA events with synchronization" if device.type == "cuda" else "perf_counter",
        "flop_tool": "torch.profiler.profile(with_flops=True)",
        "flop_counting_note": (
            "PyTorch profiler estimates supported matrix-multiplication and linear-operator FLOPs. "
            "Elementwise comparison, multiplication, broadcast addition, masking, and softmax may be uncounted."
        ),
        "saab_elementwise_counting_note": (
            "Separate scalar-operation count includes one equality and one multiplication per pair "
            "to construct the fixed bias, plus one broadcast score addition per head and layer. "
            "This is not reported as floating-point FLOPs because equality is not a floating-point operation."
        ),
    }


def run(args) -> Path:
    import torch

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required. In Colab select Runtime > Change runtime type > GPU.")
        device = torch.device("cuda")
    elif args.device == "cpu" and args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CPU is allowed only with --device cpu --allow-cpu for a tiny smoke run.")

    if args.d_model % args.num_heads != 0:
        raise ValueError("d-model must be divisible by num-heads")
    if args.field_vocab_size < 5:
        raise ValueError("field-vocab-size must be at least 5 for fixed field IDs 3 and 4")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = vars(args).copy()
    config_payload["out_dir"] = str(args.out_dir)
    config_payload["lengths"] = list(args.lengths)
    _write_json(out_dir / "benchmark_config.json", config_payload)
    _write_json(out_dir / "environment.json", _environment(torch, device, args))

    raw_records: list[dict[str, Any]] = []
    model_info: dict[tuple[str, int], dict[str, Any]] = {}
    max_length = max(args.lengths)
    for length in args.lengths:
        inputs = _make_inputs(torch, args, length, device)
        for variant in ("baseline", "saab"):
            model = _make_model(torch, args, variant, max_length, device)
            counts = model.parameter_count()
            model.eval()
            forward_flops = _profile_forward_flops(
                torch, model, (inputs[0], inputs[1], inputs[2]), device
            )
            elementwise = (
                args.batch_size
                * length
                * length
                * (2 + args.num_layers * args.num_heads)
                if variant == "saab"
                else 0
            )
            model_info[(variant, length)] = {
                "variant": variant,
                "sequence_length": length,
                "total_parameters": counts["total"],
                "trainable_parameters": counts["trainable"],
                "forward_flops_pytorch_profiler": forward_flops,
                "saab_explicit_elementwise_operations": elementwise,
            }
            raw_records.extend(
                _run_inference_blocks(torch, model, inputs, args, device, variant, length)
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            model = _make_model(torch, args, variant, max_length, device)
            raw_records.extend(
                _run_training_blocks(torch, model, inputs, args, device, variant, length)
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summaries = _summarize(raw_records, model_info)
    overhead = _pairwise_overhead(summaries)
    model_rows = [model_info[key] for key in sorted(model_info)]
    _write_json(out_dir / "raw_timings.json", raw_records)
    _write_csv(out_dir / "raw_timings.csv", raw_records)
    _write_json(out_dir / "model_operation_summary.json", model_rows)
    _write_csv(out_dir / "model_operation_summary.csv", model_rows)
    _write_json(out_dir / "benchmark_summary.json", summaries)
    _write_csv(out_dir / "benchmark_summary.csv", summaries)
    _write_json(out_dir / "baseline_saab_overhead.json", overhead)
    _write_csv(out_dir / "baseline_saab_overhead.csv", overhead)

    expected = len(args.lengths) * 2 * 2
    if len(summaries) != expected:
        raise RuntimeError(f"Expected {expected} summary rows, found {len(summaries)}")
    if any(row["trainable_parameter_difference"] != 0 for row in overhead):
        raise RuntimeError("Baseline and SAAB trainable parameter counts differ")
    print(f"benchmark complete output={out_dir} summary_rows={len(summaries)}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/computational_overhead"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--lengths", type=_parse_lengths, default=[64, 128, 256])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inference-warmup", type=int, default=5)
    parser.add_argument("--inference-iterations", type=int, default=20)
    parser.add_argument("--training-warmup", type=int, default=2)
    parser.add_argument("--training-iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--field-vocab-size", type=int, default=6)
    parser.add_argument("--num-labels", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive = [
        args.batch_size,
        args.inference_warmup,
        args.inference_iterations,
        args.training_warmup,
        args.training_iterations,
        args.repeats,
        args.vocab_size,
        args.field_vocab_size,
        args.num_labels,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.ff_dim,
    ]
    if any(value <= 0 for value in positive):
        raise ValueError("All sizes and timing counts must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
