"""Synthetic MSM smoke training.

This is not a paper experiment. It is a tiny end-to-end check that the model,
masking, loss, logging, and optimizer paths work before real data is introduced.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from structformer.utils.config import ConfigError, deep_merge, load_config
from structformer.utils.device import DeviceError, memory_gb, select_device
from structformer.utils.env import environment_report
from structformer.utils.run_logging import RunLogger, format_duration
from structformer.utils.seed import seed_everything


DEFAULT_CONFIG: dict[str, Any] = {
    "task": "msm",
    "model": "saab",
    "seed": 42,
    "device": "auto",
    "allow_cpu": False,
    "synthetic": {
        "vocab_size": 64,
        "field_vocab_size": 6,
        "mask_field_id": 5,
        "seq_len": 16,
        "title_field_id": 4,
        "content_field_id": 3,
        "mask_probability": 0.15,
    },
    "model_config": {
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 64,
        "dropout": 0.0,
    },
    "training": {
        "max_steps": 20,
        "microbatch_size": 4,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "log_every_steps": 1,
        "checkpoint_every_steps": 0,
    },
}


def run_smoke_msm(config: dict[str, Any], *, run_dir: str | Path) -> Path:
    """Run a tiny synthetic MSM training loop and return the run directory."""

    import torch

    from structformer.models import StructuredTransformerModel, TransformerConfig
    from structformer.tasks.msm import (
        MaskedMSMBatch,
        make_synthetic_msm_batch,
        mask_field_ids,
        msm_cross_entropy,
        sample_batch_summary,
    )

    resolved = deep_merge(DEFAULT_CONFIG, config)
    seed = int(resolved["seed"])
    seed_everything(seed)

    device_info = select_device(str(resolved["device"]), allow_cpu=bool(resolved["allow_cpu"]))
    device = torch.device(device_info.name)

    logger = RunLogger(run_dir)
    logger.write_json("resolved_config.json", resolved)
    logger.write_json("environment.json", environment_report())

    synthetic = resolved["synthetic"]
    model_cfg = resolved["model_config"]
    training = resolved["training"]

    model_config = TransformerConfig(
        vocab_size=int(synthetic["vocab_size"]),
        field_vocab_size=int(synthetic["field_vocab_size"]),
        max_length=int(synthetic["seq_len"]),
        variant=str(resolved["model"]),
        num_labels=5,
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    model = StructuredTransformerModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    logger.write_json("model_summary.json", model.parameter_count())

    generator = _make_generator(torch, device, seed)

    start_time = time.monotonic()
    max_steps = int(training["max_steps"])
    log_every = int(training["log_every_steps"])
    checkpoint_every = int(training.get("checkpoint_every_steps", 0))
    last_loss = None

    for step in range(1, max_steps + 1):
        batch = make_synthetic_msm_batch(
            batch_size=int(training["microbatch_size"]),
            seq_len=int(synthetic["seq_len"]),
            vocab_size=int(synthetic["vocab_size"]),
            title_field_id=int(synthetic["title_field_id"]),
            content_field_id=int(synthetic["content_field_id"]),
            device=device,
            generator=generator,
        )
        masked_field_ids, labels, mask_positions = mask_field_ids(
            batch.field_ids,
            batch.attention_mask,
            mask_field_id=int(synthetic["mask_field_id"]),
            mask_probability=float(synthetic["mask_probability"]),
            generator=generator,
        )
        masked_batch = MaskedMSMBatch(
            input_ids=batch.input_ids,
            field_ids=masked_field_ids,
            original_field_ids=batch.field_ids,
            labels=labels,
            attention_mask=batch.attention_mask,
            mask_positions=mask_positions,
        )

        if step == 1:
            logger.write_json("sample_batch.json", sample_batch_summary(masked_batch))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(masked_batch.input_ids, masked_batch.field_ids, attention_mask=masked_batch.attention_mask)
        loss = msm_cross_entropy(output.logits, masked_batch.labels)
        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu())
        elapsed = time.monotonic() - start_time
        examples_per_sec = (step * int(training["microbatch_size"])) / max(elapsed, 1.0e-9)
        row = {
            "task": "msm_smoke",
            "model": resolved["model"],
            "seed": seed,
            "step": step,
            "max_steps": max_steps,
            "loss": last_loss,
            "lr": float(training["learning_rate"]),
            "examples_per_sec": examples_per_sec,
            "device": device_info.name,
            "memory_gb": memory_gb(device_info.type),
            "elapsed_seconds": elapsed,
        }
        logger.log_metric(row)

        if step == 1 or step % log_every == 0 or step == max_steps:
            logger.progress(
                "train",
                task="msm_smoke",
                model=resolved["model"],
                seed=seed,
                step=f"{step}/{max_steps}",
                loss=f"{last_loss:.4f}",
                lr=f"{float(training['learning_rate']):.2e}",
                ex_s=f"{examples_per_sec:.1f}",
                device=device_info.name,
                mem_gb=f"{row['memory_gb']:.2f}" if row["memory_gb"] is not None else None,
                elapsed=format_duration(elapsed),
            )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            _save_checkpoint(logger.run_dir / "checkpoints" / "latest.pt", model, optimizer, step, resolved)

    logger.write_json("final_summary.json", {"final_loss": last_loss, "steps": max_steps, "device": device_info.to_dict()})
    return logger.run_dir


def _save_checkpoint(path: Path, model, optimizer, step: int, config: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def _make_generator(torch_module, device, seed: int):
    """Create a per-run generator when the backend supports it."""

    try:
        generator = torch_module.Generator(device=device)
    except (RuntimeError, TypeError):
        return None
    generator.manual_seed(seed)
    return generator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a synthetic MSM smoke-training loop.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/msm_smoke"), help="Output run directory.")
    parser.add_argument("--model", choices=["baseline", "saab"], default=None, help="Override model variant.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU for this tiny smoke run.")
    args = parser.parse_args(argv)

    config: dict[str, Any] = {}
    if args.config is not None:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(str(exc))
            return 1
    if args.model is not None:
        config = deep_merge(config, {"model": args.model})
    if args.max_steps is not None:
        config = deep_merge(config, {"training": {"max_steps": args.max_steps}})
    if args.allow_cpu:
        config = deep_merge(config, {"allow_cpu": True})

    try:
        run_dir = run_smoke_msm(config, run_dir=args.run_dir)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            print("PyTorch is required for smoke training. Run scripts/check_env.py for guidance.")
            return 1
        raise
    except DeviceError as exc:
        print(str(exc))
        return 1

    print(f"Smoke run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
