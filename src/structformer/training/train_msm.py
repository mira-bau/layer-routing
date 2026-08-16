"""Train MSM on prepared DBpedia-style JSONL artifacts."""

from __future__ import annotations

import argparse
import math
import time
from functools import partial
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
    "data": {
        "train_jsonl": None,
        "val_jsonl": None,
        "tokenizer_json": None,
        "max_records": None,
        "train_sample_size": None,
        "val_sample_size": None,
        "sample_seed": None,
        "max_length": 256,
        "pad_token_id": 0,
        "pad_field_id": 0,
        "field_vocab_size": 6,
        "mask_field_id": 5,
        "num_labels": 5,
        "mask_probability": 0.15,
    },
    "model_config": {
        "d_model": 768,
        "num_layers": 4,
        "num_heads": 6,
        "ff_dim": 3072,
        "dropout": 0.2,
        "casa_rank": 8,
        "scale_embeddings": False,
    },
    "training": {
        "max_steps": 500,
        "microbatch_size": 64,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0e-4,
        "lr_schedule": "constant",
        "warmup_steps": 0,
        "min_lr_ratio": 0.1,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "grad_clip": 1.0,
        "log_every_steps": 10,
        "eval_every_steps": 0,
        "checkpoint_every_steps": 0,
        "checkpoint_diagnostic_steps": [],
        "log_layer_gradients": False,
        "num_workers": 0,
    },
}


def run_train_msm(config: dict[str, Any], *, run_dir: str | Path) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from structformer.data.msm_jsonl import PreparedMSMJsonlDataset, collate_prepared_msm
    from structformer.models import StructuredTransformerModel, TransformerConfig
    from structformer.tasks.msm import msm_cross_entropy, sample_batch_summary

    resolved = deep_merge(DEFAULT_CONFIG, config)
    data_cfg = resolved["data"]
    training_cfg = resolved["training"]
    model_cfg = resolved["model_config"]

    if not data_cfg["train_jsonl"]:
        raise ValueError("data.train_jsonl is required")

    seed = int(resolved["seed"])
    if (
        data_cfg.get("sample_seed") is None
        and (data_cfg.get("train_sample_size") is not None or data_cfg.get("val_sample_size") is not None)
    ):
        resolved = deep_merge(resolved, {"data": {"sample_seed": seed}})
        data_cfg = resolved["data"]

    max_records = data_cfg.get("max_records")
    train_sample_size = data_cfg.get("train_sample_size")
    val_sample_size = data_cfg.get("val_sample_size")
    sample_seed = data_cfg.get("sample_seed")
    if max_records is not None and (train_sample_size is not None or val_sample_size is not None):
        raise ValueError("data.max_records cannot be combined with train_sample_size or val_sample_size")
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be a positive integer")

    seed_everything(seed)

    device_info = select_device(str(resolved["device"]), allow_cpu=bool(resolved["allow_cpu"]))
    device = torch.device(device_info.name)
    logger = RunLogger(run_dir)
    layer_gradient_logger = (
        RunLogger(logger.run_dir / "layer_gradients")
        if bool(training_cfg.get("log_layer_gradients", False))
        else None
    )
    logger.write_json("resolved_config.json", resolved)
    logger.write_json("environment.json", environment_report())

    train_dataset = PreparedMSMJsonlDataset(
        data_cfg["train_jsonl"],
        max_records=max_records,
        sample_size=train_sample_size,
        sample_seed=sample_seed,
    )
    val_dataset = (
        PreparedMSMJsonlDataset(
            data_cfg["val_jsonl"],
            max_records=max_records,
            sample_size=val_sample_size,
            sample_seed=sample_seed,
        )
        if data_cfg.get("val_jsonl")
        else None
    )
    logger.write_json(
        "data_manifest.json",
        {
            "train_jsonl": str(data_cfg["train_jsonl"]),
            "train_records": len(train_dataset),
            "train_source_records": train_dataset.source_records,
            "train_sample_size": train_dataset.sample_size,
            "train_sample_seed": train_dataset.sample_seed,
            "train_sample_indices_hash": train_dataset.sample_indices_hash,
            "train_sample_indices_preview": train_dataset.sample_indices_preview,
            "val_jsonl": str(data_cfg["val_jsonl"]) if data_cfg.get("val_jsonl") else None,
            "val_records": len(val_dataset) if val_dataset is not None else 0,
            "val_source_records": val_dataset.source_records if val_dataset is not None else 0,
            "val_sample_size": val_dataset.sample_size if val_dataset is not None else None,
            "val_sample_seed": val_dataset.sample_seed if val_dataset is not None else None,
            "val_sample_indices_hash": val_dataset.sample_indices_hash if val_dataset is not None else None,
            "val_sample_indices_preview": val_dataset.sample_indices_preview if val_dataset is not None else [],
            "max_length": data_cfg["max_length"],
        },
    )

    vocab_size = int(data_cfg.get("vocab_size") or _infer_vocab_size(data_cfg.get("tokenizer_json"), train_dataset))
    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        field_vocab_size=int(data_cfg["field_vocab_size"]),
        max_length=int(data_cfg["max_length"]),
        variant=str(resolved["model"]),
        head_type="token",
        num_labels=int(data_cfg["num_labels"]),
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        pad_token_id=int(data_cfg["pad_token_id"]),
        casa_rank=int(model_cfg["casa_rank"]),
        scale_embeddings=bool(model_cfg.get("scale_embeddings", False)),
        saab_field_weight=float(model_cfg.get("saab_field_weight", 1.0)),
        saab_layer_mask=tuple(float(x) for x in model_cfg.get("saab_layer_mask", [])),
        saab_shuffle_bias=bool(model_cfg.get("saab_shuffle_bias", False)),
        saab_shuffle_seed=int(model_cfg.get("saab_shuffle_seed", 0)),
    )
    model = StructuredTransformerModel(transformer_config).to(device)
    betas = tuple(float(value) for value in training_cfg["betas"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        betas=betas,
    )
    logger.write_json("model_summary.json", model.parameter_count())

    collate = partial(
        collate_prepared_msm,
        pad_token_id=int(data_cfg["pad_token_id"]),
        pad_field_id=int(data_cfg["pad_field_id"]),
        max_length=int(data_cfg["max_length"]),
    )
    train_shuffle_generator = _make_data_loader_generator(torch, seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["microbatch_size"]),
        shuffle=True,
        generator=train_shuffle_generator,
        num_workers=int(training_cfg["num_workers"]),
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(training_cfg["microbatch_size"]),
            shuffle=False,
            num_workers=int(training_cfg["num_workers"]),
            collate_fn=collate,
        )
        if val_dataset is not None
        else None
    )

    generator = _make_generator(torch, device, seed)
    start_time = time.monotonic()
    max_steps = int(training_cfg["max_steps"])
    base_learning_rate = float(training_cfg["learning_rate"])
    log_every = int(training_cfg["log_every_steps"])
    eval_every = int(training_cfg.get("eval_every_steps", 0))
    checkpoint_every = int(training_cfg.get("checkpoint_every_steps", 0))
    diag_steps = set(int(s) for s in training_cfg.get("checkpoint_diagnostic_steps", []))
    train_iter = iter(train_loader)
    last_loss = None
    last_eval: dict[str, float | int] | None = None

    for step in range(1, max_steps + 1):
        current_lr = _learning_rate_for_step(
            base_learning_rate=base_learning_rate,
            step=step,
            max_steps=max_steps,
            schedule=str(training_cfg.get("lr_schedule", "constant")),
            warmup_steps=int(training_cfg.get("warmup_steps", 0)),
            min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.1)),
        )
        _set_optimizer_lr(optimizer, current_lr)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for microstep in range(gradient_accumulation_steps):
            try:
                collated = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                collated = next(train_iter)
            batch = _batch_to_device(collated.batch, device)
            masked_batch = _mask_batch(batch, data_cfg, generator)
            if step == 1 and microstep == 0:
                logger.write_json("sample_batch.json", sample_batch_summary(masked_batch))
                logger.write_json("sample_rows.json", {"row_ids": collated.row_ids, "tokens": collated.tokens})

            output = model(masked_batch.input_ids, masked_batch.field_ids, attention_mask=masked_batch.attention_mask)
            loss = msm_cross_entropy(output.logits, masked_batch.labels)
            accumulated_loss += float(loss.detach().cpu())
            (loss / gradient_accumulation_steps).backward()

        layer_gradient_norms = (
            _attention_projection_gradient_norms(model)
            if layer_gradient_logger is not None
            else None
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip"]))
        if layer_gradient_logger is not None and layer_gradient_norms is not None:
            ratio = (
                layer_gradient_norms[-1] / layer_gradient_norms[-2]
                if len(layer_gradient_norms) >= 2 and layer_gradient_norms[-2] > 0.0
                else None
            )
            layer_gradient_logger.log_metric(
                {
                    "phase": "train_gradient",
                    "task": "msm",
                    "model": str(resolved["model"]),
                    "seed": seed,
                    "step": step,
                    "max_steps": max_steps,
                    "loss": accumulated_loss / gradient_accumulation_steps,
                    "lr": current_lr,
                    "global_grad_norm_pre_clip": float(grad_norm.detach().cpu()),
                    "grad_clip": float(training_cfg["grad_clip"]),
                    "qkv_grad_norm_per_layer": layer_gradient_norms,
                    "grad_norm_ratio_last_penultimate": ratio,
                    "objective": "msm_cross_entropy",
                    "measurement_point": "after_gradient_accumulation_before_global_clipping",
                }
            )
        optimizer.step()

        last_loss = accumulated_loss / gradient_accumulation_steps
        elapsed = time.monotonic() - start_time
        examples_seen = step * int(training_cfg["microbatch_size"]) * gradient_accumulation_steps
        examples_per_sec = examples_seen / max(elapsed, 1.0e-9)
        _log_progress(
            logger,
            phase="train",
            task="msm",
            model_name=str(resolved["model"]),
            seed=seed,
            step=step,
            max_steps=max_steps,
            loss=last_loss,
            lr=current_lr,
            examples_per_sec=examples_per_sec,
            device_name=device_info.name,
            memory=memory_gb(device_info.type),
            elapsed=elapsed,
            split="train",
            grad_norm=float(grad_norm.detach().cpu()),
        )

        if step == 1 or step % log_every == 0 or step == max_steps:
            logger.progress(
                "train",
                task="msm",
                model=resolved["model"],
                seed=seed,
                step=f"{step}/{max_steps}",
                loss=f"{last_loss:.4f}",
                lr=f"{current_lr:.2e}",
                ex_s=f"{examples_per_sec:.1f}",
                device=device_info.name,
                elapsed=format_duration(elapsed),
            )

        if val_loader is not None and eval_every > 0 and (step % eval_every == 0 or step == max_steps):
            eval_metrics = _evaluate(model, val_loader, data_cfg, generator, device)
            last_eval = {"step": step, **eval_metrics}
            _log_progress(
                logger,
                phase="eval",
                task="msm",
                model_name=str(resolved["model"]),
                seed=seed,
                step=step,
                max_steps=max_steps,
                loss=eval_metrics["loss"],
                lr=current_lr,
                examples_per_sec=None,
                device_name=device_info.name,
                memory=memory_gb(device_info.type),
                elapsed=time.monotonic() - start_time,
                split="val",
                accuracy=eval_metrics["accuracy"],
            )
            logger.progress(
                "eval",
                task="msm",
                model=resolved["model"],
                seed=seed,
                step=step,
                val_loss=f"{eval_metrics['loss']:.4f}",
                val_acc=f"{eval_metrics['accuracy']:.4f}",
            )

        if checkpoint_every > 0 and step % checkpoint_every == 0:
            _save_checkpoint(logger.run_dir / "checkpoints" / "latest.pt", model, optimizer, step, resolved)

        if step in diag_steps:
            _save_diagnostic_checkpoint(
                logger.run_dir / "checkpoints" / f"step_{step:04d}.pt",
                model, step, resolved,
            )

    final_summary: dict[str, Any] = {
        "final_loss": last_loss,
        "steps": max_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": int(training_cfg["microbatch_size"]) * gradient_accumulation_steps,
        "layer_gradient_logging": layer_gradient_logger is not None,
        "device": device_info.to_dict(),
    }
    if last_eval is not None:
        final_summary["final_eval"] = last_eval
    logger.write_json("final_summary.json", final_summary)
    return logger.run_dir


def _mask_batch(batch, data_cfg: dict[str, Any], generator) -> Any:
    from structformer.tasks.msm import MaskedMSMBatch, mask_field_ids

    masked_field_ids, labels, mask_positions = mask_field_ids(
        batch.field_ids,
        batch.attention_mask,
        mask_field_id=int(data_cfg["mask_field_id"]),
        mask_probability=float(data_cfg["mask_probability"]),
        generator=generator,
    )
    return MaskedMSMBatch(
        input_ids=batch.input_ids,
        field_ids=masked_field_ids,
        original_field_ids=batch.field_ids,
        labels=labels,
        attention_mask=batch.attention_mask,
        mask_positions=mask_positions,
    )


def _attention_projection_gradient_norms(model) -> list[float]:
    """Return the joint Q/K/V weight-gradient norm for each encoder layer.

    The value for a layer is the L2 norm over all gradient entries in its
    attention Q, K, and V projection weight matrices. Call this only after
    gradient accumulation and before clipping.
    """

    import torch

    norms: list[float] = []
    for layer in model.encoder.layers:
        squared_norm = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
        for projection in (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj):
            gradient = projection.weight.grad
            if gradient is not None:
                squared_norm = squared_norm + gradient.detach().float().pow(2).sum()
        norms.append(float(squared_norm.sqrt().cpu()))
    return norms


def _batch_to_device(batch, device):
    from structformer.tasks.msm import MSMBatch

    return MSMBatch(
        input_ids=batch.input_ids.to(device),
        field_ids=batch.field_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
    )


def _evaluate(model, loader, data_cfg: dict[str, Any], generator, device) -> dict[str, float]:
    import torch

    from structformer.tasks.msm import MSM_IGNORE_INDEX, msm_cross_entropy

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    batches = 0
    with torch.no_grad():
        for collated in loader:
            batch = _batch_to_device(collated.batch, device)
            masked_batch = _mask_batch(batch, data_cfg, generator)
            output = model(masked_batch.input_ids, masked_batch.field_ids, attention_mask=masked_batch.attention_mask)
            loss = msm_cross_entropy(output.logits, masked_batch.labels)
            mask = masked_batch.labels.ne(MSM_IGNORE_INDEX)
            predictions = output.logits.argmax(dim=-1)
            total_correct += int(predictions[mask].eq(masked_batch.labels[mask]).sum().item())
            total_masked += int(mask.sum().item())
            total_loss += float(loss.detach().cpu())
            batches += 1
    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": total_correct / max(total_masked, 1),
    }


def _log_progress(
    logger: RunLogger,
    *,
    phase: str,
    task: str,
    model_name: str,
    seed: int,
    step: int,
    max_steps: int,
    loss: float,
    lr: float,
    examples_per_sec: float | None,
    device_name: str,
    memory: float | None,
    elapsed: float,
    split: str,
    accuracy: float | None = None,
    grad_norm: float | None = None,
) -> None:
    logger.log_metric(
        {
            "phase": phase,
            "task": task,
            "model": model_name,
            "seed": seed,
            "split": split,
            "step": step,
            "max_steps": max_steps,
            "loss": loss,
            "accuracy": accuracy,
            "lr": lr,
            "grad_norm": grad_norm,
            "examples_per_sec": examples_per_sec,
            "device": device_name,
            "memory_gb": memory,
            "elapsed_seconds": elapsed,
        }
    )


def _learning_rate_for_step(
    *,
    base_learning_rate: float,
    step: int,
    max_steps: int,
    schedule: str,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if schedule == "constant":
        return base_learning_rate
    if schedule != "linear_warmup_cosine":
        raise ValueError(f"Unsupported training.lr_schedule: {schedule}")
    if warmup_steps < 0:
        raise ValueError("training.warmup_steps must be non-negative")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("training.min_lr_ratio must be in [0, 1]")

    step_index = step - 1
    if warmup_steps > 0 and step_index < warmup_steps:
        warmup_fraction = step_index / max(warmup_steps, 1)
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * warmup_fraction
        return base_learning_rate * factor

    decay_steps = max(max_steps - warmup_steps, 1)
    decay_progress = min(max((step_index - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor
    return base_learning_rate * factor


def _set_optimizer_lr(optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _infer_vocab_size(tokenizer_json: str | None, dataset) -> int:
    if tokenizer_json:
        from tokenizers import Tokenizer

        return int(Tokenizer.from_file(str(tokenizer_json)).get_vocab_size())
    max_id = max(max(record.input_ids) for record in dataset.records)
    return int(max_id + 1)


def _make_generator(torch_module, device, seed: int):
    try:
        generator = torch_module.Generator(device=device)
    except (RuntimeError, TypeError):
        return None
    generator.manual_seed(seed)
    return generator


def _make_data_loader_generator(torch_module, seed: int):
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


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


def _save_diagnostic_checkpoint(path: Path, model, step: int, config: dict[str, Any]) -> None:
    """Lightweight checkpoint for attention diagnostics — model weights only, no optimizer."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train MSM on prepared DBpedia JSONL artifacts.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/msm"))
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--tokenizer-json", type=Path, default=None)
    parser.add_argument("--model", choices=["baseline", "saab", "casa"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--train-sample-size", type=int, default=None)
    parser.add_argument("--val-sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr-schedule", choices=["constant", "linear_warmup_cosine"], default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--scale-embeddings", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override model_config.num_layers (e.g. 4, 6, 8, 12).",
    )
    parser.add_argument(
        "--saab-field-weight",
        type=float,
        default=None,
        help="Override saab_field_weight in model_config (e.g. 0.0, 0.25, 0.5, 1.0, 2.0).",
    )
    parser.add_argument(
        "--saab-layer-mask",
        type=str,
        default=None,
        help=(
            "Comma-separated per-layer bias multipliers (one per layer). "
            "Empty layers receive no bias. "
            "Examples: '0,0,0,1' for L3-only; '1,1,1,0' for all layers except L3."
        ),
    )
    parser.add_argument(
        "--saab-shuffle-bias",
        action="store_true",
        help=(
            "Bias-content control: build the SAAB attention bias from a per-example "
            "permutation of field_ids over valid non-padding positions (same valid-token "
            "label multiset, bias magnitude, and number of biased valid-token pairs, but "
            "unaligned with visible structure). Embeddings use the unshuffled field_ids."
        ),
    )
    parser.add_argument(
        "--saab-shuffle-seed",
        type=int,
        default=0,
        help="Seed for the --saab-shuffle-bias permutation (default 0).",
    )
    parser.add_argument(
        "--diagnostic-steps",
        type=str,
        default="",
        help="Comma-separated steps to save lightweight diagnostic checkpoints, e.g. 1,10,50,100,200,300",
    )
    parser.add_argument(
        "--log-layer-gradients",
        action="store_true",
        help=(
            "Log per-step Q/K/V attention weight-gradient norms after gradient "
            "accumulation and before global clipping."
        ),
    )
    args = parser.parse_args(argv)

    config: dict[str, Any] = {}
    if args.config is not None:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(str(exc))
            return 1
    if args.train_jsonl is not None:
        config = deep_merge(config, {"data": {"train_jsonl": str(args.train_jsonl)}})
    if args.val_jsonl is not None:
        config = deep_merge(config, {"data": {"val_jsonl": str(args.val_jsonl)}})
    if args.tokenizer_json is not None:
        config = deep_merge(config, {"data": {"tokenizer_json": str(args.tokenizer_json)}})
    if args.model is not None:
        config = deep_merge(config, {"model": args.model})
    if args.seed is not None:
        config = deep_merge(config, {"seed": args.seed})
    if args.max_steps is not None:
        config = deep_merge(config, {"training": {"max_steps": args.max_steps}})
    if args.max_records is not None:
        config = deep_merge(config, {"data": {"max_records": args.max_records}})
    if args.train_sample_size is not None:
        config = deep_merge(config, {"data": {"train_sample_size": args.train_sample_size}})
    if args.val_sample_size is not None:
        config = deep_merge(config, {"data": {"val_sample_size": args.val_sample_size}})
    if args.sample_seed is not None:
        config = deep_merge(config, {"data": {"sample_seed": args.sample_seed}})
    if args.gradient_accumulation_steps is not None:
        config = deep_merge(config, {"training": {"gradient_accumulation_steps": args.gradient_accumulation_steps}})
    if args.lr_schedule is not None:
        config = deep_merge(config, {"training": {"lr_schedule": args.lr_schedule}})
    if args.warmup_steps is not None:
        config = deep_merge(config, {"training": {"warmup_steps": args.warmup_steps}})
    if args.min_lr_ratio is not None:
        config = deep_merge(config, {"training": {"min_lr_ratio": args.min_lr_ratio}})
    if args.scale_embeddings:
        config = deep_merge(config, {"model_config": {"scale_embeddings": True}})
    if args.num_layers is not None:
        config = deep_merge(config, {"model_config": {"num_layers": args.num_layers}})
    if args.saab_field_weight is not None:
        config = deep_merge(config, {"model_config": {"saab_field_weight": args.saab_field_weight}})
    if args.saab_layer_mask is not None:
        mask = [float(x.strip()) for x in args.saab_layer_mask.split(",") if x.strip()]
        config = deep_merge(config, {"model_config": {"saab_layer_mask": mask}})
    if args.saab_shuffle_bias:
        config = deep_merge(config, {"model_config": {"saab_shuffle_bias": True}})
        config = deep_merge(config, {"model_config": {"saab_shuffle_seed": args.saab_shuffle_seed}})
    if args.allow_cpu:
        config = deep_merge(config, {"allow_cpu": True})
    if args.diagnostic_steps:
        diag_steps = [int(s.strip()) for s in args.diagnostic_steps.split(",") if s.strip()]
        config = deep_merge(config, {"training": {"checkpoint_diagnostic_steps": diag_steps}})
    if args.log_layer_gradients:
        config = deep_merge(config, {"training": {"log_layer_gradients": True}})

    try:
        run_dir = run_train_msm(config, run_dir=args.run_dir)
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "tokenizers"}:
            print(f"{exc.name} is required. Run scripts/check_env.py for guidance.")
            return 1
        raise
    except (DeviceError, ValueError) as exc:
        print(str(exc))
        return 1

    print(f"MSM run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
