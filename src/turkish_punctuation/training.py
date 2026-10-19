"""Word-normalized BERT training steps and shared word-level validation metrics."""

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import islice
from math import ceil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import BertForTokenClassification, PreTrainedTokenizerFast

from turkish_punctuation.metrics import compute_metrics
from turkish_punctuation.preprocessing import make_dataloader
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID


@dataclass
class TrainingProgress:
    epoch: int = 0
    next_row: int = 0
    global_step: int = 0
    checkpoint_serial: int = 0
    epoch_loss_sum: float = 0.0
    epoch_words: int = 0
    best_score: float = -1.0
    best_epoch: int | None = None
    history: list[dict] = field(default_factory=list)


def validate_training_config(config: dict) -> None:
    """Reject numerical and storage policies the paired FP32 runner cannot honor."""
    opt, data, runtime = (config[key] for key in ("optimization", "data", "runtime"))
    fixed = {
        "optimizer": "torch.optim.AdamW",
        "world_size": 1,
        "flush_accumulation_at_epoch_end": True,
        "scheduler": "linear_with_warmup",
        "scheduler_step_unit": "successful_optimizer_update",
        "warmup_steps_rounding": "ceil_total_optimizer_steps_times_ratio",
        "early_stopping": False,
        "exclude_from_weight_decay": ["bias", "LayerNorm_parameters"],
        "loss": {
            "name": "cross_entropy",
            "class_weights": None,
            "label_smoothing": 0.0,
            "include_O": True,
            "reduction": "sum_then_divide_by_valid_words_in_accumulation_window",
        },
    }
    if any(opt.get(key) != value for key, value in fixed.items()):
        raise ValueError(
            "Unsupported optimizer/loss/scheduler policy; revise implementation together."
        )
    if (
        runtime["target_device"] != "cuda"
        or runtime["precision"] != "fp32"
        or runtime["dataloader_workers"] != 0
        or runtime["allow_tf32"] is not False
        or data["drop_last"] is not False
        or data["truncation"] is not False
        or data["padding"] != "dynamic"
        or data["pad_to_multiple_of"] is not None
        or data["alignment"] != "first_subword"
        or data["ignore_index"] != IGNORE_INDEX
        or data["train_split"] != "train"
        or data["validation_split"] != "validation"
        or data["test_access_during_training"] is not False
        or config["hyperparameter_search"]["enabled"] is not False
    ):
        raise ValueError(
            "This implementation requires CUDA/FP32, fixed splits and strict alignment."
        )
    selection = config["selection"]
    if (
        selection["evaluation_frequency"] != "epoch"
        or selection["split"] != "validation"
        or selection["condition"] != "clean"
        or selection["metric"] != "punctuation_macro.f1"
        or selection["greater_is_better"] is not True
        or selection["tie_break"] != "earliest_epoch"
        or selection["candidate_checkpoints"] != "completed_epoch_ends_only"
        or selection["noisy_validation_used_for_selection"] is not False
    ):
        raise ValueError("Only clean epoch-end macro-F1 selection is implemented.")
    positive_ints = [
        opt[k] for k in ("epochs", "per_device_train_batch_size", "gradient_accumulation_steps")
    ]
    positive_ints += [
        runtime["evaluation_batch_size"],
        config["artifacts"]["resumable_checkpoint_every_optimizer_steps"],
    ]
    if any(type(value) is not int or value < 1 for value in positive_ints):
        raise ValueError("Epochs, batch sizes and checkpoint intervals must be positive integers.")
    if (
        opt["per_device_train_batch_size"] * opt["gradient_accumulation_steps"]
        != opt["effective_batch_size"]
    ):
        raise ValueError("Physical batch × accumulation differs from effective batch.")
    if not 0 < opt["learning_rate"] < 1 or not 0 <= opt["warmup_ratio"] < 1:
        raise ValueError("Invalid learning rate/warmup ratio.")
    if opt["max_grad_norm"] <= 0 or opt["weight_decay"] < 0 or opt["epsilon"] <= 0:
        raise ValueError("Invalid optimizer scale/regularization.")
    if len(opt["betas"]) != 2 or any(not 0 <= beta < 1 for beta in opt["betas"]):
        raise ValueError("Adam betas must be in [0, 1).")
    if config["artifacts"]["keep_latest_resumable_checkpoints"] < 2:
        raise ValueError("Keep at least two committed resumable checkpoints.")
    artifacts = config["artifacts"]
    if artifacts.get("checkpoint_layout", "legacy_serial") not in (
        "legacy_serial",
        "two_reused_slots",
    ):
        raise ValueError("Unknown checkpoint layout.")
    if (
        artifacts.get("checkpoint_layout") == "two_reused_slots"
        and artifacts["keep_latest_resumable_checkpoints"] != 2
    ):
        raise ValueError("The fixed-slot protocol retains exactly two slots.")
    if (
        artifacts["overwrite_existing_runs"] is not False
        or artifacts["keep_candidate_epoch_exports"] is not True
        or artifacts["keep_best_and_final_model_exports"] is not True
        or artifacts["save_optimizer_scheduler_rng_and_sampler_state"] is not True
        or artifacts["archive_sources_configs_and_input_hashes"] is not True
        or selection["keep_final_checkpoint_for_matched_step_analysis"] is not True
    ):
        raise ValueError("Artifact/selection policy differs from the implemented retention rules.")
    model = config["model"]
    if (
        model["architecture"] != "BertForTokenClassification"
        or model["trainable_parameters"] != "all"
        or model["attention_implementation"] != "eager"
        or model["require_base_tokenizer_compatibility"] is not True
    ):
        raise ValueError("Unsupported model initialization policy.")
    if any(
        not 0 <= model[key] < 1
        for key in ("hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout")
    ):
        raise ValueError("Dropout probabilities must be in [0, 1).")


def training_budget(records: int, config: dict) -> dict:
    opt = config["optimization"]
    batches = ceil(records / opt["per_device_train_batch_size"])
    updates = ceil(batches / opt["gradient_accumulation_steps"])
    total = updates * opt["epochs"]
    return {
        "microbatches_per_epoch": batches,
        "updates_per_epoch": updates,
        "total_updates": total,
        "warmup_updates": ceil(total * opt["warmup_ratio"]),
    }


def epoch_permutation(records: int, seed: int, epoch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(10 * seed + epoch)
    return torch.randperm(records, generator=generator)


def make_optimizer(model: BertForTokenClassification, config: dict) -> torch.optim.AdamW:
    opt = config["optimization"]
    norm_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.LayerNorm)
        for parameter in module.parameters(recurse=False)
    }
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if name.endswith(".bias") or id(parameter) in norm_parameters else decay
        target.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": opt["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=opt["learning_rate"],
        betas=tuple(opt["betas"]),
        eps=opt["epsilon"],
        foreach=False,
        fused=False,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, budget: dict
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup, total = budget["warmup_updates"], budget["total_updates"]

    def scale(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def supervised_loss(
    model: BertForTokenClassification, batch: dict, *, device: torch.device
) -> torch.Tensor:
    labels = batch["labels"].to(device)
    inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
    logits = model(**inputs).logits
    if logits.shape[:2] != labels.shape or logits.shape[-1] != len(LABEL_TO_ID):
        raise RuntimeError("Logits do not match aligned labels.")
    return F.cross_entropy(
        logits.reshape(-1, len(LABEL_TO_ID)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )


def train_epoch(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    model: BertForTokenClassification,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    progress: TrainingProgress,
    permutation: torch.Tensor,
    config: dict,
    *,
    on_update: Callable[[dict], None],
    stop_at_step: int | None = None,
) -> None:
    """Accumulate summed word losses; callbacks run only after complete optimizer steps."""
    opt = config["optimization"]
    device = next(model.parameters()).device
    remaining = dataset.select(permutation[progress.next_row :].tolist())
    loader = iter(
        make_dataloader(remaining, tokenizer, batch_size=opt["per_device_train_batch_size"])
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while True:
        torch.cuda.synchronize(device)
        started = perf_counter()
        window = list(islice(loader, opt["gradient_accumulation_steps"]))
        if not window:
            return
        word_count = sum(int((batch["labels"] != IGNORE_INDEX).sum()) for batch in window)
        if word_count < 1:
            raise ValueError("Accumulation window has no supervised words.")
        loss_total = 0.0
        for batch in window:
            loss = supervised_loss(model, batch, device=device)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Training loss is nonfinite; no update was committed.")
            loss_total += float(loss.detach().item())
            (loss / word_count).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), opt["max_grad_norm"], error_if_nonfinite=True
        )
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        seconds = perf_counter() - started
        progress.next_row += sum(batch["input_ids"].shape[0] for batch in window)
        progress.global_step += 1
        progress.epoch_loss_sum += loss_total
        progress.epoch_words += word_count
        on_update(
            {
                "epoch": progress.epoch + 1,
                "global_step": progress.global_step,
                "records_in_epoch": progress.next_row,
                "window_words": word_count,
                "window_loss": loss_total / word_count,
                "lr_used": lr_used,
                "lr_next": optimizer.param_groups[0]["lr"],
                "gradient_norm_before_clip": float(grad_norm.item()),
                "optimizer_step_seconds": seconds,
            }
        )
        if stop_at_step is not None and progress.global_step >= stop_at_step:
            return


def evaluate_validation(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    model: BertForTokenClassification,
    *,
    batch_size: int,
    prediction_path: Path,
) -> dict:
    """Score one word target once; never pass reference class values into forward()."""
    import json

    device = next(model.parameters()).device
    order = sorted(range(len(dataset)), key=lambda i: (len(dataset[i]["input_ids"]), i))
    ordered = dataset.select(order)
    loader = make_dataloader(ordered, tokenizer, batch_size=batch_size)
    row_ids = list(ordered["row_id"])
    reference, predicted = [], []
    loss_sum, cursor = 0.0, 0
    started = perf_counter()
    model.eval()
    with prediction_path.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            logits = model(**inputs).logits
            if logits.shape[:2] != labels.shape or logits.shape[-1] != len(LABEL_TO_ID):
                raise RuntimeError("Validation logits/labels differ in shape.")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("Validation produced nonfinite logits.")
            loss_sum += float(
                F.cross_entropy(
                    logits.reshape(-1, len(LABEL_TO_ID)),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                ).item()
            )
            guesses, targets = logits.argmax(-1).cpu(), labels.cpu()
            for guess, target in zip(guesses, targets, strict=True):
                positions = target != IGNORE_INDEX
                true, pred = target[positions].tolist(), guess[positions].tolist()
                reference.extend(true)
                predicted.extend(pred)
                stream.write(
                    json.dumps(
                        {"row_id": row_ids[cursor], "reference_ids": true, "prediction_ids": pred},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                cursor += 1
    if cursor != len(dataset):
        raise RuntimeError("Validation did not visit every eligible record.")
    return {
        "metrics": compute_metrics(reference, predicted),
        "word_loss": loss_sum / len(reference),
        "seconds": perf_counter() - started,
    }
