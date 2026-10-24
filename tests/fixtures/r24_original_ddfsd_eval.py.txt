"""Few-shot binary evaluation helpers for DDFSD."""

import math
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.amp import autocast

from datasets.ddfsd_datasets import (
    load_ddfsd_class_dataset,
    make_subset_loader,
    sample_support_query_indices,
)
from model.ddfsd_losses import compute_alpha, compute_prototypes, compute_query_logits, compute_support_sigmas


VALID_MODEL_MODES = {"dual", "rgb-only", "freq-only"}


def binary_metrics(labels: List[int], fake_scores: List[float], preds: List[int]) -> Dict[str, float]:
    labels_np = np.asarray(labels, dtype=np.int64)
    scores_np = np.asarray(fake_scores, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.int64)
    if not (labels_np.size == scores_np.size == preds_np.size) or labels_np.size == 0:
        raise ValueError("labels, fake_scores, and preds must have the same non-zero length.")
    if set(labels_np.tolist()) != {0, 1}:
        raise ValueError("Binary metrics require both real (0) and fake (1) query labels.")
    if not np.isfinite(scores_np).all():
        raise ValueError("fake_scores must contain only finite values.")
    if not set(preds_np.tolist()).issubset({0, 1}):
        raise ValueError("Binary predictions must contain only 0 (real) or 1 (fake).")

    real_mask = labels_np == 0
    fake_mask = labels_np == 1
    real_acc = float((preds_np[real_mask] == labels_np[real_mask]).mean())
    fake_acc = float((preds_np[fake_mask] == labels_np[fake_mask]).mean())
    metrics = {
        "acc": float((preds_np == labels_np).mean()),
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "balanced_acc": (real_acc + fake_acc) / 2.0,
        "ap": float(average_precision_score(labels_np, scores_np)),
        "auc": float(roc_auc_score(labels_np, scores_np)),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Binary metrics produced a non-finite value: {metrics}")
    return metrics


def encode_batch(model, images: torch.Tensor, device: torch.device, use_fp16: bool):
    images = images.to(device=device, non_blocking=True)
    autocast_enabled = use_fp16 and device.type == "cuda"
    with autocast(device_type="cuda", enabled=autocast_enabled):
        outputs = model(images)
    z_rgb = outputs.get("z_rgb")
    z_freq = outputs.get("z_freq")
    return (z_rgb.float() if z_rgb is not None else None, z_freq.float() if z_freq is not None else None)


def _stack_dataset_items(dataset, indices):
    return torch.stack([dataset[index][0] for index in indices], dim=0)


@torch.no_grad()
def evaluate_binary_few_shot(
    model,
    data_root: str,
    fake_class: str,
    support_shot: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_fp16: bool,
    tau: float,
    tau_r: float,
    max_query_per_class: int = 0,
    model_mode: str = "dual",
    branch_mode: str = None,
    return_indices: bool = False,
    strict_images: bool = False,
) -> Dict[str, Any]:
    """Evaluate real-vs-fake with fixed support and remaining val images as query."""

    if model_mode not in VALID_MODEL_MODES:
        raise ValueError(f"Unknown model_mode '{model_mode}', expected one of {VALID_MODEL_MODES}")
    branch_mode = branch_mode or model_mode
    if branch_mode not in VALID_MODEL_MODES:
        raise ValueError(f"Unknown branch_mode '{branch_mode}', expected one of {VALID_MODEL_MODES}")
    if model_mode != "dual" and branch_mode != model_mode:
        raise ValueError(f"A {model_mode} checkpoint can only be evaluated with branch_mode={model_mode}.")

    model.eval()
    real_dataset = load_ddfsd_class_dataset(
        data_root, "real", "val", strict_images=strict_images
    )
    fake_dataset = load_ddfsd_class_dataset(
        data_root, fake_class, "val", strict_images=strict_images
    )
    max_query = max_query_per_class if max_query_per_class > 0 else None

    real_support_idx, real_query_idx = sample_support_query_indices(
        len(real_dataset), support_shot=support_shot, seed=seed, max_query=max_query
    )
    fake_support_idx, fake_query_idx = sample_support_query_indices(
        len(fake_dataset), support_shot=support_shot, seed=seed, max_query=max_query
    )

    support_images = torch.cat(
        [
            _stack_dataset_items(real_dataset, real_support_idx),
            _stack_dataset_items(fake_dataset, fake_support_idx),
        ],
        dim=0,
    )
    support_rgb_flat, support_freq_flat = encode_batch(model, support_images, device, use_fp16)
    support_rgb = (
        support_rgb_flat.reshape(2, support_shot, -1).permute(1, 0, 2).unsqueeze(0)
        if support_rgb_flat is not None
        else None
    )
    support_freq = (
        support_freq_flat.reshape(2, support_shot, -1).permute(1, 0, 2).unsqueeze(0)
        if support_freq_flat is not None
        else None
    )
    proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq, model_mode=model_mode)
    alpha = None
    if model_mode == "dual":
        sigma_rgb, sigma_freq = compute_support_sigmas(support_rgb, support_freq, proto_rgb, proto_freq)
        alpha = compute_alpha(sigma_rgb, sigma_freq, tau_r=tau_r)

    all_labels: List[int] = []
    all_scores: List[float] = []
    all_preds: List[int] = []

    for label, dataset, query_idx in (
        (0, real_dataset, real_query_idx),
        (1, fake_dataset, fake_query_idx),
    ):
        loader = make_subset_loader(
            dataset,
            query_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for images, _ in loader:
            query_rgb, query_freq = encode_batch(model, images, device, use_fp16)
            logits = compute_query_logits(
                query_rgb=query_rgb.unsqueeze(0) if query_rgb is not None else None,
                query_freq=query_freq.unsqueeze(0) if query_freq is not None else None,
                proto_rgb=proto_rgb,
                proto_freq=proto_freq,
                alpha=alpha,
                tau=tau,
                branch_mode=branch_mode,
                model_mode=model_mode,
            )["logits"].squeeze(0)
            prob = logits.softmax(dim=-1)
            all_scores.extend(prob[:, 1].detach().cpu().tolist())
            all_preds.extend(prob.argmax(dim=-1).detach().cpu().tolist())
            all_labels.extend([label] * prob.shape[0])

    metrics = binary_metrics(all_labels, all_scores, all_preds)
    metrics.update(
        {
            "num_real_support": len(real_support_idx),
            "num_fake_support": len(fake_support_idx),
            "num_real_query": len(real_query_idx),
            "num_fake_query": len(fake_query_idx),
            "alpha_mean": float(alpha.detach().float().mean().cpu()) if alpha is not None else "",
            "alpha_min": float(alpha.detach().float().min().cpu()) if alpha is not None else "",
            "alpha_max": float(alpha.detach().float().max().cpu()) if alpha is not None else "",
        }
    )
    if return_indices:
        metrics["sampling"] = {
            "real": {
                "support_indices": list(real_support_idx),
                "query_indices": list(real_query_idx),
                "paths": list(real_dataset.paths),
            },
            fake_class: {
                "support_indices": list(fake_support_idx),
                "query_indices": list(fake_query_idx),
                "paths": list(fake_dataset.paths),
            },
        }
    return metrics
