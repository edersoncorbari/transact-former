from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.amp as _amp
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# from .model import TransactionTransformer

# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _auc_from_scores(labels: list[float], scores: list[float]) -> float:
    """Exact AUC via sorting (no sklearn dependency)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = auc = 0.0
    prev_fp = 0.0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp * (fp - prev_fp) / (n_pos * n_neg)  # trapezoidal area
            prev_fp = fp  # noqa: SIM113 — intentional
    return auc


def _binary_metrics(labels: list[float], scores: list[float]) -> dict[str, float]:
    auc = _auc_from_scores(labels, scores)
    preds = [1 if s >= 0.0 else 0 for s in scores]  # logit threshold 0
    correct = sum(p == int(l) for p, l in zip(preds, labels))
    return {
        "auc":      round(auc, 6),
        "accuracy": round(correct / max(len(labels), 1), 6),
    }


