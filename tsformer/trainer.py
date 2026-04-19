"""
transactFormer — trainer.py

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Training loops for all three phases described in the paper:

    Phase 1 — Pre-training   : NTP on raw transaction sequences (Section 3.1)
    Phase 2 — Fine-tuning    : Binary classification with LoRA (Section 3.2)
    Phase 3 — Joint Fusion   : End-to-end transactformer training (Section 3.3)

Each trainer follows the same interface:

    trainer.fit(train_loader, val_loader)
    trainer.evaluate(loader) -> metrics dict
"""

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

from .model import TransactionTransformer, transactFormer


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


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _build_optimizer_and_scheduler(
    model:        nn.Module,
    lr:           float,
    weight_decay: float,
    n_steps:      int,
    warmup_steps: int,
) -> tuple[AdamW, SequentialLR]:
    # Separate weight-decayed and non-decayed params (biases & norms skip decay)
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.ndim >= 2]
    no_decay     = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.ndim < 2]
    opt = AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay,     "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )

    warmup   = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine   = CosineAnnealingLR(opt, T_max=max(1, n_steps - warmup_steps), eta_min=lr * 0.1)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return opt, scheduler


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _log(step: int, total: int, metrics: dict, phase: str, t0: float) -> None:
    elapsed = time.time() - t0
    mstr = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    print(f"[{phase}] step {step:>6}/{total}  {mstr}  elapsed={elapsed:.0f}s")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Pre-training  (Section 3.1)
# ──────────────────────────────────────────────────────────────────────────────

class PreTrainer:
    """
    Trains TransactionTransformer with the Next-Token Prediction (NTP) objective.
    Uses mixed-precision (fp16/bf16) for GPU efficiency.
    """

    def __init__(
        self,
        model:          TransactionTransformer,
        device:         torch.device,
        lr:             float = 3e-4,
        weight_decay:   float = 0.1,
        max_grad_norm:  float = 1.0,
        warmup_ratio:   float = 0.03,
        amp_dtype:      str   = "bf16",   # "fp16" | "bf16" | "none"
        checkpoint_dir: Optional[str] = None,
        log_every:      int   = 50,
    ) -> None:
        self.model          = model.to(device)
        self.device         = device
        self.max_grad_norm  = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_every      = log_every

        self._amp_dtype = (
            torch.bfloat16 if amp_dtype == "bf16"
            else torch.float16 if amp_dtype == "fp16"
            else None
        )
        self._use_amp = self._amp_dtype is not None
        self._scaler  = GradScaler(enabled=(self._amp_dtype == torch.float16))

        self._lr           = lr
        self._weight_decay = weight_decay
        self._warmup_ratio = warmup_ratio

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
        epochs:       int = 3,
    ) -> None:
        n_steps      = len(train_loader) * epochs
        warmup_steps = max(1, int(n_steps * self._warmup_ratio))
        opt, sched   = _build_optimizer_and_scheduler(
            self.model, self._lr, self._weight_decay, n_steps, warmup_steps
        )

        global_step = 0
        t0          = time.time()
        best_val    = float("inf")

        print(f"\n{'='*60}")
        print(f"  Phase 1 — Pre-training  |  {n_steps} steps  |  {epochs} epochs")
        counts = self.model.param_count() if hasattr(self.model, "param_count") else {}
        if counts:
            print(f"  Params: total={counts['total']:,}  trainable={counts['trainable']:,}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                batch = _to_device(batch, self.device)

                with _amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                    out  = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    loss = out["loss"]

                self._scaler.scale(loss).backward()
                self._scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(opt)
                self._scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

                running_loss += loss.item()
                global_step  += 1

                if global_step % self.log_every == 0:
                    avg = running_loss / self.log_every
                    _log(global_step, n_steps, {"loss": avg, "lr": sched.get_last_lr()[0]},
                         f"pretrain ep{epoch}", t0)
                    running_loss = 0.0

            # Validation
            if val_loader is not None:
                val_loss = self._eval_loss(val_loader)
                print(f"  [epoch {epoch}] val_loss={val_loss:.4f}")
                if val_loss < best_val and self.checkpoint_dir:
                    best_val = val_loss
                    self._save("pretrain_best.pt")

        if self.checkpoint_dir:
            self._save("pretrain_final.pt")

    @torch.no_grad()
    def _eval_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            batch = _to_device(batch, self.device)
            with _amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
            total += out["loss"].item() * batch["input_ids"].size(0)
            n     += batch["input_ids"].size(0)
        self.model.train()
        return total / max(n, 1)

    def _save(self, name: str) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / name
        torch.save({"model_state": self.model.state_dict()}, path)
        print(f"  ✓ Checkpoint saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Fine-tuning with LoRA  (Section 3.2)
# ──────────────────────────────────────────────────────────────────────────────

class FineTuneTrainer:
    """
    Fine-tunes a pre-trained TransactionTransformer + MLP head for binary
    classification, using LoRA to prevent catastrophic forgetting (Section 3.2).
    """

    def __init__(
        self,
        model:          nn.Module,     # TransactionTransformer + ClassificationHead wrapper
        device:         torch.device,
        lr:             float = 1e-4,
        weight_decay:   float = 0.01,
        max_grad_norm:  float = 1.0,
        warmup_ratio:   float = 0.05,
        amp_dtype:      str   = "bf16",
        checkpoint_dir: Optional[str] = None,
        log_every:      int   = 50,
        pos_weight:     Optional[float] = None,  # for class imbalance
    ) -> None:
        self.model          = model.to(device)
        self.device         = device
        self.max_grad_norm  = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_every      = log_every

        self._amp_dtype = (
            torch.bfloat16 if amp_dtype == "bf16"
            else torch.float16 if amp_dtype == "fp16"
            else None
        )
        self._use_amp = self._amp_dtype is not None
        self._scaler  = GradScaler(enabled=(self._amp_dtype == torch.float16))
        self._pos_weight = (
            torch.tensor([pos_weight], device=device) if pos_weight else None
        )

        self._lr           = lr
        self._weight_decay = weight_decay
        self._warmup_ratio = warmup_ratio

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
        epochs:       int = 5,
    ) -> None:
        n_steps      = len(train_loader) * epochs
        warmup_steps = max(1, int(n_steps * self._warmup_ratio))
        opt, sched   = _build_optimizer_and_scheduler(
            self.model, self._lr, self._weight_decay, n_steps, warmup_steps
        )

        global_step = 0
        t0          = time.time()
        best_auc    = 0.0

        print(f"\n{'='*60}")
        print(f"  Phase 2 — Fine-tuning (LoRA)  |  {n_steps} steps  |  {epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                batch = _to_device(batch, self.device)

                with _amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                    logits = self._forward_logits(batch)
                    loss   = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, batch["labels"], pos_weight=self._pos_weight
                    )

                self._scaler.scale(loss).backward()
                self._scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self._scaler.step(opt)
                self._scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

                running_loss += loss.item()
                global_step  += 1

                if global_step % self.log_every == 0:
                    avg = running_loss / self.log_every
                    _log(global_step, n_steps,
                         {"loss": avg, "lr": sched.get_last_lr()[0]},
                         f"finetune ep{epoch}", t0)
                    running_loss = 0.0

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                print(f"  [epoch {epoch}] val: {metrics}")
                if metrics["auc"] > best_auc and self.checkpoint_dir:
                    best_auc = metrics["auc"]
                    self._save("finetune_best.pt")

        if self.checkpoint_dir:
            self._save("finetune_final.pt")

    def _forward_logits(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement _forward_logits")

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_labels, all_scores = [], []
        total_loss = 0.0

        for batch in loader:
            batch = _to_device(batch, self.device)
            with _amp.autocast(device_type=self.device.type, dtype=self._amp_dtype, enabled=self._use_amp):
                logits = self._forward_logits(batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch["labels"]
            )
            total_loss  += loss.item() * batch["labels"].size(0)
            all_labels  += batch["labels"].cpu().tolist()
            all_scores  += logits.cpu().tolist()

        self.model.train()
        metrics = _binary_metrics(all_labels, all_scores)
        metrics["loss"] = round(total_loss / max(len(all_labels), 1), 6)
        return metrics

    def _save(self, name: str) -> None:
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self.checkpoint_dir / name
            torch.save({"model_state": self.model.state_dict()}, path)
            print(f"  ✓ Checkpoint saved → {path}")


class EmbeddingClassifier(nn.Module):
    """
    Thin wrapper used in Phase 2: TransactionTransformer + classification MLP.
    (Figure 4 in the paper)
    """

    def __init__(self, transformer: TransactionTransformer, d_model: int,
                 hidden_dims: tuple[int, ...] = (128,), dropout: float = 0.1) -> None:
        super().__init__()
        from .model import ClassificationHead
        self.transformer = transformer
        self.head        = ClassificationHead(d_model, hidden_dims, dropout)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.transformer.get_user_embedding(input_ids, attention_mask)
        return self.head(emb)

    def param_count(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class EmbeddingClassifierTrainer(FineTuneTrainer):
    """Phase 2 trainer for EmbeddingClassifier."""

    def _forward_logits(self, batch: dict) -> torch.Tensor:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Joint Fusion (transactformerr)  (Section 3.3)
# ──────────────────────────────────────────────────────────────────────────────

class JointFusionTrainer(FineTuneTrainer):
    """
    Trains the full transactformerr model end-to-end:
    TransactionTransformer ⊕ PLR ⊕ DCNv2 → ClassificationHead.
    Gradient flows through both branches simultaneously (Figure 5, right).
    """

    def _forward_logits(self, batch: dict) -> torch.Tensor:
        out = self.model(
            input_ids=batch["input_ids"],
            tabular_feats=batch["tabular_feats"],
            attention_mask=batch.get("attention_mask"),
        )
        return out["logits"]

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
        epochs:       int = 5,
    ) -> None:
        print(f"\n{'='*60}")
        print(f"  Phase 3 — Joint Fusion (transactformerr)")
        if hasattr(self.model, "param_count"):
            counts = self.model.param_count()
            print(f"  Params: total={counts['total']:,}  trainable={counts['trainable']:,}")
        print(f"{'='*60}\n")

        # Delegate to parent — logic is identical, only _forward_logits differs
        super().fit(train_loader, val_loader, epochs)
