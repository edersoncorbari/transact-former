"""
transactFormer — config.py

Typed dataclasses for every hyper-parameter in the pipeline.

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Usage
─────
    from transactformer.config import transactformerConfig, TrainConfig, load_config, save_config

    cfg = transactformerConfig()    # sensible defaults
    cfg.model.d_model = 1024        # paper-330M override

    save_config(cfg, "run.json")
    cfg2 = load_config("run.json")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Model architecture
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """Section 3.1 — TransactionTransformer backbone."""
    d_model:     int   = 256     # 256 → ~24 M params  |  1024 → ~330 M params
    n_layers:    int   = 6       # paper uses 24 for both sizes
    n_heads:     int   = 8       # paper uses 16 for both sizes
    d_ff:        Optional[int] = None   # None → 4 × d_model
    max_seq_len: int   = 2048    # paper explores 512 / 1024 / 2048
    dropout:     float = 0.1

    # Convenience presets matching the paper
    @classmethod
    def small(cls) -> "TransformerConfig":
        """Fast local testing — ~115 K params."""
        return cls(d_model=64, n_layers=2, n_heads=4, max_seq_len=512)

    @classmethod
    def medium(cls) -> "TransformerConfig":
        """Development preset — ~4 M params."""
        return cls(d_model=256, n_layers=6, n_heads=8, max_seq_len=1024)

    @classmethod
    def paper_24m(cls) -> "TransformerConfig":
        """Matches the 24 M parameter model in the paper."""
        return cls(d_model=256, n_layers=24, n_heads=16, max_seq_len=2048)

    @classmethod
    def paper_330m(cls) -> "TransformerConfig":
        """Matches the 330 M parameter model in the paper."""
        return cls(d_model=1024, n_layers=24, n_heads=16, max_seq_len=2048)


@dataclass
class LoRAConfig:
    """Section 3.2 — Low-Rank Adaptation for fine-tuning."""
    enabled:        bool  = True
    rank:           int   = 8
    alpha:          float = 16.0
    dropout:        float = 0.05
    target_modules: tuple = ("qkv", "proj")   # attention layers to adapt


@dataclass
class PLRConfig:
    """Section 3.3 — Periodic Linear embedding for numerical tabular features."""
    n_frequencies: int = 32    # number of sin/cos frequencies per feature
    d_out:         int = 32    # output dim per feature before DCNv2


@dataclass
class DCNv2Config:
    """Section 3.3 — Deep & Cross Network V2 for tabular features."""
    d_cross:  int   = 128
    n_cross:  int   = 3
    d_deep:   int   = 256
    n_deep:   int   = 2
    d_out:    int   = 64
    dropout:  float = 0.1


@dataclass
class HeadConfig:
    """Classification MLP head (Figure 4)."""
    hidden_dims: tuple = (256, 128)
    dropout:     float = 0.1


@dataclass
class ModelConfig:
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    lora:        LoRAConfig        = field(default_factory=LoRAConfig)
    plr:         PLRConfig         = field(default_factory=PLRConfig)
    dcn:         DCNv2Config       = field(default_factory=DCNv2Config)
    head:        HeadConfig        = field(default_factory=HeadConfig)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PreTrainConfig:
    """Section 3.1 — NTP pre-training."""
    lr:            float = 3e-4
    weight_decay:  float = 0.1
    warmup_ratio:  float = 0.03
    max_grad_norm: float = 1.0
    epochs:        int   = 3
    batch_size:    int   = 32
    val_frac:      float = 0.05
    amp_dtype:     str   = "bf16"   # "bf16" | "fp16" | "none"
    log_every:     int   = 50


@dataclass
class FineTuneConfig:
    """Section 3.2 — LoRA fine-tuning for classification."""
    lr:            float = 1e-4
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.05
    max_grad_norm: float = 1.0
    epochs:        int   = 5
    batch_size:    int   = 32
    val_frac:      float = 0.10
    amp_dtype:     str   = "bf16"
    log_every:     int   = 50
    pos_weight:    Optional[float] = None   # None → auto-computed from data


@dataclass
class JointFusionConfig:
    """Section 3.3 — transactformer joint fusion training."""
    lr:            float = 5e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.05
    max_grad_norm: float = 1.0
    epochs:        int   = 5
    batch_size:    int   = 32
    val_frac:      float = 0.10
    amp_dtype:     str   = "bf16"
    log_every:     int   = 50
    pos_weight:    Optional[float] = None


@dataclass
class TrainConfig:
    pretrain:     PreTrainConfig     = field(default_factory=PreTrainConfig)
    finetune:     FineTuneConfig     = field(default_factory=FineTuneConfig)
    joint_fusion: JointFusionConfig  = field(default_factory=JointFusionConfig)
    num_workers:  int  = 4
    seed:         int  = 42


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    data_dir:       str = "./transactformer_data"
    checkpoint_dir: str = "./checkpoints"
    results_dir:    str = "./results"
    vocab_min_freq: int = 3


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class transactformerConfig:
    model:  ModelConfig = field(default_factory=ModelConfig)
    train:  TrainConfig = field(default_factory=TrainConfig)
    paths:  PathConfig  = field(default_factory=PathConfig)
    device: str         = "auto"   # "auto" | "cpu" | "cuda" | "cuda:1"

    @classmethod
    def for_paper_24m(cls) -> "transactformerConfig":
        cfg = cls()
        cfg.model.transformer = TransformerConfig.paper_24m()
        cfg.train.pretrain.epochs        = 3
        cfg.train.pretrain.batch_size    = 64
        cfg.train.finetune.epochs        = 5
        cfg.train.joint_fusion.epochs    = 5
        cfg.train.joint_fusion.batch_size = 64
        return cfg

    @classmethod
    def for_paper_330m(cls) -> "transactformerConfig":
        cfg = cls()
        cfg.model.transformer = TransformerConfig.paper_330m()
        cfg.train.pretrain.epochs        = 3
        cfg.train.pretrain.batch_size    = 32   # larger model → smaller batch
        cfg.train.finetune.epochs        = 5
        cfg.train.joint_fusion.epochs    = 5
        cfg.train.joint_fusion.batch_size = 32
        return cfg

    @classmethod
    def for_local_test(cls) -> "transactformerConfig":
        cfg = cls()
        cfg.model.transformer = TransformerConfig.small()
        cfg.train.pretrain.epochs        = 1
        cfg.train.pretrain.batch_size    = 16
        cfg.train.pretrain.amp_dtype     = "none"
        cfg.train.finetune.epochs        = 2
        cfg.train.finetune.batch_size    = 16
        cfg.train.finetune.amp_dtype     = "none"
        cfg.train.joint_fusion.epochs    = 2
        cfg.train.joint_fusion.batch_size = 16
        cfg.train.joint_fusion.amp_dtype = "none"
        cfg.train.num_workers            = 0
        cfg.device                       = "cpu"
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_dict(obj) -> dict:
    """Recursively convert dataclasses to plain dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    return obj


def save_config(cfg: transactformerConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"  Config saved → {path}")


def load_config(path: str | Path) -> transactformerConfig:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    def _build(cls, data):
        import dataclasses
        if not dataclasses.is_dataclass(cls):
            return data
        hints = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}
        for k, v in data.items():
            field_type = hints.get(k)
            if field_type and isinstance(v, dict):
                try:
                    sub_cls = eval(field_type) if isinstance(field_type, str) else field_type
                    kwargs[k] = _build(sub_cls, v)
                    continue
                except Exception:
                    pass
            # tuples are serialised as lists by json
            if isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        return cls(**kwargs)

    return _build(transactformerConfig, d)