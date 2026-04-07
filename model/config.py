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


