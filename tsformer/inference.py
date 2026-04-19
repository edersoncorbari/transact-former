"""
tsFormer — inference.py

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Production-ready serving utilities.

Design principle
────────────────
Checkpoints (.pt) store only model_state_dict.
Architecture is reconstructed from config.json (auto-saved by train_pipeline).
This avoids brittle coupling between checkpoint format and model class signatures.
"""

from __future__ import annotations

import json
import time
import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tsformer.config import load_config, tsFormerConfig
from tsformer.data import Vocabulary, tokenise_member
from tsformer.model import (
    TransactionTransformer,
    tsFormer,
    apply_lora_to_transformer,
)
from tsformer.trainer import EmbeddingClassifier


# ──────────────────────────────────────────────────────────────────────────────
# Input / output schemas
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MemberInput:
    """
    Single member inference request.

    transactions:     list of dicts, each with:
                        amount (float), date (str YYYY-MM-DD), description (str)
    tabular_features: dict[str, float] — raw (un-normalised) feature values.
                      Missing keys are filled with 0.0.
                      Only needed for tsFormerPredictor.
    member_id:        optional identifier, returned in PredictionResult.
    """

    transactions: list[dict]
    tabular_features: dict = field(default_factory=dict)
    member_id: Optional[str] = None


@dataclass
class PredictionResult:
    member_id: Optional[str]
    logit: float
    probability: float
    latency_ms: float


# ──────────────────────────────────────────────────────────────────────────────
# Tabular normalisation metadata helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_tabular_meta(parquet_path: str | Path) -> dict:
    """
    Compute tabular normalisation metadata from the training parquet.
    Returns a dict with keys: tab_cols, tab_means, tab_stds.
    """
    import pandas as pd

    tab_df = pd.read_parquet(parquet_path)
    meta_cols = {"member_id", "score_date", "label"}
    string_cols = set(tab_df.select_dtypes(include="object").columns) - meta_cols
    exclude = meta_cols | string_cols
    cols = [c for c in tab_df.columns if c not in exclude]

    tab_df[cols] = tab_df[cols].fillna(0.0).astype(float)
    means = tab_df[cols].mean().to_dict()
    stds = tab_df[cols].std().fillna(1.0).replace(0.0, 1.0).to_dict()
    return {"tab_cols": cols, "tab_means": means, "tab_stds": stds}


def save_tabular_meta(
    parquet_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Compute and save tabular_meta.json alongside the checkpoints."""
    meta = build_tabular_meta(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Tabular meta saved → {output_path}  ({len(meta['tab_cols'])} features)")
    return meta


def load_tabular_meta(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Internal: model builders from config
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_state(checkpoint_path: str | Path) -> dict:
    """Load checkpoint and return the state_dict regardless of format."""
    raw = torch.load(checkpoint_path, map_location="cpu")
    return raw.get("model_state", raw)


def _build_transformer(cfg: tsFormerConfig, vocab_size: int) -> TransactionTransformer:
    tc = cfg.model.transformer
    return TransactionTransformer(
        vocab_size=vocab_size,
        d_model=tc.d_model,
        n_layers=tc.n_layers,
        n_heads=tc.n_heads,
        d_ff=tc.d_ff,
        max_seq_len=tc.max_seq_len,
        dropout=tc.dropout,
    )


def _build_embedding_classifier(
    cfg: tsFormerConfig, vocab_size: int, head_in: Optional[int] = None
) -> EmbeddingClassifier:
    tc, lc, hc = cfg.model.transformer, cfg.model.lora, cfg.model.head
    tfm = _build_transformer(cfg, vocab_size)

    if lc.enabled:
        apply_lora_to_transformer(
            tfm,
            rank=lc.rank,
            alpha=lc.alpha,
            dropout=lc.dropout,
            target_modules=lc.target_modules,
        )

    final_head_in = head_in if head_in is not None else (tc.d_model * 2)

    return EmbeddingClassifier(
        transformer=tfm,
        d_model=tc.d_model,
        hidden_dims=tuple(hc.hidden_dims),
        dropout=hc.dropout,
        head_in=final_head_in,
    )


def _build_nuformer(cfg: tsFormerConfig, vocab_size: int, n_tabular: int) -> tsFormer:
    tc, lc, pc, dc, hc = (
        cfg.model.transformer,
        cfg.model.lora,
        cfg.model.plr,
        cfg.model.dcn,
        cfg.model.head,
    )
    return tsFormer(
        vocab_size=vocab_size,
        n_tabular=n_tabular,
        d_model=tc.d_model,
        n_layers=tc.n_layers,
        n_heads=tc.n_heads,
        max_seq_len=tc.max_seq_len,
        transformer_dropout=tc.dropout,
        n_frequencies=pc.n_frequencies,
        plr_d_out=pc.d_out,
        d_cross=dc.d_cross,
        n_cross=dc.n_cross,
        d_deep=dc.d_deep,
        n_deep=dc.n_deep,
        dcn_d_out=dc.d_out,
        dcn_dropout=dc.dropout,
        head_hidden=hc.hidden_dims,
        head_dropout=hc.dropout,
        lora_rank=lc.rank if lc.enabled else 0,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared sequence encoding mixin
# ──────────────────────────────────────────────────────────────────────────────


class _BasePredictor:
    def __init__(self, vocab: Vocabulary, device: torch.device, max_seq_len: int):
        self.vocab = vocab
        self.device = device
        self.max_seq_len = max_seq_len

    def _encode_sequences(
        self, inputs: list[MemberInput]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_ids, max_len = [], 0
        for inp in inputs:
            tokens = tokenise_member(inp.transactions, max_seq_len=self.max_seq_len)
            ids = self.vocab.encode_sequence(tokens)
            all_ids.append(ids)
            max_len = max(max_len, len(ids))

        pad_id = self.vocab.pad_id
        padded, masks = [], []
        for ids in all_ids:
            pad = max_len - len(ids)
            padded.append(ids + [pad_id] * pad)
            masks.append([True] * len(ids) + [False] * pad)

        return (
            torch.tensor(padded, dtype=torch.long, device=self.device),
            torch.tensor(masks, dtype=torch.bool, device=self.device),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 1. tsFormerPredictor (joint fusion, Section 3.3)
# ──────────────────────────────────────────────────────────────────────────────


class tsFormerPredictor(_BasePredictor):
    """Score members with the full tsFormer joint-fusion model."""

    def __init__(
        self,
        model: tsFormer,
        vocab: Vocabulary,
        tab_cols: list[str],
        tab_means: dict[str, float],
        tab_stds: dict[str, float],
        device: torch.device,
        max_seq_len: int,
    ) -> None:
        super().__init__(vocab, device, max_seq_len)
        self.model = model.to(device).eval()
        self.tab_cols = tab_cols
        self.tab_means = tab_means
        self.tab_stds = tab_stds

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        device: str | torch.device = "auto",
        ckpt_name: str = "tsformer_final.pt",
        meta_name: str = "tabular_meta.json",
    ) -> "tsFormerPredictor":
        """
        Load from a training run directory containing:
            config.json, vocabulary.json, tsformer_final.pt, tabular_meta.json
        """
        run_dir = Path(run_dir)
        meta_path = run_dir / meta_name
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Ops... tabular_meta.json not found at {meta_path}."
            )

        return cls.from_checkpoint(
            checkpoint_path=run_dir / ckpt_name,
            vocab_path=run_dir / "vocabulary.json",
            config_path=run_dir / "config.json",
            tabular_meta_path=meta_path,
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        config_path: str | Path,
        tabular_meta_path: str | Path,
        device: str | torch.device = "auto",
    ) -> "tsFormerPredictor":
        """
        Args
        ────
        checkpoint_path    finetune_final.pt (or finetune_best.pt)
        vocab_path         vocabulary.json
        config_path        config.json (auto-saved by train_pipeline)
        tabular_meta_path  tabular_meta.json (generate with save_tabular_meta())
        """
        device = _resolve_device(device)
        cfg = load_config(config_path)
        vocab = Vocabulary.load(vocab_path)
        meta = load_tabular_meta(tabular_meta_path)

        model = _build_nuformer(cfg, len(vocab), len(meta["tab_cols"]))
        state_dict = _load_state(checkpoint_path)
        own_keys = set(model.state_dict().keys())
        filtered = {k: v for k, v in state_dict.items() if k in own_keys}
        missing = own_keys - set(filtered.keys())
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing {len(missing)} required keys. "
                f"Is this a nuformer checkpoint? Missing: {list(missing)[:3]}"
            )
        model.load_state_dict(filtered, strict=True)

        return cls(
            model=model,
            vocab=vocab,
            tab_cols=meta["tab_cols"],
            tab_means=meta["tab_means"],
            tab_stds=meta["tab_stds"],
            device=device,
            max_seq_len=cfg.model.transformer.max_seq_len,
        )

    # ── Tabular encoding ──────────────────────────────────────────────────────

    def _encode_tabular(self, inputs: list[MemberInput]) -> torch.Tensor:
        rows = []
        for inp in inputs:
            row = []
            for col in self.tab_cols:
                raw = float((inp.tabular_features or {}).get(col, 0.0))
                mu = self.tab_means.get(col, 0.0)
                std = self.tab_stds.get(col, 1.0)
                row.append((raw - mu) / max(std, 1e-9))
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    # ── Scoring ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(self, inp: MemberInput) -> PredictionResult:
        t0 = time.perf_counter()
        ids, mask = self._encode_sequences([inp])
        tab = self._encode_tabular([inp])
        out = self.model(ids, tab, mask)
        logit = out["logits"].item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        return PredictionResult(
            member_id=inp.member_id,
            logit=round(logit, 6),
            probability=round(prob, 6),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    @torch.no_grad()
    def score_batch(self, inputs: list[MemberInput]) -> list[PredictionResult]:
        t0 = time.perf_counter()
        ids, mask = self._encode_sequences(inputs)
        tab = self._encode_tabular(inputs)
        out = self.model(ids, tab, mask)
        logits = out["logits"].cpu().tolist()
        probs = torch.sigmoid(out["logits"]).cpu().tolist()
        elapsed = (time.perf_counter() - t0) * 1000 / max(len(inputs), 1)
        return [
            PredictionResult(inp.member_id, round(l, 6), round(p, 6), round(elapsed, 2))
            for inp, l, p in zip(inputs, logits, probs)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 2. LateFusionPredictor (embedding classifier — no tabular required)
# ──────────────────────────────────────────────────────────────────────────────


class LateFusionPredictor(_BasePredictor):
    """
    Score members using only the fine-tuned TransactionTransformer + MLP head.
    No tabular features required — useful when features are unavailable or for
    real-time streaming contexts.
    """

    def __init__(
        self,
        model: EmbeddingClassifier,
        vocab: Vocabulary,
        device: torch.device,
        max_seq_len: int,
    ) -> None:
        super().__init__(vocab, device, max_seq_len)
        self.model = model.to(device).eval()

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        device: str | torch.device = "auto",
        ckpt_name: str = "finetune_final.pt",
    ) -> "LateFusionPredictor":
        """Load from run dir containing config.json, vocabulary.json, finetune_final.pt."""
        run_dir = Path(run_dir)
        return cls.from_checkpoint(
            checkpoint_path=run_dir / ckpt_name,
            vocab_path=run_dir / "vocabulary.json",
            config_path=run_dir / "config.json",
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        config_path: str | Path,
        device: str | torch.device = "auto",
    ) -> "LateFusionPredictor":
        device = _resolve_device(device)
        cfg = load_config(config_path)
        vocab = Vocabulary.load(vocab_path)
        state_dict = _load_state(checkpoint_path)

        # 1. Automatic Discovery of Head Architecture
        try:
            # Detects whether the checkpoint has a value of 192 (Joint) or 128 (Late).
            actual_head_in = state_dict["head.net.0.weight"].shape[1]
        except KeyError:
            actual_head_in = cfg.model.transformer.d_model * 2

        # 2. Instance the model respecting the LoRA from the config
        model = _build_embedding_classifier(cfg, len(vocab), head_in=actual_head_in)

        # 3. Professional Key Filtering (Only what's needed for Late Fusion)
        own_keys = model.state_dict().keys()
        # We filter to ignore plr.*, dcn.* and emb_norm.* which are in the file
        filtered_state = {k: v for k, v in state_dict.items() if k in own_keys}

        # 4. Secure Loading
        model.load_state_dict(filtered_state, strict=True)
        return cls(model, vocab, device, cfg.model.transformer.max_seq_len)

    @torch.no_grad()
    def score(self, inp: MemberInput) -> PredictionResult:
        t0 = time.perf_counter()
        ids, mask = self._encode_sequences([inp])
        logit = self.model(ids, mask).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        return PredictionResult(
            member_id=inp.member_id,
            logit=round(logit, 6),
            probability=round(prob, 6),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    @torch.no_grad()
    def score_batch(self, inputs: list[MemberInput]) -> list[PredictionResult]:
        t0 = time.perf_counter()
        ids, mask = self._encode_sequences(inputs)
        logits = self.model(ids, mask).cpu().tolist()
        probs = torch.sigmoid(torch.tensor(logits)).tolist()
        elapsed = (time.perf_counter() - t0) * 1000 / max(len(inputs), 1)
        return [
            PredictionResult(inp.member_id, round(l, 6), round(p, 6), round(elapsed, 2))
            for inp, l, p in zip(inputs, logits, probs)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 3. EmbeddingExtractor (raw backbone)
# ──────────────────────────────────────────────────────────────────────────────


class EmbeddingExtractor(_BasePredictor):
    """
    Extract d_model-dimensional user embeddings from the pre-trained backbone.
    No classification head — use for GBT late fusion, clustering, or visualisation.
    """

    def __init__(
        self,
        transformer: TransactionTransformer,
        vocab: Vocabulary,
        device: torch.device,
        max_seq_len: int,
    ) -> None:
        super().__init__(vocab, device, max_seq_len)
        self.transformer = transformer.to(device).eval()

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        device: str | torch.device = "auto",
        ckpt_name: str = "pretrain_final.pt",
    ) -> "EmbeddingExtractor":
        """Load from run dir containing config.json, vocabulary.json, pretrain_final.pt."""
        run_dir = Path(run_dir)
        return cls.from_checkpoint(
            checkpoint_path=run_dir / ckpt_name,
            vocab_path=run_dir / "vocabulary.json",
            config_path=run_dir / "config.json",
            device=device,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        config_path: str | Path,
        device: str | torch.device = "auto",
    ) -> "EmbeddingExtractor":
        device = _resolve_device(device)
        cfg = load_config(config_path)
        vocab = Vocabulary.load(vocab_path)

        model = _build_transformer(cfg, len(vocab))
        state_dict = _load_state(checkpoint_path)

        # Accept partial load — pretrain checkpoint may not have LoRA keys
        compatible = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(compatible, strict=False)

        return cls(
            transformer=model,
            vocab=vocab,
            device=device,
            max_seq_len=cfg.model.transformer.max_seq_len,
        )

    @torch.no_grad()
    def embed(self, transactions: list[dict]) -> list[float]:
        """Return one member's embedding as a float list (length = d_model)."""
        ids, mask = self._encode_sequences([MemberInput(transactions)])
        emb = self.transformer.get_user_embedding(ids, mask)
        return emb.squeeze(0).cpu().tolist()

    @torch.no_grad()
    def embed_batch(self, batch_transactions: list[list[dict]]) -> list[list[float]]:
        """Return embeddings for a list of members."""
        members = [MemberInput(txns) for txns in batch_transactions]
        ids, mask = self._encode_sequences(members)
        embs = self.transformer.get_user_embedding(ids, mask)
        return embs.cpu().tolist()
