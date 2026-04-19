"""
tsFormer — data.py

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Vocabulary, tokenizer and PyTorch Dataset / DataLoader utilities.

Implements the transaction tokenisation described in Section 3.1:
    τ(t) = [φ_sign, φ_amt, φ_month, φ_day, φ_weekday] ⊕ φ_BPE(desc)

and the member-sequence construction of Equation (2):
    x_i = SEP-delimited concatenation of all τ(t), truncated to max_seq_len.
"""

from __future__ import annotations

import json
import re
import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Special token definitions
# ──────────────────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
MASK_TOKEN = "<MASK>"  # reserved for MLM ablations

SIGN_TOKENS = ["<PAID>", "<RCVD>"]

AMOUNT_EDGES = [0, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, float("inf")]
AMOUNT_TOKENS = [f"<AMT:{i}>" for i in range(len(AMOUNT_EDGES))]  # 12 buckets

MONTH_TOKENS = [
    f"<M:{m}>"
    for m in [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
]
DAY_TOKENS = [f"<D:{d:02d}>" for d in range(1, 32)]  # 31 tokens
WEEKDAY_TOKENS = [
    f"<WD:{w}>" for w in ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
]

SPECIAL_TOKENS = (
    [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, MASK_TOKEN]
    + SIGN_TOKENS
    + AMOUNT_TOKENS
    + MONTH_TOKENS
    + DAY_TOKENS
    + WEEKDAY_TOKENS
)

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────


class Vocabulary:
    """
    Maps tokens → integer IDs and back.

    Special tokens are always assigned IDs 0 … len(SPECIAL_TOKENS)-1.
    Text tokens (pseudo-BPE words from descriptions) are appended on top.
    """

    def __init__(self) -> None:
        self._tok2id: dict[str, int] = {}
        self._id2tok: list[str] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    # ── construction ──

    def _add(self, token: str) -> int:
        if token not in self._tok2id:
            idx = len(self._id2tok)
            self._tok2id[token] = idx
            self._id2tok.append(token)
        return self._tok2id[token]

    def build_from_corpus(
        self, token_lists: list[list[str]], min_freq: int = 5
    ) -> "Vocabulary":
        """Scan tokenised sequences and add text tokens with freq ≥ min_freq."""
        from collections import Counter

        counts: Counter = Counter()
        for toks in token_lists:
            for t in toks:
                if t not in self._tok2id:
                    counts[t] += 1
        for tok, cnt in counts.items():
            if cnt >= min_freq:
                self._add(tok)
        return self

    # ── I/O ──

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._id2tok, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        vocab = cls.__new__(cls)
        vocab._id2tok = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab._tok2id = {t: i for i, t in enumerate(vocab._id2tok)}
        return vocab

    # ── access ──

    def __len__(self) -> int:
        return len(self._id2tok)

    def __contains__(self, token: str) -> bool:
        return token in self._tok2id

    def encode(self, token: str) -> int:
        return self._tok2id.get(token, self._tok2id[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        return self._id2tok[idx] if 0 <= idx < len(self._id2tok) else UNK_TOKEN

    def encode_sequence(self, tokens: list[str]) -> list[int]:
        return [self.encode(t) for t in tokens]

    def decode_sequence(self, ids: list[int]) -> list[str]:
        return [self.decode(i) for i in ids]

    @property
    def pad_id(self) -> int:
        return self._tok2id[PAD_TOKEN]

    @property
    def eos_id(self) -> int:
        return self._tok2id[EOS_TOKEN]

    @property
    def bos_id(self) -> int:
        return self._tok2id[BOS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self._tok2id[SEP_TOKEN]

    @property
    def mask_id(self) -> int:
        return self._tok2id[MASK_TOKEN]


# ──────────────────────────────────────────────────────────────────────────────
# Transaction tokeniser  (Section 3.1 — Eq. 1 & 2)
# ──────────────────────────────────────────────────────────────────────────────


def _amount_bucket(amount: float) -> int:
    abs_amt = abs(amount)
    for i, edge in enumerate(AMOUNT_EDGES[1:], start=1):
        if abs_amt < edge:
            return i - 1
    return len(AMOUNT_EDGES) - 1


def _simple_word_tokenise(text: str) -> list[str]:
    """
    Pseudo-BPE: uppercase + split on non-alphanumeric boundaries.
    Replace with a real BPE tokeniser (e.g. tiktoken / sentencepiece)
    when scaling to production.
    """
    text = text.upper().strip()
    words = re.split(r"[^A-Z0-9]+", text)
    return [w for w in words if w]


def tokenise_transaction(txn: dict) -> list[str]:
    """
    Eq. (1): τ(t) = [φ_sign, φ_amt, φ_month, φ_day, φ_weekday] ⊕ φ_BPE(desc)

    txn keys expected: amount (float), date (str YYYY-MM-DD or dict with
    month/day/weekday), description (str).
    """
    # ── amount ──
    amount = float(txn.get("amount", 0))
    sign_tok = SIGN_TOKENS[0] if amount >= 0 else SIGN_TOKENS[1]
    amt_tok = AMOUNT_TOKENS[_amount_bucket(amount)]

    # ── date ──
    date_str = txn.get("date", "")
    try:
        from datetime import datetime as _dt

        d = _dt.strptime(date_str[:10], "%Y-%m-%d")
        month_tok = MONTH_TOKENS[d.month - 1]
        day_tok = DAY_TOKENS[d.day - 1]
        weekday_tok = WEEKDAY_TOKENS[d.weekday()]
    except (ValueError, TypeError):
        month_tok = day_tok = weekday_tok = UNK_TOKEN

    # ── description (pseudo-BPE) ──
    desc = txn.get("description", "")
    desc_tokens = _simple_word_tokenise(desc)

    return [sign_tok, amt_tok, month_tok, day_tok, weekday_tok] + desc_tokens


def tokenise_member(
    transactions: list[dict],
    max_seq_len: int = 2048,
) -> list[str]:
    """
    Eq. (2): x_i = SEP-delimited transaction strings, truncated to max_seq_len
    by keeping the most RECENT transactions (paper uses causal attention →
    recency matters).
    """
    # Build per-transaction token chunks newest-first, then reverse
    chunks: list[list[str]] = []
    total = 0
    for txn in reversed(transactions):
        chunk = tokenise_transaction(txn) + [SEP_TOKEN]
        if total + len(chunk) > max_seq_len - 2:  # reserve BOS + EOS
            break
        chunks.append(chunk)
        total += len(chunk)

    tokens = [BOS_TOKEN]
    for chunk in reversed(chunks):
        tokens.extend(chunk)
    tokens.append(EOS_TOKEN)
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Datasets
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PreTrainSample:
    input_ids: torch.Tensor  # (seq_len,)  long
    labels: torch.Tensor  # (seq_len,)  long  — shifted NTP targets


@dataclass
class FineTuneSample:
    input_ids: torch.Tensor  # (seq_len,)  long
    tabular_feats: torch.Tensor  # (n_tab,)    float
    label: torch.Tensor  # ()          float


class PreTrainDataset(Dataset):
    """
    Reads tokenized_sequences.jsonl produced by the data generator.
    Each member sequence is used as a next-token-prediction training example.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        vocab: Vocabulary,
        max_seq_len: int = 2048,
    ) -> None:
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.records: list[dict] = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PreTrainSample:
        rec = self.records[idx]
        tokens = rec["tokens"][: self.max_seq_len]
        ids = self.vocab.encode_sequence(tokens)

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)

        return PreTrainSample(input_ids=input_ids, labels=labels)


class FineTuneDataset(Dataset):
    """
    Combines tokenized sequences with tabular features + binary label.
    Supports both late-fusion (frozen embeddings) and joint-fusion setups.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        parquet_path: str | Path,
        vocab: Vocabulary,
        max_seq_len: int = 2048,
        tabular_cols: Optional[list[str]] = None,
    ) -> None:
        import pandas as pd

        self.vocab = vocab
        self.max_seq_len = max_seq_len

        # Load tabular features
        tab_df = pd.read_parquet(parquet_path)

        # Select numeric/categorical tabular columns (exclude meta columns)
        meta_cols = {"member_id", "score_date", "label"}
        string_cols = set(tab_df.select_dtypes(include="object").columns) - meta_cols
        exclude = meta_cols | string_cols
        if tabular_cols is not None:
            self.tab_cols = [c for c in tabular_cols if c in tab_df.columns]
        else:
            self.tab_cols = [c for c in tab_df.columns if c not in exclude]

        # Fill NaN and normalise
        tab_df[self.tab_cols] = tab_df[self.tab_cols].fillna(0.0).astype(float)
        means = tab_df[self.tab_cols].mean()
        stds = tab_df[self.tab_cols].std().replace(0, 1)
        tab_df[self.tab_cols] = (tab_df[self.tab_cols] - means) / stds

        # Index tabular rows by (member_id, score_date)
        self._tab_index: dict[tuple, np.ndarray] = {}
        for _, row in tab_df.iterrows():
            key = (row["member_id"], row["score_date"])
            self._tab_index[key] = row[self.tab_cols].values.astype(np.float32)

        # Load sequences
        self.records: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = (rec["member_id"], rec["score_date"])
                if key in self._tab_index:
                    self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def n_tabular(self) -> int:
        return len(self.tab_cols)

    def __getitem__(self, idx: int) -> FineTuneSample:
        rec = self.records[idx]
        tokens = rec["tokens"][: self.max_seq_len]
        ids = self.vocab.encode_sequence(tokens)
        key = (rec["member_id"], rec["score_date"])

        input_ids = torch.tensor(ids, dtype=torch.long)
        tabular_feats = torch.tensor(self._tab_index[key], dtype=torch.float32)
        label = torch.tensor(float(rec["label"]), dtype=torch.float32)

        return FineTuneSample(
            input_ids=input_ids,
            tabular_feats=tabular_feats,
            label=label,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Collate functions
# ──────────────────────────────────────────────────────────────────────────────


def pretrain_collate(
    batch: list[PreTrainSample], pad_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(s.input_ids.size(0) for s in batch)

    input_ids_list, labels_list, mask_list = [], [], []
    for s in batch:
        L = s.input_ids.size(0)
        pad = max_len - L
        input_ids_list.append(
            torch.cat([s.input_ids, torch.full((pad,), pad_id, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([s.labels, torch.full((pad,), -100, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat(
                [torch.ones(L, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]
            )
        )

    return {
        "input_ids": torch.stack(input_ids_list),  # (B, T)
        "labels": torch.stack(labels_list),  # (B, T)
        "attention_mask": torch.stack(mask_list),  # (B, T)
    }


def finetune_collate(
    batch: list[FineTuneSample], pad_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(s.input_ids.size(0) for s in batch)

    input_ids_list, mask_list = [], []
    for s in batch:
        L = s.input_ids.size(0)
        pad = max_len - L
        input_ids_list.append(
            torch.cat([s.input_ids, torch.full((pad,), pad_id, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat(
                [torch.ones(L, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)]
            )
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
        "tabular_feats": torch.stack([s.tabular_feats for s in batch]),
        "labels": torch.stack([s.label for s in batch]),
    }


def _unwrap(ds):
    """Traverse Subset chain to reach the root Dataset."""
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds


def build_pretrain_loader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    base = _unwrap(dataset)
    pad_id = base.vocab.pad_id
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: pretrain_collate(b, pad_id),
    )


def build_finetune_loader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    base = _unwrap(dataset)
    pad_id = base.vocab.pad_id
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda b: finetune_collate(b, pad_id),
    )
