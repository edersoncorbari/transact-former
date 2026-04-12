from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Special token definitions
# ──────────────────────────────────────────────────────────────────────────────

PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
BOS_TOKEN   = "<BOS>"
EOS_TOKEN   = "<EOS>"
SEP_TOKEN   = "<SEP>"
MASK_TOKEN  = "<MASK>" 

SIGN_TOKENS = ["<PAID>", "<RCVD>"]

AMOUNT_EDGES = [0, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, float("inf")]
AMOUNT_TOKENS = [f"<AMT:{i}>" for i in range(len(AMOUNT_EDGES))]  # 12 buckets

MONTH_TOKENS   = [f"<M:{m}>" for m in
                  ["JAN","FEB","MAR","APR","MAY","JUN",
                   "JUL","AUG","SEP","OCT","NOV","DEC"]]
DAY_TOKENS     = [f"<D:{d:02d}>" for d in range(1, 32)]           # 31 tokens
WEEKDAY_TOKENS = [f"<WD:{w}>" for w in
                  ["MON","TUE","WED","THU","FRI","SAT","SUN"]]

SPECIAL_TOKENS = (
    [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, MASK_TOKEN]
    + SIGN_TOKENS
    + AMOUNT_TOKENS
    + MONTH_TOKENS
    + DAY_TOKENS
    + WEEKDAY_TOKENS
)

_MONTH_MAP = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12,
    "JANUARY":1,"FEBRUARY":2,"MARCH":3,"APRIL":4,"JUNE":6,
    "JULY":7,"AUGUST":8,"SEPTEMBER":9,"OCTOBER":10,"NOVEMBER":11,"DECEMBER":12,
}
_WEEKDAY_MAP = {
    "MONDAY":0,"TUESDAY":1,"WEDNESDAY":2,"THURSDAY":3,
    "FRIDAY":4,"SATURDAY":5,"SUNDAY":6,
    "MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6,
}

class Vocabulary:
    """
    Maps tokens → integer IDs and back.

    Special tokens are always assigned IDs 0 ... len(SPECIAL_TOKENS)-1.
    Text tokens (pseudo-BPE words from descriptions) are appended on top.
    """

    def __init__(self) -> None:
        self._tok2id: dict[str, int] = {}
        self._id2tok: list[str] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)

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

