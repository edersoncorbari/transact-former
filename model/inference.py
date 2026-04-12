from __future__ import annotations
 
import json
import time
import torch

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .data import Vocabulary, tokenise_member

@dataclass
class MemberInput:
    transactions:     list[dict]
    tabular_features: dict = field(default_factory=dict)
    member_id:        Optional[str] = None
 
 
@dataclass
class PredictionResult:
    member_id:   Optional[str]
    logit:       float
    probability: float
    latency_ms:  float
 
 

class _BasePredictor:
 
    def __init__(self, vocab: Vocabulary, device: torch.device, max_seq_len: int):
        self.vocab       = vocab
        self.device      = device
        self.max_seq_len = max_seq_len
 
    def _encode_sequences(
        self, inputs: list[MemberInput]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_ids, max_len = [], 0
        for inp in inputs:
            tokens = tokenise_member(inp.transactions, max_seq_len=self.max_seq_len)
            ids    = self.vocab.encode_sequence(tokens)
            all_ids.append(ids)
            max_len = max(max_len, len(ids))
 
        pad_id  = self.vocab.pad_id
        padded, masks = [], []
        for ids in all_ids:
            pad = max_len - len(ids)
            padded.append(ids + [pad_id] * pad)
            masks.append([True] * len(ids) + [False] * pad)
 
        return (
            torch.tensor(padded, dtype=torch.long, device=self.device),
            torch.tensor(masks,  dtype=torch.bool,  device=self.device),
        )


class LateFusionPredictor(_BasePredictor):
 
    def __init__(
        self,
        model:       EmbeddingClassifier,
        vocab:       Vocabulary,
        device:      torch.device,
        max_seq_len: int,
    ) -> None:
        super().__init__(vocab, device, max_seq_len)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def score(self, inp: MemberInput) -> PredictionResult:
        t0        = time.perf_counter()
        ids, mask = self._encode_sequences([inp])
        logit     = self.model(ids, mask).item()
        prob      = torch.sigmoid(torch.tensor(logit)).item()
        return PredictionResult(
            member_id   = inp.member_id,
            logit       = round(logit, 6),
            probability = round(prob,  6),
            latency_ms  = round((time.perf_counter() - t0) * 1000, 2),
        )

