"""
transactFormer — model.py

Full model architecture implementing Sections 3.1, 3.2 and 3.3 of the paper.

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers" (transactformer)

Components
──────────
TransactionTransformer   — causal GPT-like backbone with NoPE 
LoRALinear               — low-rank adaptation for fine-tuning 
PLREmbedding             — periodic / linear embedding for numerical features
DCNv2Block               — deep & cross network for tabular features 
transactFormer           — joint fusion model with end-to-end training
"""

from __future__ import annotations
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction — lighter than LayerNorm)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention WITHOUT positional encodings (NoPE).
    Supports optional Flash Attention via scaled_dot_product_attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = math.sqrt(self.head_dim)

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                         # (B, T, D)
        attention_mask: Optional[torch.Tensor],  # (B, T) bool — True = keep
    ) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                 # Each (B, T, H, Dh)
        q = q.transpose(1, 2)                       # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build additive causal + padding mask
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        attn_bias = torch.zeros(B, 1, T, T, device=x.device)
        attn_bias.masked_fill_(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            # (B, 1, 1, T) — mask out padding keys
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_bias.masked_fill_(~pad_mask, float("-inf"))

        # Use PyTorch's efficient attention when available
        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.drop.p if self.training else 0.0,
            )
        except TypeError: # older PyTorch fallback
            scores = (q @ k.transpose(-2, -1)) / self.scale + attn_bias
            scores = F.softmax(scores, dim=-1)
            scores = self.drop(scores)
            out    = scores @ v

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2. Transaction Transformer backbone  
# ──────────────────────────────────────────────────────────────────────────────

class TransactionTransformer(nn.Module):
    """
    Causal GPT-like transformer trained with next-token prediction (NTP).
    No positional embeddings (NoPE). Sizes: 24 M and 330 M params.

    Args
    ────
    vocab_size  : vocabulary size (|V|)
    d_model     : embedding / hidden dim (256 for 24M, 1024 for 330M)
    n_layers    : number of transformer blocks (paper: 24)
    n_heads     : attention heads (paper: 16)
    d_ff        : feed-forward inner dim (default: 4 × d_model)
    max_seq_len : maximum context length (512 / 1024 / 2048)
    dropout     : dropout rate
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 256,
        n_layers:    int   = 6,
        n_heads:     int   = 8,
        d_ff:        Optional[int] = None,
        max_seq_len: int   = 2048,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm     = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (standard in LM literature)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def param_count(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def encode(
        self,
        input_ids:      torch.Tensor,               # (B, T)
        attention_mask: Optional[torch.Tensor],     # (B, T)
    ) -> torch.Tensor:                              # (B, T, D)
        x = self.drop(self.embedding(input_ids))
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.norm(x)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(input_ids, attention_mask)  # (B, T, D)
        logits = self.lm_head(hidden)                    # (B, T, V)

        out: dict[str, torch.Tensor] = {"logits": logits, "hidden": hidden}

        if labels is not None:
            # NTP loss — ignore padding (-100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out

    def get_user_embedding(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns the final (non-padding) token embedding — the user representation
        used for downstream tasks.
        """
        hidden = self.encode(input_ids, attention_mask)  # (B, T, D)

        if attention_mask is not None:
            # Index of last real token per sequence
            lengths  = attention_mask.sum(dim=1) - 1           # (B,)
            lengths  = lengths.clamp(min=0)
            idx      = lengths.unsqueeze(-1).unsqueeze(-1)     # (B, 1, 1)
            idx      = idx.expand(-1, 1, hidden.size(-1))      # (B, 1, D)
            user_emb = hidden.gather(1, idx).squeeze(1)        # (B, D)
        else:
            user_emb = hidden[:, -1, :]                         # (B, D)

        return user_emb


# ──────────────────────────────────────────────────────────────────────────────
# 3.  LoRA linear layer
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a low-rank adaptation ΔW = B·A.
    Use LoRA to help prevent overfitting and catastrophic forgetting.
    """

    def __init__(
        self,
        linear:  nn.Linear,
        rank:    int   = 8,
        alpha:   float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.linear  = linear
        self.rank    = rank
        self.scaling = alpha / rank

        in_features  = linear.in_features
        out_features = linear.out_features

        self.lora_A  = nn.Linear(in_features,  rank,         bias=False)
        self.lora_B  = nn.Linear(rank,          out_features, bias=False)
        self.drop    = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the base weights
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.drop(x))) * self.scaling
        return base + lora


def apply_lora_to_transformer(
    transformer: TransactionTransformer,
    rank:    int   = 8,
    alpha:   float = 16.0,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] = ("qkv", "proj"),
) -> TransactionTransformer:
    """
    Replace targeted linear layers in every TransformerBlock with LoRA variants.
    All other parameters are frozen.
    """
    # First freeze everything
    for p in transformer.parameters():
        p.requires_grad = False

    for block in transformer.blocks:
        attn = block.attn
        if "qkv" in target_modules:
            attn.qkv  = LoRALinear(attn.qkv,  rank=rank, alpha=alpha, dropout=dropout)
        if "proj" in target_modules:
            attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha, dropout=dropout)

    return transformer


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PLR — Periodic Linear embedding for numerical features
# ──────────────────────────────────────────────────────────────────────────────

class PLREmbedding(nn.Module):
    """
    Periodic + Linear embedding for numerical tabular features.
    "embedding numerical attributes … using periodic activations at different
    (learned) frequencies" — Section 3.3, citing Gorishniy et al. 2022.

    Maps a scalar x → d-dimensional vector via:
        e(x) = [sin(ω₁x + φ₁), cos(ω₁x + φ₁), …, sin(ωₖx + φₖ), cos(ωₖx + φₖ), linear_proj(x)]
    """

    def __init__(self, n_features: int, n_frequencies: int = 32, d_out: int = 64) -> None:
        super().__init__()
        self.n_features    = n_features
        self.n_frequencies = n_frequencies
        self.d_out         = d_out

        # Learnable frequencies and phases per feature
        self.omega = nn.Parameter(torch.randn(n_features, n_frequencies))
        self.phi   = nn.Parameter(torch.randn(n_features, n_frequencies))

        # Project [sin, cos] × n_freq → d_out
        self.proj = nn.Linear(2 * n_frequencies, d_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        B, F = x.shape
        assert F == self.n_features

        x_exp = x.unsqueeze(-1)                     # (B, F, 1)
        arg   = x_exp * self.omega + self.phi       # (B, F, n_freq)
        feats = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)  # (B, F, 2*n_freq)
        out   = self.proj(feats)                    # (B, F, d_out)
        return out.reshape(B, F * self.d_out)       # (B, F*d_out)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  DCNv2  
# ──────────────────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    """
    One DCNv2 cross layer: x_{l+1} = x_0 ⊙ (W_l·x_l + b_l) + x_l
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.W = nn.Linear(d, d, bias=True)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.W(x) + x


class DCNv2Block(nn.Module):
    """
    Deep & Cross Network V2 for tabular feature modelling.
    Architecture: parallel cross-network + deep network → concat → output proj.

    Paper customisation: only the embedded tabular features pass through the
    cross layers; result is projected to low-dim before concatenation with
    the transaction embedding.
    """

    def __init__(
        self,
        d_in:      int,
        d_cross:   int = 128,
        n_cross:   int = 3,
        d_deep:    int = 256,
        n_deep:    int = 2,
        d_out:     int = 64,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_cross),
            nn.LayerNorm(d_cross),
            nn.ReLU(),
        )

        # Cross network
        self.cross_layers = nn.ModuleList([CrossLayer(d_cross) for _ in range(n_cross)])

        # Deep network
        deep_layers: list[nn.Module] = []
        prev = d_cross
        for _ in range(n_deep):
            deep_layers += [nn.Linear(prev, d_deep), nn.ReLU(), nn.Dropout(dropout)]
            prev = d_deep
        self.deep_net = nn.Sequential(*deep_layers)

        # Output projection (cross + deep → d_out)
        self.out_proj = nn.Linear(d_cross + d_deep, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in)
        h  = self.input_proj(x)          # (B, d_cross)
        x0 = h

        # Cross network
        xc = h
        for layer in self.cross_layers:
            xc = layer(x0, xc)           # (B, d_cross)

        # Deep network
        xd = self.deep_net(h)            # (B, d_deep)

        # Combine
        combined = torch.cat([xc, xd], dim=-1)   # (B, d_cross + d_deep)
        return self.out_proj(combined)           # (B, d_out)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Classification head
# ──────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """MLP that maps user embedding → binary score."""

    def __init__(self, d_in: int, hidden_dims: tuple[int, ...] = (128,), dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  transactFormer — Joint Fusion  (Section 3.3, Figure 5)
# ──────────────────────────────────────────────────────────────────────────────

class transactFormer(nn.Module):
    """
    Full joint-fusion model described in Section 3.3.

    Training signal propagates end-to-end through both branches.
    """

    def __init__(
        self,
        vocab_size:      int,
        n_tabular:       int,
        # Transformer
        d_model:         int   = 256,
        n_layers:        int   = 6,
        n_heads:         int   = 8,
        max_seq_len:     int   = 2048,
        transformer_dropout: float = 0.1,
        # PLR
        n_frequencies:   int   = 32,
        plr_d_out:       int   = 32,
        # DCNv2
        d_cross:         int   = 128,
        n_cross:         int   = 3,
        d_deep:          int   = 256,
        n_deep:          int   = 2,
        dcn_d_out:       int   = 64,
        dcn_dropout:     float = 0.1,
        # Head
        head_hidden:     tuple[int, ...] = (256, 128),
        head_dropout:    float = 0.1,
        # LoRA (set lora_rank > 0 to enable)
        lora_rank:       int   = 0,
        lora_alpha:      float = 16.0,
        lora_dropout:    float = 0.05,
    ) -> None:
        super().__init__()

        self.transformer = TransactionTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=transformer_dropout,
        )

        if lora_rank > 0:
            apply_lora_to_transformer(
                self.transformer,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        self.plr = PLREmbedding(
            n_features=n_tabular,
            n_frequencies=n_frequencies,
            d_out=plr_d_out,
        )

        plr_out_dim = n_tabular * plr_d_out
        self.dcn = DCNv2Block(
            d_in=plr_out_dim,
            d_cross=d_cross,
            n_cross=n_cross,
            d_deep=d_deep,
            n_deep=n_deep,
            d_out=dcn_d_out,
            dropout=dcn_dropout,
        )

        # Normalise transaction embeddings before concatenation (Section 3.3)
        self.emb_norm = nn.LayerNorm(d_model)

        head_in = d_model + dcn_d_out
        self.head = ClassificationHead(head_in, head_hidden, head_dropout)

    def forward(
        self,
        input_ids:      torch.Tensor,               # (B, T)
        tabular_feats:  torch.Tensor,               # (B, n_tab)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        labels:         Optional[torch.Tensor] = None,  # (B,)
    ) -> dict[str, torch.Tensor]:

        # ── Transaction branch ──
        user_emb  = self.transformer.get_user_embedding(input_ids, attention_mask)
        user_emb  = self.emb_norm(user_emb)          # (B, d_model)

        # ── Tabular branch ──
        tab_emb   = self.plr(tabular_feats)           # (B, n_tab * plr_d_out)
        tab_emb   = self.dcn(tab_emb)                 # (B, dcn_d_out)

        # ── Fusion ──
        fused  = torch.cat([user_emb, tab_emb], dim=-1)  # (B, d_model + dcn_d_out)
        logits = self.head(fused)                        # (B,)

        out: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            out["loss"] = loss

        return out

    def trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
