"""
tsFormer — train_pipeline.py

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

End-to-end training pipeline driven by tsFormerConfig dataclasses.

CLI
───
    python -m tsformer.train_pipeline --preset local_test --data-dir ./tsformer_data
    python -m tsformer.train_pipeline --preset paper_24m  --data-dir ./data --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import torch

from pathlib import Path
from torch.utils.data import random_split

from tsformer.config import tsFormerConfig, save_config, load_config
from tsformer.data import (
    PreTrainDataset,
    FineTuneDataset,
    Vocabulary,
    build_pretrain_loader,
    build_finetune_loader,
)
from tsformer.model import TransactionTransformer, tsFormer, apply_lora_to_transformer
from tsformer.trainer import (
    PreTrainer,
    EmbeddingClassifier,
    EmbeddingClassifierTrainer,
    JointFusionTrainer,
)
from tsformer.evaluate import (
    run_final_backtest,
    compute_perplexity,
    embedding_similarity_report,
)

PRESETS = {
    "local_test": tsFormerConfig.for_local_test,
    "medium": tsFormerConfig,
    "paper_24m": tsFormerConfig.for_paper_24m,
    "paper_330m": tsFormerConfig.for_paper_330m,
}


def resolve_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _pos_weight(ds):
    n_pos = sum(r["label"] for r in ds.records)
    n_neg = len(ds.records) - n_pos
    return n_neg / max(n_pos, 1)


def build_vocabulary(jsonl_path, vocab_path, min_freq):
    if vocab_path.exists():
        print(f"  Loading vocabulary  ({vocab_path})")
        return Vocabulary.load(vocab_path)

    print("  Building vocabulary …")
    vocab = Vocabulary()
    token_lists = []

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                token_lists.append(json.loads(line).get("tokens", []))

    vocab.build_from_corpus(token_lists, min_freq=min_freq)
    vocab.save(vocab_path)
    print(f"  Vocab size: {len(vocab):,}")
    return vocab


def pretrain(cfg, vocab, jsonl_path, ckpt_dir, device):
    print("\n" + "─" * 60 + "\n  STEP 2 — Pre-training (NTP)\n" + "─" * 60)
    ckpt_path = ckpt_dir / "pretrain_final.pt"
    tc, pc = cfg.model.transformer, cfg.train.pretrain

    model = TransactionTransformer(
        vocab_size=len(vocab),
        d_model=tc.d_model,
        n_layers=tc.n_layers,
        n_heads=tc.n_heads,
        d_ff=tc.d_ff,
        max_seq_len=tc.max_seq_len,
        dropout=tc.dropout,
    )

    if ckpt_path.exists():
        print("  Checkpoint found — skipping.")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return model

    full_ds = PreTrainDataset(jsonl_path, vocab, max_seq_len=tc.max_seq_len)
    n_val = max(1, int(len(full_ds) * pc.val_frac))

    tr_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )
    tr_l = build_pretrain_loader(
        tr_ds, pc.batch_size, num_workers=cfg.train.num_workers
    )
    val_l = build_pretrain_loader(
        val_ds, pc.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )

    PreTrainer(
        model,
        device,
        lr=pc.lr,
        weight_decay=pc.weight_decay,
        max_grad_norm=pc.max_grad_norm,
        warmup_ratio=pc.warmup_ratio,
        amp_dtype=pc.amp_dtype,
        checkpoint_dir=str(ckpt_dir),
        log_every=pc.log_every,
    ).fit(tr_l, val_l, epochs=pc.epochs)
    return model


def finetune(cfg, transformer, vocab, jsonl_path, parquet_path, ckpt_dir, device):
    print("\n" + "─" * 60 + "\n  STEP 3 — Fine-tuning with LoRA\n" + "─" * 60)
    ckpt_path = ckpt_dir / "finetune_final.pt"

    tc = cfg.model.transformer
    fc = cfg.train.finetune
    lc = cfg.model.lora
    hc = cfg.model.head

    h_dims = hc.hidden_dims
    if h_dims is None:
        h_dims = (64,)
    elif isinstance(h_dims, int):
        h_dims = (h_dims,)

    tfm = copy.deepcopy(transformer)

    if lc.enabled:
        apply_lora_to_transformer(
            tfm,
            rank=lc.rank,
            alpha=lc.alpha,
            dropout=lc.dropout,
            target_modules=lc.target_modules,
        )

    clf = EmbeddingClassifier(tfm, tc.d_model, hidden_dims=h_dims, dropout=hc.dropout)

    if ckpt_path.exists():
        print("  Checkpoint found — skipping.")
        state = torch.load(ckpt_path, map_location="cpu")
        clf.load_state_dict(state["model_state"])
        return clf

    full_ds = FineTuneDataset(
        jsonl_path, parquet_path, vocab, max_seq_len=tc.max_seq_len
    )
    pw = fc.pos_weight or _pos_weight(full_ds)
    print(f"  pos_weight={pw:.2f}")

    n_val = max(1, int(len(full_ds) * fc.val_frac))
    tr_ds, val_ds = random_split(
        full_ds,
        [len(full_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    tr_l = build_finetune_loader(
        tr_ds, fc.batch_size, num_workers=cfg.train.num_workers
    )
    val_l = build_finetune_loader(
        val_ds, fc.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )

    EmbeddingClassifierTrainer(
        clf,
        device,
        lr=fc.lr,
        weight_decay=fc.weight_decay,
        max_grad_norm=fc.max_grad_norm,
        warmup_ratio=fc.warmup_ratio,
        amp_dtype=fc.amp_dtype,
        checkpoint_dir=str(ckpt_dir),
        log_every=fc.log_every,
        pos_weight=pw,
    ).fit(tr_l, val_l, epochs=fc.epochs)

    return clf


def joint_fusion(cfg, transformer, vocab, jsonl_path, parquet_path, ckpt_dir, device):
    print("\n" + "─" * 60 + "\n  STEP 4 — Joint Fusion (tsFormer)\n" + "─" * 60)
    ckpt_path = ckpt_dir / "tsformer_final.pt"
    tc, jc, lc, pc, dc, hc = (
        cfg.model.transformer,
        cfg.train.joint_fusion,
        cfg.model.lora,
        cfg.model.plr,
        cfg.model.dcn,
        cfg.model.head,
    )

    probe = FineTuneDataset(jsonl_path, parquet_path, vocab, max_seq_len=tc.max_seq_len)
    n_tab = probe.n_tabular

    print(f"  Tabular features: {n_tab}")
    model = tsFormer(
        vocab_size=len(vocab),
        n_tabular=n_tab,
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
    model.transformer.load_state_dict(transformer.state_dict(), strict=False)

    if ckpt_path.exists():
        print("  Checkpoint found — skipping.")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        return model

    pw = jc.pos_weight or _pos_weight(probe)

    print(f"  pos_weight={pw:.2f}")
    n_val = max(1, int(len(probe) * jc.val_frac))
    tr_ds, val_ds = random_split(
        probe,
        [len(probe) - n_val, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )
    tr_l = build_finetune_loader(
        tr_ds, jc.batch_size, num_workers=cfg.train.num_workers
    )
    val_l = build_finetune_loader(
        val_ds, jc.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )

    JointFusionTrainer(
        model,
        device,
        lr=jc.lr,
        weight_decay=jc.weight_decay,
        max_grad_norm=jc.max_grad_norm,
        warmup_ratio=jc.warmup_ratio,
        amp_dtype=jc.amp_dtype,
        checkpoint_dir=str(ckpt_dir),
        log_every=jc.log_every,
        pos_weight=pw,
    ).fit(tr_l, val_l, epochs=jc.epochs)

    return model


def evaluate(
    cfg,
    transformer,
    ft_model,
    nuf_model,
    vocab,
    jsonl_path,
    parquet_path,
    results_dir,
    device,
):
    print("\n" + "─" * 60 + "\n  STEP 5 — Evaluation\n" + "─" * 60)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-save tabular_meta.json so inference works out of the box
    _meta_path = Path(cfg.paths.checkpoint_dir) / "tabular_meta.json"
    if not _meta_path.exists():
        from tsformer.inference import save_tabular_meta as _stm

        _stm(parquet_path, _meta_path)

    tc = cfg.model.transformer
    all_r = {}
    all_r["perplexity"] = compute_perplexity(
        transformer,
        vocab,
        jsonl_path,
        device,
        max_seq_len=tc.max_seq_len,
        batch_size=cfg.train.pretrain.batch_size,
    )

    all_r["embedding_similarity"] = embedding_similarity_report(
        transformer, vocab, jsonl_path, device, max_seq_len=tc.max_seq_len
    )

    backtest = run_final_backtest(
        ft_model,
        nuf_model,
        vocab,
        jsonl_path,
        parquet_path,
        device,
        max_seq_len=tc.max_seq_len,
        batch_size=cfg.train.joint_fusion.batch_size,
        output_dir=results_dir,
    )

    all_r["backtest"] = {
        k: v
        for k, v in backtest.items()
        if not k.startswith("roc_") and not k.startswith("pr_")
    }

    with open(results_dir / "full_results.json", "w") as f:
        json.dump(all_r, f, indent=2)

    print(f"\n  Results → {results_dir}/full_results.json")
    return all_r


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="local_test", choices=list(PRESETS))
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default="./tsformer_data")
    p.add_argument("--checkpoint-dir", default="./checkpoints")
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--device", default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--pretrain-epochs", type=int, default=None)
    p.add_argument("--finetune-epochs", type=int, default=None)
    p.add_argument("--fusion-epochs", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--vocab-min-freq", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else PRESETS[args.preset]()

    if args.data_dir:
        cfg.paths.data_dir = args.data_dir

    if args.checkpoint_dir:
        cfg.paths.checkpoint_dir = args.checkpoint_dir

    if args.results_dir:
        cfg.paths.results_dir = args.results_dir

    if args.device:
        cfg.device = args.device

    if args.max_seq_len:
        cfg.model.transformer.max_seq_len = args.max_seq_len

    if args.batch_size:
        cfg.train.pretrain.batch_size = cfg.train.finetune.batch_size = (
            cfg.train.joint_fusion.batch_size
        ) = args.batch_size

    if args.pretrain_epochs:
        cfg.train.pretrain.epochs = args.pretrain_epochs

    if args.finetune_epochs:
        cfg.train.finetune.epochs = args.finetune_epochs

    if args.fusion_epochs:
        cfg.train.joint_fusion.epochs = args.fusion_epochs

    if args.lora_rank:
        cfg.model.lora.rank = args.lora_rank

    if args.num_workers:
        cfg.train.num_workers = args.num_workers

    device = resolve_device(cfg.device)
    data_dir = Path(cfg.paths.data_dir)
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg.paths.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path, parquet_path = (
        data_dir / "tokenized_sequences.jsonl",
        data_dir / "tabular_features.parquet",
    )

    assert jsonl_path.exists(), f"Missing: {jsonl_path}"
    assert parquet_path.exists(), f"Missing: {parquet_path}"
    save_config(cfg, ckpt_dir / "config.json")

    print("\n" + "=" * 60)
    print("  tsFormer Training Pipeline")
    print("=" * 60)

    tc = cfg.model.transformer
    gpu = f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""

    print(f"  Device  : {device}{gpu}")
    print(
        f"  Model   : d={tc.d_model}  L={tc.n_layers}  H={tc.n_heads}  ctx={tc.max_seq_len}"
    )
    print(
        f"  Phases  : pretrain={cfg.train.pretrain.epochs}ep  "
        f"finetune={cfg.train.finetune.epochs}ep  fusion={cfg.train.joint_fusion.epochs}ep"
    )
    print("=" * 60)

    vocab = build_vocabulary(
        jsonl_path, ckpt_dir / "vocabulary.json", args.vocab_min_freq
    )
    transformer = pretrain(cfg, vocab, jsonl_path, ckpt_dir, device)

    ft_model = finetune(
        cfg, transformer, vocab, jsonl_path, parquet_path, ckpt_dir, device
    )
    nuf_model = joint_fusion(
        cfg, transformer, vocab, jsonl_path, parquet_path, ckpt_dir, device
    )

    return evaluate(
        cfg,
        transformer,
        ft_model,
        nuf_model,
        vocab,
        jsonl_path,
        parquet_path,
        res_dir,
        device,
    )


if __name__ == "__main__":
    main()
