"""
tsFormer — scaling_analysis.py

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Reproduces the scaling experiments from Section 4.3 of the paper:

    Section 4.3.2  Impact of Model Size      (Table 3)
    Section 4.3.3  Effect of Context Length  (Figure 6)
    Section 4.3.4  Effect of Training Data Volume  (Figure 7)

Each experiment trains multiple tsFormer variants and records AUC gain
vs the tabular-only baseline, then prints results in the paper's format.

Usage
─────
    # All experiments
    python -m tsformer.scaling_analysis \
        --data-dir ./tsformer_data \
        --results-dir ./scaling_results \
        --experiment all

    # Just model size (fastest)
    python -m tsformer.scaling_analysis --experiment model_size

    # Just data volume
    python -m tsformer.scaling_analysis --experiment data_volume
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import torch

from pathlib import Path
from torch.utils.data import random_split, Subset

from tsformer.config import tsFormerConfig
from tsformer.data import FineTuneDataset, Vocabulary, build_finetune_loader
from tsformer.model import tsFormer
from tsformer.trainer import JointFusionTrainer
from tsformer.train_pipeline import (
    build_vocabulary,
    resolve_device,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared training helper
# ──────────────────────────────────────────────────────────────────────────────


def _train_and_eval_tsformer(
    cfg: tsFormerConfig,
    vocab: Vocabulary,
    jsonl_path: Path,
    parquet_path: Path,
    device: torch.device,
    n_train_rows: int | None = None,  # None = use all
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 32,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    label: str = "",
) -> dict:
    """
    Train a tsFormer from the given config and return AUC on the held-out test set.
    Suppresses per-step logging for clean experiment output.
    """
    tc = cfg.model.transformer
    full_ds = FineTuneDataset(
        jsonl_path, parquet_path, vocab, max_seq_len=tc.max_seq_len
    )

    n_total = len(full_ds)
    n_test = max(10, int(n_total * test_frac))
    n_pool = n_total - n_test

    pool_ds, test_ds = random_split(
        full_ds,
        [n_pool, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    # Optionally subsample the training pool
    if n_train_rows is not None and n_train_rows < n_pool:
        idxs = torch.randperm(n_pool, generator=torch.Generator().manual_seed(seed))
        idxs = idxs[:n_train_rows].tolist()
        pool_ds = Subset(pool_ds, idxs)

    n_val = max(5, int(len(pool_ds) * val_frac))
    n_train = len(pool_ds) - n_val
    train_ds, val_ds = random_split(
        pool_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed + 1),
    )

    train_loader = build_finetune_loader(train_ds, batch_size, num_workers=0)
    val_loader = build_finetune_loader(val_ds, batch_size, num_workers=0, shuffle=False)
    test_loader = build_finetune_loader(
        test_ds, batch_size, num_workers=0, shuffle=False
    )

    n_pos = sum(r["label"] for r in full_ds.records)
    n_neg = len(full_ds.records) - n_pos
    pw = n_neg / max(n_pos, 1)

    lc, pc, dc, hc = cfg.model.lora, cfg.model.plr, cfg.model.dcn, cfg.model.head
    model = tsFormer(
        vocab_size=len(vocab),
        n_tabular=full_ds.n_tabular,
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
    )

    trainer = JointFusionTrainer(
        model=model,
        device=device,
        lr=cfg.train.joint_fusion.lr,
        weight_decay=cfg.train.joint_fusion.weight_decay,
        amp_dtype=cfg.train.joint_fusion.amp_dtype,
        pos_weight=pw,
        log_every=999_999,
    )

    t0 = time.time()
    trainer.fit(train_loader, val_loader, epochs=epochs)
    elapsed = time.time() - t0

    # Evaluate on test set
    metrics = trainer.evaluate(test_loader)
    n_params = sum(p.numel() for p in model.parameters())

    result = {
        "label": label,
        "auc": metrics["auc"],
        "n_params": n_params,
        "n_train": n_train,
        "d_model": tc.d_model,
        "n_layers": tc.n_layers,
        "max_seq_len": tc.max_seq_len,
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
    }
    print(
        f"    {label:<30}  AUC={metrics['auc']:.4f}  "
        f"params={n_params:,}  t={elapsed:.0f}s"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Model Size  (Table 3)
# ──────────────────────────────────────────────────────────────────────────────


def experiment_model_size(
    vocab: Vocabulary,
    jsonl_path: Path,
    parquet_path: Path,
    device: torch.device,
    base_cfg: tsFormerConfig,
    epochs: int = 3,
    batch_size: int = 32,
) -> list[dict]:
    """
    Train tsFormer at different model sizes and record AUC gain.
    Mirrors Table 3 / Figure 6 approach from the paper.
    """
    print("\n" + "═" * 60)
    print("  Experiment: Model Size Scaling  (Table 3)")
    print("═" * 60)

    size_variants = {
        "tiny   (d=64,  L=2)": dict(d_model=64, n_layers=2, n_heads=4),
        "small  (d=128, L=4)": dict(d_model=128, n_layers=4, n_heads=8),
        "medium (d=256, L=6)": dict(d_model=256, n_layers=6, n_heads=8),
    }

    results = []
    for label, dims in size_variants.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.model.transformer.d_model = dims["d_model"]
        cfg.model.transformer.n_layers = dims["n_layers"]
        cfg.model.transformer.n_heads = dims["n_heads"]
        cfg.train.joint_fusion.batch_size = batch_size
        cfg.train.joint_fusion.amp_dtype = "none"

        r = _train_and_eval_tsformer(
            cfg,
            vocab,
            jsonl_path,
            parquet_path,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            label=label,
        )
        results.append(r)

    # Print Table 3 style
    baseline_auc = results[0]["auc"]
    print(f"\n  Model Size Results (baseline = {baseline_auc:.4f}):")
    print(f"  {'Variant':<30} {'# Params':>10} {'AUC':>8} {'Δ AUC':>8}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 8} {'─' * 8}")
    for r in results:
        delta = r["auc"] - baseline_auc
        print(
            f"  {r['label']:<30} {r['n_params']:>10,} {r['auc']:>8.4f} {delta:>+8.4f}"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Context Length  (Figure 6)
# ──────────────────────────────────────────────────────────────────────────────


def experiment_context_length(
    vocab: Vocabulary,
    jsonl_path: Path,
    parquet_path: Path,
    device: torch.device,
    base_cfg: tsFormerConfig,
    context_lengths: list[int] = (128, 256, 512),
    model_sizes: dict | None = None,
    epochs: int = 3,
    batch_size: int = 32,
) -> dict[str, list[dict]]:
    """
    Train tsFormer at different context lengths for multiple model sizes.
    Mirrors Figure 6 of the paper (512 / 1024 / 2048 in the paper;
    adapted to 128 / 256 / 512 for synthetic data).
    """
    print("\n" + "═" * 60)
    print("  Experiment: Context Length Scaling  (Figure 6)")
    print("═" * 60)

    if model_sizes is None:
        model_sizes = {
            "small  (d=128)": dict(d_model=128, n_layers=4, n_heads=8),
            "medium (d=256)": dict(d_model=256, n_layers=6, n_heads=8),
        }

    all_results: dict[str, list[dict]] = {name: [] for name in model_sizes}

    for ctx in context_lengths:
        for size_name, dims in model_sizes.items():
            label = f"ctx={ctx}  {size_name}"
            cfg = copy.deepcopy(base_cfg)
            cfg.model.transformer.d_model = dims["d_model"]
            cfg.model.transformer.n_layers = dims["n_layers"]
            cfg.model.transformer.n_heads = dims["n_heads"]
            cfg.model.transformer.max_seq_len = ctx
            cfg.train.joint_fusion.amp_dtype = "none"

            r = _train_and_eval_tsformer(
                cfg,
                vocab,
                jsonl_path,
                parquet_path,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                label=label,
            )
            r["context_length"] = ctx
            all_results[size_name].append(r)

    # Print Figure 6 style table
    print("\n  Context Length Results:")
    size_names = list(model_sizes.keys())
    header = f"  {'Context':>10}  " + "  ".join(f"{n:>20}" for n in size_names)
    print(header)
    print("  " + "─" * (12 + 22 * len(size_names)))
    for i, ctx in enumerate(context_lengths):
        row = f"  {ctx:>10}  "
        for name in size_names:
            auc = all_results[name][i]["auc"]
            row += f"  {auc:>20.4f}"
        print(row)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Data Volume  (Figure 7)
# ──────────────────────────────────────────────────────────────────────────────


def experiment_data_volume(
    vocab: Vocabulary,
    jsonl_path: Path,
    parquet_path: Path,
    device: torch.device,
    base_cfg: tsFormerConfig,
    volume_fracs: list[float] = (0.05, 0.10, 0.25, 0.50, 1.0),
    model_sizes: dict | None = None,
    epochs: int = 3,
    batch_size: int = 32,
) -> dict[str, list[dict]]:
    """
    Train tsFormer at different data volumes for two model sizes.
    Mirrors Figure 7 of the paper.
    """
    print("\n" + "═" * 60)
    print("  Experiment: Data Volume Scaling  (Figure 7)")
    print("═" * 60)

    if model_sizes is None:
        model_sizes = {
            "small  (d=64)": dict(d_model=64, n_layers=2, n_heads=4),
            "medium (d=128)": dict(d_model=128, n_layers=4, n_heads=8),
        }

    # Get total dataset size
    probe_ds = FineTuneDataset(jsonl_path, parquet_path, vocab, max_seq_len=512)
    n_total = len(probe_ds)
    test_n = max(10, int(n_total * 0.10))
    pool_n = n_total - test_n
    print(f"  Total rows: {n_total:,}  test: {test_n:,}  pool: {pool_n:,}")

    all_results: dict[str, list[dict]] = {name: [] for name in model_sizes}

    for frac in volume_fracs:
        n_use = max(20, int(pool_n * frac))
        for size_name, dims in model_sizes.items():
            label = f"frac={frac:.0%} n={n_use:,}  {size_name}"
            cfg = copy.deepcopy(base_cfg)
            cfg.model.transformer.d_model = dims["d_model"]
            cfg.model.transformer.n_layers = dims["n_layers"]
            cfg.model.transformer.n_heads = dims["n_heads"]
            cfg.model.transformer.max_seq_len = 256
            cfg.train.joint_fusion.amp_dtype = "none"

            r = _train_and_eval_tsformer(
                cfg,
                vocab,
                jsonl_path,
                parquet_path,
                device=device,
                n_train_rows=n_use,
                epochs=epochs,
                batch_size=batch_size,
                label=label,
            )
            r["frac"] = frac
            r["n_used"] = n_use
            all_results[size_name].append(r)

    # Print Figure 7 style table
    print("\n  Data Volume Results:")
    size_names = list(model_sizes.keys())
    header = f"  {'N rows':>10}  {'Frac':>6}  " + "  ".join(
        f"{n:>20}" for n in size_names
    )
    print(header)
    print("  " + "─" * (22 + 22 * len(size_names)))
    for i, frac in enumerate(volume_fracs):
        n_use = all_results[size_names[0]][i]["n_used"]
        row = f"  {n_use:>10,}  {frac:>5.0%}  "
        for name in size_names:
            auc = all_results[name][i]["auc"]
            row += f"  {auc:>20.4f}"
        print(row)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Results persistence
# ──────────────────────────────────────────────────────────────────────────────


def save_scaling_results(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scaling_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Scaling results saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="tsFormer scaling analysis (Section 4.3)")
    p.add_argument("--data-dir", type=str, default="./tsformer_data")
    p.add_argument("--results-dir", type=str, default="./scaling_results")
    p.add_argument(
        "--vocab-dir",
        type=str,
        default="./checkpoints",
        help="Dir containing vocabulary.json",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "model_size", "context_length", "data_volume"],
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--vocab-min-freq", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    res_dir = Path(args.results_dir)
    vocab_dir = Path(args.vocab_dir)

    jsonl_path = data_dir / "tokenized_sequences.jsonl"
    parquet_path = data_dir / "tabular_features.parquet"
    vocab_path = vocab_dir / "vocabulary.json"

    assert jsonl_path.exists(), f"Missing: {jsonl_path}"
    assert parquet_path.exists(), f"Missing: {parquet_path}"

    # Build or load vocabulary
    if vocab_path.exists():
        vocab = Vocabulary.load(vocab_path)
        print(f"  Loaded vocabulary: {len(vocab):,} tokens")
    else:
        vocab = build_vocabulary(jsonl_path, vocab_path, args.vocab_min_freq)

    # Base config: small model for speed; override per experiment
    base_cfg = tsFormerConfig.for_local_test()
    base_cfg.train.joint_fusion.amp_dtype = "none"

    all_results = {}

    if args.experiment in ("all", "model_size"):
        r = experiment_model_size(
            vocab,
            jsonl_path,
            parquet_path,
            device,
            base_cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_results["model_size"] = r

    if args.experiment in ("all", "context_length"):
        r = experiment_context_length(
            vocab,
            jsonl_path,
            parquet_path,
            device,
            base_cfg,
            context_lengths=[128, 256, 512],
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_results["context_length"] = r

    if args.experiment in ("all", "data_volume"):
        r = experiment_data_volume(
            vocab,
            jsonl_path,
            parquet_path,
            device,
            base_cfg,
            volume_fracs=[0.05, 0.10, 0.25, 0.50, 1.0],
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_results["data_volume"] = r

    save_scaling_results(all_results, res_dir)


if __name__ == "__main__":
    main()
