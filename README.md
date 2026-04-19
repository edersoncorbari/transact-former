# transact-former

Transformer-based framework for modeling financial transaction sequences and predicting user behavior through multi-stage training.

## Overview

`transact-former` is a deep learning framework designed to learn user financial behavior from transaction data. It combines sequence modeling with tabular features using a multi-phase training pipeline:

- **Phase 1 — Pre-training**: Next-Token Prediction (NTP) on raw transaction sequences  
- **Phase 2 — Fine-tuning**: Binary classification with Transformer embeddings  
- **Phase 3 — Joint Fusion**: End-to-end training with sequence and tabular features  

The architecture is inspired by recent research on applying Transformers to financial behavior modeling.

---

## Key Features

- Transformer-based sequence modeling for transaction data  
- Multi-stage training pipeline (pretrain → finetune → fusion)  
- Mixed precision training (fp16 / bf16)  
- Support for GPU acceleration (CUDA)  
- Modular design for experimentation  

---

## Dataset Generator

Example 1:

```bash
python -m tools.generate_dataset \
    --members 10000 \
    --fraud-rate 0.12 \
    --output ./data \
    --seed 42
```

Example 2:

```bash
python -m tools.generate_dataset.py \
    --members 50000 \
    --score-dates 2 \
    --history-months 12 \
    --output ./data \
    --fraud-rate 0.12 \
    --seed 42
```

---

## Train Model

Example 1:

```bash
python -m tsformer.train_pipeline \
    --preset local_test \
    --data-dir ./data \
    --checkpoint-dir ./checkpoints \
    --results-dir ./results \
    --pretrain-epochs 1 \
    --finetune-epochs 1 \
    --fusion-epochs 5 \
    --vocab-min-freq 2 \
    --device cuda
```
