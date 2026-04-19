## Synthetic Financial Transaction Dataset Generator


## Dataset Generator

Example 1:

```bash
python -m tools.generate_dataset \
    --members 10000 \
    --fraud-rate 0.12 \
    --output ./data \
    --seed 42 \
    --device cuda
```

Example 2:

```bash
python -m tools.generate_dataset.py \
    --members 50000 \
    --score-dates 2 \ 
    --history-months 12 \
    --output ./data \
    --fraud-rate 0.12 \
    --seed 42 \
    --device cuda
```

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
