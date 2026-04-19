## Synthetic Financial Transaction Dataset Generator

```bash
python -m tools.generate_dataset \
    --members 10000 \
    --fraud-rate 0.12 \
    --output ./data \
    --seed 42 \
    --device cuda
```

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