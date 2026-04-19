import torch

from tsformer.model import tsFormer
from tsformer.data import Vocabulary


def load_model(ckpt_path, vocab_path):
    vocab = Vocabulary.load(vocab_path)

    model = tsFormer(
        vocab_size=len(vocab),
        n_tabular=113,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, vocab


def prepare_tabular_data(raw_row):
    # Skip member_id (0), score_date (1), label (2) -> 113 columns remain
    numeric_features = raw_row[3:]
    processed = []

    # Mapping dictionary for the strings present in your data
    mapping = {
        "fair": 1.0,
        "medium": 2.0,
        "SP": 25.0,
        "RJ": 19.0,
        "45-54": 4.0,
        "25-34": 2.0,
        "very_high": 5.0,
    }

    for val in numeric_features:
        if isinstance(val, str):
            processed.append(float(mapping.get(val, 0.0)))
        else:
            processed.append(float(val if val is not None else 0.0))

    return torch.tensor([processed], dtype=torch.float32)


# --- EXECUTION ---
ckpt = "checkpoints/finetune_final.pt"
vcb = "checkpoints/vocabulary.json"

model, vocab = load_model(ckpt, vcb)

tokens_example = ["<RCVD>", "<AMOUNT:50-100>", "<MONTH:DEC>", "SUPERMERCADO", "<SEP>"]
tabular_example = [
    "MBR_00000000",
    "2023-03-25",
    0,
    17,
    2179.86,
    128.23,
    655.17,
    5,
    35,
    3316.45,
    94.76,
    655.17,
    10,
    61,
    6461.32,
    105.92,
    658.15,
    12,
    92,
    8880.69,
    96.53,
    658.15,
    13,
    206,
    19386.03,
    168,
    13,
    37,
    860.53,
    23.26,
    221.63,
    3,
    70,
    2221.87,
    31.74,
    347.53,
    8,
    139,
    4347.43,
    31.28,
    347.53,
    16,
    224,
    6885.18,
    30.74,
    347.53,
    22,
    374,
    10699.96,
    206,
    12,
    0,
    0.0,
    0.0,
    0.0,
    0,
    4,
    2944.73,
    736.18,
    1461.48,
    0,
    12,
    5576.03,
    464.67,
    1461.48,
    2,
    30,
    9815.68,
    327.19,
    1461.48,
    3,
    57,
    16736.98,
    52,
    11,
    54,
    3040.39,
    56.3,
    655.17,
    8,
    109,
    8483.05,
    77.83,
    1461.48,
    18,
    212,
    16384.78,
    77.29,
    1461.48,
    30,
    346,
    25581.55,
    73.94,
    1461.48,
    38,
    637,
    46822.97,
    426,
    13,
    1,
    1,
    524,
    "fair",
    1417,
    1,
    1,
    1,
    1,
    "medium",
    "SP",
    "45-54",
    0.5567,
    0.3943,
]

ids = torch.tensor([vocab.encode(tokens_example)], dtype=torch.long)
tabs = prepare_tabular_data(tabular_example)

with torch.no_grad():
    output = model(input_ids=ids, tabular_feats=tabs)
    prob = torch.sigmoid(output["logits"]).item()

print(f"\nUser: {tabular_example[0]}")
print(f"Propensity Score: {prob:.4%}")
