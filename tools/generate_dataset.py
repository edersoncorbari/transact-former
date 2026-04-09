"""
Synthetic Financial Transaction Dataset Generator

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers" (nuFormer)

Generates a realistic synthetic dataset that mirrors the structure described in the paper:
  - Members with time-ordered transaction sequences
  - Each transaction: amount, date, description (text)
  - Binary recommendation label (positive interaction in ~6 months)
  - Three transaction sources: A (credit card), B (debit card), C (bill/other)
  - Tabular features (hand-crafted, mirroring the 291-feature baseline)

Usage:
    python generate_dataset.py --members 10000 --output ./data --seed 42
"""

import argparse
import json
import math
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker

# ──────────────────────────────────────────────────────────────────────────────
# Merchant / description vocabulary per source
# ──────────────────────────────────────────────────────────────────────────────

MERCHANTS_A = [  # Credit card = varied retail + services
    "NETFLIX", "SPOTIFY", "AMAZON", "UBER", "IFOOD", "RAPPI",
    "AMERICANAS", "MAGAZINE LUIZA", "MERCADO LIVRE", "SHOPEE",
    "POSTO IPIRANGA", "SHELL", "STARBUCKS", "MCDONALD'S", "BURGER KING",
    "FARMACIA DROGASIL", "DROGARIA SAO PAULO", "RENNER", "RIACHUELO",
    "DECATHLON", "CENTAURO", "STEAM", "PLAYSTATION", "APPLE", "GOOGLE",
    "BOOKING.COM", "AIRBNB", "LATAM AIRLINES", "GOL AIRLINES",
    "CLARO FATURA", "VIVO FATURA", "TIM FATURA",
    "SUPERMERCADO EXTRA", "CARREFOUR", "PAO DE ACUCAR",
    "ACADEMIA SMART FIT", "UBER EATS", "ZOMATO",
    "SARAIVA LIVRARIA", "FNAC", "ETSY",
]

MERCHANTS_B = [  # Debit card = everyday / local / small
    "PADARIA DO BAIRRO", "MERCADINHO LOCAL", "BANCA DE JORNAL",
    "FARMACIA LOCAL", "ACOUGUE DO JOAO", "FEIRINHA", "HORTIFRUTI",
    "LANCHONETE", "RESTAURANTE CASEIRO", "SORVETERIA",
    "POSTO DE GASOLINA", "ESTACIONAMENTO", "LAVAGEM DE CARRO",
    "SALAO DE BELEZA", "BARBEARIA", "LAVANDERIA",
    "PAPELARIA", "LOJA DE CONVENIENCIA", "QUENTINHA",
    "BOTECO", "ACAI DA ESQUINA", "TAPIOCA",
]

MERCHANTS_C = [  # Bill / structured payments
    "PAGAMENTO CARTAO CREDITO", "TARIFA BANCARIA",
    "SEGURO AUTO", "SEGURO VIDA", "SEGURO RESIDENCIAL",
    "ALUGUEL", "CONDOMINIO", "IPTU PARCELA",
    "IPVA PARCELA", "DPVAT", "DAS MEI",
    "BOLETO ENERGIA ELETRICA", "BOLETO AGUA", "BOLETO GAS",
    "PARCELA EMPRESTIMO", "CDC VEICULO",
    "PLANO DE SAUDE", "MENSALIDADE ESCOLA",
    "PIX ENVIADO", "TED ENVIADA", "DOC ENVIADO",
]

DESCRIPTION_SUFFIXES = [
    "", " PGTO", " *ONLINE", " APP", " DEBIT",
    " BR", " SP", " RJ", " MG", " BH",
    " 01", " 02", " 03",
]

# ──────────────────────────────────────────────────────────────────────────────
# Amount distributions per source (realistic BRL ranges)
# ──────────────────────────────────────────────────────────────────────────────

SOURCE_AMOUNT_PARAMS = {
    "A": {"mean": 120.0, "std": 200.0, "min": 5.0,  "max": 8000.0},
    "B": {"mean": 35.0,  "std": 50.0,  "min": 1.5,  "max": 500.0},
    "C": {"mean": 350.0, "std": 400.0, "min": 20.0, "max": 5000.0},
}

SOURCE_FREQ_PARAMS = {
    "A": {"mean": 18, "std": 8},
    "B": {"mean": 30, "std": 12},
    "C": {"mean": 5,  "std": 2},
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lognormal_amount(p: dict, rng: np.random.Generator) -> float:
    mu = math.log(p["mean"] ** 2 / math.sqrt(p["std"] ** 2 + p["mean"] ** 2))
    sigma = math.sqrt(math.log(1 + (p["std"] / p["mean"]) ** 2))
    val = rng.lognormal(mu, sigma)
    return round(_clamp(val, p["min"], p["max"]), 2)


def _random_description(source: str, rng: np.random.Generator) -> str:
    pool = {"A": MERCHANTS_A, "B": MERCHANTS_B, "C": MERCHANTS_C}[source]
    merchant = rng.choice(pool)
    suffix = rng.choice(DESCRIPTION_SUFFIXES)
    return (merchant + suffix).strip()


def _seasonal_factor(date: datetime) -> float:
    month = date.month
    profile = {1: 0.85, 2: 0.75, 3: 0.90, 4: 0.95,
               5: 1.00, 6: 1.05, 7: 1.10, 8: 1.00,
               9: 0.95, 10: 1.05, 11: 1.20, 12: 1.40}
    return profile[month]


# ──────────────────────────────────────────────────────────────────────────────
# Transaction generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_transactions(
    member_id: str,
    source: str,
    start_date: datetime,
    end_date: datetime,
    rng: np.random.Generator,
) -> list[dict]:
    freq_params = SOURCE_FREQ_PARAMS[source]
    amount_params = SOURCE_AMOUNT_PARAMS[source]

    txns = []
    current = start_date
    while current < end_date:
        # How many transactions this month?
        season = _seasonal_factor(current)
        monthly_n = max(0, int(rng.normal(
            freq_params["mean"] * season,
            freq_params["std"]
        )))
        days_in_month = 30
        days_remaining = (end_date - current).days
        days_span = min(days_in_month, days_remaining)

        for _ in range(monthly_n):
            day_offset = int(rng.uniform(0, days_span))
            txn_date = current + timedelta(days=day_offset)
            if txn_date >= end_date:
                break

            amount = _lognormal_amount(amount_params, rng)
            # ~15 % chance of inflow (refund / cashback)
            sign = -1.0 if rng.random() < 0.15 else 1.0
            
            txns.append({
                "member_id": member_id,
                "source": source,
                "date": txn_date.strftime("%Y-%m-%d"),
                "amount": round(sign * amount, 2),
                "description": _random_description(source, rng),
            })

        current += timedelta(days=days_span)

    # Sort chronologically
    txns.sort(key=lambda t: t["date"])
    return txns


# ──────────────────────────────────────────────────────────────────────────────
# Tabular feature generator
# Mimics the paper's 291-feature baseline (numeric + categorical)
# ──────────────────────────────────────────────────────────────────────────────

def generate_tabular_features(
    member_id: str,
    transactions: list[dict],
    label: int,
    score_date: datetime,
    rng: np.random.Generator,
) -> dict:
    txns_df = pd.DataFrame(transactions) if transactions else pd.DataFrame(
        columns=["member_id", "source", "date", "amount", "description"]
    )

    def source_txns(src):
        if txns_df.empty:
            return pd.DataFrame(columns=txns_df.columns)
        return txns_df[txns_df["source"] == src].copy()

    def period_stats(df, days, prefix):
        feats = {}
        cutoff = (score_date - timedelta(days=days)).strftime("%Y-%m-%d")
        sub = df[df["date"] >= cutoff] if not df.empty else df
        feats[f"{prefix}_n_txns_{days}d"] = len(sub)
        feats[f"{prefix}_total_spend_{days}d"] = round(sub["amount"].clip(lower=0).sum(), 2) if not sub.empty else 0.0
        feats[f"{prefix}_avg_spend_{days}d"] = round(sub["amount"].clip(lower=0).mean(), 2) if not sub.empty else 0.0
        feats[f"{prefix}_max_spend_{days}d"] = round(sub["amount"].clip(lower=0).max(), 2) if not sub.empty else 0.0
        feats[f"{prefix}_n_inflow_{days}d"] = int((sub["amount"] < 0).sum()) if not sub.empty else 0
        return feats

    row: dict = {"member_id": member_id, "score_date": score_date.strftime("%Y-%m-%d"), "label": label}

    # ── Per-source transaction features ──
    for src in ["A", "B", "C"]:
        sdf = source_txns(src)
        for days in [30, 60, 90, 180]:
            row.update(period_stats(sdf, days, f"src{src}"))
        row[f"src{src}_lifetime_txns"] = len(sdf)
        row[f"src{src}_lifetime_spend"] = round(sdf["amount"].clip(lower=0).sum(), 2) if not sdf.empty else 0.0
        row[f"src{src}_unique_merchants"] = sdf["description"].nunique() if not sdf.empty else 0
        row[f"src{src}_months_active"] = sdf["date"].str[:7].nunique() if not sdf.empty else 0

    # ── Aggregate across all sources ──
    for days in [30, 60, 90, 180]:
        row.update(period_stats(txns_df, days, "all"))

    row["total_lifetime_txns"] = len(txns_df)
    row["total_lifetime_spend"] = round(txns_df["amount"].clip(lower=0).sum(), 2) if not txns_df.empty else 0.0
    row["unique_merchants_all"] = txns_df["description"].nunique() if not txns_df.empty else 0
    row["months_active_all"] = txns_df["date"].str[:7].nunique() if not txns_df.empty else 0

    # ── Seasonality features ──
    row["has_dec_spend"] = int(
        not txns_df.empty and txns_df[txns_df["date"].str[5:7] == "12"]["amount"].clip(lower=0).sum() > 0
    )
    row["has_nov_spend"] = int(
        not txns_df.empty and txns_df[txns_df["date"].str[5:7] == "11"]["amount"].clip(lower=0).sum() > 0
    )

    # ── External / bureau features (synthetic noise around label signal) ──
    base_score = 650 + label * 80 + rng.normal(0, 60)
    row["bureau_score"] = int(_clamp(base_score, 300, 900))
    row["bureau_score_band"] = (
        "poor" if row["bureau_score"] < 500
        else "fair" if row["bureau_score"] < 650
        else "good" if row["bureau_score"] < 750
        else "excellent"
    )

    row["account_age_days"] = int(rng.uniform(30, 1825))
    row["n_products_active"] = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.15, 0.10]))
    row["has_credit_card"] = int(len(source_txns("A")) > 0)
    row["has_debit_card"] = int(len(source_txns("B")) > 0)
    row["has_bill_pay"] = int(len(source_txns("C")) > 0)
    row["income_band"] = rng.choice(["low", "medium", "high", "very_high"], p=[0.25, 0.40, 0.25, 0.10])
    row["state"] = rng.choice(["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "CE", "PE", "GO"],
                              p=[0.35, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.04])
    row["age_group"] = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"],
                                  p=[0.18, 0.32, 0.25, 0.15, 0.10])

    # ── Derived ratios ──
    spend_30 = row.get("all_total_spend_30d", 0)
    spend_90 = row.get("all_total_spend_90d", 0)
    row["spend_growth_ratio"] = round(spend_30 / (spend_90 / 3 + 1e-6), 4)
    row["digital_spend_share"] = round(
        row.get("srcA_total_spend_90d", 0) / (row.get("all_total_spend_90d", 0) + 1e-6), 4
    )

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Label generation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_label_probability(transactions: list[dict], bureau_score: float) -> float:
    n_txns = len(transactions)
    spend = sum(t["amount"] for t in transactions if t["amount"] > 0)
    unique_merchants = len({t["description"] for t in transactions})

    # Logistic-like signal (calibrated for ~15 % positive rate)
    logit = (
        -7.2
        + 0.0005 * n_txns
        + 0.00002 * spend
        + 0.01 * unique_merchants
        + 0.002 * (bureau_score - 650)
    )
    prob = 1 / (1 + math.exp(-logit))
    return _clamp(prob, 0.02, 0.80)


# ──────────────────────────────────────────────────────────────────────────────
# Main member generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_member(
    member_id: str,
    score_date: datetime,
    history_months: int,
    rng: np.random.Generator,
    sources: tuple[str, ...] = ("A", "B", "C"),
) -> tuple[list[dict], dict]:
    start_date = score_date - timedelta(days=history_months * 30)
    end_date = score_date

    transactions: list[dict] = []
    for src in sources:
        # Not all members use all sources
        if rng.random() < 0.85 or src == "A":
            transactions.extend(
                generate_transactions(member_id, src, start_date, end_date, rng)
            )

    # Preliminary bureau score to influence label
    rough_bureau = 650 + rng.normal(0, 80)
    label_prob = _compute_label_probability(transactions, rough_bureau)
    label = int(rng.random() < label_prob)

    tabular = generate_tabular_features(member_id, transactions, label, score_date, rng)

    # Attach member metadata to each transaction row
    for t in transactions:
        t["score_date"] = score_date.strftime("%Y-%m-%d")
        t["label"] = label

    return transactions, tabular


# ──────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    n_members: int,
    n_score_dates_per_member: int,
    history_months: int,
    base_date: datetime,
    seed: int,
    sources: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # fake = Faker("pt_BR")
    Faker.seed(seed)

    all_transactions: list[dict] = []
    all_tabular: list[dict] = []

    print(f"Generating {n_members:,} members × {n_score_dates_per_member} score date(s) …")

    for i in range(n_members):
        if (i + 1) % max(1, n_members // 10) == 0:
            print(f"  {i + 1:,} / {n_members:,}")

        member_id = f"MBR_{i:08d}"

        for _ in range(n_score_dates_per_member):
            # Score dates spread over a 12-month window before base_date
            offset_days = int(rng.uniform(0, 365))
            score_date = base_date - timedelta(days=offset_days)

            txns, tab = generate_member(
                member_id=member_id,
                score_date=score_date,
                history_months=history_months,
                rng=rng,
                sources=sources,
            )
            all_transactions.extend(txns)
            all_tabular.append(tab)

    transactions_df = pd.DataFrame(all_transactions)
    tabular_df = pd.DataFrame(all_tabular)

    # Sort transactions by member and date
    if not transactions_df.empty:
        transactions_df = transactions_df.sort_values(["member_id", "score_date", "date"]).reset_index(drop=True)

    tabular_df = tabular_df.sort_values(["member_id", "score_date"]).reset_index(drop=True)

    print(f"\nDone.")
    print(f"  Transactions : {len(transactions_df):,} rows")
    print(f"  Tabular rows : {len(tabular_df):,} rows")
    print(f"  Positive rate: {tabular_df['label'].mean():.2%}")

    return transactions_df, tabular_df


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

AMOUNT_BINS = [-8000, -500, -100, -50, -20, -10, -5,
               0, 5, 10, 20, 50, 100, 200, 500, 1000,
               2000, 5000, 8000, float("inf")]

def tokenize_transaction(txn: dict) -> list[str]:
    amount = float(txn["amount"])
    date = datetime.strptime(txn["date"], "%Y-%m-%d")

    # Sign token
    sign_tok = "<PAID>" if amount < 0 else "<RCVD>"

    # Amount bucket token
    bucket = next(
        (i for i, edge in enumerate(AMOUNT_BINS[1:], start=1) if abs(amount) < edge),
        len(AMOUNT_BINS) - 1,
    )

    low = AMOUNT_BINS[bucket - 1] if bucket > 0 else 0
    high = AMOUNT_BINS[bucket]
    amt_tok = f"<AMOUNT:{low:.0f}-{high:.0f}>"

    # Date tokens
    month_tok = f"<MONTH:{date.strftime('%b').upper()}>"
    day_tok = f"<DAY:{date.day:02d}>"
    weekday_tok = f"<WEEKDAY:{date.strftime('%A').upper()}>"

    # Pseudo-BPE: simple whitespace split on description
    desc_tokens = txn["description"].split()

    # Order follows Eq. (1): φ_sign, φ_amt, φ_month, φ_day, φ_weekday ⊕ BPE(desc)
    return [sign_tok, amt_tok, month_tok, day_tok, weekday_tok] + desc_tokens

def tokenize_member_sequence(
    member_transactions: list[dict],
    max_tokens: int = 2048,
    sep_token: str = "<SEP>",
) -> list[str]:

    tokens_per_txn = [
        tokenize_transaction(txn) + [sep_token]
        for txn in member_transactions
    ]

    # Get most recent transactions
    selected, total = [], 0
    
    for tok in reversed(tokens_per_txn):
        if total + len(tok) > max_tokens:
            break
        selected.append(tok)
        total += len(tok)

    # Restore chronological order (without reversing tokens)
    selected.reverse()

    return [t for txn in selected for t in txn]


# ──────────────────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_dataset(
    transactions_df: pd.DataFrame,
    tabular_df: pd.DataFrame,
    output_dir: Path,
    max_tokens: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw transactions (parquet + CSV)
    transactions_df.to_parquet(output_dir / "transactions.parquet", index=False)
    transactions_df.to_csv(output_dir / "transactions.csv", index=False)
    print(f"  Saved transactions → {output_dir / 'transactions.parquet'}")

    # 2. Tabular features (parquet + CSV)
    tabular_df.to_parquet(output_dir / "tabular_features.parquet", index=False)
    tabular_df.to_csv(output_dir / "tabular_features.csv", index=False)
    print(f"  Saved tabular features → {output_dir / 'tabular_features.parquet'}")

    # 3. Tokenized member sequences (JSONL — one line per modelling row)
    jsonl_path = output_dir / "tokenized_sequences.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, tab_row in tabular_df.iterrows():
            mid = tab_row["member_id"]
            sd = tab_row["score_date"]
            member_txns = transactions_df[
                (transactions_df["member_id"] == mid) &
                (transactions_df["score_date"] == sd)
            ].to_dict("records")
            tokens = tokenize_member_sequence(member_txns, max_tokens=max_tokens)
            record = {
                "member_id": mid,
                "score_date": sd,
                "label": int(tab_row["label"]),
                "n_tokens": len(tokens),
                "tokens": tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved tokenized sequences → {jsonl_path}")

    # 4. Dataset statistics
    stats = {
        "n_members": tabular_df["member_id"].nunique(),
        "n_modelling_rows": len(tabular_df),
        "n_transactions": len(transactions_df),
        "positive_rate": round(float(tabular_df["label"].mean()), 4),
        "avg_txns_per_member": round(
            len(transactions_df) / max(1, tabular_df["member_id"].nunique()), 1
        ),
        "source_distribution": (
            transactions_df["source"].value_counts().to_dict()
            if not transactions_df.empty else {}
        ),
        "label_dist": tabular_df["label"].value_counts().to_dict(),
        "tabular_feature_count": len(tabular_df.columns) - 3,  # excl. member_id, score_date, label
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats → {output_dir / 'dataset_stats.json'}")
    print(f"\nDataset statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction dataset"
    )
    parser.add_argument("--members", type=int, default=5_000,
                        help="Number of unique members (default: 5000)")
    parser.add_argument("--score-dates", type=int, default=1,
                        help="Score date snapshots per member (default: 1)")
    parser.add_argument("--history-months", type=int, default=12,
                        help="Months of transaction history per member (default: 12)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max token context length for sequences (default: 2048)")
    parser.add_argument("--sources", type=str, default="A,B,C",
                        help="Comma-separated transaction sources to include (default: A,B,C)")
    parser.add_argument("--base-date", type=str, default="2024-01-01",
                        help="Reference date for score window (default: 2024-01-01)")
    parser.add_argument("--output", type=str, default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources = tuple(s.strip().upper() for s in args.sources.split(","))
    base_date = datetime.strptime(args.base_date, "%Y-%m-%d")
    output_dir = Path(args.output)

    print("=" * 60)
    print("  Synthetic Dataset Generator")
    print("=" * 60)
    print(f"  Members       : {args.members:,}")
    print(f"  Score dates   : {args.score_dates}")
    print(f"  History       : {args.history_months} months")
    print(f"  Sources       : {sources}")
    print(f"  Max tokens    : {args.max_tokens}")
    print(f"  Base date     : {args.base_date}")
    print(f"  Output dir    : {output_dir}")
    print(f"  Seed          : {args.seed}")
    print("=" * 60)

    transactions_df, tabular_df = build_dataset(
        n_members=args.members,
        n_score_dates_per_member=args.score_dates,
        history_months=args.history_months,
        base_date=base_date,
        seed=args.seed,
        sources=sources,
    )

    print("\nSaving dataset …")
    save_dataset(transactions_df, tabular_df, output_dir, args.max_tokens)
    print("\nAll files written successfully.")


if __name__ == "__main__":
    main()

