"""
Synthetic Financial Transaction Dataset Generator

Author: Ederson Corbari <ecorbari@proton.me>
Created: 2026-04
Based on: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

Generates a realistic synthetic dataset that mirrors the structure described in the paper:
  - Members with time-ordered transaction sequences
  - Each transaction: amount, date, description (text)
  - Binary recommendation label (positive interaction in ~6 months)
  - Three transaction sources: A (credit card), B (debit card), C (bill/other)
  - Tabular features (hand-crafted, mirroring the 291-feature baseline)

Usage:
    python generate_dataset.py --members 10000 --fraud-rate 0.12 --output ./data --seed 42
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

MERCHANTS_A = [
    "NETFLIX",
    "SPOTIFY",
    "AMAZON",
    "UBER",
    "IFOOD",
    "RAPPI",
    "AMERICANAS",
    "MAGAZINE LUIZA",
    "MERCADO LIVRE",
    "SHOPEE",
    "POSTO IPIRANGA",
    "SHELL",
    "STARBUCKS",
    "MCDONALD'S",
    "BURGER KING",
    "FARMACIA DROGASIL",
    "DROGARIA SAO PAULO",
    "RENNER",
    "RIACHUELO",
    "DECATHLON",
    "CENTAURO",
    "STEAM",
    "PLAYSTATION",
    "APPLE",
    "GOOGLE",
    "BOOKING.COM",
    "AIRBNB",
    "LATAM AIRLINES",
    "GOL AIRLINES",
    "CLARO FATURA",
    "VIVO FATURA",
    "TIM FATURA",
    "SUPERMERCADO EXTRA",
    "CARREFOUR",
    "PAO DE ACUCAR",
    "ACADEMIA SMART FIT",
    "UBER EATS",
    "ZOMATO",
    "SARAIVA LIVRARIA",
    "FNAC",
    "ETSY",
]

MERCHANTS_B = [
    "PADARIA DO BAIRRO",
    "MERCADINHO LOCAL",
    "BANCA DE JORNAL",
    "FARMACIA LOCAL",
    "ACOUGUE DO JOAO",
    "FEIRINHA",
    "HORTIFRUTI",
    "LANCHONETE",
    "RESTAURANTE CASEIRO",
    "SORVETERIA",
    "POSTO DE GASOLINA",
    "ESTACIONAMENTO",
    "LAVAGEM DE CARRO",
    "SALAO DE BELEZA",
    "BARBEARIA",
    "LAVANDERIA",
    "PAPELARIA",
    "LOJA DE CONVENIENCIA",
    "QUENTINHA",
    "BOTECO",
    "ACAI DA ESQUINA",
    "TAPIOCA",
]

MERCHANTS_C = [
    "PAGAMENTO CARTAO CREDITO",
    "TARIFA BANCARIA",
    "SEGURO AUTO",
    "SEGURO VIDA",
    "SEGURO RESIDENCIAL",
    "ALUGUEL",
    "CONDOMINIO",
    "IPTU PARCELA",
    "IPVA PARCELA",
    "DPVAT",
    "DAS MEI",
    "BOLETO ENERGIA ELETRICA",
    "BOLETO AGUA",
    "BOLETO GAS",
    "PARCELA EMPRESTIMO",
    "CDC VEICULO",
    "PLANO DE SAUDE",
    "MENSALIDADE ESCOLA",
    "PIX ENVIADO",
    "TED ENVIADA",
    "DOC ENVIADO",
]

# Merchants typical in fraud scenarios (high-risk CNP / digital goods / gift cards)
MERCHANTS_FRAUD = [
    "GIFT CARD ONLINE",
    "CRYPTO EXCHANGE",
    "WESTERN UNION",
    "MONEYGRAM",
    "CASINO ONLINE",
    "BET365",
    "LOJA GENERICA ONLINE",
    "MARKETPLACE XYZ",
    "FOREIGN TRANSFER",
    "DIGITAL GOODS STORE",
    "PREPAID CARD RELOAD",
    "STEAM WALLET CODE",
]

DESCRIPTION_SUFFIXES = [
    "",
    " PGTO",
    " *ONLINE",
    " APP",
    " DEBIT",
    " BR",
    " SP",
    " RJ",
    " MG",
    " BH",
    " 01",
    " 02",
    " 03",
]

# ──────────────────────────────────────────────────────────────────────────────
# Amount distributions per source (realistic BRL ranges)
# ──────────────────────────────────────────────────────────────────────────────

SOURCE_AMOUNT_PARAMS = {
    "A": {"mean": 120.0, "std": 200.0, "min": 5.0, "max": 8000.0},
    "B": {"mean": 35.0, "std": 50.0, "min": 1.5, "max": 500.0},
    "C": {"mean": 350.0, "std": 400.0, "min": 20.0, "max": 5000.0},
}

SOURCE_FREQ_PARAMS = {
    "A": {"mean": 18, "std": 8},
    "B": {"mean": 30, "std": 12},
    "C": {"mean": 5, "std": 2},
}

# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer bins (for amount bucketing in sequence models)
# ──────────────────────────────────────────────────────────────────────────────

AMOUNT_BINS = [
    -8000,
    -500,
    -100,
    -50,
    -20,
    -10,
    -5,
    0,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
    8000,
    float("inf"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Fraud archetypes
# ──────────────────────────────────────────────────────────────────────────────

FRAUD_ARCHETYPES = [
    "velocity_burst",  # Many txns in a short window (card testing + draining)
    "large_cnp",  # Few large CNP/digital transactions
    "night_pattern",  # Concentrated activity between 00:00–05:00
    "ticket_spike",  # Sudden increase in ticket relative to member's avg
    "micro_charges",  # Many small charges (card testing pattern)
    "geo_anomaly",  # International / distant merchant mixed with local
    "account_takeover",  # Abrupt behavioral shift: new merchants, new amounts
]

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


def _random_description(
    source: str, rng: np.random.Generator, fraud_merchant: bool = False
) -> str:
    if fraud_merchant:
        merchant = rng.choice(MERCHANTS_FRAUD)
    else:
        pool = {"A": MERCHANTS_A, "B": MERCHANTS_B, "C": MERCHANTS_C}[source]
        merchant = rng.choice(pool)
    suffix = rng.choice(DESCRIPTION_SUFFIXES)
    return (merchant + suffix).strip()


def _seasonal_factor(date: datetime) -> float:
    month = date.month
    profile = {
        1: 0.85,
        2: 0.75,
        3: 0.90,
        4: 0.95,
        5: 1.00,
        6: 1.05,
        7: 1.10,
        8: 1.00,
        9: 0.95,
        10: 1.05,
        11: 1.20,
        12: 1.40,
    }
    return profile[month]


# ──────────────────────────────────────────────────────────────────────────────
# Fraud transaction injectors
# ──────────────────────────────────────────────────────────────────────────────


def _inject_velocity_burst(
    member_id: str, anchor: datetime, rng: np.random.Generator
) -> list[dict]:
    """Many small/medium transactions within a 2-hour window (card testing then draining)."""
    txns = []
    n = int(rng.integers(15, 35))
    for _ in range(n):
        offset_min = int(rng.integers(0, 120))
        txn_dt = anchor + timedelta(minutes=offset_min)
        amount = round(
            float(
                rng.choice([1.0, 2.0, 5.0, 9.99, 19.99, 49.99, 200.0, 500.0, 1000.0])
            ),
            2,
        )
        txns.append(
            {
                "member_id": member_id,
                "source": "A",
                "date": txn_dt.strftime("%Y-%m-%d"),
                "hour": txn_dt.hour,
                "amount": amount,
                "description": _random_description(
                    "A", rng, fraud_merchant=rng.random() < 0.6
                ),
                "is_fraud_txn": True,
            }
        )
    return txns


def _inject_large_cnp(
    member_id: str, anchor: datetime, rng: np.random.Generator
) -> list[dict]:
    """Few large card-not-present transactions at fraud merchants."""
    txns = []
    n = int(rng.integers(2, 6))
    for _ in range(n):
        offset_hours = int(rng.integers(0, 48))
        txn_dt = anchor + timedelta(hours=offset_hours)
        amount = round(float(rng.uniform(1500.0, 7500.0)), 2)
        txns.append(
            {
                "member_id": member_id,
                "source": "A",
                "date": txn_dt.strftime("%Y-%m-%d"),
                "hour": txn_dt.hour,
                "amount": amount,
                "description": _random_description("A", rng, fraud_merchant=True),
                "is_fraud_txn": True,
            }
        )
    return txns


def _inject_night_pattern(
    member_id: str, anchor: datetime, rng: np.random.Generator
) -> list[dict]:
    """Concentrated transactions between 00:00 and 05:00 over multiple nights."""
    txns = []
    n_nights = int(rng.integers(3, 8))
    for night in range(n_nights):
        n_txns = int(rng.integers(3, 8))
        base_dt = anchor + timedelta(days=night)
        for _ in range(n_txns):
            hour = int(rng.integers(0, 5))
            minute = int(rng.integers(0, 60))
            txn_dt = base_dt.replace(hour=hour, minute=minute)
            amount = round(float(rng.uniform(50.0, 800.0)), 2)
            txns.append(
                {
                    "member_id": member_id,
                    "source": "A",
                    "date": txn_dt.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "amount": amount,
                    "description": _random_description(
                        "A", rng, fraud_merchant=rng.random() < 0.4
                    ),
                    "is_fraud_txn": True,
                }
            )
    return txns


def _inject_micro_charges(
    member_id: str, anchor: datetime, rng: np.random.Generator
) -> list[dict]:
    """Many R$0.01–R$5.00 charges (classic card testing pattern)."""
    txns = []
    n = int(rng.integers(20, 50))
    for _ in range(n):
        offset_min = int(rng.integers(0, 240))
        txn_dt = anchor + timedelta(minutes=offset_min)
        amount = round(float(rng.uniform(0.01, 5.0)), 2)
        txns.append(
            {
                "member_id": member_id,
                "source": "A",
                "date": txn_dt.strftime("%Y-%m-%d"),
                "hour": txn_dt.hour,
                "amount": amount,
                "description": _random_description("A", rng, fraud_merchant=True),
                "is_fraud_txn": True,
            }
        )
    return txns


def _inject_geo_anomaly(
    member_id: str, anchor: datetime, rng: np.random.Generator
) -> list[dict]:
    """International/distant merchants suddenly appearing."""
    FOREIGN_MERCHANTS = [
        "AMAZON USA",
        "PAYPAL INTL",
        "ALIEXPRESS",
        "WISH.COM",
        "FOREIGN CASINO",
        "OVERSEAS TRANSFER",
        "BINANCE INTL",
    ]
    txns = []
    n = int(rng.integers(3, 8))
    for _ in range(n):
        offset_days = int(rng.integers(0, 10))
        txn_dt = anchor + timedelta(days=offset_days)
        amount = round(float(rng.uniform(200.0, 3000.0)), 2)
        txns.append(
            {
                "member_id": member_id,
                "source": "A",
                "date": txn_dt.strftime("%Y-%m-%d"),
                "hour": int(rng.integers(0, 24)),
                "amount": amount,
                "description": rng.choice(FOREIGN_MERCHANTS),
                "is_fraud_txn": True,
            }
        )
    return txns


def _inject_account_takeover(
    member_id: str, anchor: datetime, rng: np.random.Generator
) -> list[dict]:
    """Abrupt change: new categories, unusual amounts, PIX draining."""
    txns = []
    # First: rapid PIX outflows
    for _ in range(int(rng.integers(3, 7))):
        offset_hours = int(rng.integers(0, 6))
        txn_dt = anchor + timedelta(hours=offset_hours)
        amount = round(float(rng.uniform(500.0, 5000.0)), 2)
        txns.append(
            {
                "member_id": member_id,
                "source": "C",
                "date": txn_dt.strftime("%Y-%m-%d"),
                "hour": txn_dt.hour,
                "amount": amount,
                "description": "PIX ENVIADO",
                "is_fraud_txn": True,
            }
        )
    # Then: unusual high-value purchases
    for _ in range(int(rng.integers(2, 5))):
        offset_hours = int(rng.integers(6, 24))
        txn_dt = anchor + timedelta(hours=offset_hours)
        amount = round(float(rng.uniform(1000.0, 6000.0)), 2)
        txns.append(
            {
                "member_id": member_id,
                "source": "A",
                "date": txn_dt.strftime("%Y-%m-%d"),
                "hour": txn_dt.hour,
                "amount": amount,
                "description": _random_description("A", rng, fraud_merchant=True),
                "is_fraud_txn": True,
            }
        )
    return txns


ARCHETYPE_INJECTORS = {
    "velocity_burst": _inject_velocity_burst,
    "large_cnp": _inject_large_cnp,
    "night_pattern": _inject_night_pattern,
    "micro_charges": _inject_micro_charges,
    "geo_anomaly": _inject_geo_anomaly,
    "account_takeover": _inject_account_takeover,
}


# ──────────────────────────────────────────────────────────────────────────────
# Regular transaction generator
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
        season = _seasonal_factor(current)
        monthly_n = max(
            0, int(rng.normal(freq_params["mean"] * season, freq_params["std"]))
        )
        days_in_month = 30
        days_remaining = (end_date - current).days
        days_span = min(days_in_month, days_remaining)

        for _ in range(monthly_n):
            day_offset = int(rng.uniform(0, days_span))
            hour = int(rng.integers(6, 23))  # Normal hours: 6am–11pm
            txn_date = current + timedelta(days=day_offset)
            if txn_date >= end_date:
                break

            amount = _lognormal_amount(amount_params, rng)
            sign = -1.0 if rng.random() < 0.15 else 1.0

            txns.append(
                {
                    "member_id": member_id,
                    "source": source,
                    "date": txn_date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "amount": round(sign * amount, 2),
                    "description": _random_description(source, rng),
                    "is_fraud_txn": False,
                }
            )

        current += timedelta(days=days_span)

    txns.sort(key=lambda t: t["date"])
    return txns


# ──────────────────────────────────────────────────────────────────────────────
# Label generation — STRENGTHENED SIGNAL
# ──────────────────────────────────────────────────────────────────────────────


def _compute_fraud_probability(
    transactions: list[dict],
    bureau_score: float,
    archetype: str | None,
) -> float:
    """
    Stronger logistic model for fraud probability.
    Key changes vs original:
      - Higher absolute coefficients
      - Penalizes high bureau score (fraud ≠ bad credit)
      - Archetype adds a large additive boost to logit
      - Behavioral features: micro charges, night ratio, velocity
    """
    if not transactions:
        base_logit = -5.0
        if archetype:
            base_logit += 4.5
        return _clamp(1 / (1 + math.exp(-base_logit)), 0.01, 0.98)

    amounts = [t["amount"] for t in transactions if t["amount"] > 0]
    hours = [t.get("hour", 12) for t in transactions]

    n_txns = len(transactions)
    total_spend = sum(amounts)
    avg_ticket = total_spend / max(len(amounts), 1)
    unique_merchants = len({t["description"] for t in transactions})

    # Velocity: txns in last 24h window (simulate by looking at same-date clusters)
    from collections import Counter

    date_counts = Counter(t["date"] for t in transactions)
    max_daily_velocity = max(date_counts.values()) if date_counts else 0

    # Night ratio: transactions between 00:00 and 05:00
    night_txns = sum(1 for h in hours if h < 5)
    night_ratio = night_txns / max(n_txns, 1)

    # Micro charges: amounts < R$5
    micro_count = sum(1 for a in amounts if a < 5.0)

    # Ticket spike: max amount / avg ticket
    max_amount = max(amounts) if amounts else 0.0
    ticket_spike = max_amount / max(avg_ticket, 1.0)

    # Fraud merchants
    fraud_merchant_count = sum(1 for t in transactions if t.get("is_fraud_txn", False))

    logit = (
        -4.5  # Base intercept (calibrated for ~10-15% rate)
        + 0.008 * max_daily_velocity  # High daily velocity → suspicious
        + 3.5 * night_ratio  # Night transactions → strong signal
        + 0.04 * micro_count  # Card testing pattern
        + 0.002 * ticket_spike  # Ticket spike
        + 0.015 * fraud_merchant_count  # Known fraud merchants
        - 0.003 * (bureau_score - 650)  # High bureau score slightly reduces prob
        + 0.0001 * total_spend  # Higher spend slightly increases prob
        + 0.005 * unique_merchants  # Many unique merchants = slight risk
    )

    # Archetype boost — this is the key amplifier
    archetype_boost = {
        "velocity_burst": 4.0,
        "large_cnp": 3.5,
        "night_pattern": 3.8,
        "ticket_spike": 3.2,
        "micro_charges": 4.2,
        "geo_anomaly": 3.0,
        "account_takeover": 4.5,
    }
    if archetype:
        logit += archetype_boost.get(archetype, 3.0)

    prob = 1 / (1 + math.exp(-logit))
    return _clamp(prob, 0.01, 0.97)


# ──────────────────────────────────────────────────────────────────────────────
# Tabular feature generator (fraud-enriched)
# ──────────────────────────────────────────────────────────────────────────────


def generate_tabular_features(
    member_id: str,
    transactions: list[dict],
    label: int,
    score_date: datetime,
    bureau_score: int,
    archetype: str | None,
    rng: np.random.Generator,
) -> dict:
    txns_df = (
        pd.DataFrame(transactions)
        if transactions
        else pd.DataFrame(
            columns=[
                "member_id",
                "source",
                "date",
                "hour",
                "amount",
                "description",
                "is_fraud_txn",
            ]
        )
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
        feats[f"{prefix}_total_spend_{days}d"] = (
            round(sub["amount"].clip(lower=0).sum(), 2) if not sub.empty else 0.0
        )
        feats[f"{prefix}_avg_spend_{days}d"] = (
            round(sub["amount"].clip(lower=0).mean(), 2) if not sub.empty else 0.0
        )
        feats[f"{prefix}_max_spend_{days}d"] = (
            round(sub["amount"].clip(lower=0).max(), 2) if not sub.empty else 0.0
        )
        feats[f"{prefix}_n_inflow_{days}d"] = (
            int((sub["amount"] < 0).sum()) if not sub.empty else 0
        )
        return feats

    row: dict = {
        "member_id": member_id,
        "score_date": score_date.strftime("%Y-%m-%d"),
        "label": label,
        "fraud_archetype": archetype or "none",
    }

    # ── Per-source transaction features ──
    for src in ["A", "B", "C"]:
        sdf = source_txns(src)
        for days in [30, 60, 90, 180]:
            row.update(period_stats(sdf, days, f"src{src}"))
        row[f"src{src}_lifetime_txns"] = len(sdf)
        row[f"src{src}_lifetime_spend"] = (
            round(sdf["amount"].clip(lower=0).sum(), 2) if not sdf.empty else 0.0
        )
        row[f"src{src}_unique_merchants"] = (
            sdf["description"].nunique() if not sdf.empty else 0
        )
        row[f"src{src}_months_active"] = (
            sdf["date"].str[:7].nunique() if not sdf.empty else 0
        )

    # ── Aggregate across all sources ──
    for days in [30, 60, 90, 180]:
        row.update(period_stats(txns_df, days, "all"))

    row["total_lifetime_txns"] = len(txns_df)
    row["total_lifetime_spend"] = (
        round(txns_df["amount"].clip(lower=0).sum(), 2) if not txns_df.empty else 0.0
    )
    row["unique_merchants_all"] = (
        txns_df["description"].nunique() if not txns_df.empty else 0
    )
    row["months_active_all"] = (
        txns_df["date"].str[:7].nunique() if not txns_df.empty else 0
    )

    # ── Seasonality features ──
    row["has_dec_spend"] = int(
        not txns_df.empty
        and txns_df[txns_df["date"].str[5:7] == "12"]["amount"].clip(lower=0).sum() > 0
    )
    row["has_nov_spend"] = int(
        not txns_df.empty
        and txns_df[txns_df["date"].str[5:7] == "11"]["amount"].clip(lower=0).sum() > 0
    )

    # ── Bureau / external (consistent — score generated BEFORE label) ──
    row["bureau_score"] = bureau_score
    row["bureau_score_band"] = (
        "poor"
        if bureau_score < 500
        else "fair"
        if bureau_score < 650
        else "good"
        if bureau_score < 750
        else "excellent"
    )

    row["account_age_days"] = int(rng.uniform(30, 1825))
    row["n_products_active"] = int(rng.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.15, 0.10]))
    row["has_credit_card"] = int(len(source_txns("A")) > 0)
    row["has_debit_card"] = int(len(source_txns("B")) > 0)
    row["has_bill_pay"] = int(len(source_txns("C")) > 0)
    row["income_band"] = rng.choice(
        ["low", "medium", "high", "very_high"], p=[0.25, 0.40, 0.25, 0.10]
    )
    row["state"] = rng.choice(
        ["SP", "RJ", "MG", "RS", "PR", "BA", "SC", "CE", "PE", "GO"],
        p=[0.35, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.04],
    )
    row["age_group"] = rng.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"], p=[0.18, 0.32, 0.25, 0.15, 0.10]
    )

    # ── Derived ratios ──
    spend_30 = row.get("all_total_spend_30d", 0)
    spend_90 = row.get("all_total_spend_90d", 0)
    row["spend_growth_ratio"] = round(spend_30 / (spend_90 / 3 + 1e-6), 4)
    row["digital_spend_share"] = round(
        row.get("srcA_total_spend_90d", 0) / (row.get("all_total_spend_90d", 0) + 1e-6),
        4,
    )

    # ── FRAUD-SPECIFIC FEATURES ──
    if not txns_df.empty:
        from collections import Counter

        amounts_pos = txns_df[txns_df["amount"] > 0]["amount"]
        hours_all = (
            txns_df["hour"] if "hour" in txns_df.columns else pd.Series(dtype=float)
        )
        date_counts = Counter(txns_df["date"].tolist())

        # Velocity: max transactions in a single day
        row["velocity_max_daily"] = max(date_counts.values()) if date_counts else 0

        # Night ratio (00:00–05:00)
        night_mask = hours_all < 5
        row["night_txn_ratio"] = (
            round(float(night_mask.mean()), 4) if len(hours_all) > 0 else 0.0
        )
        row["night_txn_count_30d"] = (
            int(
                (
                    txns_df[
                        (
                            txns_df["date"]
                            >= (score_date - timedelta(days=30)).strftime("%Y-%m-%d")
                        )
                        & (txns_df["hour"] < 5)
                    ].shape[0]
                )
            )
            if "hour" in txns_df.columns
            else 0
        )

        # Micro-charge count (< R$5, positive)
        row["micro_charge_count_30d"] = int(
            txns_df[
                (txns_df["amount"] > 0)
                & (txns_df["amount"] < 5.0)
                & (
                    txns_df["date"]
                    >= (score_date - timedelta(days=30)).strftime("%Y-%m-%d")
                )
            ].shape[0]
        )

        # Ticket spike ratio (max / avg)
        avg_t = float(amounts_pos.mean()) if len(amounts_pos) > 0 else 1.0
        max_t = float(amounts_pos.max()) if len(amounts_pos) > 0 else 0.0
        row["ticket_spike_ratio"] = round(max_t / max(avg_t, 1.0), 4)

        # Fraud merchants ratio
        if "is_fraud_txn" in txns_df.columns:
            row["fraud_merchant_ratio"] = round(
                float(txns_df["is_fraud_txn"].mean()), 4
            )
        else:
            row["fraud_merchant_ratio"] = 0.0

        # CNP proxy: source A + high amount
        cnp_mask = (txns_df["source"] == "A") & (txns_df["amount"] > 500)
        row["cnp_high_value_count_30d"] = int(
            txns_df[
                cnp_mask
                & (
                    txns_df["date"]
                    >= (score_date - timedelta(days=30)).strftime("%Y-%m-%d")
                )
            ].shape[0]
        )

        # Geo anomaly flag (inferred from foreign-like merchant names)
        foreign_keywords = [
            "USA",
            "INTL",
            "ALIEXPRESS",
            "WISH",
            "OVERSEAS",
            "FOREIGN",
            "BINANCE",
        ]
        has_foreign = (
            txns_df["description"]
            .str.contains("|".join(foreign_keywords), case=False)
            .any()
        )
        row["geo_anomaly_flag"] = int(has_foreign)

    else:
        row["velocity_max_daily"] = 0
        row["night_txn_ratio"] = 0.0
        row["night_txn_count_30d"] = 0
        row["micro_charge_count_30d"] = 0
        row["ticket_spike_ratio"] = 0.0
        row["fraud_merchant_ratio"] = 0.0
        row["cnp_high_value_count_30d"] = 0
        row["geo_anomaly_flag"] = 0

    return row


# ──────────────────────────────────────────────────────────────────────────────
# Member generator
# ──────────────────────────────────────────────────────────────────────────────


def generate_member(
    member_id: str,
    score_date: datetime,
    history_months: int,
    rng: np.random.Generator,
    sources: tuple[str, ...] = ("A", "B", "C"),
    target_fraud_rate: float = 0.12,
) -> tuple[list[dict], dict]:
    start_date = score_date - timedelta(days=history_months * 30)
    end_date = score_date

    # Generate regular transactions
    transactions: list[dict] = []
    for src in sources:
        if rng.random() < 0.85 or src == "A":
            transactions.extend(
                generate_transactions(member_id, src, start_date, end_date, rng)
            )

    # Bureau score generated BEFORE label (causal consistency)
    bureau_score = int(_clamp(650 + rng.normal(0, 80), 300, 900))

    # Determine if this member is a fraud candidate
    # We pre-sample archetype to inject anomalous txns BEFORE computing label
    pre_fraud = rng.random() < (target_fraud_rate * 3.0)
    archetype: str | None = None

    avg_ticket = (
        float(np.mean([t["amount"] for t in transactions if t["amount"] > 0]))
        if transactions
        else 100.0
    )
    anchor_date = end_date - timedelta(days=int(rng.integers(7, 30)))

    if pre_fraud:
        archetype = str(rng.choice(list(ARCHETYPE_INJECTORS.keys())))
        injector = ARCHETYPE_INJECTORS[archetype]
        if archetype == "ticket_spike":
            fraud_txns = injector(member_id, anchor_date, rng, avg_ticket)
        else:
            fraud_txns = injector(member_id, anchor_date, rng)
        transactions.extend(fraud_txns)
        transactions.sort(key=lambda t: t["date"])

    # Compute label probability with actual behavioral features
    label_prob = _compute_fraud_probability(transactions, bureau_score, archetype)
    label = int(rng.random() < label_prob)

    # If label=0 but archetype was set, clear archetype (not actually fraud)
    if label == 0:
        archetype = None

    tabular = generate_tabular_features(
        member_id, transactions, label, score_date, bureau_score, archetype, rng
    )

    # Attach metadata to transaction rows
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
    target_fraud_rate: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    Faker.seed(seed)

    all_transactions: list[dict] = []
    all_tabular: list[dict] = []

    print(
        f"Generating {n_members:,} members × {n_score_dates_per_member} score date(s) …"
    )
    print(f"  Target fraud rate: {target_fraud_rate:.1%}")

    for i in range(n_members):
        if (i + 1) % max(1, n_members // 10) == 0:
            print(f"  {i + 1:,} / {n_members:,}")

        member_id = f"MBR_{i:08d}"

        for _ in range(n_score_dates_per_member):
            offset_days = int(rng.uniform(0, 365))
            score_date = base_date - timedelta(days=offset_days)

            txns, tab = generate_member(
                member_id=member_id,
                score_date=score_date,
                history_months=history_months,
                rng=rng,
                sources=sources,
                target_fraud_rate=target_fraud_rate,
            )
            all_transactions.extend(txns)
            all_tabular.append(tab)

    transactions_df = pd.DataFrame(all_transactions)
    tabular_df = pd.DataFrame(all_tabular)

    if not transactions_df.empty:
        transactions_df = transactions_df.sort_values(
            ["member_id", "score_date", "date"]
        ).reset_index(drop=True)

    tabular_df = tabular_df.sort_values(["member_id", "score_date"]).reset_index(
        drop=True
    )

    actual_fraud_rate = tabular_df["label"].mean()
    print("\nDone.")
    print(f"  Transactions  : {len(transactions_df):,} rows")
    print(f"  Tabular rows  : {len(tabular_df):,} rows")
    print(f"  Positive rate : {actual_fraud_rate:.2%}")

    if actual_fraud_rate < 0.03:
        print("    Fraud rate below 3% — consider increasing --fraud-rate")

    return transactions_df, tabular_df


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────


def tokenize_transaction(txn: dict) -> list[str]:
    amount = float(txn["amount"])
    date = datetime.strptime(txn["date"], "%Y-%m-%d")

    sign_tok = "<PAID>" if amount < 0 else "<RCVD>"

    bucket = next(
        (i for i, edge in enumerate(AMOUNT_BINS[1:], start=1) if abs(amount) < edge),
        len(AMOUNT_BINS) - 1,
    )

    low = AMOUNT_BINS[bucket - 1] if bucket > 0 else 0
    high = AMOUNT_BINS[bucket]
    amt_tok = f"<AMOUNT:{low:.0f}-{high:.0f}>"

    month_tok = f"<MONTH:{date.strftime('%b').upper()}>"
    day_tok = f"<DAY:{date.day:02d}>"
    weekday_tok = f"<WEEKDAY:{date.strftime('%A').upper()}>"

    # Hour token (NEW — captures night pattern signal for model)
    hour = txn.get("hour", 12)
    if hour < 5:
        hour_tok = "<HOUR:NIGHT>"
    elif hour < 12:
        hour_tok = "<HOUR:MORNING>"
    elif hour < 18:
        hour_tok = "<HOUR:AFTERNOON>"
    else:
        hour_tok = "<HOUR:EVENING>"

    desc_tokens = txn["description"].split()

    return [sign_tok, amt_tok, month_tok, day_tok, weekday_tok, hour_tok] + desc_tokens


def tokenize_member_sequence(
    member_transactions: list[dict],
    max_tokens: int = 2048,
    sep_token: str = "<SEP>",
) -> list[str]:
    tokens_per_txn = [
        tokenize_transaction(txn) + [sep_token] for txn in member_transactions
    ]

    selected, total = [], 0

    for tok in reversed(tokens_per_txn):
        if total + len(tok) > max_tokens:
            break
        selected.append(tok)
        total += len(tok)

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

    transactions_df.to_parquet(output_dir / "transactions.parquet", index=False)
    transactions_df.to_csv(output_dir / "transactions.csv", index=False)
    print(f"  Saved transactions → {output_dir / 'transactions.parquet'}")

    tabular_df.to_parquet(output_dir / "tabular_features.parquet", index=False)
    tabular_df.to_csv(output_dir / "tabular_features.csv", index=False)
    print(f"  Saved tabular features → {output_dir / 'tabular_features.parquet'}")

    jsonl_path = output_dir / "tokenized_sequences.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, tab_row in tabular_df.iterrows():
            mid = tab_row["member_id"]
            sd = tab_row["score_date"]
            member_txns = transactions_df[
                (transactions_df["member_id"] == mid)
                & (transactions_df["score_date"] == sd)
            ].to_dict("records")
            tokens = tokenize_member_sequence(member_txns, max_tokens=max_tokens)
            record = {
                "member_id": mid,
                "score_date": sd,
                "label": int(tab_row["label"]),
                "fraud_archetype": tab_row.get("fraud_archetype", "none"),
                "n_tokens": len(tokens),
                "tokens": tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved tokenized sequences → {jsonl_path}")

    archetype_dist = (
        tabular_df[tabular_df["label"] == 1]["fraud_archetype"].value_counts().to_dict()
        if "fraud_archetype" in tabular_df.columns
        else {}
    )

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
            if not transactions_df.empty
            else {}
        ),
        "label_dist": tabular_df["label"].value_counts().to_dict(),
        "fraud_archetype_dist": archetype_dist,
        "tabular_feature_count": len(tabular_df.columns) - 3,
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats → {output_dir / 'dataset_stats.json'}")
    print("\nDataset statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud transaction dataset (tsFormer)"
    )
    parser.add_argument(
        "--members",
        type=int,
        default=5_000,
        help="Number of unique members (default: 5000)",
    )
    parser.add_argument(
        "--score-dates",
        type=int,
        default=1,
        help="Score date snapshots per member (default: 1)",
    )
    parser.add_argument(
        "--history-months",
        type=int,
        default=12,
        help="Months of transaction history per member (default: 12)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max token context length for sequences (default: 2048)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="A,B,C",
        help="Comma-separated transaction sources (default: A,B,C)",
    )
    parser.add_argument(
        "--base-date",
        type=str,
        default="2024-01-01",
        help="Reference date for score window (default: 2024-01-01)",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.12,
        help="Target fraud rate, 0.0–1.0 (default: 0.12)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources = tuple(s.strip().upper() for s in args.sources.split(","))
    base_date = datetime.strptime(args.base_date, "%Y-%m-%d")
    output_dir = Path(args.output)

    print("=" * 60)
    print("  Synthetic Fraud Dataset Generator (tsFormer)")
    print("=" * 60)
    print(f"  Members       : {args.members:,}")
    print(f"  Score dates   : {args.score_dates}")
    print(f"  History       : {args.history_months} months")
    print(f"  Sources       : {sources}")
    print(f"  Max tokens    : {args.max_tokens}")
    print(f"  Base date     : {args.base_date}")
    print(f"  Target fraud  : {args.fraud_rate:.1%}")
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
        target_fraud_rate=args.fraud_rate,
    )

    print("\nSaving dataset …")
    save_dataset(transactions_df, tabular_df, output_dir, args.max_tokens)
    print("\nAll files written successfully.")


if __name__ == "__main__":
    main()
