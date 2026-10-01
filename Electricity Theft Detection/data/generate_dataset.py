"""
generate_dataset.py
Generates a realistic synthetic smart-meter electricity consumption dataset
for the Electricity Theft Detection capstone project.

Structure mirrors real-world smart meter datasets (e.g., SGCC):
- Each row = one consumer
- Columns = daily energy consumption (kWh) over N days
- Label = 0 (normal / honest consumer), 1 (theft / anomalous consumer)

Consumption is modeled with:
- Seasonal trend (sinusoidal, yearly)
- Weekly pattern (weekday vs weekend)
- Random noise
- Theft patterns injected: sudden drops, zero-consumption stretches,
  capped/flattened peaks, and gradual under-reporting (typical real theft signatures)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CONSUMERS = 3000
N_DAYS = 180  # 6 months of daily readings
THEFT_RATIO = 0.15  # ~15% theft cases (realistic class imbalance)

start_date = datetime(2025, 1, 1)
date_cols = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(N_DAYS)]


def base_consumption_profile(n_days, base_load, seasonal_amp, weekly_amp, noise_std):
    t = np.arange(n_days)
    seasonal = seasonal_amp * np.sin(2 * np.pi * t / 365 + np.random.uniform(0, 2 * np.pi))
    weekly = weekly_amp * np.sin(2 * np.pi * t / 7 + np.random.uniform(0, 2 * np.pi))
    noise = np.random.normal(0, noise_std, n_days)
    consumption = base_load + seasonal + weekly + noise
    return np.clip(consumption, 0, None)


def inject_theft_pattern(profile, pattern_type):
    p = profile.copy()
    n = len(p)
    if pattern_type == "sudden_drop":
        start = np.random.randint(int(n * 0.3), int(n * 0.6))
        p[start:] *= np.random.uniform(0.15, 0.35)
    elif pattern_type == "zero_stretches":
        n_stretches = np.random.randint(3, 8)
        for _ in range(n_stretches):
            s = np.random.randint(0, n - 10)
            length = np.random.randint(2, 6)
            p[s:s + length] = np.random.uniform(0, 0.2)
    elif pattern_type == "flattened_cap":
        cap = np.percentile(p, 40)
        p = np.minimum(p, cap) * np.random.uniform(0.8, 1.0)
    elif pattern_type == "gradual_underreport":
        decay = np.linspace(1.0, np.random.uniform(0.25, 0.45), n)
        p = p * decay
    elif pattern_type == "periodic_bypass":
        # tampering only on specific weekdays (e.g., meter bypassed on weekends)
        mask = (np.arange(n) % 7 >= 5)
        p[mask] *= np.random.uniform(0.1, 0.3)
    return np.clip(p, 0, None)


records = []
labels = []
consumer_ids = []
theft_types = []

for i in range(N_CONSUMERS):
    consumer_id = f"CUST{i+1:05d}"
    base_load = np.random.uniform(3, 25)       # kWh/day base load varies by household/industry
    seasonal_amp = np.random.uniform(0.5, 5)
    weekly_amp = np.random.uniform(0.2, 3)
    noise_std = np.random.uniform(0.3, 2)

    profile = base_consumption_profile(N_DAYS, base_load, seasonal_amp, weekly_amp, noise_std)

    is_theft = np.random.rand() < THEFT_RATIO
    theft_type = "none"
    if is_theft:
        theft_type = np.random.choice(
            ["sudden_drop", "zero_stretches", "flattened_cap", "gradual_underreport", "periodic_bypass"]
        )
        profile = inject_theft_pattern(profile, theft_type)

    records.append(profile)
    labels.append(int(is_theft))
    consumer_ids.append(consumer_id)
    theft_types.append(theft_type)

df = pd.DataFrame(records, columns=date_cols)
df.insert(0, "consumer_id", consumer_ids)
df["theft_type"] = theft_types  # kept for reference / EDA only, dropped before training
df["label"] = labels  # 1 = theft, 0 = normal

# introduce a small amount of missing values (real smart meter data has gaps)
mask = np.random.rand(*df[date_cols].shape) < 0.01
df.loc[:, date_cols] = df[date_cols].mask(mask)

out_path = "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/data/smart_meter_data.csv"
df.to_csv(out_path, index=False)
print(f"Dataset saved to {out_path}")
print(f"Shape: {df.shape}")
print(f"Theft ratio: {df['label'].mean():.3f}")
print(df.head())
