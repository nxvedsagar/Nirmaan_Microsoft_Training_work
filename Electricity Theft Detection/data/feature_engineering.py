"""
feature_engineering.py
Converts the raw wide-format smart-meter dataset (one row per consumer,
one column per day) into an ML-ready feature table.

This mirrors how utilities engineer features from AMI (Advanced Metering
Infrastructure) daily-read data before feeding a classifier, since raw
day-by-day readings are too high-dimensional / non-generalisable to use
directly as model input.
"""

import numpy as np
import pandas as pd

RAW_PATH = "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/data/smart_meter_data.csv"
OUT_PATH = "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/data/theft_features_dataset.csv"


def rolling_week_totals(series):
    n_weeks = len(series) // 7
    trimmed = series[: n_weeks * 7].reshape(n_weeks, 7)
    return trimmed.sum(axis=1)


def build_features(df):
    date_cols = [c for c in df.columns if c not in ("consumer_id", "theft_type", "label")]
    # fill small missing-value gaps (linear interpolation, then median fallback)
    daily = df[date_cols].astype(float)
    daily = daily.interpolate(axis=1, limit_direction="both")
    daily = daily.fillna(daily.median(axis=1), axis=0)

    values = daily.values
    n_days = values.shape[1]
    first_half = values[:, : n_days // 2]
    second_half = values[:, n_days // 2:]

    rows = []
    for i in range(len(df)):
        series = values[i]
        weekly_totals = rolling_week_totals(series)

        mean_val = series.mean()
        std_val = series.std()
        cv = std_val / (mean_val + 1e-6)
        min_val = series.min()
        max_val = series.max()
        zero_frac = np.mean(series < 0.15)
        median_val = np.median(series)

        # trend: compare first half average vs second half average
        # (captures "gradual under-reporting" theft pattern)
        first_half_mean = first_half[i].mean() if i < first_half.shape[0] else first_half.mean(axis=0).mean()
        # NOTE: index i already selects the row, redo correctly below
        first_half_mean = first_half[i].mean()
        second_half_mean = second_half[i].mean()
        trend_ratio = second_half_mean / (first_half_mean + 1e-6)

        # weekday vs weekend variability (captures periodic_bypass pattern)
        weekday_vals = series[np.arange(n_days) % 7 < 5]
        weekend_vals = series[np.arange(n_days) % 7 >= 5]
        weekday_weekend_ratio = (weekend_vals.mean() + 1e-6) / (weekday_vals.mean() + 1e-6)

        # sudden drop count: consecutive-day drop greater than 50% of mean
        diffs = np.diff(series)
        sudden_drop_count = int(np.sum(diffs < -0.5 * (mean_val + 1e-6)))

        # week-to-week volatility (captures zero_stretches / flattened_cap)
        week_std = weekly_totals.std()
        week_mean = weekly_totals.mean()
        week_cv = week_std / (week_mean + 1e-6)

        # flattening indicator: ratio of days near the 40th percentile (capped-looking) 
        p40 = np.percentile(series, 40)
        near_cap_fraction = np.mean(np.abs(series - p40) < 0.05 * (mean_val + 1e-6))

        rows.append({
            "consumer_id": df.loc[i, "consumer_id"],
            "mean_daily_kwh": round(mean_val, 3),
            "std_daily_kwh": round(std_val, 3),
            "coefficient_of_variation": round(cv, 4),
            "min_daily_kwh": round(min_val, 3),
            "max_daily_kwh": round(max_val, 3),
            "median_daily_kwh": round(median_val, 3),
            "zero_day_fraction": round(zero_frac, 4),
            "trend_ratio": round(trend_ratio, 4),
            "weekday_weekend_ratio": round(weekday_weekend_ratio, 4),
            "sudden_drop_count": sudden_drop_count,
            "weekly_cv": round(week_cv, 4),
            "near_cap_fraction": round(near_cap_fraction, 4),
            "max_to_mean_ratio": round(max_val / (mean_val + 1e-6), 4),
            "theft_type": df.loc[i, "theft_type"],
            "Label": "Theft" if df.loc[i, "label"] == 1 else "Normal",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    feat_df = build_features(raw)
    feat_df = feat_df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    feat_df.to_csv(OUT_PATH, index=False)
    print("Feature dataset shape:", feat_df.shape)
    print(feat_df["Label"].value_counts())
    print(feat_df.head())
