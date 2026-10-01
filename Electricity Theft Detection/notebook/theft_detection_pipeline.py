# %% [markdown]
# # Electricity Theft Detection in Distribution Networks using Machine Learning
# ### B.Tech EEE Capstone Project
#
# **Objective:** Build an ML classifier that flags consumers likely to be
# committing electricity theft / meter tampering, using engineered features
# derived from smart-meter (AMI) daily consumption data, so utilities can
# prioritise field inspections instead of random checks.
#
# **Workflow:** Data Collection → Preprocessing → EDA → Feature Engineering →
# Model Training → Evaluation → Deployment (Streamlit demo)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, ConfusionMatrixDisplay
)
import joblib

sns.set_style("whitegrid")
RANDOM_STATE = 42

# %% [markdown]
# ## 1. Load Raw Data
# Raw dataset: 3000 consumers x 180 days of daily kWh readings, simulated to
# mirror real AMI smart-meter data (seasonal + weekly patterns + noise), with
# ~14% of consumers exhibiting one of five realistic theft/tampering signatures:
# `sudden_drop`, `zero_stretches`, `flattened_cap`, `gradual_underreport`,
# `periodic_bypass`.

# %%
raw = pd.read_csv("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/data/smart_meter_data.csv")
print("Raw shape:", raw.shape)
raw[["consumer_id", "theft_type", "label"]].head()

# %% [markdown]
# ## 2. Data Preprocessing
# Real smart-meter feeds have communication gaps -> small % missing values
# were injected. We interpolate along each consumer's time series (linear),
# falling back to the consumer's own median for any remaining gaps.

# %%
date_cols = [c for c in raw.columns if c not in ("consumer_id", "theft_type", "label")]
print("Missing values before cleaning:", raw[date_cols].isna().sum().sum())

daily = raw[date_cols].astype(float)
daily = daily.interpolate(axis=1, limit_direction="both")
daily = daily.fillna(daily.median(axis=1), axis=0)
print("Missing values after cleaning:", daily.isna().sum().sum())

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sample_normal = daily[raw["label"] == 0].iloc[0]
sample_theft = daily[raw["label"] == 1].iloc[0]
axes[0].plot(sample_normal.values, color="seagreen")
axes[0].set_title("Sample Normal Consumer - Daily kWh")
axes[1].plot(sample_theft.values, color="firebrick")
axes[1].set_title(f"Sample Theft Consumer ({raw[raw['label']==1]['theft_type'].iloc[0]})")
plt.tight_layout()
plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/sample_series.png", dpi=150)
plt.show()

# %%
plt.figure(figsize=(5, 4))
raw["label"].map({0: "Normal", 1: "Theft"}).value_counts().plot(
    kind="bar", color=["seagreen", "firebrick"]
)
plt.title("Class Distribution")
plt.ylabel("Number of Consumers")
plt.tight_layout()
plt.savefig("../report_assets/class_distribution.png", dpi=150)
plt.show()

# %%
plt.figure(figsize=(6, 4))
sns.countplot(data=raw[raw.label == 1], x="theft_type",
              order=raw[raw.label == 1]["theft_type"].value_counts().index,
              hue="theft_type", palette="Reds_r", legend=False)
plt.xticks(rotation=30)
plt.title("Distribution of Theft Pattern Types")
plt.tight_layout()
plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/theft_type_distribution.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Feature Engineering
# Raw 180-day series are high-dimensional and don't generalise well as direct
# model input. We engineer statistical / behavioural features per consumer
# that are known (from DGA-analogous domain reasoning for load profiles) to
# separate normal usage from tampering signatures:
#
# - `mean_daily_kwh`, `std_daily_kwh`, `coefficient_of_variation`
# - `zero_day_fraction` — catches bypass/zero-stretch tampering
# - `trend_ratio` (2nd half mean / 1st half mean) — catches gradual under-reporting
# - `weekday_weekend_ratio` — catches periodic/weekend-only bypass
# - `sudden_drop_count` — catches abrupt meter tampering events
# - `weekly_cv`, `near_cap_fraction` — catches flattened/capped readings

# %%
import sys
sys.path.append("./Users/nivedsagark/Documents/Projects/Electricity Theft Detection/data")
from feature_engineering import build_features

feat_df = build_features(raw)
feat_df.to_csv("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/data/theft_features_dataset.csv", index=False)
feat_df.head()

# %%
plt.figure(figsize=(10, 6))
corr = feat_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/correlation_heatmap.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
plot_feats = ["mean_daily_kwh", "coefficient_of_variation", "zero_day_fraction",
              "trend_ratio", "sudden_drop_count", "weekly_cv"]
for ax, feat in zip(axes.flat, plot_feats):
    sns.boxplot(data=feat_df, x="Label", y=feat, hue="Label",
                palette={"Normal": "seagreen", "Theft": "firebrick"}, ax=ax, legend=False)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/feature_boxplots.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Prepare Data for Modelling

# %%
feature_cols = [
    "mean_daily_kwh", "std_daily_kwh", "coefficient_of_variation",
    "min_daily_kwh", "max_daily_kwh", "median_daily_kwh", "zero_day_fraction",
    "trend_ratio", "weekday_weekend_ratio", "sudden_drop_count",
    "weekly_cv", "near_cap_fraction", "max_to_mean_ratio",
]

X = feat_df[feature_cols]
y = LabelEncoder().fit_transform(feat_df["Label"])  # Normal=0, Theft=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Theft ratio in train:", y_train.mean().round(3), " test:", y_test.mean().round(3))

# %% [markdown]
# ## 6. Model Training & Comparison
# Because theft cases are a minority class (~14%), we use `class_weight="balanced"`
# / equivalent so the model doesn't just predict "Normal" for everyone.

# %%
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
}

results = []
fitted_models = {}
for name, model in models.items():
    if name in ("Logistic Regression", "SVM (RBF)"):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    results.append({"Model": name, "Accuracy": acc, "F1-Score (Theft)": f1, "ROC-AUC": auc})
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values("F1-Score (Theft)", ascending=False)
results_df

# %%
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
print("Best model selected:", best_name)

# %% [markdown]
# ## 7. Detailed Evaluation of Best Model

# %%
if best_name in ("Logistic Regression", "SVM (RBF)"):
    best_preds = best_model.predict(X_test_scaled)
    best_proba = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    best_preds = best_model.predict(X_test)
    best_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, best_preds, target_names=["Normal", "Theft"]))

# %%
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Theft"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/confusion_matrix.png", dpi=150)
plt.show()

# %%
fpr, tpr, _ = roc_curve(y_test, best_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"{best_name} (AUC = {roc_auc_score(y_test, best_proba):.3f})", color="firebrick")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/roc_curve.png", dpi=150)
plt.show()

# %%
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="teal")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/feature_importance.png", dpi=150)
    plt.show()

# %% [markdown]
# ## 8. Save Final Model Artefacts

# %%
joblib.dump(best_model, "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/model/theft_detection_model.pkl")
joblib.dump(scaler, "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/model/scaler.pkl")
joblib.dump(feature_cols, "/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/model/feature_columns.json")
results_df.to_csv("/Users/nivedsagark/Documents/Projects/Electricity Theft Detection/report_assets/model_comparison.csv", index=False)
print("Saved model, scaler and feature list to ../model/")
print(results_df)

# %% [markdown]
# ## 9. Conclusion
# The engineered behavioural features (zero-day fraction, trend ratio,
# weekday/weekend ratio, sudden-drop count, weekly CV) successfully separate
# theft/tampering consumers from normal consumers. The best-performing model
# is deployed in a Streamlit app (`app/streamlit_app.py`) that lets a utility
# analyst either enter a consumer's recent daily readings or upload a CSV of
# multiple consumers, and get back a theft-risk prediction with probability,
# to prioritise field inspection.
