from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def explore_data(df: pd.DataFrame) -> dict:
    stats = {}
    for col in NUMERIC_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
                "std": float(s.std()),
            }

    return {
        "numeric_stats": stats,
        "target_counts": {str(k): int(v) for k, v in df["HeartDisease"].value_counts().items()},
        "chest_pain_counts": {str(k): int(v) for k, v in df["ChestPainType"].value_counts().items()},
        "age_by_target_mean": {str(k): float(v) for k, v in df.groupby("HeartDisease")["Age"].mean().items()},
        "chol_by_target_mean": {str(k): float(v) for k, v in df.groupby("HeartDisease")["Cholesterol"].mean().items()},
    }


def plot_age_distribution(df: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Age", hue="HeartDisease", kde=True, bins=20, ax=ax)
    ax.set_title("Age Distribution by HeartDisease")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_target_distribution(df: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="HeartDisease", ax=ax)
    ax.set_title("HeartDisease Class Distribution")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_chest_pain_distribution(df: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="ChestPainType", hue="HeartDisease", ax=ax)
    ax.set_title("ChestPainType vs HeartDisease")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_cholesterol_boxplot(df: pd.DataFrame, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="HeartDisease", y="Cholesterol", ax=ax)
    ax.set_title("Cholesterol by HeartDisease")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None):
    cols = [c for c in ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak", "HeartDisease"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
