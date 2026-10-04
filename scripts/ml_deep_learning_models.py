# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:13:51 2026

@author: ak7u24
"""
"""
Plot deep-learning model use across selected studies and create an UpSet-style
plot of exact deep-learning model combinations.

Input:
    input/ml_deep_learning_models.xlsx

Each row represents one study. Studies are treated as using deep learning if the
column 'ml_model_type' is either 'Deep Learning' or 'Both'.

Outputs:
    output/deep_learning_model_use_plot.png/.pdf/.svg
    output/deep_learning_model_use_summary.csv
    output/deep_learning_model_upset_plot.png/.pdf/.svg
    output/deep_learning_model_combination_summary.csv
"""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "ml_deep_learning_models.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_BAR_PNG = OUTPUT_DIR / "deep_learning_model_use_plot.png"
OUTPUT_BAR_PDF = OUTPUT_DIR / "deep_learning_model_use_plot.pdf"
OUTPUT_BAR_SVG = OUTPUT_DIR / "deep_learning_model_use_plot.svg"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "deep_learning_model_use_summary.csv"

OUTPUT_UPSET_PNG = OUTPUT_DIR / "deep_learning_model_upset_plot.png"
OUTPUT_UPSET_PDF = OUTPUT_DIR / "deep_learning_model_upset_plot.pdf"
OUTPUT_UPSET_SVG = OUTPUT_DIR / "deep_learning_model_upset_plot.svg"
OUTPUT_COMBINATION_CSV = OUTPUT_DIR / "deep_learning_model_combination_summary.csv"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

STUDY_ID_COLUMN = "Study Reference"
ML_MODEL_TYPE_COLUMN = "ml_model_type"

# Column name, abbreviation, display label
DEEP_LEARNING_MODEL_COLUMNS = [
    ("cnn", "CNN", "Convolutional neural network"),
    ("rnn", "RNN", "Recurrent neural network"),
    ("lstm", "LSTM", "Long short-term memory"),
    ("gru", "GRU", "Gated recurrent unit"),
    ("autoencoders", "AE", "Autoencoder"),
    ("attention", "Attention", "Attention mechanism"),
    ("transformer", "Transformer", "Transformer"),
    ("simple_multi_layer_model", "MLP", "Simple multi-layer model"),
]

DROP_ZERO_USE_MODELS_FROM_PLOT = False
SORT_MODELS_BY_COUNT = True
TOP_N_COMBINATIONS = 21
EXCLUDE_NO_MODEL_ROWS_FROM_COMBINATIONS = True
OUTPUT_DPI = 600


# --------------------------------------------------
# Plot style
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalise_text(value) -> str:
    """Normalise text for robust matching of column names and labels."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("‘", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def cell_to_bool(value) -> bool:
    """Convert TRUE/FALSE-like Excel values to Boolean."""
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    key = str(value).strip().lower()

    true_values = {"true", "t", "yes", "y", "1", "used", "use", "present", "included"}
    false_values = {
        "false", "f", "no", "n", "0", "not used", "absent",
        "na", "n/a", "nan", "none", "not reported", "unclear", ""
    }

    if key in true_values:
        return True
    if key in false_values:
        return False
    return False


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the Excel file and clean column names."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    if STUDY_ID_COLUMN not in df.columns:
        df[STUDY_ID_COLUMN] = np.arange(1, len(df) + 1)

    return df


def validate_and_resolve_columns(df: pd.DataFrame) -> list[dict]:
    """Resolve configured model columns against actual Excel columns."""
    normalised_to_actual = {normalise_text(col): col for col in df.columns}

    if ML_MODEL_TYPE_COLUMN not in df.columns:
        normalised_type_col = normalise_text(ML_MODEL_TYPE_COLUMN)
        if normalised_type_col in normalised_to_actual:
            df.rename(columns={normalised_to_actual[normalised_type_col]: ML_MODEL_TYPE_COLUMN}, inplace=True)
        else:
            raise KeyError(
                f"Required column not found: {ML_MODEL_TYPE_COLUMN}\n"
                f"Available columns: {list(df.columns)}"
            )

    model_info = []
    missing_columns = []

    for raw_column, abbreviation, display_label in DEEP_LEARNING_MODEL_COLUMNS:
        normalised_raw = normalise_text(raw_column)

        if raw_column in df.columns:
            actual_column = raw_column
        elif normalised_raw in normalised_to_actual:
            actual_column = normalised_to_actual[normalised_raw]
        else:
            missing_columns.append(raw_column)
            continue

        model_info.append({
            "configured_column": raw_column,
            "actual_column": actual_column,
            "abbreviation": abbreviation,
            "display_label": display_label,
        })

    if missing_columns:
        raise KeyError(
            f"Missing deep-learning model columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    return model_info


def identify_deep_learning_studies(df: pd.DataFrame) -> pd.Series:
    """Return a Boolean mask identifying studies using deep learning."""
    ml_type = df[ML_MODEL_TYPE_COLUMN].astype(str).str.strip().str.lower()
    return ml_type.isin({"deep learning", "both"})


def create_deep_learning_model_summary(df: pd.DataFrame, model_info: list[dict]) -> pd.DataFrame:
    """Create count and percentage summary for deep-learning model use."""
    deep_learning_mask = identify_deep_learning_studies(df)
    denominator = int(deep_learning_mask.sum())

    if denominator == 0:
        raise ValueError("No studies were identified as 'Deep Learning' or 'Both'.")

    records = []
    for info in model_info:
        bool_values = df.loc[deep_learning_mask, info["actual_column"]].apply(cell_to_bool)
        count = int(bool_values.sum())
        percentage = round(count / denominator * 100, 1)
        records.append({
            "Model abbreviation": info["abbreviation"],
            "Model label": info["display_label"],
            "Model column": info["actual_column"],
            "Number of studies": count,
            "Percentage": percentage,
            "Percentage denominator": denominator,
        })

    summary = pd.DataFrame(records)
    if SORT_MODELS_BY_COUNT:
        summary = summary.sort_values(
            ["Number of studies", "Model abbreviation"],
            ascending=[False, True],
        ).reset_index(drop=True)
    return summary


def create_deep_learning_model_matrix(df: pd.DataFrame, model_info: list[dict]) -> pd.DataFrame:
    """Create a boolean matrix of DL model use for DL studies only."""
    dl_mask = identify_deep_learning_studies(df)
    dl_df = df.loc[dl_mask].copy()

    matrix = pd.DataFrame(index=dl_df.index)
    for info in model_info:
        matrix[info["abbreviation"]] = dl_df[info["actual_column"]].apply(cell_to_bool).astype(bool)

    return dl_df, matrix


def create_combination_usage_table(dl_df: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Create exact deep-learning model-combination count table."""
    records = []

    for idx, row in matrix.iterrows():
        used_models = tuple(matrix.columns[row.to_numpy()].tolist())

        if EXCLUDE_NO_MODEL_ROWS_FROM_COMBINATIONS and len(used_models) == 0:
            continue

        combination_label = " + ".join(used_models) if used_models else "No listed model"
        records.append({
            "Study Reference": dl_df.loc[idx, STUDY_ID_COLUMN],
            "Model combination": combination_label,
            "Models tuple": used_models,
            "Number of models": len(used_models),
        })

    long_table = pd.DataFrame(records)
    if long_table.empty:
        return pd.DataFrame(
            columns=["Combination ID", "Model combination", "Number of studies", "Number of models", "Models tuple"]
        )

    summary = (
        long_table
        .groupby(["Model combination", "Number of models"], as_index=False)
        .agg(
            **{
                "Number of studies": ("Study Reference", "nunique"),
                "Models tuple": ("Models tuple", "first"),
            }
        )
    )

    summary = summary.sort_values(
        ["Number of studies", "Number of models", "Model combination"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    summary["Combination ID"] = [f"C{i + 1}" for i in range(len(summary))]
    return summary[["Combination ID", "Model combination", "Number of studies", "Number of models", "Models tuple"]]


def plot_deep_learning_model_use(summary: pd.DataFrame) -> None:
    """Draw a horizontal bar chart of deep-learning model use."""
    plot_df = summary.copy()
    if DROP_ZERO_USE_MODELS_FROM_PLOT:
        plot_df = plot_df[plot_df["Number of studies"] > 0].copy()
    if plot_df.empty:
        raise ValueError("No nonzero deep-learning model categories available for plotting.")

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    figure_width = 8.8
    figure_height = max(4.8, 0.42 * len(plot_df) + 1.5)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    bars = ax.barh(
        plot_df["Model abbreviation"],
        plot_df["Number of studies"],
        height=0.62,
        edgecolor="black",
        linewidth=0.5,
    )

    max_count = max(int(plot_df["Number of studies"].max()), 1)
    ax.set_xlim(0, max_count * 1.30)

    for bar, count, pct in zip(bars, plot_df["Number of studies"], plot_df["Percentage"]):
        ax.text(
            bar.get_width() + max_count * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("Deep-learning model")
    ax.set_title("Deep-learning model use across studies")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUTPUT_BAR_PNG, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_BAR_PDF, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_BAR_SVG, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.show()
    print(f"Bar plot saved to: {OUTPUT_BAR_PNG.resolve()}")


def plot_deep_learning_model_upset(summary: pd.DataFrame, combination_summary: pd.DataFrame) -> None:
    """Draw a compact UpSet-style plot for deep-learning model combinations."""
    if DROP_ZERO_USE_MODELS_FROM_PLOT:
        model_plot_df = summary[summary["Number of studies"] > 0].copy()
    else:
        model_plot_df = summary.copy()

    if model_plot_df.empty:
        raise ValueError("No DL model categories available for the UpSet plot.")
    if combination_summary.empty:
        raise ValueError("No deep-learning model combinations were found.")

    combination_plot = combination_summary.head(TOP_N_COMBINATIONS).copy()
    models_for_y = model_plot_df["Model abbreviation"].tolist()
    n_models = len(models_for_y)
    n_combinations = len(combination_plot)

    figure_width = max(8.5, 0.36 * n_combinations + 3.0)
    figure_height = max(5.2, 0.28 * n_models + 2.8)

    fig = plt.figure(figsize=(figure_width, figure_height))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[2.4, 2.8], hspace=0.02)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[1, 0], sharex=ax_top)

    x_positions = np.arange(n_combinations)
    x_min = -0.45
    x_max = n_combinations - 0.55
    top_counts = combination_plot["Number of studies"].to_numpy()

    ax_top.bar(x_positions, top_counts, width=0.62, edgecolor="black", linewidth=0.5)
    max_top_count = max(int(top_counts.max()), 1)

    for x, count in zip(x_positions, top_counts):
        ax_top.text(x, count + max_top_count * 0.03, str(int(count)), ha="center", va="bottom", fontsize=8.5)

    ax_top.set_ylabel("Studies")
    ax_top.set_title("")
    ax_top.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylim(0, max_top_count * 1.28)
    ax_top.set_xlim(x_min, x_max)
    ax_top.margins(x=0)

    model_y = {model: n_models - 1 - i for i, model in enumerate(models_for_y)}

    for x in x_positions:
        ax_matrix.scatter([x] * n_models, [model_y[m] for m in models_for_y], s=20, alpha=0.18)

    for x, model_combo in zip(x_positions, combination_plot["Model combination"]):
        used_models = [m.strip() for m in str(model_combo).split("+")]
        used_models = [m for m in used_models if m in model_y]
        if not used_models:
            continue
        ys = [model_y[m] for m in used_models]
        if len(ys) > 1:
            ax_matrix.plot([x, x], [min(ys), max(ys)], linewidth=1.3)
        ax_matrix.scatter([x] * len(ys), ys, s=55)

    ax_matrix.set_yticks([model_y[m] for m in models_for_y])
    ax_matrix.set_yticklabels(models_for_y)
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels(combination_plot["Combination ID"], rotation=0)
    ax_matrix.set_xlabel("Model-combination pattern")
    ax_matrix.set_ylim(-0.8, n_models - 0.2)
    ax_matrix.set_xlim(x_min, x_max)
    ax_matrix.margins(x=0)
    ax_matrix.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["left"].set_visible(False)

    fig.savefig(OUTPUT_UPSET_PNG, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_UPSET_PDF, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_UPSET_SVG, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.show()
    print(f"UpSet-style plot saved to: {OUTPUT_UPSET_PNG.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run the deep-learning model-use analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {INPUT_FILE.resolve()}")
    df = load_data(INPUT_FILE)
    model_info = validate_and_resolve_columns(df)

    summary = create_deep_learning_model_summary(df, model_info)
    dl_df, matrix = create_deep_learning_model_matrix(df, model_info)
    combination_summary = create_combination_usage_table(dl_df, matrix)

    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    combination_summary.to_csv(OUTPUT_COMBINATION_CSV, index=False)

    plot_deep_learning_model_use(summary)
    plot_deep_learning_model_upset(summary, combination_summary)

    n_dl_studies = int(identify_deep_learning_studies(df).sum())
    print(f"Deep-learning studies identified: {n_dl_studies}")
    print(f"Summary table saved to: {OUTPUT_SUMMARY_CSV.resolve()}")
    print(f"Combination table saved to: {OUTPUT_COMBINATION_CSV.resolve()}")
    print("\nDeep-learning model-use analysis completed successfully.")


if __name__ == "__main__":
    main()
