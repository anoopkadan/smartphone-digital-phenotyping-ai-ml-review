# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:37:17 2026

@author: ak7u24
"""
"""
Plot interpretability / explainability use across included studies.

Input:
    input/ml_deep_learning_models.xlsx

Required columns:
    Study Reference
    ml_model_type
    interpretability

Coding rules:
    1. A study is counted as using interpretability if the 'interpretability'
       column contains anything other than NA, N/A, blank, NaN, None,
       not reported, or unclear.
    2. A study is counted as using SHAP if the 'interpretability' column
       contains the text 'SHAP' case-insensitively.
    3. 'Classical ML' studies are rows where ml_model_type is Classical ML
       or Both.
    4. 'Deep learning' studies are rows where ml_model_type is Deep Learning
       or Both.
       Therefore, studies coded as Both contribute to both denominators.

Outputs:
    output/interpretability_summary_plot.png/.pdf/.svg
    output/interpretability_overall_plot.png/.pdf/.svg
    output/interpretability_by_model_type_plot.png/.pdf/.svg
    output/interpretability_overall_summary.csv
    output/interpretability_by_model_type_summary.csv
"""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "ml_deep_learning_models.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_COMBINED_PNG = OUTPUT_DIR / "interpretability_summary_plot.png"
OUTPUT_COMBINED_PDF = OUTPUT_DIR / "interpretability_summary_plot.pdf"
OUTPUT_COMBINED_SVG = OUTPUT_DIR / "interpretability_summary_plot.svg"

OUTPUT_OVERALL_PNG = OUTPUT_DIR / "interpretability_overall_plot.png"
OUTPUT_OVERALL_PDF = OUTPUT_DIR / "interpretability_overall_plot.pdf"
OUTPUT_OVERALL_SVG = OUTPUT_DIR / "interpretability_overall_plot.svg"

OUTPUT_BY_TYPE_PNG = OUTPUT_DIR / "interpretability_by_model_type_plot.png"
OUTPUT_BY_TYPE_PDF = OUTPUT_DIR / "interpretability_by_model_type_plot.pdf"
OUTPUT_BY_TYPE_SVG = OUTPUT_DIR / "interpretability_by_model_type_plot.svg"

OUTPUT_OVERALL_CSV = OUTPUT_DIR / "interpretability_overall_summary.csv"
OUTPUT_BY_TYPE_CSV = OUTPUT_DIR / "interpretability_by_model_type_summary.csv"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

STUDY_ID_COLUMN = "Study Reference"
ML_MODEL_TYPE_COLUMN = "ml_model_type"
INTERPRETABILITY_COLUMN = "interpretability"

OUTPUT_DPI = 600

# Whether to produce the individual one-panel figures in addition to the
# combined two-panel figure.
SAVE_SEPARATE_PLOTS = True
SAVE_COMBINED_PLOT = True


# --------------------------------------------------
# Plot style
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
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


def resolve_column(df: pd.DataFrame, required_column: str) -> str:
    """Resolve a required column even if there are small spacing/case differences."""
    if required_column in df.columns:
        return required_column

    lookup = {normalise_text(col): col for col in df.columns}
    key = normalise_text(required_column)

    if key not in lookup:
        raise KeyError(
            f"Required column not found: {required_column}\n"
            f"Available columns: {list(df.columns)}"
        )

    return lookup[key]


def is_noninformative(value) -> bool:
    """Return True if a value means no interpretability information was reported."""
    if pd.isna(value):
        return True

    text = str(value).strip()
    key = normalise_text(text)

    noninformative_values = {
        "", "na", "n_a", "nan", "none", "no", "false",
        "not_reported", "not_mentioned", "unclear", "not_clear",
        "not_applicable", "not_available",
    }

    return key in noninformative_values


def has_interpretability(value) -> bool:
    """Interpretability is present if the cell has anything other than NA-like content."""
    return not is_noninformative(value)


def has_shap(value) -> bool:
    """Return True if the interpretability text includes SHAP."""
    if pd.isna(value):
        return False

    return bool(re.search(r"\bSHAP\b", str(value), flags=re.IGNORECASE))


def load_and_prepare_data(file_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load Excel data, resolve columns, and remove empty trailing rows."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    resolved = {
        "study_id": resolve_column(df, STUDY_ID_COLUMN),
        "ml_model_type": resolve_column(df, ML_MODEL_TYPE_COLUMN),
        "interpretability": resolve_column(df, INTERPRETABILITY_COLUMN),
    }

    # Keep only actual study rows. This removes blank summary rows sometimes
    # left at the bottom of manually edited Excel sheets.
    study_col = resolved["study_id"]
    df = df[df[study_col].notna()].copy()

    return df, resolved


def create_analysis_tables(df: pd.DataFrame, columns: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create overall and modelling-approach interpretability summary tables."""
    interpret_col = columns["interpretability"]
    ml_type_col = columns["ml_model_type"]

    included_denominator = len(df)

    any_interpretability = df[interpret_col].apply(has_interpretability)
    shap = df[interpret_col].apply(has_shap)

    n_interpretability = int(any_interpretability.sum())
    n_shap = int((any_interpretability & shap).sum())

    overall_records = [
        {
            "Metric": "Any interpretability",
            "Number of studies": n_interpretability,
            "Percentage denominator": included_denominator,
            "Percentage": round(n_interpretability / included_denominator * 100, 1) if included_denominator else 0.0,
            "Label for plot": f"{n_interpretability}/{included_denominator} ({round(n_interpretability / included_denominator * 100, 1):.1f}%)" if included_denominator else "0/0 (0.0%)",
        },
        {
            "Metric": "SHAP among interpretable studies",
            "Number of studies": n_shap,
            "Percentage denominator": n_interpretability,
            "Percentage": round(n_shap / n_interpretability * 100, 1) if n_interpretability else 0.0,
            "Label for plot": f"{n_shap}/{n_interpretability} ({round(n_shap / n_interpretability * 100, 1):.1f}%)" if n_interpretability else "0/0 (0.0%)",
        },
        {
            "Metric": "SHAP among all included studies",
            "Number of studies": n_shap,
            "Percentage denominator": included_denominator,
            "Percentage": round(n_shap / included_denominator * 100, 1) if included_denominator else 0.0,
            "Label for plot": f"{n_shap}/{included_denominator} ({round(n_shap / included_denominator * 100, 1):.1f}%)" if included_denominator else "0/0 (0.0%)",
        },
    ]

    overall_table = pd.DataFrame(overall_records)

    ml_type = df[ml_type_col].astype(str).str.strip().str.lower()

    group_masks = {
        "Classical ML": ml_type.isin({"classical ml", "machine learning", "both"})
        | ml_type.str.contains("classical|machine learning", na=False),
        "Deep learning": ml_type.isin({"deep learning", "both"})
        | ml_type.str.contains("deep learning", na=False),
    }

    by_type_records = []

    for group_name, group_mask in group_masks.items():
        denominator = int(group_mask.sum())
        n_any = int((group_mask & any_interpretability).sum())
        n_shap_group = int((group_mask & shap).sum())
        percentage_any = round(n_any / denominator * 100, 1) if denominator else 0.0
        percentage_shap = round(n_shap_group / denominator * 100, 1) if denominator else 0.0

        by_type_records.append({
            "Modelling approach": group_name,
            "Number of studies in group": denominator,
            "Number using any interpretability": n_any,
            "Percentage using any interpretability": percentage_any,
            "Number using SHAP": n_shap_group,
            "Percentage using SHAP": percentage_shap,
            "Label for plot": f"{n_any}/{denominator} ({percentage_any:.1f}%)",
        })

    by_type_table = pd.DataFrame(by_type_records)

    return overall_table, by_type_table


# --------------------------------------------------
# Plotting functions
# --------------------------------------------------

def add_value_labels(ax, bars, labels, max_count: int) -> None:
    """Add count/percentage labels to horizontal bars."""
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() + max_count * 0.025,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=8.5,
        )


def style_horizontal_bar_axis(ax, xlabel: str, ylabel: str = "") -> None:
    """Apply journal-style formatting to a horizontal bar chart."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overall_interpretability(overall_table: pd.DataFrame) -> None:
    """Plot overall interpretability and SHAP use."""
    plot_df = overall_table[overall_table["Metric"].isin([
        "Any interpretability",
        "SHAP among interpretable studies",
    ])].copy()

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    bars = ax.barh(
        plot_df["Metric"],
        plot_df["Number of studies"],
        height=0.58,
        edgecolor="black",
        linewidth=0.5,
    )

    max_count = max(int(plot_df["Number of studies"].max()), 1)
    ax.set_xlim(0, max_count * 1.45)
    add_value_labels(ax, bars, plot_df["Label for plot"], max_count)
    style_horizontal_bar_axis(ax, xlabel="Number of studies")
    ax.set_title("Interpretability and SHAP use")

    fig.savefig(OUTPUT_OVERALL_PNG, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_OVERALL_PDF, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_OVERALL_SVG, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.show()


def plot_interpretability_by_model_type(by_type_table: pd.DataFrame) -> None:
    """Plot interpretability use by classical ML and deep-learning study groups."""
    plot_df = by_type_table.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.8, 3.4))

    bars = ax.barh(
        plot_df["Modelling approach"],
        plot_df["Number using any interpretability"],
        height=0.58,
        edgecolor="black",
        linewidth=0.5,
    )

    max_count = max(int(plot_df["Number using any interpretability"].max()), 1)
    ax.set_xlim(0, max_count * 1.45)
    add_value_labels(ax, bars, plot_df["Label for plot"], max_count)
    style_horizontal_bar_axis(ax, xlabel="Number of studies using interpretability")
    ax.set_title("Interpretability use by modelling approach")

    fig.savefig(OUTPUT_BY_TYPE_PNG, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_BY_TYPE_PDF, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_BY_TYPE_SVG, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.show()


def plot_combined_summary(overall_table: pd.DataFrame, by_type_table: pd.DataFrame) -> None:
    """Create a two-panel journal-style summary figure."""
    overall_plot = overall_table[overall_table["Metric"].isin([
        "Any interpretability",
        "SHAP among interpretable studies",
    ])].copy().iloc[::-1].reset_index(drop=True)

    by_type_plot = by_type_table.iloc[::-1].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))

    # Panel A
    ax = axes[0]
    bars = ax.barh(
        overall_plot["Metric"],
        overall_plot["Number of studies"],
        height=0.58,
        edgecolor="black",
        linewidth=0.5,
    )
    max_count = max(int(overall_plot["Number of studies"].max()), 1)
    ax.set_xlim(0, max_count * 1.45)
    add_value_labels(ax, bars, overall_plot["Label for plot"], max_count)
    style_horizontal_bar_axis(ax, xlabel="Number of studies")
    ax.set_title("A. Overall interpretability")

    # Panel B
    ax = axes[1]
    bars = ax.barh(
        by_type_plot["Modelling approach"],
        by_type_plot["Number using any interpretability"],
        height=0.58,
        edgecolor="black",
        linewidth=0.5,
    )
    max_count = max(int(by_type_plot["Number using any interpretability"].max()), 1)
    ax.set_xlim(0, max_count * 1.45)
    add_value_labels(ax, bars, by_type_plot["Label for plot"], max_count)
    style_horizontal_bar_axis(ax, xlabel="Number of studies using interpretability")
    ax.set_title("B. By modelling approach")

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUTPUT_COMBINED_PNG, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_COMBINED_PDF, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(OUTPUT_COMBINED_SVG, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.show()


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run interpretability analysis and generate plots/tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {INPUT_FILE.resolve()}")

    df, columns = load_and_prepare_data(INPUT_FILE)
    overall_table, by_type_table = create_analysis_tables(df, columns)

    overall_table.to_csv(OUTPUT_OVERALL_CSV, index=False)
    by_type_table.to_csv(OUTPUT_BY_TYPE_CSV, index=False)

    if SAVE_COMBINED_PLOT:
        plot_combined_summary(overall_table, by_type_table)

    if SAVE_SEPARATE_PLOTS:
        plot_overall_interpretability(overall_table)
        plot_interpretability_by_model_type(by_type_table)

    print(f"Included studies analysed: {len(df)}")
    print(f"Overall summary saved to: {OUTPUT_OVERALL_CSV.resolve()}")
    print(f"By-model-type summary saved to: {OUTPUT_BY_TYPE_CSV.resolve()}")
    print("\nInterpretability analysis completed successfully.")


if __name__ == "__main__":
    main()
