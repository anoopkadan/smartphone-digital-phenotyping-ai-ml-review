# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:22:28 2026

@author: ak7u24
"""
"""
Plot non-smartphone device use by modelling approach.

Input:
    multimodal.xlsx

Required columns:
    Study Reference
    non_smartphone_device
    ml_model_type

Definitions:
    A study is treated as using a non-smartphone device if the
    'non_smartphone_device' column is anything other than 'Not used',
    blank, NA, or similar missing/unclear labels.

    Statistical studies:
        ml_model_type == 'Statistical Study'

    Classical ML studies:
        ml_model_type == 'Classical ML' or 'Both'

    Deep-learning studies:
        ml_model_type == 'Deep Learning' or 'Both'

    Studies labelled 'Both' contribute to both Classical ML and Deep-learning
    denominators.

Outputs:
    rq3_multimodal_output/non_smartphone_device_by_model_type_plot.png
    rq3_multimodal_output/non_smartphone_device_by_model_type_plot.pdf
    rq3_multimodal_output/non_smartphone_device_by_model_type_plot.svg
    rq3_multimodal_output/non_smartphone_device_by_model_type_summary.csv
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "multimodal.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_PNG = OUTPUT_DIR / "non_smartphone_device_by_model_type_plot.png"
OUTPUT_PDF = OUTPUT_DIR / "non_smartphone_device_by_model_type_plot.pdf"
OUTPUT_SVG = OUTPUT_DIR / "non_smartphone_device_by_model_type_plot.svg"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "non_smartphone_device_by_model_type_summary.csv"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

STUDY_ID_COLUMN = "Study Reference"
NON_SMARTPHONE_DEVICE_COLUMN = "non_smartphone_device"
ML_MODEL_TYPE_COLUMN = "ml_model_type"

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
    """Normalise text for robust matching."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("‘", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalise_column_name(value) -> str:
    """Normalise a column name to handle minor spelling/spacing differences."""
    text = normalise_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def resolve_required_column(df: pd.DataFrame, required_column: str) -> str:
    """Resolve a required column name against actual columns in the Excel file."""
    if required_column in df.columns:
        return required_column

    lookup = {normalise_column_name(col): col for col in df.columns}
    key = normalise_column_name(required_column)

    if key in lookup:
        return lookup[key]

    raise KeyError(
        f"Required column not found: {required_column}\n"
        f"Available columns: {list(df.columns)}"
    )


def is_non_smartphone_device_used(value) -> bool:
    """
    Identify whether a study used a non-smartphone device.

    The main exclusion label is 'Not used'. Blank, NA, N/A, None,
    and similar missing/unclear labels are also treated as not used.
    """
    key = normalise_text(value)

    not_used_values = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "not used",
        "not reported",
        "unclear",
        "no",
        "false",
        "0",
    }

    return key not in not_used_values


def load_data(file_path: Path) -> pd.DataFrame:
    """Load Excel file and clean column names."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    if STUDY_ID_COLUMN not in df.columns:
        df[STUDY_ID_COLUMN] = np.arange(1, len(df) + 1)

    return df


def create_model_type_masks(df: pd.DataFrame, model_type_column: str) -> dict[str, pd.Series]:
    """
    Create modelling-approach masks.

    Studies labelled 'Both' are counted under both Classical ML and Deep learning.
    """
    model_type = df[model_type_column].apply(normalise_text)

    masks = {
        "Statistical study": model_type.eq("statistical study"),
        "Classical ML": model_type.isin({"classical ml", "both"}),
        "Deep learning": model_type.isin({"deep learning", "both"}),
    }

    return masks


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create count and percentage table by modelling approach."""
    device_col = resolve_required_column(df, NON_SMARTPHONE_DEVICE_COLUMN)
    model_type_col = resolve_required_column(df, ML_MODEL_TYPE_COLUMN)

    used_device = df[device_col].apply(is_non_smartphone_device_used)
    model_type_masks = create_model_type_masks(df, model_type_col)

    records = []

    for approach, mask in model_type_masks.items():
        total_studies = int(mask.sum())
        used_count = int((mask & used_device).sum())
        not_used_count = int(total_studies - used_count)
        percentage = round(used_count / total_studies * 100, 1) if total_studies else 0.0

        records.append({
            "Modelling approach": approach,
            "Studies using non-smartphone device": used_count,
            "Studies not using non-smartphone device": not_used_count,
            "Total studies": total_studies,
            "Percentage using non-smartphone device": percentage,
        })

    summary = pd.DataFrame(records)

    return summary


def plot_summary(summary: pd.DataFrame) -> None:
    """Plot percentage of studies using non-smartphone devices by modelling approach."""
    plot_df = summary.copy()

    # Reverse so the first category appears at the top.
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    bars = ax.barh(
        plot_df["Modelling approach"],
        plot_df["Percentage using non-smartphone device"],
        height=0.58,
        edgecolor="black",
        linewidth=0.5,
    )

    max_pct = max(float(plot_df["Percentage using non-smartphone device"].max()), 1.0)
    ax.set_xlim(0, min(100, max_pct * 1.35))

    for bar, used, total, pct in zip(
        bars,
        plot_df["Studies using non-smartphone device"],
        plot_df["Total studies"],
        plot_df["Percentage using non-smartphone device"],
    ):
        ax.text(
            bar.get_width() + max_pct * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{int(used)}/{int(total)} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Studies using a non-smartphone device (%)")
    ax.set_ylabel("Modelling approach")
    ax.set_title("Non-smartphone device use by modelling approach")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(
        OUTPUT_PNG,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    fig.savefig(
        OUTPUT_SVG,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )

    plt.show()

    print(f"Plot saved to: {OUTPUT_PNG.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run non-smartphone device analysis by modelling approach."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {INPUT_FILE.resolve()}")

    df = load_data(INPUT_FILE)
    summary = create_summary_table(df)

    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    plot_summary(summary)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSummary table saved to: {OUTPUT_SUMMARY_CSV.resolve()}")
    print("Non-smartphone device analysis completed successfully.")


if __name__ == "__main__":
    main()
