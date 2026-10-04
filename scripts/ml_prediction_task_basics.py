# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 11:23:52 2026

@author: ak7u24
"""

"""
Plot prediction task types and machine-learning approach types for RQ1.

Input:
    inputs/ml_prediction_task_basics.xlsx

Required columns:
    Study Reference
    prediction_task_type
    ML_approach

Outputs:
    output/ml_prediction_task_and_approach_summary.xlsx
    output/prediction_task_type_bar_plot.png/.pdf/.svg
    output/ml_approach_bar_plot.png/.pdf/.svg

Notes:
    - Non-prediction/statistical-analysis rows are excluded from prediction-task statistics.
    - Statistical-study rows are excluded from ML-approach statistics.
    - If a study has multiple ML approaches separated by semicolon, the study is counted once
      under each relevant approach.
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

INPUT_FILE = PROJECT_ROOT / "input" / "ml_prediction_task_basics.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "ml_prediction_task_and_approach_summary.xlsx"

OUTPUT_PREDICTION_TASK_PNG = OUTPUT_DIR / "prediction_task_type_bar_plot.png"
OUTPUT_PREDICTION_TASK_PDF = OUTPUT_DIR / "prediction_task_type_bar_plot.pdf"
OUTPUT_PREDICTION_TASK_SVG = OUTPUT_DIR / "prediction_task_type_bar_plot.svg"

OUTPUT_ML_APPROACH_PNG = OUTPUT_DIR / "ml_approach_bar_plot.png"
OUTPUT_ML_APPROACH_PDF = OUTPUT_DIR / "ml_approach_bar_plot.pdf"
OUTPUT_ML_APPROACH_SVG = OUTPUT_DIR / "ml_approach_bar_plot.svg"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

STUDY_ID_COLUMN = "Study Reference"
PREDICTION_TASK_COLUMN = "prediction_task_type"
ML_APPROACH_COLUMN = "ML_approach"


# --------------------------------------------------
# Analysis options
# --------------------------------------------------

# Percent denominator options:
#   "included_after_exclusion" = percentage among studies retained for that analysis
#   "all_studies" = percentage among all rows/studies in the Excel file
PREDICTION_PERCENT_DENOMINATOR = "included_after_exclusion"
ML_APPROACH_PERCENT_DENOMINATOR = "included_after_exclusion"

# Set a category to False to remove it from the plot and output table.
INCLUDE_PREDICTION_TASK = {
    "Prediction/classification approaches": True,
    "Prognostic/forecasting/long-term outcome prediction": True,
    "Symptom monitoring/progression prediction": True,
    "Treatment-response/treatment-effect prediction": True,
}

INCLUDE_ML_APPROACH = {
    "Supervised": True,
    "Unsupervised": True,
}

PREDICTION_TASK_ORDER = [
    "Prediction/classification approaches",
    "Symptom monitoring/progression prediction",
    "Prognostic/forecasting/long-term outcome prediction",
    "Treatment-response/treatment-effect prediction",
]

ML_APPROACH_ORDER = [
    "Supervised",
    "Unsupervised",
]

SORT_BARS_BY_COUNT = False
SHOW_PERCENT_LABELS = True
OUTPUT_DPI = 600


# --------------------------------------------------
# Plot style
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "Arial",
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
# Text cleaning helpers
# --------------------------------------------------

def normalise_text(value) -> str:
    """Normalise text for robust category matching."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("’", "'").replace("‘", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_multi_value_cell(value) -> list[str]:
    """Split semicolon-separated multi-response cells."""
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    tokens = [token.strip() for token in text.split(";")]
    return [token for token in tokens if token]


# --------------------------------------------------
# Category mapping
# --------------------------------------------------

def map_prediction_task(token: str) -> str | None:
    """Map raw prediction-task labels to the four standard RQ1 categories."""
    key = normalise_text(token)

    if not key:
        return None

    # Exclude non-prediction/statistical-analysis rows.
    if (
        "not a prediction model" in key
        or "statistical analysis" in key
        or "statistical study" in key
        or key in {"not applicable", "na", "n a", "unclear", "not reported"}
    ):
        return None

    if "symptom monitoring" in key or "progression" in key:
        return "Symptom monitoring/progression prediction"

    if "prognostic" in key or "forecast" in key or "long term outcome" in key:
        return "Prognostic/forecasting/long-term outcome prediction"

    if "treatment response" in key or "treatment effect" in key:
        return "Treatment-response/treatment-effect prediction"

    # Includes exact labels such as:
    # "Prediction/classification approaches"
    # "Diagnostic / Prediction/Classification"
    if (
        "prediction classification" in key
        or "classification" in key
        or "diagnostic" in key
        or "diagnosis" in key
    ):
        return "Prediction/classification approaches"

    return None


def map_ml_approach(token: str) -> str | None:
    """Map raw ML-approach labels to Supervised/Unsupervised."""
    key = normalise_text(token)

    if not key:
        return None

    # Exclude statistical-study labels.
    if "statistical" in key or key in {"not applicable", "na", "n a", "unclear", "not reported"}:
        return None

    # Important: check 'unsupervised' before 'supervised', because
    # the word 'unsupervised' contains 'supervised'.
    if re.search(r"\bunsupervised\b", key):
        return "Unsupervised"

    if re.search(r"\bsupervised\b", key):
        return "Supervised"

    return None


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """Load Excel file and validate required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    # Remove accidental leading/trailing spaces in Excel headers.
    df.columns = df.columns.str.strip()

    required_columns = [STUDY_ID_COLUMN, PREDICTION_TASK_COLUMN, ML_APPROACH_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns after cleaning: {list(df.columns)}"
        )

    df = df.copy()
    df[STUDY_ID_COLUMN] = df[STUDY_ID_COLUMN].fillna("missing_study_id")

    return df


# --------------------------------------------------
# Summary-table functions
# --------------------------------------------------

def create_prediction_task_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create counts and percentages for prediction task types."""
    records = []

    for _, row in df.iterrows():
        study_id = row[STUDY_ID_COLUMN]
        tokens = split_multi_value_cell(row[PREDICTION_TASK_COLUMN])

        for token in tokens:
            category = map_prediction_task(token)
            if category is None:
                continue
            if not INCLUDE_PREDICTION_TASK.get(category, False):
                continue

            records.append({
                "Study Reference": study_id,
                "Raw prediction_task_type": token,
                "Prediction task type": category,
            })

    long_table = pd.DataFrame(records)

    selected_categories = [
        category for category in PREDICTION_TASK_ORDER
        if INCLUDE_PREDICTION_TASK.get(category, False)
    ]

    if long_table.empty:
        summary = pd.DataFrame({
            "Prediction task type": selected_categories,
            "Number of studies": 0,
            "Percentage": 0.0,
        })
        return summary, long_table

    long_table = long_table.drop_duplicates(
        subset=["Study Reference", "Prediction task type"]
    )

    if PREDICTION_PERCENT_DENOMINATOR == "all_studies":
        denominator = df[STUDY_ID_COLUMN].nunique()
    elif PREDICTION_PERCENT_DENOMINATOR == "included_after_exclusion":
        denominator = long_table["Study Reference"].nunique()
    else:
        raise ValueError(
            "PREDICTION_PERCENT_DENOMINATOR must be 'all_studies' "
            "or 'included_after_exclusion'."
        )

    summary = (
        long_table
        .groupby("Prediction task type", as_index=False)
        .agg(**{"Number of studies": ("Study Reference", "nunique")})
    )

    summary = (
        pd.DataFrame({"Prediction task type": selected_categories})
        .merge(summary, on="Prediction task type", how="left")
        .fillna({"Number of studies": 0})
    )

    summary["Number of studies"] = summary["Number of studies"].astype(int)
    summary["Percentage"] = (summary["Number of studies"] / denominator * 100).round(1)
    summary["Percentage denominator"] = denominator

    if SORT_BARS_BY_COUNT:
        summary = summary.sort_values(
            ["Number of studies", "Prediction task type"],
            ascending=[False, True],
        )

    return summary, long_table


def create_ml_approach_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create counts and percentages for supervised/unsupervised ML approaches."""
    records = []

    for _, row in df.iterrows():
        study_id = row[STUDY_ID_COLUMN]
        tokens = split_multi_value_cell(row[ML_APPROACH_COLUMN])

        for token in tokens:
            category = map_ml_approach(token)
            if category is None:
                continue
            if not INCLUDE_ML_APPROACH.get(category, False):
                continue

            records.append({
                "Study Reference": study_id,
                "Raw ML_approach": token,
                "ML approach": category,
            })

    long_table = pd.DataFrame(records)

    selected_categories = [
        category for category in ML_APPROACH_ORDER
        if INCLUDE_ML_APPROACH.get(category, False)
    ]

    if long_table.empty:
        summary = pd.DataFrame({
            "ML approach": selected_categories,
            "Number of studies": 0,
            "Percentage": 0.0,
        })
        return summary, long_table

    long_table = long_table.drop_duplicates(
        subset=["Study Reference", "ML approach"]
    )

    if ML_APPROACH_PERCENT_DENOMINATOR == "all_studies":
        denominator = df[STUDY_ID_COLUMN].nunique()
    elif ML_APPROACH_PERCENT_DENOMINATOR == "included_after_exclusion":
        denominator = long_table["Study Reference"].nunique()
    else:
        raise ValueError(
            "ML_APPROACH_PERCENT_DENOMINATOR must be 'all_studies' "
            "or 'included_after_exclusion'."
        )

    summary = (
        long_table
        .groupby("ML approach", as_index=False)
        .agg(**{"Number of studies": ("Study Reference", "nunique")})
    )

    summary = (
        pd.DataFrame({"ML approach": selected_categories})
        .merge(summary, on="ML approach", how="left")
        .fillna({"Number of studies": 0})
    )

    summary["Number of studies"] = summary["Number of studies"].astype(int)
    summary["Percentage"] = (summary["Number of studies"] / denominator * 100).round(1)
    summary["Percentage denominator"] = denominator

    if SORT_BARS_BY_COUNT:
        summary = summary.sort_values(
            ["Number of studies", "ML approach"],
            ascending=[False, True],
        )

    return summary, long_table


# --------------------------------------------------
# Plot functions
# --------------------------------------------------

def plot_horizontal_bar(
    table: pd.DataFrame,
    label_column: str,
    count_column: str,
    percentage_column: str,
    title: str,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
) -> None:
    """Create a horizontal count-and-percentage bar chart."""
    plot_df = table.copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    figure_height = max(3.8, 0.55 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, figure_height))

    bars = ax.barh(
        plot_df[label_column],
        plot_df[count_column],
        height=0.62,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("")
    ax.set_title(title)

    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_count = max(int(plot_df[count_column].max()), 1)
    ax.set_xlim(0, max_count * 1.28)

    if SHOW_PERCENT_LABELS:
        for bar, count, pct in zip(
            bars,
            plot_df[count_column],
            plot_df[percentage_column],
        ):
            ax.text(
                bar.get_width() + max_count * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count)} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
            )

    fig.tight_layout()

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()
    print(f"Plot saved to: {output_png.resolve()}")


# --------------------------------------------------
# Save summary workbook
# --------------------------------------------------

def save_summary_workbook(
    prediction_task_summary: pd.DataFrame,
    prediction_task_long: pd.DataFrame,
    ml_approach_summary: pd.DataFrame,
    ml_approach_long: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save all summary tables to an Excel workbook."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        prediction_task_summary.to_excel(
            writer,
            sheet_name="Prediction task summary",
            index=False,
        )
        prediction_task_long.to_excel(
            writer,
            sheet_name="Prediction task long",
            index=False,
        )
        ml_approach_summary.to_excel(
            writer,
            sheet_name="ML approach summary",
            index=False,
        )
        ml_approach_long.to_excel(
            writer,
            sheet_name="ML approach long",
            index=False,
        )

    print(f"Summary workbook saved to: {output_path.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run analysis and generate plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {INPUT_FILE.resolve()}")
    df = load_data(INPUT_FILE)

    prediction_task_summary, prediction_task_long = create_prediction_task_table(df)
    ml_approach_summary, ml_approach_long = create_ml_approach_table(df)

    save_summary_workbook(
        prediction_task_summary=prediction_task_summary,
        prediction_task_long=prediction_task_long,
        ml_approach_summary=ml_approach_summary,
        ml_approach_long=ml_approach_long,
        output_path=OUTPUT_TABLE,
    )

    plot_horizontal_bar(
        table=prediction_task_summary,
        label_column="Prediction task type",
        count_column="Number of studies",
        percentage_column="Percentage",
        title="Prediction tasks addressed by conventional ML models",
        output_png=OUTPUT_PREDICTION_TASK_PNG,
        output_pdf=OUTPUT_PREDICTION_TASK_PDF,
        output_svg=OUTPUT_PREDICTION_TASK_SVG,
    )

    plot_horizontal_bar(
        table=ml_approach_summary,
        label_column="ML approach",
        count_column="Number of studies",
        percentage_column="Percentage",
        title="Machine-learning approaches used across studies",
        output_png=OUTPUT_ML_APPROACH_PNG,
        output_pdf=OUTPUT_ML_APPROACH_PDF,
        output_svg=OUTPUT_ML_APPROACH_SVG,
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()