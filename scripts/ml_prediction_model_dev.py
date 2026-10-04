# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:51:40 2026

@author: ak7u24
"""
"""
Grouped horizontal bar chart for conventional ML model-development characteristics.

This script creates one figure with four grouped sections:
    1. Preprocessing approaches
    2. Feature type
    3. Data-splitting approaches
    4. Model-validation approaches

Input:
    inputs/ml_prediction_model_dev.xlsx

Required columns:
    Study Reference
    pre_processing
    featute_type
    data_splitting
    model_validation

Outputs:
    output/model_development_characteristics_summary.xlsx
    output/model_development_characteristics_grouped_bar_plot.png
    output/model_development_characteristics_grouped_bar_plot.pdf
    output/model_development_characteristics_grouped_bar_plot.svg

Notes:
    - Multiple values separated by semicolons are counted separately.
    - NA / Na / N/A / missing values are excluded.
    - Each study is counted once per category within each domain.
    - "Summary and Longitudinal" is counted under both Summary features and
      Longitudinal features.
"""

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "ml_prediction_model_dev.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "model_development_characteristics_summary.xlsx"

OUTPUT_PLOT_PNG = OUTPUT_DIR / "model_development_characteristics_grouped_bar_plot.png"
OUTPUT_PLOT_PDF = OUTPUT_DIR / "model_development_characteristics_grouped_bar_plot.pdf"
OUTPUT_PLOT_SVG = OUTPUT_DIR / "model_development_characteristics_grouped_bar_plot.svg"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

STUDY_ID_COLUMN = "Study Reference"
PREPROCESSING_COLUMN = "pre_processing"

# This column is intentionally written as "featute_type" because that is the
# column name in the Excel file provided.
FEATURE_TYPE_COLUMN = "featute_type"

DATA_SPLITTING_COLUMN = "data_splitting"
MODEL_VALIDATION_COLUMN = "model_validation"


# --------------------------------------------------
# Analysis options
# --------------------------------------------------

# Percentage denominator options:
#   "valid_domain_studies" = denominator is the number of unique studies with a
#                            non-NA value for that domain.
#   "all_studies" = denominator is all unique studies in the Excel file.
PERCENT_DENOMINATOR_MODE = "valid_domain_studies"

OUTPUT_DPI = 300
SHOW_PERCENT_LABELS = True
SHOW_LEGEND = True

# Set any category to False to remove it from the plot and summary table.
INCLUDE_PREPROCESSING = {
    "Feature scaling/standardization": True,
    "Data transformation": True,
    "Data cleaning": True,
    "Data windowing": True,
    "Clustering": True,
    "Missing data imputation": True,
    "Oversampling (SMOTE)": True,
}

INCLUDE_FEATURE_TYPE = {
    "Summary features": True,
    "Longitudinal features": True,
}

INCLUDE_DATA_SPLITTING = {
    "K-Fold": True,
    "K-Fold+Test": True,
    "LOOCV": True,
    "Train-Test/Val": True,
    "Train-Test-Val": True,
}

INCLUDE_MODEL_VALIDATION = {
    "Internal validation": True,
    "Separate holdout validation": True,
    "External validation": True,
}

PREPROCESSING_ORDER = [
    "Feature scaling/standardization",
    "Data transformation",
    "Data cleaning",
    "Data windowing",
    "Clustering",
    "Missing data imputation",
    "Oversampling (SMOTE)",
]

FEATURE_TYPE_ORDER = [
    "Summary features",
    "Longitudinal features",
]

DATA_SPLITTING_ORDER = [
    "K-Fold",
    "K-Fold+Test",
    "LOOCV",
    "Train-Test/Val",
    "Train-Test-Val",
]

MODEL_VALIDATION_ORDER = [
    "Internal validation",
    "Separate holdout validation",
    "External validation",
]

DOMAIN_ORDER = [
    "Preprocessing approaches",
    "Feature type",
    "Data splitting",
    "Model validation",
]

# Muted, journal-style colours.
DOMAIN_COLORS = {
    "Preprocessing approaches": "#4E79A7",  # muted blue
    "Feature type": "#59A14F",              # muted green
    "Data splitting": "#F28E2B",            # muted orange
    "Model validation": "#B07AA1",          # muted purple
}


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
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# --------------------------------------------------
# Text helpers
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
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_missing_like(value) -> bool:
    """Return True for NA-like or uninformative values."""
    key = normalise_text(value)

    return key in {
        "",
        "na",
        "n a",
        "nan",
        "none",
        "not reported",
        "not mentioned",
        "unclear",
        "not clear",
        "not applicable",
        "not available",
    }


def split_semicolon_cell(value) -> list[str]:
    """Split semicolon-separated multi-response cells."""
    if is_missing_like(value):
        return []

    text = str(value).strip()
    tokens = [token.strip() for token in text.split(";")]
    tokens = [token for token in tokens if not is_missing_like(token)]

    return tokens


def study_has_valid_domain_value(value) -> bool:
    """Whether a study has a coded, non-NA value for a given domain."""
    return len(split_semicolon_cell(value)) > 0


# --------------------------------------------------
# Category mappers
# --------------------------------------------------

def map_preprocessing_token(token: str) -> str | None:
    """Map raw preprocessing text to a standard category."""
    key = normalise_text(token)

    if is_missing_like(key):
        return None

    if "smote" in key or "oversampling" in key or "over sampling" in key:
        return "Oversampling (SMOTE)"

    if "missing" in key or "imputation" in key or "impute" in key:
        return "Missing data imputation"

    if (
        "scaling" in key
        or "standardization" in key
        or "standardisation" in key
        or "normalization" in key
        or "normalisation" in key
    ):
        return "Feature scaling/standardization"

    if "transformation" in key or "transform" in key:
        return "Data transformation"

    if "cleaning" in key or "clean" in key:
        return "Data cleaning"

    if "windowing" in key or "window" in key or "segmentation" in key or "segment" in key:
        return "Data windowing"

    if "clustering" in key or "cluster" in key:
        return "Clustering"

    return None


def map_feature_type_token(token: str) -> list[str]:
    """
    Map raw feature-type text to one or more standard feature categories.

    "Summary and Longitudinal" is counted under both categories.
    """
    key = normalise_text(token)

    if is_missing_like(key):
        return []

    categories = []

    if "summary" in key:
        categories.append("Summary features")

    if "longitudinal" in key or "time series" in key or "temporal" in key:
        categories.append("Longitudinal features")

    deduplicated = []
    for category in categories:
        if category not in deduplicated:
            deduplicated.append(category)

    return deduplicated


def map_data_splitting_token(token: str) -> str | None:
    """Map raw data-splitting text to a standard category."""
    raw = str(token).strip()
    key = normalise_text(raw)

    if is_missing_like(key):
        return None

    # Order matters: check K-Fold+Test before K-Fold.
    if "k fold test" in key or "kfold test" in key:
        return "K-Fold+Test"

    if key in {"k fold", "kfold", "cross validation", "cross validation cv", "cv"}:
        return "K-Fold"

    if "loocv" in key or "leave one out" in key:
        return "LOOCV"

    # Keep these two categories separate, as requested.
    if "train test val" in key or "train test validation" in key:
        if "/" in raw:
            return "Train-Test/Val"
        return "Train-Test-Val"

    if "train test" in key and "val" in key:
        if "/" in raw:
            return "Train-Test/Val"
        return "Train-Test-Val"

    return None


def map_model_validation_token(token: str) -> str | None:
    """Map raw model-validation text to a standard validation category."""
    key = normalise_text(token)

    if is_missing_like(key):
        return None

    # Order matters: external first, because external validation is distinct.
    if "external" in key or "independent cohort" in key or "independent dataset" in key:
        return "External validation"

    if (
        "separate holdout" in key
        or "holdout" in key
        or "held out" in key
        or "heldout" in key
        or "test set" in key
        or "separate test" in key
    ):
        return "Separate holdout validation"

    if (
        "internal" in key
        or "cross validation" in key
        or "k fold" in key
        or "kfold" in key
        or "loocv" in key
        or "leave one out" in key
        or "train test" in key
        or "validation set" in key
    ):
        return "Internal validation"

    return None


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """Load Excel file and validate required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    # Clean accidental spaces in column names.
    df.columns = df.columns.str.strip()

    required_columns = [
        STUDY_ID_COLUMN,
        PREPROCESSING_COLUMN,
        FEATURE_TYPE_COLUMN,
        DATA_SPLITTING_COLUMN,
        MODEL_VALIDATION_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns after cleaning: {list(df.columns)}\n"
            f"Add a column named '{MODEL_VALIDATION_COLUMN}' with values such as "
            "'internal validation', 'separate holdout validation', or "
            "'external validation'."
        )

    df = df.copy()
    df[STUDY_ID_COLUMN] = df[STUDY_ID_COLUMN].fillna("missing_study_id")

    return df


# --------------------------------------------------
# Counting functions
# --------------------------------------------------

def get_denominator(df: pd.DataFrame, domain_column: str) -> int:
    """Get denominator for percentages."""
    if PERCENT_DENOMINATOR_MODE == "all_studies":
        return int(df[STUDY_ID_COLUMN].nunique())

    if PERCENT_DENOMINATOR_MODE == "valid_domain_studies":
        valid_df = df[df[domain_column].apply(study_has_valid_domain_value)]
        return int(valid_df[STUDY_ID_COLUMN].nunique())

    raise ValueError(
        "PERCENT_DENOMINATOR_MODE must be either 'valid_domain_studies' "
        "or 'all_studies'."
    )


def create_domain_long_table(
    df: pd.DataFrame,
    domain_name: str,
    source_column: str,
    mapper_function,
    allowed_categories: dict[str, bool],
) -> pd.DataFrame:
    """Create long-format table for one analysis domain."""
    records = []

    for _, row in df.iterrows():
        study_id = row[STUDY_ID_COLUMN]
        tokens = split_semicolon_cell(row[source_column])

        for token in tokens:
            mapped = mapper_function(token)

            if mapped is None:
                continue

            if isinstance(mapped, str):
                mapped_categories = [mapped]
            else:
                mapped_categories = list(mapped)

            for category in mapped_categories:
                if not allowed_categories.get(category, False):
                    continue

                records.append({
                    "Study Reference": study_id,
                    "Domain": domain_name,
                    "Raw value": token,
                    "Category": category,
                })

    long_table = pd.DataFrame(records)

    if long_table.empty:
        return pd.DataFrame(
            columns=["Study Reference", "Domain", "Raw value", "Category"]
        )

    long_table = long_table.drop_duplicates(
        subset=["Study Reference", "Domain", "Category"]
    )

    return long_table


def create_summary_table(
    long_table: pd.DataFrame,
    domain_name: str,
    ordered_categories: list[str],
    include_categories: dict[str, bool],
    denominator: int,
) -> pd.DataFrame:
    """Create count and percentage summary for one domain."""
    selected_categories = [
        category for category in ordered_categories
        if include_categories.get(category, False)
    ]

    base = pd.DataFrame({
        "Domain": domain_name,
        "Category": selected_categories,
    })

    if long_table.empty:
        base["Number of studies"] = 0
    else:
        counts = (
            long_table
            .groupby(["Domain", "Category"], as_index=False)
            .agg(**{"Number of studies": ("Study Reference", "nunique")})
        )

        base = (
            base
            .merge(counts, on=["Domain", "Category"], how="left")
            .fillna({"Number of studies": 0})
        )

    base["Number of studies"] = base["Number of studies"].astype(int)

    if denominator == 0:
        base["Percentage"] = 0.0
    else:
        base["Percentage"] = (
            base["Number of studies"] / denominator * 100
        ).round(1)

    base["Percentage denominator"] = denominator

    return base


def create_all_summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create combined long and summary tables for all four domains."""
    preprocessing_long = create_domain_long_table(
        df=df,
        domain_name="Preprocessing approaches",
        source_column=PREPROCESSING_COLUMN,
        mapper_function=map_preprocessing_token,
        allowed_categories=INCLUDE_PREPROCESSING,
    )

    feature_long = create_domain_long_table(
        df=df,
        domain_name="Feature type",
        source_column=FEATURE_TYPE_COLUMN,
        mapper_function=map_feature_type_token,
        allowed_categories=INCLUDE_FEATURE_TYPE,
    )

    splitting_long = create_domain_long_table(
        df=df,
        domain_name="Data splitting",
        source_column=DATA_SPLITTING_COLUMN,
        mapper_function=map_data_splitting_token,
        allowed_categories=INCLUDE_DATA_SPLITTING,
    )

    validation_long = create_domain_long_table(
        df=df,
        domain_name="Model validation",
        source_column=MODEL_VALIDATION_COLUMN,
        mapper_function=map_model_validation_token,
        allowed_categories=INCLUDE_MODEL_VALIDATION,
    )

    long_table = pd.concat(
        [preprocessing_long, feature_long, splitting_long, validation_long],
        ignore_index=True,
    )

    preprocessing_summary = create_summary_table(
        long_table=preprocessing_long,
        domain_name="Preprocessing approaches",
        ordered_categories=PREPROCESSING_ORDER,
        include_categories=INCLUDE_PREPROCESSING,
        denominator=get_denominator(df, PREPROCESSING_COLUMN),
    )

    feature_summary = create_summary_table(
        long_table=feature_long,
        domain_name="Feature type",
        ordered_categories=FEATURE_TYPE_ORDER,
        include_categories=INCLUDE_FEATURE_TYPE,
        denominator=get_denominator(df, FEATURE_TYPE_COLUMN),
    )

    splitting_summary = create_summary_table(
        long_table=splitting_long,
        domain_name="Data splitting",
        ordered_categories=DATA_SPLITTING_ORDER,
        include_categories=INCLUDE_DATA_SPLITTING,
        denominator=get_denominator(df, DATA_SPLITTING_COLUMN),
    )

    validation_summary = create_summary_table(
        long_table=validation_long,
        domain_name="Model validation",
        ordered_categories=MODEL_VALIDATION_ORDER,
        include_categories=INCLUDE_MODEL_VALIDATION,
        denominator=get_denominator(df, MODEL_VALIDATION_COLUMN),
    )

    summary_table = pd.concat(
        [preprocessing_summary, feature_summary, splitting_summary, validation_summary],
        ignore_index=True,
    )

    return summary_table, long_table


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def build_plot_table(summary_table: pd.DataFrame) -> pd.DataFrame:
    """Create a hierarchical plot table with domain headers and category rows."""
    plot_rows = []

    for domain in DOMAIN_ORDER:
        domain_df = summary_table[summary_table["Domain"] == domain].copy()

        if domain_df.empty:
            continue

        plot_rows.append({
            "Label": domain,
            "Domain": domain,
            "Category": None,
            "Number of studies": np.nan,
            "Percentage": np.nan,
            "Is header": True,
        })

        for _, row in domain_df.iterrows():
            plot_rows.append({
                "Label": "   " + row["Category"],
                "Domain": domain,
                "Category": row["Category"],
                "Number of studies": row["Number of studies"],
                "Percentage": row["Percentage"],
                "Is header": False,
            })

    plot_table = pd.DataFrame(plot_rows)

    # Reverse so the first domain appears at the top in a horizontal bar plot.
    plot_table = plot_table.iloc[::-1].reset_index(drop=True)

    return plot_table


def plot_grouped_horizontal_bar(
    summary_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
) -> None:
    """Create grouped horizontal bar chart."""
    plot_table = build_plot_table(summary_table)

    bar_positions = []
    bar_values = []
    bar_colors = []

    for i, row in plot_table.iterrows():
        if not row["Is header"]:
            bar_positions.append(i)
            bar_values.append(float(row["Number of studies"]))
            bar_colors.append(DOMAIN_COLORS.get(row["Domain"], "#808080"))

    fig_height = max(8.2, 0.44 * len(plot_table) + 1.8)
    fig, ax = plt.subplots(figsize=(11.8, fig_height))

    bars = ax.barh(
        bar_positions,
        bar_values,
        height=0.72,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(plot_table)))
    ax.set_yticklabels(plot_table["Label"])

    # Bold domain headers.
    for tick_label, is_header in zip(ax.get_yticklabels(), plot_table["Is header"]):
        if is_header and tick_label.get_text().strip():
            tick_label.set_fontweight("bold")

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("")
    ax.set_title(
        "Preprocessing, feature-type, data-splitting, and model-validation approaches"
    )

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

    max_count = max(bar_values) if bar_values else 1
    ax.set_xlim(0, max_count * 1.34)

    if SHOW_PERCENT_LABELS:
        for bar, row_index in zip(bars, bar_positions):
            row = plot_table.loc[row_index]
            count = int(row["Number of studies"])
            percentage = float(row["Percentage"])

            ax.text(
                count + max_count * 0.02,
                row_index,
                f"{count} ({percentage:.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
            )

    if SHOW_LEGEND:
        legend_handles = [
            Patch(facecolor=DOMAIN_COLORS[domain], edgecolor="black", label=domain)
            for domain in DOMAIN_ORDER
            if domain in DOMAIN_COLORS
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=True,
            title="Domain",
        )

    if PERCENT_DENOMINATOR_MODE == "valid_domain_studies":
        note = (
            "Percentages use the number of studies with non-NA coded data within "
            "each methodological domain as the denominator."
        )
    else:
        note = (
            "Percentages use all unique studies in the input file as the denominator."
        )

    fig.text(
        0.01,
        0.01,
        (
            "Categories may be non-mutually exclusive; studies could contribute to "
            "more than one category within a domain. " + note
        ),
        ha="left",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()

    print(f"Plot saved to: {output_png.resolve()}")


# --------------------------------------------------
# Save workbook
# --------------------------------------------------

def save_summary_workbook(
    summary_table: pd.DataFrame,
    long_table: pd.DataFrame,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save summary, long-format mapping, and cleaned source columns."""
    cleaned_source = df[
        [
            STUDY_ID_COLUMN,
            PREPROCESSING_COLUMN,
            FEATURE_TYPE_COLUMN,
            DATA_SPLITTING_COLUMN,
            MODEL_VALIDATION_COLUMN,
        ]
    ].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_table.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        long_table.to_excel(
            writer,
            sheet_name="Long format mapping",
            index=False,
        )

        cleaned_source.to_excel(
            writer,
            sheet_name="Cleaned source columns",
            index=False,
        )

    print(f"Summary workbook saved to: {output_path.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run full workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {INPUT_FILE.resolve()}")

    df = load_data(INPUT_FILE)

    summary_table, long_table = create_all_summary_tables(df)

    save_summary_workbook(
        summary_table=summary_table,
        long_table=long_table,
        df=df,
        output_path=OUTPUT_TABLE,
    )

    plot_grouped_horizontal_bar(
        summary_table=summary_table,
        output_png=OUTPUT_PLOT_PNG,
        output_pdf=OUTPUT_PLOT_PDF,
        output_svg=OUTPUT_PLOT_SVG,
    )

    print("\nModel-development characteristics analysis completed successfully.")


if __name__ == "__main__":
    main()
