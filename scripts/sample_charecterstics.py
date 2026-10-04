# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 19:53:25 2026

@author: ak7u24
"""

"""
Three-panel study-characteristics plot.

Input:
    inputs/sample_charecterstics.xlsx

Required columns:
    sample_type
    sample_size
    data_collection_duration_days

Outputs:
    output/sample_characteristics_summary.xlsx
    output/sample_characteristics_three_panel_plot.png
    output/sample_characteristics_three_panel_plot.pdf
    output/sample_characteristics_three_panel_plot.svg
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

INPUT_FILE = PROJECT_ROOT / "input" / "sample_charecterstics.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "sample_characteristics_summary.xlsx"
OUTPUT_PLOT_PNG = OUTPUT_DIR / "sample_characteristics_three_panel_plot.png"
OUTPUT_PLOT_PDF = OUTPUT_DIR / "sample_characteristics_three_panel_plot.pdf"
OUTPUT_PLOT_SVG = OUTPUT_DIR / "sample_characteristics_three_panel_plot.svg"


# --------------------------------------------------
# Column names
# --------------------------------------------------

SAMPLE_TYPE_COLUMN = "sample_type"
SAMPLE_SIZE_COLUMN = "sample_size"
DURATION_COLUMN = "data_collection_duration_days"
STUDY_REFERENCE_COLUMN = "Study Reference"


# --------------------------------------------------
# Plot options
# --------------------------------------------------

USE_LOG_SCALE_FOR_SAMPLE_SIZE = False
USE_LOG_SCALE_FOR_DURATION = False
SHOW_FIGURE_NOTE = False

OUTPUT_DPI = 300

SAMPLE_TYPE_ORDER = [
    "Students",
    "Patients",
    "Mixed source",
    "Public",
    "Employees",
    "Crowdsourced",
    "Unknown/unspecified",
]

SAMPLE_TYPE_DISPLAY_LABELS = {
    "Students": "Students",
    "Patients": "Clinical",
    "Mixed source": "Mixed source",
    "Public": "Public",
    "Employees": "Employees",
    "Crowdsourced": "Crowdsourced",
    "Unknown/unspecified": "Unknown/unspecified",
}


# --------------------------------------------------
# Matplotlib publication settings
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# --------------------------------------------------
# Text cleaning utilities
# --------------------------------------------------

def normalise_text(value: str) -> str:
    """
    Normalise text for robust matching.
    """
    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace("’", "'").replace("‘", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_missing_or_unclear(value) -> bool:
    """
    Identify missing, unclear, or not-reported values.
    """
    if pd.isna(value):
        return True

    key = normalise_text(value)

    missing_terms = {
        "",
        "na",
        "n a",
        "nan",
        "none",
        "not clear",
        "unclear",
        "not mentioned",
        "not reported",
        "unknown",
        "unknown unspecified",
        "unspecified",
    }

    return key in missing_terms


def extract_numeric_value(value):
    """
    Extract a numeric value from a cell.

    Returns np.nan for not clear, not mentioned, or non-numeric values.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()

    if is_missing_or_unclear(text):
        return np.nan

    text = text.replace(",", "")

    match = re.search(r"[-+]?\d*\.?\d+", text)

    if match:
        return float(match.group())

    return np.nan


# --------------------------------------------------
# Sample type cleaning
# --------------------------------------------------

def split_sample_type_cell(value) -> list[str]:
    """
    Split sample_type cell using semicolon.

    Example:
        'Students; Patients' -> ['Students', 'Patients']
    """
    if pd.isna(value):
        return ["Unknown/unspecified"]

    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)

    if is_missing_or_unclear(text):
        return ["Unknown/unspecified"]

    tokens = [token.strip() for token in text.split(";")]
    tokens = [token for token in tokens if token]

    if not tokens:
        return ["Unknown/unspecified"]

    return tokens


def standardise_sample_type_token(token: str) -> str:
    """
    Map raw sample-type tokens to standard categories.

    Standard categories:
        Students
        Patients
        Mixed source
        Public
        Employees
        Crowdsourced
        Unknown/unspecified
    """
    key = normalise_text(token)

    if key in {
        "",
        "na",
        "n a",
        "nan",
        "unknown",
        "unspecified",
        "unknown unspecified",
        "not clear",
        "not mentioned",
        "not reported",
    }:
        return "Unknown/unspecified"

    if re.search(r"\b(student|students|undergraduate|undergraduates|college|university)\b", key):
        return "Students"

    if re.search(r"\b(patient|patients|clinical|clinic)\b", key):
        return "Patients"

    if re.search(r"\b(mixed|mixed general population|mixed source)\b", key):
        return "Mixed source"

    if re.search(r"\b(public|community|general population|population)\b", key):
        return "Public"

    if re.search(r"\b(employee|employees|worker|workers|workplace)\b", key):
        return "Employees"

    if re.search(r"\b(crowdsourced|crowdsource|online|prolific|mturk|amazon mechanical turk)\b", key):
        return "Crowdsourced"

    return "Unknown/unspecified"


# --------------------------------------------------
# Data loading and preparation
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load Excel file and validate required columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    required_columns = [
        SAMPLE_TYPE_COLUMN,
        SAMPLE_SIZE_COLUMN,
        DURATION_COLUMN,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["study_row_id"] = np.arange(1, len(df) + 1)

    if STUDY_REFERENCE_COLUMN not in df.columns:
        df[STUDY_REFERENCE_COLUMN] = df["study_row_id"].astype(str)

    return df


def create_exploded_sample_type_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one row per study-sample-type combination.

    If a study has 'Students; Patients', it contributes once to Students
    and once to Patients.
    """
    records = []
    mapping_records = []

    for _, row in df.iterrows():
        raw_sample_type = row[SAMPLE_TYPE_COLUMN]
        raw_tokens = split_sample_type_cell(raw_sample_type)

        standardised_types = set()

        for token in raw_tokens:
            standard_type = standardise_sample_type_token(token)
            standardised_types.add(standard_type)

            mapping_records.append({
                "study_row_id": row["study_row_id"],
                "study_reference": row[STUDY_REFERENCE_COLUMN],
                "sample_type_raw": raw_sample_type,
                "sample_type_token": token,
                "sample_type_standardised": standard_type,
            })

        for sample_type in standardised_types:
            records.append({
                "study_row_id": row["study_row_id"],
                "study_reference": row[STUDY_REFERENCE_COLUMN],
                "sample_type": sample_type,
                "sample_size_raw": row[SAMPLE_SIZE_COLUMN],
                "sample_size": extract_numeric_value(row[SAMPLE_SIZE_COLUMN]),
                "data_collection_duration_days_raw": row[DURATION_COLUMN],
                "data_collection_duration_days": extract_numeric_value(row[DURATION_COLUMN]),
            })

    exploded_df = pd.DataFrame(records)
    mapping_df = pd.DataFrame(mapping_records)

    exploded_df["sample_type"] = pd.Categorical(
        exploded_df["sample_type"],
        categories=SAMPLE_TYPE_ORDER,
        ordered=True,
    )

    return exploded_df, mapping_df


# --------------------------------------------------
# Summary tables
# --------------------------------------------------

def create_panel_a_table(exploded_df: pd.DataFrame, total_studies: int) -> pd.DataFrame:
    """
    Create sample-type count table for Panel A.
    """
    counts = (
        exploded_df
        .drop_duplicates(subset=["study_row_id", "sample_type"])
        .groupby("sample_type", observed=False)
        .agg(number_of_studies=("study_row_id", "nunique"))
        .reset_index()
    )

    counts["percentage_of_included_studies"] = (
        counts["number_of_studies"] / total_studies * 100
    ).round(1)

    return counts


def summarise_numeric_by_sample_type(
    exploded_df: pd.DataFrame,
    value_column: str
) -> pd.DataFrame:
    """
    Summarise sample size or data-collection duration by sample type.
    """
    clean_df = exploded_df.dropna(subset=[value_column]).copy()
    clean_df = clean_df[clean_df[value_column] > 0].copy()

    summary = (
        clean_df
        .groupby("sample_type", observed=False)
        .agg(
            number_of_studies=("study_row_id", "nunique"),
            median=(value_column, "median"),
            q1=(value_column, lambda x: x.quantile(0.25)),
            q3=(value_column, lambda x: x.quantile(0.75)),
            minimum=(value_column, "min"),
            maximum=(value_column, "max"),
        )
        .reset_index()
    )

    for col in ["median", "q1", "q3", "minimum", "maximum"]:
        summary[col] = summary[col].round(2)

    return summary


def save_summary_tables(
    panel_a_table: pd.DataFrame,
    sample_size_summary: pd.DataFrame,
    duration_summary: pd.DataFrame,
    exploded_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save all summary tables to Excel.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded_sample_size = exploded_df[
        exploded_df["sample_size"].isna()
    ][
        [
            "study_row_id",
            "study_reference",
            "sample_type",
            "sample_size_raw",
        ]
    ].copy()

    excluded_duration = exploded_df[
        exploded_df["data_collection_duration_days"].isna()
    ][
        [
            "study_row_id",
            "study_reference",
            "sample_type",
            "data_collection_duration_days_raw",
        ]
    ].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        panel_a_table.to_excel(
            writer,
            sheet_name="Panel A sample types",
            index=False,
        )

        sample_size_summary.to_excel(
            writer,
            sheet_name="Panel B sample size",
            index=False,
        )

        duration_summary.to_excel(
            writer,
            sheet_name="Panel C duration",
            index=False,
        )

        exploded_df.to_excel(
            writer,
            sheet_name="Cleaned exploded data",
            index=False,
        )

        mapping_df.to_excel(
            writer,
            sheet_name="Sample type mapping",
            index=False,
        )

        if not excluded_sample_size.empty:
            excluded_sample_size.to_excel(
                writer,
                sheet_name="Excluded sample size",
                index=False,
            )

        if not excluded_duration.empty:
            excluded_duration.to_excel(
                writer,
                sheet_name="Excluded duration",
                index=False,
            )

    print(f"Summary Excel table saved to: {output_path.resolve()}")


# --------------------------------------------------
# Plotting helper functions
# --------------------------------------------------

def add_panel_label(ax, label: str) -> None:
    """
    Add panel label such as A, B, or C.
    """
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_panel_a(ax, panel_a_table: pd.DataFrame) -> None:
    """
    Panel A: horizontal bar chart of sample-type distribution.
    """
    plot_df = panel_a_table.copy()
    plot_df = plot_df.sort_values("number_of_studies", ascending=True)
    
    display_labels = [
        SAMPLE_TYPE_DISPLAY_LABELS.get(sample_type, sample_type)
        for sample_type in plot_df["sample_type"].astype(str)
        ]
    bars = ax.barh(
        display_labels,
        plot_df["number_of_studies"],
        height=0.65,
        edgecolor="black",
        linewidth=0.6,
        )

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("")
    ax.set_title("Sample type distribution")

    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    max_count = plot_df["number_of_studies"].max()

    if pd.isna(max_count) or max_count == 0:
        max_count = 1

    ax.set_xlim(0, max_count * 1.20)

    for bar, count, percentage in zip(
        bars,
        plot_df["number_of_studies"],
        plot_df["percentage_of_included_studies"],
    ):
        ax.text(
            bar.get_width() + max_count * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({percentage:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_boxplot_with_points(
    ax,
    exploded_df: pd.DataFrame,
    value_column: str,
    y_label: str,
    title: str,
    use_log_scale: bool,
) -> None:
    """
    Draw grouped boxplot with jittered individual study points.
    """
    plot_df = exploded_df.dropna(subset=[value_column]).copy()
    plot_df = plot_df[plot_df[value_column] > 0].copy()

    available_sample_types = [
        sample_type
        for sample_type in SAMPLE_TYPE_ORDER
        if sample_type in plot_df["sample_type"].astype(str).unique()
    ]

    data_by_group = [
        plot_df.loc[
            plot_df["sample_type"].astype(str) == sample_type,
            value_column,
        ].to_numpy()
        for sample_type in available_sample_types
    ]

    positions = np.arange(1, len(available_sample_types) + 1)

    ax.boxplot(
        data_by_group,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=False,
        medianprops={"linewidth": 1.5},
    )

    rng = np.random.default_rng(42)

    for position, values in zip(positions, data_by_group):
        jitter = rng.normal(loc=0, scale=0.055, size=len(values))

        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            alpha=0.60,
            linewidth=0.4,
            edgecolor="black",
        )
        
    display_labels = [
        SAMPLE_TYPE_DISPLAY_LABELS.get(sample_type, sample_type)
        for sample_type in available_sample_types
        ]
    
    ax.set_xticks(positions)
    ax.set_xticklabels(
        display_labels,
        rotation=45,
        ha="right",
        )

    ax.set_ylabel(y_label)
    ax.set_title(title)

    if use_log_scale:
        ax.set_yscale("log")

    ax.grid(
        True,
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------
# Main three-panel plot
# --------------------------------------------------

def plot_three_panel_figure(
    panel_a_table: pd.DataFrame,
    exploded_df: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
) -> None:
    """
    Create the three-panel figure.
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5.5),
        gridspec_kw={
            "width_ratios": [1.15, 1.45, 1.45],
            "wspace": 0.35,
        },
    )

    ax_a, ax_b, ax_c = axes

    plot_panel_a(
        ax=ax_a,
        panel_a_table=panel_a_table,
    )

    plot_boxplot_with_points(
        ax=ax_b,
        exploded_df=exploded_df,
        value_column="sample_size",
        y_label="Sample size",
        title="Sample size by sample type",
        use_log_scale=USE_LOG_SCALE_FOR_SAMPLE_SIZE,
    )

    plot_boxplot_with_points(
        ax=ax_c,
        exploded_df=exploded_df,
        value_column="data_collection_duration_days",
        y_label="Data-collection duration, days",
        title="Data-collection duration by sample type",
        use_log_scale=USE_LOG_SCALE_FOR_DURATION,
    )

    add_panel_label(ax_a, "A")
    add_panel_label(ax_b, "B")
    add_panel_label(ax_c, "C")

    scale_note_sample_size = "log scale" if USE_LOG_SCALE_FOR_SAMPLE_SIZE else "linear scale"
    scale_note_duration = "log scale" if USE_LOG_SCALE_FOR_DURATION else "linear scale"
    
    if SHOW_FIGURE_NOTE:
        fig.text(
            0.01,
            0.01,
            (
                "Note: Studies with multiple sample types were counted once under each relevant sample type. "
                f"Panel B uses {scale_note_sample_size}; Panel C uses {scale_note_duration}. "
                "Unclear, not reported, and non-numeric values were excluded from numeric boxplots."
                ),
            fontsize=8,
            ha="left",
            )
        fig.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        fig.tight_layout(rect=[0, 0.02, 1, 1])

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()

    print(f"PNG plot saved to: {output_png.resolve()}")
    print(f"PDF plot saved to: {output_pdf.resolve()}")
    print(f"SVG plot saved to: {output_svg.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """
    Run the full workflow.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {INPUT_FILE.resolve()}")

    df = load_data(INPUT_FILE)

    total_studies = df.shape[0]

    exploded_df, mapping_df = create_exploded_sample_type_data(df)

    panel_a_table = create_panel_a_table(
        exploded_df=exploded_df,
        total_studies=total_studies,
    )

    sample_size_summary = summarise_numeric_by_sample_type(
        exploded_df=exploded_df,
        value_column="sample_size",
    )

    duration_summary = summarise_numeric_by_sample_type(
        exploded_df=exploded_df,
        value_column="data_collection_duration_days",
    )

    save_summary_tables(
        panel_a_table=panel_a_table,
        sample_size_summary=sample_size_summary,
        duration_summary=duration_summary,
        exploded_df=exploded_df,
        mapping_df=mapping_df,
        output_path=OUTPUT_TABLE,
    )

    plot_three_panel_figure(
        panel_a_table=panel_a_table,
        exploded_df=exploded_df,
        output_png=OUTPUT_PLOT_PNG,
        output_pdf=OUTPUT_PLOT_PDF,
        output_svg=OUTPUT_PLOT_SVG,
    )

    print("\nThree-panel sample characteristics figure completed successfully.")


if __name__ == "__main__":
    main()