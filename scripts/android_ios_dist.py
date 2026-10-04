# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:19:25 2026

@author: ak7u24
"""

"""
Plot yearly use of Android, iOS, and both smartphone operating systems.

Input:
    inputs/age_smartphone_os.xlsx

Required columns:
    publication_year
    smartphone_os

Outputs:
    output/smartphone_os_by_year_table.xlsx
    output/smartphone_os_by_year_plot.png
    output/smartphone_os_by_year_plot.pdf
    output/smartphone_os_by_year_plot.svg
"""

from pathlib import Path
import re
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "age_smartphone_os.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "smartphone_os_by_year_table.xlsx"
OUTPUT_PLOT_PNG = OUTPUT_DIR / "smartphone_os_by_year_plot.png"
OUTPUT_PLOT_PDF = OUTPUT_DIR / "smartphone_os_by_year_plot.pdf"
OUTPUT_PLOT_SVG = OUTPUT_DIR / "smartphone_os_by_year_plot.svg"


# --------------------------------------------------
# Column names
# --------------------------------------------------

PUBLICATION_YEAR_COLUMN = "publication_year"
SMARTPHONE_OS_COLUMN = "smartphone_os"

# If each row is one study, keep this as None.
# If each row is one participant and you have a study identifier column,
# set this to that column name, for example:
# STUDY_ID_COLUMN = "study_id"
STUDY_ID_COLUMN = None


# --------------------------------------------------
# Plot options
# --------------------------------------------------

OUTPUT_DPI = 300
SHOW_VALUE_LABELS = True

OS_ORDER = [
    "Android",
    "iOS",
    "Both",
]


# --------------------------------------------------
# Matplotlib publication settings
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
# Text utilities
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


def is_missing_or_not_reported(value) -> bool:
    """
    Identify missing or not-reported smartphone OS values.
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
        "not reported",
        "not mentioned",
        "not clear",
        "unclear",
        "unknown",
        "unknown unspecified",
        "unspecified",
    }

    return key in missing_terms


def standardise_smartphone_os(value) -> str | None:
    """
    Map raw smartphone OS labels to:
        Android
        iOS
        Both

    Returns None for not-reported or unclear values.
    """
    if is_missing_or_not_reported(value):
        return None

    key = normalise_text(value)

    has_android = bool(
        re.search(r"\b(android|google android)\b", key)
    )

    has_ios = bool(
        re.search(r"\b(ios|iphone|apple)\b", key)
    )

    has_both_word = bool(
        re.search(r"\b(both|android and ios|ios and android|android ios)\b", key)
    )

    if has_both_word or (has_android and has_ios):
        return "Both"

    if has_android:
        return "Android"

    if has_ios:
        return "iOS"

    return None


# --------------------------------------------------
# Data loading and cleaning
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load Excel file and validate required columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    required_columns = [
        PUBLICATION_YEAR_COLUMN,
        SMARTPHONE_OS_COLUMN,
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    if STUDY_ID_COLUMN is not None and STUDY_ID_COLUMN not in df.columns:
        raise KeyError(
            f"STUDY_ID_COLUMN is set to '{STUDY_ID_COLUMN}', "
            f"but this column was not found in the Excel file."
        )

    return df


def clean_smartphone_os_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean publication year and smartphone OS columns.

    Returns:
        clean_df: rows included in the analysis
        excluded_df: rows excluded because year or OS was missing/unclear
    """
    cleaned = df.copy()

    cleaned["publication_year_clean"] = pd.to_numeric(
        cleaned[PUBLICATION_YEAR_COLUMN],
        errors="coerce"
    )

    cleaned["smartphone_os_clean"] = cleaned[SMARTPHONE_OS_COLUMN].apply(
        standardise_smartphone_os
    )

    included_mask = (
        cleaned["publication_year_clean"].notna()
        & cleaned["smartphone_os_clean"].notna()
    )

    clean_df = cleaned[included_mask].copy()
    excluded_df = cleaned[~included_mask].copy()

    clean_df["publication_year_clean"] = (
        clean_df["publication_year_clean"]
        .astype(int)
    )

    return clean_df, excluded_df


# --------------------------------------------------
# Create count tables
# --------------------------------------------------

def create_os_by_year_tables(
    clean_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create long and wide count tables.

    If STUDY_ID_COLUMN is None:
        counts rows as studies.

    If STUDY_ID_COLUMN is provided:
        counts unique studies per publication year and smartphone OS.
    """
    if STUDY_ID_COLUMN is None:
        long_table = (
            clean_df
            .groupby(
                ["publication_year_clean", "smartphone_os_clean"],
                as_index=False
            )
            .size()
            .rename(columns={
                "publication_year_clean": "Publication year",
                "smartphone_os_clean": "Smartphone OS",
                "size": "Number of studies",
            })
        )

    else:
        long_table = (
            clean_df
            .groupby(
                ["publication_year_clean", "smartphone_os_clean"],
                as_index=False
            )
            .agg(**{
                "Number of studies": (STUDY_ID_COLUMN, "nunique")
            })
            .rename(columns={
                "publication_year_clean": "Publication year",
                "smartphone_os_clean": "Smartphone OS",
            })
        )

    wide_table = (
        long_table
        .pivot(
            index="Publication year",
            columns="Smartphone OS",
            values="Number of studies"
        )
        .fillna(0)
        .astype(int)
    )

    # Ensure consistent column order
    for os_name in OS_ORDER:
        if os_name not in wide_table.columns:
            wide_table[os_name] = 0

    wide_table = wide_table[OS_ORDER]
    wide_table = wide_table.sort_index()

    wide_table["Total studies"] = wide_table.sum(axis=1)

    wide_table = wide_table.reset_index()

    return long_table, wide_table


# --------------------------------------------------
# Save Excel table
# --------------------------------------------------

def save_tables(
    long_table: pd.DataFrame,
    wide_table: pd.DataFrame,
    clean_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save summary tables to Excel.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wide_table.to_excel(
            writer,
            sheet_name="OS by year wide",
            index=False
        )

        long_table.to_excel(
            writer,
            sheet_name="OS by year long",
            index=False
        )

        clean_df.to_excel(
            writer,
            sheet_name="Cleaned included rows",
            index=False
        )

        if not excluded_df.empty:
            excluded_df.to_excel(
                writer,
                sheet_name="Excluded rows",
                index=False
            )

    print(f"Smartphone OS summary table saved to: {output_path.resolve()}")


# --------------------------------------------------
# Plot horizontal stacked bar chart
# --------------------------------------------------

def plot_smartphone_os_by_year(
    wide_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path
) -> None:
    """
    Plot horizontal stacked bar chart of smartphone OS use by publication year.
    """
    plot_df = wide_table.copy()
    plot_df = plot_df.sort_values("Publication year", ascending=True)

    fig_height = max(5.0, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, fig_height))

    left_values = pd.Series([0] * len(plot_df), index=plot_df.index)

    for os_name in OS_ORDER:
        bars = ax.barh(
            y=plot_df["Publication year"].astype(str),
            width=plot_df[os_name],
            left=left_values,
            label=os_name,
            edgecolor="black",
            linewidth=0.4,
            height=0.68,
        )

        if SHOW_VALUE_LABELS:
            for bar, value in zip(bars, plot_df[os_name]):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(value)),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        left_values = left_values + plot_df[os_name]

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("Publication year")
    ax.set_title("Yearly use of smartphone operating systems in included studies")

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

    ax.legend(
        title="Smartphone OS",
        loc="lower right",
        frameon=True,
    )

    fig.tight_layout()

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
    Run the full smartphone OS by publication year workflow.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {INPUT_FILE.resolve()}")

    df = load_data(INPUT_FILE)

    clean_df, excluded_df = clean_smartphone_os_data(df)

    if clean_df.empty:
        raise ValueError(
            "No valid rows were available after excluding not-reported smartphone OS values."
        )

    long_table, wide_table = create_os_by_year_tables(clean_df)

    save_tables(
        long_table=long_table,
        wide_table=wide_table,
        clean_df=clean_df,
        excluded_df=excluded_df,
        output_path=OUTPUT_TABLE,
    )

    plot_smartphone_os_by_year(
        wide_table=wide_table,
        output_png=OUTPUT_PLOT_PNG,
        output_pdf=OUTPUT_PLOT_PDF,
        output_svg=OUTPUT_PLOT_SVG,
    )

    print("\nSmartphone OS yearly distribution analysis completed successfully.")


if __name__ == "__main__":
    main()