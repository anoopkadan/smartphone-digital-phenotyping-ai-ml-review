# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:52:12 2026

@author: ak7u24
"""
"""
Temporal trend plot for systematic review model types.

Folder structure:
    project_root/
    ├── inputs/
    │   └── summary_data.xlsx
    ├── outputs/
    └── scripts/
        └── plot_temporal_trend.py

Outputs:
    outputs/temporal_trend_table.xlsx
    outputs/combined_temporal_trend_plot.png
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "summary_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "temporal_trend_table.xlsx"
OUTPUT_COMBINED_PLOT = OUTPUT_DIR / "combined_temporal_trend_plot.png"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

PUBLICATION_YEAR_COLUMN = "publication_year"
MODEL_TYPE_COLUMN = "ml_model_type"
MULTIMODAL_COLUMN = "multimodal"


# --------------------------------------------------
# Plot configuration
# --------------------------------------------------

HISTOGRAM_ALPHA = 0.25  # Change this to control bar opacity: 0.1 = faint, 1.0 = solid

SHOW_SHADOW_TRENDS = True
SHADOW_TREND_ALPHA = 0.18
SHADOW_TREND_WIDTH = 4

'''
SHADOW_TREND_ALPHA = 0.10  # very faint shadow
SHADOW_TREND_ALPHA = 0.25  # stronger shadow
SHADOW_TREND_WIDTH = 8     # thinner shadow
SHADOW_TREND_WIDTH = 14    # wider shadow
'''


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the Excel file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    return pd.read_excel(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the required columns for temporal trend analysis.
    """
    required_columns = [
        PUBLICATION_YEAR_COLUMN,
        MODEL_TYPE_COLUMN,
        MULTIMODAL_COLUMN,
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns are: {list(df.columns)}"
        )

    cleaned_df = df[required_columns].copy()

    cleaned_df[PUBLICATION_YEAR_COLUMN] = pd.to_numeric(
        cleaned_df[PUBLICATION_YEAR_COLUMN],
        errors="coerce"
    )

    cleaned_df = cleaned_df.dropna(subset=[PUBLICATION_YEAR_COLUMN])
    cleaned_df[PUBLICATION_YEAR_COLUMN] = cleaned_df[PUBLICATION_YEAR_COLUMN].astype(int)

    cleaned_df[MODEL_TYPE_COLUMN] = (
        cleaned_df[MODEL_TYPE_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned_df[MULTIMODAL_COLUMN] = (
        cleaned_df[MULTIMODAL_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return cleaned_df


# --------------------------------------------------
# Temporal trend table
# --------------------------------------------------

def create_temporal_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create yearly counts for each modelling approach.

    Coding rules:
    - 'Statistical Study' counts as Statistical Approach.
    - 'Classical ML' counts as Classical ML.
    - 'Deep Learning' counts as Deep Learning.
    - 'Both' counts as both Classical ML and Deep Learning.
    - multimodal != 'Not used' counts as Multimodal/Hybrid.
    """
    model_type = df[MODEL_TYPE_COLUMN].str.lower().str.strip()
    multimodal = df[MULTIMODAL_COLUMN].str.lower().str.strip()

    coded_df = pd.DataFrame()
    coded_df["Year"] = df[PUBLICATION_YEAR_COLUMN]

    coded_df["Statistical Approach"] = (
        model_type == "statistical study"
    ).astype(int)

    coded_df["Classical ML"] = (
        (model_type == "classical ml") |
        (model_type == "both")
    ).astype(int)

    coded_df["Deep Learning"] = (
        (model_type == "deep learning") |
        (model_type == "both")
    ).astype(int)

    coded_df["Multimodal/Hybrid"] = (
        (multimodal != "not used") &
        (multimodal != "") &
        (multimodal != "nan")
    ).astype(int)

    trend_table = (
        coded_df
        .groupby("Year", as_index=False)
        .sum()
        .sort_values("Year")
    )

    return trend_table


def create_year_histogram_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create yearly total study counts from the publication_year column.

    This is used as the histogram/bar background in the combined plot.
    """
    year_counts = (
        df[PUBLICATION_YEAR_COLUMN]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    year_counts.columns = ["Year", "Total studies"]

    return year_counts


def combine_trend_and_histogram_tables(
    trend_table: pd.DataFrame,
    year_histogram_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine model-type trend counts with total study counts per year.
    """
    combined_table = pd.merge(
        year_histogram_table,
        trend_table,
        on="Year",
        how="outer"
    )

    combined_table = combined_table.sort_values("Year").fillna(0)

    count_columns = [
        "Total studies",
        "Statistical Approach",
        "Classical ML",
        "Deep Learning",
        "Multimodal/Hybrid",
    ]

    combined_table[count_columns] = combined_table[count_columns].astype(int)

    return combined_table


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

def save_trend_table(trend_table: pd.DataFrame, output_path: Path) -> None:
    """
    Save the temporal trend table as an Excel file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trend_table.to_excel(output_path, index=False)
    print(f"Trend table saved to: {output_path.resolve()}")


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def add_shadow_trend_line(
    x_values: pd.Series,
    y_values: pd.Series,
    original_line,
) -> None:
    """
    Add a soft shadow trend behind the observed line.

    The observed solid line shows yearly counts.
    The shadow line shows the overall linear trend.
    """
    x = pd.to_numeric(x_values, errors="coerce").to_numpy()
    y = pd.to_numeric(y_values, errors="coerce").to_numpy()

    valid_mask = ~np.isnan(x) & ~np.isnan(y)

    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < 2:
        return

    slope, intercept = np.polyfit(x, y, 1)
    trend_values = slope * x + intercept

    plt.plot(
        x,
        trend_values,
        linestyle="-",
        linewidth=SHADOW_TREND_WIDTH,
        alpha=SHADOW_TREND_ALPHA,
        color=original_line.get_color(),
        solid_capstyle="butt",
        zorder=2
    )

def plot_combined_temporal_trend(
    combined_table: pd.DataFrame,
    output_path: Path,
    histogram_alpha: float = 0.25,
    show_shadow_trends: bool = True,
    shadow_alpha: float = 0.19,
    shadow_width: int = 4
) -> None:
    """
    Plot total yearly studies as a semi-transparent histogram/bar chart
    and overlay temporal trend lines for modelling approaches.

    Solid lines show the observed yearly counts.
    Soft shadow lines show the overall linear trend for each modelling approach.
    """

    if not 0 <= histogram_alpha <= 1:
        raise ValueError("histogram_alpha must be between 0 and 1.")

    if not 0 <= shadow_alpha <= 1:
        raise ValueError("shadow_alpha must be between 0 and 1.")

    plt.figure(figsize=(12, 7))

    # --------------------------------------------------
    # Background histogram/bar chart: total studies per year
    # --------------------------------------------------
    plt.bar(
        combined_table["Year"],
        combined_table["Total studies"],
        alpha=histogram_alpha,
        label="Total studies per year",
        zorder=1
    )

    # --------------------------------------------------
    # Observed yearly trend lines
    # --------------------------------------------------
    statistical_line, = plt.plot(
        combined_table["Year"],
        combined_table["Statistical Approach"],
        marker="o",
        linewidth=2,
        label="Statistical Approach",
        zorder=3
    )

    classical_ml_line, = plt.plot(
        combined_table["Year"],
        combined_table["Classical ML"],
        marker="o",
        linewidth=2,
        label="Classical ML",
        zorder=3
    )

    deep_learning_line, = plt.plot(
        combined_table["Year"],
        combined_table["Deep Learning"],
        marker="o",
        linewidth=2,
        label="Deep Learning",
        zorder=3
    )

    multimodal_line, = plt.plot(
        combined_table["Year"],
        combined_table["Multimodal/Hybrid"],
        marker="o",
        linewidth=2,
        label="Multimodal/Hybrid",
        zorder=3
    )

    # --------------------------------------------------
    # Soft shadow trend lines
    # --------------------------------------------------
    if show_shadow_trends:
        trend_columns = [
            ("Statistical Approach", statistical_line),
            ("Classical ML", classical_ml_line),
            ("Deep Learning", deep_learning_line),
            ("Multimodal/Hybrid", multimodal_line),
        ]

        x = combined_table["Year"].to_numpy()

        for column_name, original_line in trend_columns:
            y = combined_table[column_name].to_numpy()

            if len(x) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                trend_values = slope * x + intercept

                # Counts cannot be negative, so clip visual trend at zero.
                trend_values = np.clip(trend_values, a_min=0, a_max=None)

                plt.plot(
                    x,
                    trend_values,
                    linestyle="-",
                    linewidth=shadow_width,
                    alpha=shadow_alpha,
                    color=original_line.get_color(),
                    solid_capstyle="round",
                    zorder=2
                )

    # --------------------------------------------------
    # Labels and formatting
    # --------------------------------------------------
    plt.xlabel("Publication year")
    plt.ylabel("Number of studies")
    plt.title("Temporal trend in modelling approaches over publication years")

    # No rotation for year labels
    plt.xticks(combined_table["Year"], rotation=0)

    # Light horizontal and vertical grid
    plt.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    # Keep grid behind bars, shadows, and lines
    plt.gca().set_axisbelow(True)

    plt.legend(ncol=2)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Combined temporal trend plot saved to: {output_path.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """
    Run the temporal trend workflow.
    """
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_FILE)
    df = clean_data(df)

    trend_table = create_temporal_trend_table(df)
    year_histogram_table = create_year_histogram_table(df)

    combined_table = combine_trend_and_histogram_tables(
        trend_table=trend_table,
        year_histogram_table=year_histogram_table
    )

    save_trend_table(combined_table, OUTPUT_TABLE)

    plot_combined_temporal_trend(
        combined_table=combined_table,
        output_path=OUTPUT_COMBINED_PLOT,
        histogram_alpha=HISTOGRAM_ALPHA
    )

    print("\nTemporal trend analysis completed successfully.")


if __name__ == "__main__":
    main()