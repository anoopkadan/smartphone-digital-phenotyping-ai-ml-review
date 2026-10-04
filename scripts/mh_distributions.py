# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:59:04 2026

@author: ak7u24
"""
"""
Plot mental health condition distribution of included studies.

Input:
    input/mh_data.xlsx

Required input column:
    mh_domain

Each row is one included study.
For multi-condition studies, conditions are separated by semicolon (;).

Standard categories:
    Depression
    Anxiety
    Stress
    Bipolar disorder
    Schizophrenia
    Suicide risk
    ADHD
    Others

Outputs:
    output/mental_health_condition_distribution_table.xlsx
    output/mental_health_condition_distribution_plot.png
    output/mental_health_condition_distribution_plot.pdf
    output/mental_health_condition_distribution_plot.svg
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

INPUT_FILE = PROJECT_ROOT / "input" / "mh_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "mental_health_condition_distribution_table.xlsx"
OUTPUT_PLOT_PNG = OUTPUT_DIR / "mental_health_condition_distribution_plot.png"
OUTPUT_PLOT_PDF = OUTPUT_DIR / "mental_health_condition_distribution_plot.pdf"
OUTPUT_PLOT_SVG = OUTPUT_DIR / "mental_health_condition_distribution_plot.svg"


# --------------------------------------------------
# Column and study configuration
# --------------------------------------------------

MH_DOMAIN_COLUMN = "mh_domain"
STUDY_REFERENCE_COLUMN = "Study Reference"

TOTAL_INCLUDED_STUDIES = 117

STANDARD_CONDITIONS = [
    "Depression",
    "Anxiety",
    "Stress",
    "Bipolar disorder",
    "Schizophrenia",
    "Suicide risk",
    "ADHD",
    "Others",
]


# --------------------------------------------------
# Plot configuration
# --------------------------------------------------

SORT_BY_COUNT = True
KEEP_OTHERS_LAST = True

FIGURE_WIDTH = 8.5
FIGURE_HEIGHT = 5.2
OUTPUT_DPI = 300


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
# Text cleaning
# --------------------------------------------------

def normalise_text(value: str) -> str:
    """
    Convert text into a simple lowercase matching form.
    """
    text = str(value).strip()
    text = text.replace("’", "'").replace("‘", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_mh_domain_cell(value: str) -> list[str]:
    """
    Split the mh_domain cell into raw condition tokens.

    Main separator is semicolon (;), but this function also handles obvious
    combined labels such as 'Depression and anxiety'.
    """
    text = str(value).strip()

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Preserve semicolon as primary separator.
    tokens = [token.strip() for token in text.split(";")]

    tokens = [token for token in tokens if token]

    return tokens


# --------------------------------------------------
# Mapping raw labels to standard categories
# --------------------------------------------------

def map_token_to_standard_conditions(token: str) -> list[str]:
    """
    Map a raw mental-health token to one or more standard categories.

    Any mental-health problem not belonging to the seven named categories
    is mapped to 'Others'.
    """
    key = normalise_text(token)

    if key in {"", "nan", "none", "not mentioned", "not reported"}:
        return []

    mapped_conditions = set()

    # Depression
    if re.search(
        r"\b(depression|depressive|major depressive disorder|mdd)\b",
        key
    ):
        mapped_conditions.add("Depression")

    # Anxiety
    if re.search(
        r"\b(anxiety|generalized anxiety disorder|gad|social anxiety disorder|sad|panic disorder)\b",
        key
    ):
        mapped_conditions.add("Anxiety")

    # Stress
    if re.search(r"\b(stress|stressed|perceived stress)\b", key):
        mapped_conditions.add("Stress")

    # Bipolar disorder
    if re.search(
        r"\b(bipolar|bipolar disorder|bipolar disorders|bipolar affective disorder|mania|manic)\b",
        key
    ):
        mapped_conditions.add("Bipolar disorder")

    # Schizophrenia
    if re.search(
        r"\b(schizophrenia|schizophrenic)\b",
        key
    ):
        mapped_conditions.add("Schizophrenia")

    # Suicide risk
    if re.search(
        r"\b(suicide|suicidal|suicidality|self harm|self harming|self injury|self injurious)\b",
        key
    ):
        mapped_conditions.add("Suicide risk")

    # ADHD
    if re.search(
        r"\b(adhd|attention deficit|attention deficit hyperactivity disorder)\b",
        key
    ):
        mapped_conditions.add("ADHD")

    # Explicit or residual other mental-health domains
    if re.search(
        r"\b(other|others|insomnia|sleep|loneliness|mood|wellbeing|well being)\b",
        key
    ):
        mapped_conditions.add("Others")

    # If no standard category matched but there is a non-empty mental-health label,
    # classify it as Others.
    if not mapped_conditions:
        mapped_conditions.add("Others")

    return sorted(mapped_conditions, key=STANDARD_CONDITIONS.index)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the Excel input file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    if MH_DOMAIN_COLUMN not in df.columns:
        raise KeyError(
            f"Required column '{MH_DOMAIN_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


# --------------------------------------------------
# Create mental-health condition count table
# --------------------------------------------------

def create_mh_condition_table(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a condition-level count table.

    Counting rule:
    - Each row is one study.
    - Multi-condition studies are counted once for each relevant condition.
    - A study is counted only once per condition, even if the same condition
      appears multiple times in that row.
    - Percentages use TOTAL_INCLUDED_STUDIES as the denominator.
    """
    mapped_records = []
    raw_mapping_records = []
    excluded_records = []

    for study_row_id, raw_value in df[MH_DOMAIN_COLUMN].items():
        study_reference = (
            df.loc[study_row_id, STUDY_REFERENCE_COLUMN]
            if STUDY_REFERENCE_COLUMN in df.columns
            else study_row_id
        )

        if pd.isna(raw_value):
            excluded_records.append({
                "study_row_id": study_row_id,
                "study_reference": study_reference,
                "raw_mh_domain": raw_value,
                "reason": "missing mh_domain value"
            })
            continue

        raw_tokens = split_mh_domain_cell(raw_value)

        study_conditions = set()

        for token in raw_tokens:
            mapped_conditions = map_token_to_standard_conditions(token)

            if not mapped_conditions:
                excluded_records.append({
                    "study_row_id": study_row_id,
                    "study_reference": study_reference,
                    "raw_mh_domain": raw_value,
                    "raw_token": token,
                    "reason": "empty or invalid token"
                })
                continue

            for condition in mapped_conditions:
                study_conditions.add(condition)

                raw_mapping_records.append({
                    "study_row_id": study_row_id,
                    "study_reference": study_reference,
                    "raw_mh_domain": raw_value,
                    "raw_token": token,
                    "mapped_condition": condition
                })

        for condition in study_conditions:
            mapped_records.append({
                "study_row_id": study_row_id,
                "study_reference": study_reference,
                "MH Condition": condition
            })

    mapped_df = pd.DataFrame(mapped_records)
    raw_mapping_df = pd.DataFrame(raw_mapping_records)
    excluded_df = pd.DataFrame(excluded_records)

    if mapped_df.empty:
        raise ValueError("No mental health conditions were mapped from the input data.")

    # Avoid double-counting the same study under the same condition
    mapped_df = mapped_df.drop_duplicates(
        subset=["study_row_id", "MH Condition"]
    )

    condition_counts = (
        mapped_df
        .groupby("MH Condition", as_index=False)
        .agg({"study_row_id": "nunique"})
        .rename(columns={"study_row_id": "Number of studies"})
    )

    # Ensure all standard conditions appear, even if count = 0
    all_conditions = pd.DataFrame({"MH Condition": STANDARD_CONDITIONS})

    condition_counts = all_conditions.merge(
        condition_counts,
        on="MH Condition",
        how="left"
    )

    condition_counts["Number of studies"] = (
        condition_counts["Number of studies"]
        .fillna(0)
        .astype(int)
    )

    condition_counts["Percentage of included studies"] = (
        condition_counts["Number of studies"] / TOTAL_INCLUDED_STUDIES * 100
    ).round(1)

    if SORT_BY_COUNT:
        condition_counts = condition_counts.sort_values(
            ["Number of studies", "MH Condition"],
            ascending=[False, True]
        )

    if KEEP_OTHERS_LAST:
        others_row = condition_counts[
            condition_counts["MH Condition"] == "Others"
        ]

        non_others = condition_counts[
            condition_counts["MH Condition"] != "Others"
        ]

        condition_counts = pd.concat(
            [non_others, others_row],
            ignore_index=True
        )

    return condition_counts, raw_mapping_df, excluded_df


# --------------------------------------------------
# Save Excel table
# --------------------------------------------------

def save_mh_condition_table(
    condition_counts: pd.DataFrame,
    raw_mapping_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save the mental-health condition distribution table to Excel.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        condition_counts.to_excel(
            writer,
            sheet_name="MH condition counts",
            index=False
        )

        raw_mapping_df.to_excel(
            writer,
            sheet_name="Raw label mapping",
            index=False
        )

        if not excluded_df.empty:
            excluded_df.to_excel(
                writer,
                sheet_name="Excluded values",
                index=False
            )

    print(f"Mental health condition table saved to: {output_path.resolve()}")


# --------------------------------------------------
# Plot horizontal bar chart
# --------------------------------------------------

def plot_mh_condition_distribution(
    condition_counts: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path
) -> None:
    """
    Plot a horizontal bar chart of mental health condition counts.
    """
    plot_df = condition_counts.copy()

    # Reverse order so the largest category appears at the top
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    bars = ax.barh(
        plot_df["MH Condition"],
        plot_df["Number of studies"],
        height=0.65,
        edgecolor="black",
        linewidth=0.6
    )

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("")
    ax.set_title("Distribution of mental health conditions studied")

    ax.grid(
        True,
        which="major",
        axis="x",
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    ax.set_axisbelow(True)

    # Remove unnecessary spines for a cleaner publication-style figure
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_count = int(plot_df["Number of studies"].max())

    if max_count == 0:
        max_count = 1

    ax.set_xlim(0, max_count * 1.20)

    # Add count and percentage labels at the end of each bar
    for bar, count, percentage in zip(
        bars,
        plot_df["Number of studies"],
        plot_df["Percentage of included studies"]
    ):
        label = f"{count} ({percentage:.1f}%)"

        ax.text(
            bar.get_width() + max_count * 0.02,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9
        )

    fig.text(
        0.01,
        0.01,
        (
            "Note: Studies addressing multiple conditions were counted once "
            "for each relevant condition. Percentages use 117 included studies "
            "as the denominator."
        ),
        fontsize=8,
        ha="left"
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()

    print(f"Mental health condition plot saved to: {output_png.resolve()}")
    print(f"PDF version saved to: {output_pdf.resolve()}")
    print(f"SVG version saved to: {output_svg.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """
    Run the full mental-health condition distribution workflow.
    """
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_FILE)

    condition_counts, raw_mapping_df, excluded_df = create_mh_condition_table(df)

    save_mh_condition_table(
        condition_counts=condition_counts,
        raw_mapping_df=raw_mapping_df,
        excluded_df=excluded_df,
        output_path=OUTPUT_TABLE
    )

    plot_mh_condition_distribution(
        condition_counts=condition_counts,
        output_png=OUTPUT_PLOT_PNG,
        output_pdf=OUTPUT_PLOT_PDF,
        output_svg=OUTPUT_PLOT_SVG
    )

    print("\nMental health condition distribution analysis completed successfully.")


if __name__ == "__main__":
    main()
