# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:36:31 2026

@author: ak7u24
"""

"""
Plot smartphone sensor/data-source use and condition-sensor heatmap.

Input:
    input/phenotypes.xlsx

Required columns:
    mh_domain

Sensor/data-source columns:
    call_logs, call_content, SMS_logs, SMS_content, notifications,
    app_usage, gps, screen_time, screen_lock_unlock, bluetooth, wifi,
    accelerometer, gyroscope, ambient_air, ambient_light, ambient_sound,
    keyboard, microphone, camera, battery

Outputs:
    output/sensor_data_source_summary.xlsx
    output/sensor_data_source_bar_plot.png
    output/sensor_data_source_bar_plot.pdf
    output/sensor_data_source_bar_plot.svg
    output/condition_sensor_heatmap.png
    output/condition_sensor_heatmap.pdf
    output/condition_sensor_heatmap.svg
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

INPUT_FILE = PROJECT_ROOT / "input" / "sensor_data_sourc_phenotypes.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "sensor_data_source_summary.xlsx"

OUTPUT_BAR_PNG = OUTPUT_DIR / "sensor_data_source_bar_plot.png"
OUTPUT_BAR_PDF = OUTPUT_DIR / "sensor_data_source_bar_plot.pdf"
OUTPUT_BAR_SVG = OUTPUT_DIR / "sensor_data_source_bar_plot.svg"

OUTPUT_HEATMAP_PNG = OUTPUT_DIR / "condition_sensor_heatmap.png"
OUTPUT_HEATMAP_PDF = OUTPUT_DIR / "condition_sensor_heatmap.pdf"
OUTPUT_HEATMAP_SVG = OUTPUT_DIR / "condition_sensor_heatmap.svg"


# --------------------------------------------------
# Column configuration
# --------------------------------------------------

MH_DOMAIN_COLUMN = "mh_domain"

# If each row is one study, keep this as None.
# If each row is one participant and you have a study identifier column,
# set this to that column name, for example:
# STUDY_ID_COLUMN = "Study Reference"
# STUDY_ID_COLUMN = "study_id"
STUDY_ID_COLUMN = None


SENSOR_COLUMNS = [
    "call_logs",
    "call_content",
    "SMS_logs",
    "SMS_content",
    "notifications",
    "app_usage",
    "gps",
    "screen_time",
    "screen_lock_unlock",
    "bluetooth",
    "wifi",
    "accelerometer",
    "gyroscope",
    "ambient_air",
    "ambient_light",
    "ambient_sound",
    "keyboard",
    "microphone",
    "camera",
    "battery_charging",
]


SENSOR_DISPLAY_LABELS = {
    "call_logs": "Call logs",
    "call_content": "Call content",
    "SMS_logs": "SMS logs",
    "SMS_content": "SMS content",
    "notifications": "Notifications",
    "app_usage": "App usage",
    "gps": "GPS/location",
    "screen_time": "Screen time",
    "screen_lock_unlock": "Screen lock/unlock",
    "bluetooth": "Bluetooth",
    "wifi": "Wi-Fi",
    "accelerometer": "Accelerometer",
    "gyroscope": "Gyroscope",
    "ambient_air": "Ambient air",
    "ambient_light": "Ambient light",
    "ambient_sound": "Ambient sound",
    "keyboard": "Keyboard",
    "microphone": "Microphone",
    "camera": "Camera",
    "battery_charging": "Battery/charging",
}


# --------------------------------------------------
# Mental health condition options
# --------------------------------------------------
# Set any condition to False if you want to discard it from the heatmap.

INCLUDE_CONDITION = {
    "Depression": True,
    "Anxiety": True,
    "Stress": True,
    "Bipolar disorder": True,
    "Schizophrenia": True,
    "Others": True,
}

CONDITION_ORDER = [
    "Depression",
    "Anxiety",
    "Stress",
    "Bipolar disorder",
    "Schizophrenia",
    "Others",
]


# --------------------------------------------------
# Plot options
# --------------------------------------------------

TOTAL_INCLUDED_STUDIES = None
# If None, denominator is calculated from the included data.
# If you want to force the denominator to 117, set:
# TOTAL_INCLUDED_STUDIES = 117

OUTPUT_DPI = 300

SHOW_PERCENT_LABELS = True
SHOW_VALUE_LABELS_ON_HEATMAP = True

SORT_SENSOR_BAR_BY_COUNT = True

BAR_FIGURE_WIDTH = 9.5
BAR_FIGURE_HEIGHT = 7.5

HEATMAP_FIGURE_WIDTH = 12.0
HEATMAP_FIGURE_HEIGHT = 5.5


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
# Utility functions
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


def cell_to_bool(value) -> bool:
    """
    Convert TRUE/FALSE-like Excel values to Boolean.

    Accepts:
        True, False
        "true", "false"
        "yes", "no"
        "1", "0"
        "used", "not used"
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    key = normalise_text(value)

    true_values = {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "used",
        "use",
        "included",
        "present",
    }

    false_values = {
        "false",
        "f",
        "no",
        "n",
        "0",
        "not used",
        "not included",
        "absent",
        "none",
        "na",
        "n a",
        "not reported",
        "not mentioned",
        "unclear",
        "not clear",
    }

    if key in true_values:
        return True

    if key in false_values:
        return False

    return False


def split_semicolon_cell(value) -> list[str]:
    """
    Split semicolon-separated cells.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    tokens = [token.strip() for token in text.split(";")]
    tokens = [token for token in tokens if token]

    return tokens


# --------------------------------------------------
# Mental-health condition mapping
# --------------------------------------------------

def map_mh_token_to_condition(token: str) -> str:
    """
    Map raw mh_domain label to one of the standard mental health conditions.

    ADHD and suicide risk are intentionally grouped under 'Others'.
    """
    key = normalise_text(token)

    if key in {
        "",
        "na",
        "n a",
        "nan",
        "none",
        "not reported",
        "not mentioned",
        "not clear",
        "unclear",
    }:
        return "Others"

    if re.search(r"\b(depression|depressive|mdd|major depressive disorder)\b", key):
        return "Depression"

    if re.search(r"\b(anxiety|gad|generalized anxiety|panic|social anxiety)\b", key):
        return "Anxiety"

    if re.search(r"\b(stress|stressed|perceived stress)\b", key):
        return "Stress"

    if re.search(r"\b(bipolar|mania|manic|bd i|bd ii|bdi|bdii)\b", key):
        return "Bipolar disorder"

    if re.search(r"\b(schizophrenia|schizophrenic|psychosis|psychotic)\b", key):
        return "Schizophrenia"

    # ADHD is grouped under Others
    if re.search(r"\b(adhd|attention deficit|attention deficit hyperactivity)\b", key):
        return "Others"

    # Suicide risk is grouped under Others
    if re.search(
        r"\b(suicide|suicidal|suicidality|self harm|self harming|self injury)\b",
        key
    ):
        return "Others"

    return "Others"


def extract_standard_conditions(value) -> list[str]:
    """
    Extract standard mental-health conditions from mh_domain.

    A study with multiple conditions separated by semicolon is counted once
    for each relevant condition.
    """
    tokens = split_semicolon_cell(value)

    if not tokens:
        return ["Others"]

    conditions = {
        map_mh_token_to_condition(token)
        for token in tokens
    }

    selected_conditions = [
        condition
        for condition in CONDITION_ORDER
        if condition in conditions and INCLUDE_CONDITION.get(condition, False)
    ]

    return selected_conditions


# --------------------------------------------------
# Data loading and cleaning
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the Excel file and validate columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    required_columns = [MH_DOMAIN_COLUMN] + SENSOR_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
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

    df = df.copy()

    if STUDY_ID_COLUMN is None:
        df["study_id_for_counting"] = np.arange(1, len(df) + 1)
    else:
        df["study_id_for_counting"] = df[STUDY_ID_COLUMN]

    for sensor in SENSOR_COLUMNS:
        df[sensor] = df[sensor].apply(cell_to_bool)

    df["standard_conditions"] = df[MH_DOMAIN_COLUMN].apply(
        extract_standard_conditions
    )

    return df


# --------------------------------------------------
# Sensor count table
# --------------------------------------------------

def create_sensor_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number and percentage of studies using each sensor/data source.
    """
    denominator = (
        TOTAL_INCLUDED_STUDIES
        if TOTAL_INCLUDED_STUDIES is not None
        else df["study_id_for_counting"].nunique()
    )

    records = []

    for sensor in SENSOR_COLUMNS:
        sensor_df = df[df[sensor]].copy()

        number_of_studies = sensor_df["study_id_for_counting"].nunique()

        records.append({
            "Sensor column": sensor,
            "Sensor/Data Source": SENSOR_DISPLAY_LABELS.get(sensor, sensor),
            "Number of studies": int(number_of_studies),
            "Percentage of included studies": round(
                number_of_studies / denominator * 100,
                1
            ),
        })

    sensor_count_table = pd.DataFrame(records)

    if SORT_SENSOR_BAR_BY_COUNT:
        sensor_count_table = sensor_count_table.sort_values(
            ["Number of studies", "Sensor/Data Source"],
            ascending=[False, True],
        )

    return sensor_count_table


# --------------------------------------------------
# Condition-sensor heatmap table
# --------------------------------------------------

def create_condition_sensor_heatmap_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a condition × sensor count matrix.

    Counting rule:
    - A study with multiple conditions is counted once for each condition.
    - A study using a sensor contributes 1 to that condition-sensor cell.
    - If STUDY_ID_COLUMN is provided, unique studies are counted.
    """
    records = []

    for _, row in df.iterrows():
        study_id = row["study_id_for_counting"]
        conditions = row["standard_conditions"]

        for condition in conditions:
            if not INCLUDE_CONDITION.get(condition, False):
                continue

            for sensor in SENSOR_COLUMNS:
                if row[sensor]:
                    records.append({
                        "study_id_for_counting": study_id,
                        "Mental health condition": condition,
                        "Sensor/Data Source": SENSOR_DISPLAY_LABELS.get(sensor, sensor),
                    })

    exploded = pd.DataFrame(records)

    selected_conditions = [
        condition
        for condition in CONDITION_ORDER
        if INCLUDE_CONDITION.get(condition, False)
    ]

    sensor_labels = [
        SENSOR_DISPLAY_LABELS.get(sensor, sensor)
        for sensor in SENSOR_COLUMNS
    ]

    if exploded.empty:
        return pd.DataFrame(
            0,
            index=selected_conditions,
            columns=sensor_labels,
        )

    exploded = exploded.drop_duplicates(
        subset=[
            "study_id_for_counting",
            "Mental health condition",
            "Sensor/Data Source",
        ]
    )

    heatmap_table = (
        exploded
        .groupby(
            ["Mental health condition", "Sensor/Data Source"],
            as_index=False
        )
        .agg(**{
            "Number of studies": ("study_id_for_counting", "nunique")
        })
        .pivot(
            index="Mental health condition",
            columns="Sensor/Data Source",
            values="Number of studies"
        )
        .fillna(0)
        .astype(int)
    )

    heatmap_table = heatmap_table.reindex(
        index=selected_conditions,
        columns=sensor_labels,
        fill_value=0,
    )

    # Remove all-zero condition rows if they were selected but absent
    heatmap_table = heatmap_table.loc[
        heatmap_table.sum(axis=1) > 0
    ]

    return heatmap_table


# --------------------------------------------------
# Save tables
# --------------------------------------------------

def save_summary_tables(
    sensor_count_table: pd.DataFrame,
    heatmap_table: pd.DataFrame,
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save summary tables to Excel.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    condition_mapping = df[
        [
            "study_id_for_counting",
            MH_DOMAIN_COLUMN,
            "standard_conditions",
        ]
    ].copy()

    condition_mapping["standard_conditions"] = condition_mapping[
        "standard_conditions"
    ].apply(lambda values: "; ".join(values))

    cleaned_sensor_data = df[
        ["study_id_for_counting", MH_DOMAIN_COLUMN] + SENSOR_COLUMNS
    ].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sensor_count_table.to_excel(
            writer,
            sheet_name="Sensor counts",
            index=False,
        )

        heatmap_table.to_excel(
            writer,
            sheet_name="Condition sensor heatmap",
            index=True,
        )

        condition_mapping.to_excel(
            writer,
            sheet_name="Condition mapping",
            index=False,
        )

        cleaned_sensor_data.to_excel(
            writer,
            sheet_name="Cleaned sensor data",
            index=False,
        )

    print(f"Summary tables saved to: {output_path.resolve()}")


# --------------------------------------------------
# Plot 1: Sensor/data-source counts
# --------------------------------------------------

def plot_sensor_counts(
    sensor_count_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path
) -> None:
    """
    Plot number of studies using each sensor/data source.
    """
    plot_df = sensor_count_table.copy()

    # Reverse so the largest value appears at the top
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig_height = max(6.5, 0.36 * len(plot_df))
    fig, ax = plt.subplots(figsize=(BAR_FIGURE_WIDTH, fig_height))

    bars = ax.barh(
        plot_df["Sensor/Data Source"],
        plot_df["Number of studies"],
        height=0.65,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("")
    ax.set_title("Smartphone sensor/data sources used across included studies")

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

    max_count = int(plot_df["Number of studies"].max())

    if max_count == 0:
        max_count = 1

    ax.set_xlim(0, max_count * 1.25)

    if SHOW_PERCENT_LABELS:
        for bar, count, percentage in zip(
            bars,
            plot_df["Number of studies"],
            plot_df["Percentage of included studies"],
        ):
            ax.text(
                bar.get_width() + max_count * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count)} ({percentage:.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
            )

    fig.tight_layout()

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()

    print(f"Sensor count plot saved to: {output_png.resolve()}")


# --------------------------------------------------
# Plot 2: Condition-sensor heatmap
# --------------------------------------------------

def plot_condition_sensor_heatmap(
    heatmap_table: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path
) -> None:
    """
    Plot mental-health condition × sensor heatmap.
    """
    if heatmap_table.empty:
        raise ValueError("Heatmap table is empty. No condition-sensor data to plot.")

    fig_width = max(HEATMAP_FIGURE_WIDTH, 0.55 * heatmap_table.shape[1])
    fig_height = max(HEATMAP_FIGURE_HEIGHT, 0.55 * heatmap_table.shape[0])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    data = heatmap_table.to_numpy()

    image = ax.imshow(
        data,
        aspect="auto",
        cmap="Blues",
    )

    ax.set_xticks(np.arange(heatmap_table.shape[1]))
    ax.set_yticks(np.arange(heatmap_table.shape[0]))

    ax.set_xticklabels(
        heatmap_table.columns,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax.set_yticklabels(heatmap_table.index)

    ax.set_xlabel("Sensor/data source")
    ax.set_ylabel("Mental health condition")
    ax.set_title("Mental health condition–sensor heatmap")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Number of studies")

    if SHOW_VALUE_LABELS_ON_HEATMAP:
        max_value = data.max()

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = int(data[i, j])

                if value == 0:
                    continue

                text_colour = "white" if value > max_value * 0.55 else "black"

                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color=text_colour,
                    fontsize=8,
                )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()

    print(f"Condition-sensor heatmap saved to: {output_png.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """
    Run full analysis and plotting workflow.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {INPUT_FILE.resolve()}")

    df = load_data(INPUT_FILE)

    sensor_count_table = create_sensor_count_table(df)

    heatmap_table = create_condition_sensor_heatmap_table(df)

    save_summary_tables(
        sensor_count_table=sensor_count_table,
        heatmap_table=heatmap_table,
        df=df,
        output_path=OUTPUT_TABLE,
    )

    plot_sensor_counts(
        sensor_count_table=sensor_count_table,
        output_png=OUTPUT_BAR_PNG,
        output_pdf=OUTPUT_BAR_PDF,
        output_svg=OUTPUT_BAR_SVG,
    )

    plot_condition_sensor_heatmap(
        heatmap_table=heatmap_table,
        output_png=OUTPUT_HEATMAP_PNG,
        output_pdf=OUTPUT_HEATMAP_PDF,
        output_svg=OUTPUT_HEATMAP_SVG,
    )

    print("\nSensor/data-source analysis completed successfully.")


if __name__ == "__main__":
    main()