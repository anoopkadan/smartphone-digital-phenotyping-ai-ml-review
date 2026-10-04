# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:33:12 2026

@author: ak7u24
"""

# -*- coding: utf-8 -*-
"""
Create two separate plots for ML model use across studies.

Input:
    input/ml_prediction_model_dev.xlsx

Each row represents one study. Each ML model column contains TRUE/FALSE values
indicating whether that model was used in that study.

Outputs:
    output/ml_model_upset_only_plot.png
    output/ml_model_upset_only_plot.pdf
    output/ml_model_upset_only_plot.svg
    output/ml_model_individual_use_plot.png
    output/ml_model_individual_use_plot.pdf
    output/ml_model_individual_use_plot.svg
    output/ml_model_individual_usage.csv
    output/ml_model_combination_usage.csv

Plots produced:
    1. UpSet-style plot:
       - Top bars: number of studies using each exact model-combination pattern
       - Dot matrix: models included in each combination

    2. Individual model-use plot:
       - Horizontal bars: total use of each individual ML model, regardless of combination
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

INPUT_FILE = PROJECT_ROOT / "input" / "ml_prediction_model_dev.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_UPSET_PNG = OUTPUT_DIR / "ml_model_upset_only_plot.png"
OUTPUT_UPSET_PDF = OUTPUT_DIR / "ml_model_upset_only_plot.pdf"
OUTPUT_UPSET_SVG = OUTPUT_DIR / "ml_model_upset_only_plot.svg"

OUTPUT_INDIVIDUAL_PNG = OUTPUT_DIR / "ml_model_individual_use_plot.png"
OUTPUT_INDIVIDUAL_PDF = OUTPUT_DIR / "ml_model_individual_use_plot.pdf"
OUTPUT_INDIVIDUAL_SVG = OUTPUT_DIR / "ml_model_individual_use_plot.svg"

OUTPUT_INDIVIDUAL_USAGE = OUTPUT_DIR / "ml_model_individual_usage.csv"
OUTPUT_COMBINATION_USAGE = OUTPUT_DIR / "ml_model_combination_usage.csv"


# --------------------------------------------------
# Study and model-column configuration
# --------------------------------------------------

STUDY_ID_COLUMN = "Study Reference"

# Column name, abbreviation, full display name
MODEL_COLUMNS = [
    ("linear_regression", "LiR", "Linear regression"),
    ("logistic_regression", "LoR", "Logistic regression"),
    ("ridge_regression", "Ridge", "Ridge regression"),
    ("lasso_regression", "Lasso", "Lasso regression"),
    ("elastic_net_regression", "EN", "Elastic net regression"),
    ("decision_tree", "DT", "Decision tree"),
    ("random_forest", "RF", "Random forest"),
    ("gradient_boosting_machines", "GBM", "Gradient boosting machines"),
    ("XGBoost", "XGB", "XGBoost"),
    ("lightGBM", "LGBM", "LightGBM"),
    ("CatBoost", "CB", "CatBoost"),
    ("support_vector_machines", "SVM", "Support vector machines"),
    ("K_nearest_neighbors", "KNN", "K-nearest neighbors"),
    ("neural_networks", "ANN", "Neural networks"),
    ("gaussian_mixture_models", "GMM", "Gaussian mixture models"),
    ("naïve_bayes", "NB", "Naïve Bayes"),
    ("hidden_markov_models", "HMM", "Hidden Markov models"),
]

# If a model has zero use, remove it from the plots to improve readability.
DROP_ZERO_USE_MODELS_FROM_PLOT = True

# Maximum exact model combinations to display in the UpSet matrix.
# The full combination table is still saved to CSV.
TOP_N_COMBINATIONS = 60

# Denominator for individual-model percentages:
#   "all_studies" = all rows/studies in the Excel file
#   "studies_with_any_model" = only studies with at least one model marked TRUE
PERCENT_DENOMINATOR_MODE = "studies_with_any_model"

# If True, exclude studies that did not use any of the listed ML models from the combination plot.
EXCLUDE_NO_MODEL_ROWS_FROM_COMBINATIONS = True

OUTPUT_DPI = 300


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
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    return text


def cell_to_bool(value) -> bool:
    """
    Convert Excel TRUE/FALSE-like values to Boolean.
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    key = str(value).strip().lower()

    true_values = {
        "true", "t", "yes", "y", "1", "used", "use", "present", "included"
    }

    false_values = {
        "false", "f", "no", "n", "0", "not used", "absent",
        "na", "n/a", "nan", "none", "not reported", "unclear", ""
    }

    if key in true_values:
        return True

    if key in false_values:
        return False

    return False


def load_and_prepare_data(file_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Load Excel data, validate model columns, and convert model cells to Boolean."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    # Build a robust column lookup so that minor Unicode/spacing differences can be handled.
    normalised_to_actual = {normalise_text(col): col for col in df.columns}

    if STUDY_ID_COLUMN not in df.columns:
        df[STUDY_ID_COLUMN] = np.arange(1, len(df) + 1)

    model_info = []
    missing_columns = []

    for raw_column, abbreviation, full_name in MODEL_COLUMNS:
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
            "full_name": full_name,
        })

    if missing_columns:
        raise KeyError(
            f"Missing model columns: {missing_columns}\n"
            f"Available columns after cleaning: {list(df.columns)}"
        )

    df = df.copy()

    for info in model_info:
        df[info["actual_column"]] = df[info["actual_column"]].apply(cell_to_bool)

    return df, model_info


def create_model_matrix(df: pd.DataFrame, model_info: list[dict]) -> pd.DataFrame:
    """Create a Boolean matrix with model abbreviations as columns."""
    matrix = pd.DataFrame(index=df.index)

    for info in model_info:
        matrix[info["abbreviation"]] = df[info["actual_column"]].astype(bool)

    return matrix


def create_individual_usage_table(
    df: pd.DataFrame,
    matrix: pd.DataFrame,
    model_info: list[dict],
) -> pd.DataFrame:
    """Create individual model-use count and percentage table."""
    studies_with_any_model = matrix.any(axis=1)

    if PERCENT_DENOMINATOR_MODE == "all_studies":
        denominator = len(df)
    elif PERCENT_DENOMINATOR_MODE == "studies_with_any_model":
        denominator = int(studies_with_any_model.sum())
    else:
        raise ValueError(
            "PERCENT_DENOMINATOR_MODE must be 'all_studies' or 'studies_with_any_model'."
        )

    full_name_by_abbrev = {
        info["abbreviation"]: info["full_name"]
        for info in model_info
    }

    records = []

    for abbreviation in matrix.columns:
        count = int(matrix[abbreviation].sum())
        percentage = round(count / denominator * 100, 1) if denominator else 0.0

        records.append({
            "Model abbreviation": abbreviation,
            "Model full name": full_name_by_abbrev[abbreviation],
            "Number of studies": count,
            "Percentage": percentage,
            "Percentage denominator": denominator,
        })

    table = pd.DataFrame(records)
    table = table.sort_values(
        ["Number of studies", "Model abbreviation"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return table


def create_combination_usage_table(
    df: pd.DataFrame,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Create exact model-combination count table."""
    records = []

    for idx, row in matrix.iterrows():
        used_models = tuple(matrix.columns[row.to_numpy()].tolist())

        if EXCLUDE_NO_MODEL_ROWS_FROM_COMBINATIONS and len(used_models) == 0:
            continue

        combination_label = " + ".join(used_models) if used_models else "No listed model"

        records.append({
            "Study Reference": df.loc[idx, STUDY_ID_COLUMN],
            "Model combination": combination_label,
            "Models tuple": used_models,
            "Number of models": len(used_models),
        })

    long_table = pd.DataFrame(records)

    if long_table.empty:
        return pd.DataFrame(
            columns=[
                "Combination ID",
                "Model combination",
                "Number of studies",
                "Number of models",
                "Models tuple",
            ]
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

    summary = summary[
        [
            "Combination ID",
            "Model combination",
            "Number of studies",
            "Number of models",
            "Models tuple",
        ]
    ]

    return summary


def filter_tables_for_plot(
    individual_table: pd.DataFrame,
    combination_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select nonzero models and top-N model combinations for plotting."""
    individual_plot = individual_table.copy()

    if DROP_ZERO_USE_MODELS_FROM_PLOT:
        individual_plot = individual_plot[individual_plot["Number of studies"] > 0].copy()

    combination_plot = combination_table.head(TOP_N_COMBINATIONS).copy()

    return individual_plot, combination_plot


# --------------------------------------------------
# Plot functions
# --------------------------------------------------

def plot_upset_only(
    individual_plot: pd.DataFrame,
    combination_plot: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
) -> None:
    """Draw an UpSet-style plot with top bars and dot matrix only."""
    if individual_plot.empty:
        raise ValueError("No ML models were marked as used. Cannot draw UpSet plot.")

    if combination_plot.empty:
        raise ValueError("No model combinations found. Cannot draw UpSet plot.")

    models_for_y = individual_plot["Model abbreviation"].tolist()
    n_models = len(models_for_y)
    n_combinations = len(combination_plot)

    figure_width = max(8.5, 0.36 * n_combinations + 3.0)
    figure_height = max(5.2, 0.28 * n_models + 2.8)

    fig = plt.figure(figsize=(figure_width, figure_height))
    gs = GridSpec(nrows=2, ncols=1, height_ratios= [2.4, 2.8], hspace=0.02)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[1, 0], sharex=ax_top)

    x_positions = np.arange(n_combinations)
    
    x_min = -0.45
    x_max = n_combinations - 0.55
    
    top_counts = combination_plot["Number of studies"].to_numpy()

    ax_top.bar(
        x_positions,
        top_counts,
        width=0.62,
        edgecolor="black",
        linewidth=0.5,
    )

    max_top_count = max(int(top_counts.max()), 1)

    for x, count in zip(x_positions, top_counts):
        ax_top.text(
            x,
            count + max_top_count * 0.03,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    
    ax_top.set_xlim(x_min, x_max)
    ax_matrix.set_xlim(x_min, x_max)
    ax_top.margins(x=0)
    ax_matrix.margins(x=0)

    ax_top.set_ylabel("Studies")
    ax_top.set_title("Model-combination patterns")
    ax_top.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax_top.set_axisbelow(True)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylim(0, max_top_count * 1.28)

    model_y = {model: n_models - 1 - i for i, model in enumerate(models_for_y)}

    for x in x_positions:
        ax_matrix.scatter(
            [x] * n_models,
            [model_y[m] for m in models_for_y],
            s=20,
            alpha=0.18,
        )

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
    ax_matrix.set_xlabel("Exact model-combination pattern")
    ax_matrix.set_ylim(-0.8, n_models - 0.2)
    ax_matrix.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.25)
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["left"].set_visible(False)

    #fig.text(
        #0.01,
        #0.01,
        #"Top bars show exact model-combination counts. Dots indicate which models are included in each combination.",
        #fontsize=8,
        #ha="left",
    #)

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", pad_inches=0.02, facecolor="white")

    plt.show()
    print(f"UpSet-style plot saved to: {output_png.resolve()}")


def plot_individual_model_use(
    individual_plot: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path,
) -> None:
    """Draw a separate horizontal bar chart for individual model use."""
    if individual_plot.empty:
        raise ValueError("No ML models were marked as used. Cannot draw individual-use plot.")

    plot_df = individual_plot.copy().reset_index(drop=True)
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    figure_width = 8.8
    figure_height = max(6.0, 0.38 * len(plot_df) + 1.8)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    bars = ax.barh(
        plot_df["Model abbreviation"],
        plot_df["Number of studies"],
        height=0.62,
        edgecolor="black",
        linewidth=0.5,
    )

    max_count = max(int(plot_df["Number of studies"].max()), 1)
    ax.set_xlim(0, max_count * 1.28)

    for bar, count, pct in zip(
        bars,
        plot_df["Number of studies"],
        plot_df["Percentage"],
    ):
        ax.text(
            bar.get_width() + max_count * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
        )

    ax.set_xlabel("Number of studies")
    ax.set_ylabel("ML model")
    ax.set_title("Individual ML model use across studies")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.01,
        0.01,
        "Bars show the total number and percentage of studies using each individual ML model, regardless of combination.",
        fontsize=8,
        ha="left",
    )

    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()
    print(f"Individual model-use plot saved to: {output_png.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """Run model-use analysis and draw two separate plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading input file: {INPUT_FILE.resolve()}")

    df, model_info = load_and_prepare_data(INPUT_FILE)
    matrix = create_model_matrix(df, model_info)

    individual_table = create_individual_usage_table(
        df=df,
        matrix=matrix,
        model_info=model_info,
    )

    combination_table = create_combination_usage_table(
        df=df,
        matrix=matrix,
    )

    individual_plot, combination_plot = filter_tables_for_plot(
        individual_table=individual_table,
        combination_table=combination_table,
    )

    individual_table.to_csv(OUTPUT_INDIVIDUAL_USAGE, index=False)
    combination_table.to_csv(OUTPUT_COMBINATION_USAGE, index=False)

    plot_upset_only(
        individual_plot=individual_plot,
        combination_plot=combination_plot,
        output_png=OUTPUT_UPSET_PNG,
        output_pdf=OUTPUT_UPSET_PDF,
        output_svg=OUTPUT_UPSET_SVG,
    )

    plot_individual_model_use(
        individual_plot=individual_plot,
        output_png=OUTPUT_INDIVIDUAL_PNG,
        output_pdf=OUTPUT_INDIVIDUAL_PDF,
        output_svg=OUTPUT_INDIVIDUAL_SVG,
    )

    print(f"Individual model-use table saved to: {OUTPUT_INDIVIDUAL_USAGE.resolve()}")
    print(f"Model-combination table saved to: {OUTPUT_COMBINATION_USAGE.resolve()}")
    print("\nML model-use analysis completed successfully.")


if __name__ == "__main__":
    main()
