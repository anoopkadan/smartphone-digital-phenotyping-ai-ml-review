# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:02:54 2026

@author: ak7u24
"""

"""
Plot geographical distribution of included studies.

Input:
    inputs/country_data.xlsx

Required input column:
    countries

The 'countries' column may contain one or more countries per study.
For multi-country studies, countries must be separated by semicolon (;).

Outputs:
    output/country_distribution_table.xlsx
    output/country_distribution_map.png
    output/country_distribution_map.pdf
    output/country_distribution_map.svg
"""

from pathlib import Path
import re
import unicodedata
import urllib.request

import pandas as pd
import numpy as np
import geopandas as gpd
import pycountry
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch



# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "country_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "country_distribution_table.xlsx"
OUTPUT_MAP_PNG = OUTPUT_DIR / "country_distribution_map.png"
OUTPUT_MAP_PDF = OUTPUT_DIR / "country_distribution_map.pdf"
OUTPUT_MAP_SVG = OUTPUT_DIR / "country_distribution_map.svg"

COUNTRIES_COLUMN = "countries"

NATURAL_EARTH_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "110m_cultural/ne_110m_admin_0_countries.zip"
)

MAP_CACHE_DIR = OUTPUT_DIR / "_map_cache"
NATURAL_EARTH_ZIP = MAP_CACHE_DIR / "ne_110m_admin_0_countries.zip"


# --------------------------------------------------
# Manual cleaning rules
# --------------------------------------------------

PRE_SPLIT_REPLACEMENTS = {
    r"\bIndia\s+and\s+US\b": "India; United States",
    r"\bIndia\s+and\s+USA\b": "India; United States",
    r"\bPuerto\s*;\s*Rico\b": "Puerto Rico",
}

NON_COUNTRY_KEYS = {
    "",
    "nan",
    "none",
    "not mentioned",
    "country not provided",
    "online crowdsourced",
    "social media",
    "advertisement through social media campaigns on facebook and google social media",
    "participants recruited via reddit social media",
}

COUNTRY_NAME_FIXES = {
    "us": "United States",
    "usa": "United States",
    "u s": "United States",
    "united states of america": "United States",

    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",

    "korea": "South Korea",
    "republic of korea": "South Korea",

    "new york": "United States",
    "washington": "United States",
    "boston ma": "United States",
    "warsaw": "Poland",

    "cape verde": "Cape Verde",
    "cabo verde": "Cape Verde",

    "sao tome and principe": "Sao Tome and Principe",
    "são tomé and príncipe": "Sao Tome and Principe",

    "ivory coast": "Côte d’Ivoire",
    "cote d ivoire": "Côte d’Ivoire",
    "cote divoire": "Côte d’Ivoire",

    "dr congo": "Democratic Republic of the Congo",
    "democratic republic of congo": "Democratic Republic of the Congo",
    "republic of congo": "Republic of the Congo",
}

ISO3_OVERRIDES = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "Taiwan": "TWN",
    "Cape Verde": "CPV",
    "Sao Tome and Principe": "STP",
    "Côte d’Ivoire": "CIV",
    "Democratic Republic of the Congo": "COD",
    "Republic of the Congo": "COG",
    "Puerto Rico": "PRI",
    "Kosovo": "XKX",
}


# --------------------------------------------------
# Text-cleaning functions
# --------------------------------------------------

def normalise_key(value: str) -> str:
    """
    Convert text to a simple matching key.
    """
    text = str(value).strip()
    text = text.replace("’", "'").replace("‘", "'")

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_country_cell(value: str) -> str:
    """
    Prepare a country-cell value before splitting by semicolon.
    """
    text = str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    for pattern, replacement in PRE_SPLIT_REPLACEMENTS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def clean_country_name(value: str) -> str | None:
    """
    Convert a raw country token into a standard country name.

    Returns None when the value is not a valid country field.
    """
    text = str(value).strip()
    text = text.strip(" .,;:")

    key = normalise_key(text)

    if key in NON_COUNTRY_KEYS:
        return None

    if key in COUNTRY_NAME_FIXES:
        return COUNTRY_NAME_FIXES[key]

    return text


def get_iso3(country_name: str) -> str | None:
    """
    Convert country name into ISO3 country code.
    """
    if country_name in ISO3_OVERRIDES:
        return ISO3_OVERRIDES[country_name]

    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None


# --------------------------------------------------
# Data processing
# --------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load the Excel input file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path.resolve()}")

    df = pd.read_excel(file_path)

    if COUNTRIES_COLUMN not in df.columns:
        raise KeyError(
            f"Required column '{COUNTRIES_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


def create_country_count_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create country-level count table.

    Counting rule:
    - Each row is treated as one study.
    - If a study includes multiple countries separated by semicolon (;),
      each country receives one count.
    - The same country is counted only once per study.
    """
    country_records = []
    excluded_records = []

    for study_row_id, raw_value in df[COUNTRIES_COLUMN].items():
        if pd.isna(raw_value):
            excluded_records.append({
                "study_row_id": study_row_id,
                "raw_value": raw_value,
                "token": "",
                "reason": "missing country value"
            })
            continue

        prepared_value = prepare_country_cell(raw_value)
        raw_tokens = prepared_value.split(";")

        for token in raw_tokens:
            cleaned_country = clean_country_name(token)

            if cleaned_country is None:
                excluded_records.append({
                    "study_row_id": study_row_id,
                    "raw_value": raw_value,
                    "token": str(token).strip(),
                    "reason": "not a valid country or country not provided"
                })
                continue

            country_records.append({
                "study_row_id": study_row_id,
                "country": cleaned_country,
                "raw_token": str(token).strip()
            })

    country_records_df = pd.DataFrame(country_records)
    excluded_df = pd.DataFrame(excluded_records)

    if country_records_df.empty:
        raise ValueError("No valid country names were found.")

    # Avoid double-counting the same country within the same study
    country_records_df = country_records_df.drop_duplicates(
        subset=["study_row_id", "country"]
    )

    country_counts = (
        country_records_df
        .groupby("country", as_index=False)
        .agg({"study_row_id": "nunique"})
        .rename(columns={"study_row_id": "number of studies"})
        .sort_values(["number of studies", "country"], ascending=[False, True])
    )

    country_counts["iso3"] = country_counts["country"].apply(get_iso3)

    unmatched_df = country_counts[country_counts["iso3"].isna()].copy()

    country_counts_matched = (
        country_counts
        .dropna(subset=["iso3"])
        .sort_values(["number of studies", "country"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return country_counts_matched, excluded_df, unmatched_df


def save_country_table(
    country_counts: pd.DataFrame,
    excluded_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save the country count table to Excel.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    publication_table = country_counts[["country", "number of studies"]].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        publication_table.to_excel(
            writer,
            sheet_name="Country counts",
            index=False
        )

        country_counts[["country", "iso3", "number of studies"]].to_excel(
            writer,
            sheet_name="Map table with ISO3",
            index=False
        )

        if not excluded_df.empty:
            excluded_df.to_excel(
                writer,
                sheet_name="Excluded values",
                index=False
            )

        if not unmatched_df.empty:
            unmatched_df.to_excel(
                writer,
                sheet_name="Unmatched countries",
                index=False
            )

    print(f"Country distribution table saved to: {output_path.resolve()}")


# --------------------------------------------------
# Map data
# --------------------------------------------------

def download_natural_earth_map() -> Path:
    """
    Download Natural Earth country boundaries if not already available.
    """
    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not NATURAL_EARTH_ZIP.exists():
        print("Downloading Natural Earth country boundaries...")
        urllib.request.urlretrieve(NATURAL_EARTH_URL, NATURAL_EARTH_ZIP)

    return NATURAL_EARTH_ZIP


def load_world_boundaries() -> gpd.GeoDataFrame:
    """
    Load Natural Earth country boundaries.
    """
    zip_path = download_natural_earth_map()

    world = gpd.read_file(f"zip://{zip_path}")

    iso_column = None

    for candidate in ["ADM0_A3", "ISO_A3", "ISO_A3_EH"]:
        if candidate in world.columns:
            iso_column = candidate
            break

    if iso_column is None:
        raise KeyError(
            "Could not find an ISO3 country-code column in the Natural Earth file."
        )

    world = world.rename(columns={iso_column: "iso3"})

    if "ADMIN" in world.columns:
        world = world[world["ADMIN"] != "Antarctica"]

    return world


# --------------------------------------------------
# Plotting
# --------------------------------------------------

def plot_country_distribution(
    country_counts: pd.DataFrame,
    output_png: Path,
    output_pdf: Path,
    output_svg: Path
) -> None:
    """
    Plot global country distribution of included studies.

    Countries are shaded according to the number of included studies.
    """
    world = load_world_boundaries()

    merged = world.merge(
        country_counts,
        on="iso3",
        how="left"
    )

    robinson_projection = (
        "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"
    )

    merged = merged.to_crs(robinson_projection)

    countries_with_data = merged[merged["number of studies"].notna()].copy()

    if countries_with_data.empty:
        raise ValueError("No countries could be matched to the world map.")

    max_count = int(country_counts["number of studies"].max())

    # Discrete bins make low-count countries more visible.
    # This is better than a continuous colour scale when counts are skewed.
    bin_edges = [0.5, 1.5, 2.5, 5.5, 10.5, 20.5, max_count + 0.5]
    
    bin_labels = [
        "1 study",
        "2 studies",
        "3–5 studies",
        "6–10 studies",
        "11–20 studies",
        ">20 studies"
        ]
    
    # Stronger sequential colours.
    # The first colour is deliberately visible, not near-white.
    map_colours = [
        "#9ECAE1",
        "#6BAED6",
        "#3182BD",
        "#08519C",
        "#08306B",
        "#041C45"
        ]
    
    discrete_cmap = ListedColormap(map_colours)
    discrete_norm = BoundaryNorm(bin_edges, discrete_cmap.N)
    
    fig, ax = plt.subplots(figsize=(13.5, 7.5))

    # Base world map
    merged.plot(
        ax=ax,
        color="#F2F2F2",
        edgecolor="#D0D0D0",
        linewidth=0.35
    )

    # Countries with included studies
    # Use a truncated blue colour map so that countries with count = 1
    # are still clearly visible.
    
    countries_with_data.plot(
        ax=ax,
        column="number of studies",
        cmap=discrete_cmap,
        norm=discrete_norm,
        edgecolor="#4D4D4D",
        linewidth=0.45
        )

    ax.set_title(
        "Geographical distribution of included studies",
        fontsize=15,
        pad=12
    )

    ax.set_axis_off()

    # Colour bar
    legend_handles = [
        Patch(
            facecolor=map_colours[i],
            edgecolor="#4D4D4D",
            label=bin_labels[i]
            )
        for i in range(len(bin_labels))
        ]
    
    ax.legend(
        handles=legend_handles,
        title="Number of included studies",
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=10
        )
    

    fig.tight_layout()

    fig.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_svg, bbox_inches="tight", facecolor="white")

    plt.show()

    print(f"Country distribution map saved to: {output_png.resolve()}")
    print(f"PDF version saved to: {output_pdf.resolve()}")
    print(f"SVG version saved to: {output_svg.resolve()}")


# --------------------------------------------------
# Main workflow
# --------------------------------------------------

def main() -> None:
    """
    Run the full country-distribution workflow.
    """
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_FILE)

    country_counts, excluded_df, unmatched_df = create_country_count_table(df)

    save_country_table(
        country_counts=country_counts,
        excluded_df=excluded_df,
        unmatched_df=unmatched_df,
        output_path=OUTPUT_TABLE
    )

    plot_country_distribution(
        country_counts=country_counts,
        output_png=OUTPUT_MAP_PNG,
        output_pdf=OUTPUT_MAP_PDF,
        output_svg=OUTPUT_MAP_SVG
    )

    print("\nGeographical distribution analysis completed successfully.")


if __name__ == "__main__":
    main()