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
from matplotlib.lines import Line2D

from shapely.validation import make_valid
from shapely.errors import GEOSException



# --------------------------------------------------
# Path configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_FILE = PROJECT_ROOT / "input" / "country_data.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_TABLE = OUTPUT_DIR / "country_distribution_table.xlsx"
OUTPUT_MAP_PNG = OUTPUT_DIR / "country_who_distribution_map.png"
OUTPUT_MAP_PDF = OUTPUT_DIR / "country_who_distribution_map.pdf"
OUTPUT_MAP_SVG = OUTPUT_DIR / "country_who_distribution_map.svg"

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

WHO_REGION_BY_ISO3 = {
    # WHO African Region
    "DZA": "WHO African Region",
    "AGO": "WHO African Region",
    "BEN": "WHO African Region",
    "BWA": "WHO African Region",
    "BFA": "WHO African Region",
    "BDI": "WHO African Region",
    "CPV": "WHO African Region",
    "CMR": "WHO African Region",
    "CAF": "WHO African Region",
    "TCD": "WHO African Region",
    "COM": "WHO African Region",
    "COG": "WHO African Region",
    "CIV": "WHO African Region",
    "COD": "WHO African Region",
    "GNQ": "WHO African Region",
    "ERI": "WHO African Region",
    "SWZ": "WHO African Region",
    "ETH": "WHO African Region",
    "GAB": "WHO African Region",
    "GMB": "WHO African Region",
    "GHA": "WHO African Region",
    "GIN": "WHO African Region",
    "GNB": "WHO African Region",
    "KEN": "WHO African Region",
    "LSO": "WHO African Region",
    "LBR": "WHO African Region",
    "MDG": "WHO African Region",
    "MWI": "WHO African Region",
    "MLI": "WHO African Region",
    "MRT": "WHO African Region",
    "MUS": "WHO African Region",
    "MOZ": "WHO African Region",
    "NAM": "WHO African Region",
    "NER": "WHO African Region",
    "NGA": "WHO African Region",
    "RWA": "WHO African Region",
    "STP": "WHO African Region",
    "SEN": "WHO African Region",
    "SYC": "WHO African Region",
    "SLE": "WHO African Region",
    "ZAF": "WHO African Region",
    "SSD": "WHO African Region",
    "TGO": "WHO African Region",
    "UGA": "WHO African Region",
    "TZA": "WHO African Region",
    "ZMB": "WHO African Region",
    "ZWE": "WHO African Region",

    # WHO Region of the Americas
    "ATG": "WHO Region of the Americas",
    "ARG": "WHO Region of the Americas",
    "BHS": "WHO Region of the Americas",
    "BRB": "WHO Region of the Americas",
    "BLZ": "WHO Region of the Americas",
    "BOL": "WHO Region of the Americas",
    "BRA": "WHO Region of the Americas",
    "CAN": "WHO Region of the Americas",
    "CHL": "WHO Region of the Americas",
    "COL": "WHO Region of the Americas",
    "CRI": "WHO Region of the Americas",
    "CUB": "WHO Region of the Americas",
    "DMA": "WHO Region of the Americas",
    "DOM": "WHO Region of the Americas",
    "ECU": "WHO Region of the Americas",
    "SLV": "WHO Region of the Americas",
    "GRD": "WHO Region of the Americas",
    "GTM": "WHO Region of the Americas",
    "GUY": "WHO Region of the Americas",
    "HTI": "WHO Region of the Americas",
    "HND": "WHO Region of the Americas",
    "JAM": "WHO Region of the Americas",
    "MEX": "WHO Region of the Americas",
    "NIC": "WHO Region of the Americas",
    "PAN": "WHO Region of the Americas",
    "PRY": "WHO Region of the Americas",
    "PER": "WHO Region of the Americas",
    "KNA": "WHO Region of the Americas",
    "LCA": "WHO Region of the Americas",
    "VCT": "WHO Region of the Americas",
    "SUR": "WHO Region of the Americas",
    "TTO": "WHO Region of the Americas",
    "USA": "WHO Region of the Americas",
    "URY": "WHO Region of the Americas",
    "VEN": "WHO Region of the Americas",
    "PRI": "WHO Region of the Americas",

    # WHO South-East Asia Region
    "BGD": "WHO South-East Asia Region",
    "BTN": "WHO South-East Asia Region",
    "PRK": "WHO South-East Asia Region",
    "IND": "WHO South-East Asia Region",
    "IDN": "WHO South-East Asia Region",
    "MDV": "WHO South-East Asia Region",
    "MMR": "WHO South-East Asia Region",
    "NPL": "WHO South-East Asia Region",
    "LKA": "WHO South-East Asia Region",
    "THA": "WHO South-East Asia Region",
    "TLS": "WHO South-East Asia Region",

    # WHO European Region
    "ALB": "WHO European Region",
    "AND": "WHO European Region",
    "ARM": "WHO European Region",
    "AUT": "WHO European Region",
    "AZE": "WHO European Region",
    "BLR": "WHO European Region",
    "BEL": "WHO European Region",
    "BIH": "WHO European Region",
    "BGR": "WHO European Region",
    "HRV": "WHO European Region",
    "CYP": "WHO European Region",
    "CZE": "WHO European Region",
    "DNK": "WHO European Region",
    "EST": "WHO European Region",
    "FIN": "WHO European Region",
    "FRA": "WHO European Region",
    "GEO": "WHO European Region",
    "DEU": "WHO European Region",
    "GRC": "WHO European Region",
    "HUN": "WHO European Region",
    "ISL": "WHO European Region",
    "IRL": "WHO European Region",
    "ISR": "WHO European Region",
    "ITA": "WHO European Region",
    "KAZ": "WHO European Region",
    "KGZ": "WHO European Region",
    "LVA": "WHO European Region",
    "LTU": "WHO European Region",
    "LUX": "WHO European Region",
    "MLT": "WHO European Region",
    "MCO": "WHO European Region",
    "MNE": "WHO European Region",
    "NLD": "WHO European Region",
    "MKD": "WHO European Region",
    "NOR": "WHO European Region",
    "POL": "WHO European Region",
    "PRT": "WHO European Region",
    "MDA": "WHO European Region",
    "ROU": "WHO European Region",
    "RUS": "WHO European Region",
    "SMR": "WHO European Region",
    "SRB": "WHO European Region",
    "SVK": "WHO European Region",
    "SVN": "WHO European Region",
    "ESP": "WHO European Region",
    "SWE": "WHO European Region",
    "CHE": "WHO European Region",
    "TJK": "WHO European Region",
    "TUR": "WHO European Region",
    "TKM": "WHO European Region",
    "UKR": "WHO European Region",
    "GBR": "WHO European Region",
    "UZB": "WHO European Region",
    "KOS": "WHO European Region",
    "XKX": "WHO European Region",

    # WHO Eastern Mediterranean Region
    "AFG": "WHO Eastern Mediterranean Region",
    "BHR": "WHO Eastern Mediterranean Region",
    "DJI": "WHO Eastern Mediterranean Region",
    "EGY": "WHO Eastern Mediterranean Region",
    "IRN": "WHO Eastern Mediterranean Region",
    "IRQ": "WHO Eastern Mediterranean Region",
    "JOR": "WHO Eastern Mediterranean Region",
    "KWT": "WHO Eastern Mediterranean Region",
    "LBN": "WHO Eastern Mediterranean Region",
    "LBY": "WHO Eastern Mediterranean Region",
    "MAR": "WHO Eastern Mediterranean Region",
    "OMN": "WHO Eastern Mediterranean Region",
    "PAK": "WHO Eastern Mediterranean Region",
    "PSE": "WHO Eastern Mediterranean Region",
    "QAT": "WHO Eastern Mediterranean Region",
    "SAU": "WHO Eastern Mediterranean Region",
    "SOM": "WHO Eastern Mediterranean Region",
    "SDN": "WHO Eastern Mediterranean Region",
    "SYR": "WHO Eastern Mediterranean Region",
    "TUN": "WHO Eastern Mediterranean Region",
    "ARE": "WHO Eastern Mediterranean Region",
    "YEM": "WHO Eastern Mediterranean Region",

    # WHO Western Pacific Region
    "AUS": "WHO Western Pacific Region",
    "BRN": "WHO Western Pacific Region",
    "KHM": "WHO Western Pacific Region",
    "CHN": "WHO Western Pacific Region",
    "COK": "WHO Western Pacific Region",
    "FJI": "WHO Western Pacific Region",
    "JPN": "WHO Western Pacific Region",
    "KIR": "WHO Western Pacific Region",
    "LAO": "WHO Western Pacific Region",
    "MYS": "WHO Western Pacific Region",
    "MHL": "WHO Western Pacific Region",
    "FSM": "WHO Western Pacific Region",
    "MNG": "WHO Western Pacific Region",
    "NRU": "WHO Western Pacific Region",
    "NZL": "WHO Western Pacific Region",
    "NIU": "WHO Western Pacific Region",
    "PLW": "WHO Western Pacific Region",
    "PNG": "WHO Western Pacific Region",
    "PHL": "WHO Western Pacific Region",
    "KOR": "WHO Western Pacific Region",
    "WSM": "WHO Western Pacific Region",
    "SGP": "WHO Western Pacific Region",
    "SLB": "WHO Western Pacific Region",
    "TON": "WHO Western Pacific Region",
    "TUV": "WHO Western Pacific Region",
    "VUT": "WHO Western Pacific Region",
    "VNM": "WHO Western Pacific Region",
    "TWN": "WHO Western Pacific Region",
}

WHO_REGION_ORDER = [
    "WHO African Region",
    "WHO Region of the Americas",
    "WHO South-East Asia Region",
    "WHO European Region",
    "WHO Eastern Mediterranean Region",
    "WHO Western Pacific Region",
]

WHO_REGION_OUTLINE_STYLES = {
    "WHO African Region": {
        "color": "#000000",
        "linestyle": "-",
        "linewidth": 1.0,
    },
    "WHO Region of the Americas": {
        "color": "#000000",
        "linestyle": "--",
        "linewidth": 1.0,
    },
    "WHO South-East Asia Region": {
        "color": "#000000",
        "linestyle": "-.",
        "linewidth": 1.0,
    },
    "WHO European Region": {
        "color": "#000000",
        "linestyle": ":",
        "linewidth": 1.3,
    },
    "WHO Eastern Mediterranean Region": {
        "color": "#4D4D4D",
        "linestyle": "--",
        "linewidth": 1.4,
    },
    "WHO Western Pacific Region": {
        "color": "#4D4D4D",
        "linestyle": "-",
        "linewidth": 1.4,
    },
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
    country_counts["WHO region"] = (
        country_counts["iso3"]
        .map(WHO_REGION_BY_ISO3)
        .fillna("Unclassified")
        )
    
    unmatched_df = country_counts[
        country_counts["iso3"].isna() |
        (country_counts["WHO region"] == "Unclassified")
        ].copy()

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

    Sheets:
    - Country counts: main country-count table.
    - Map table with ISO3: country counts with ISO3 code.
    - Country WHO region: country counts with WHO region.
    - WHO region summary: total study-country records by WHO region.
    - Excluded values: non-country or missing records.
    - Unmatched countries: countries without ISO3 or WHO-region mapping.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    publication_table = country_counts[["country", "number of studies"]].copy()

    who_region_table = country_counts[
        ["country", "number of studies", "WHO region"]
    ].copy()

    who_region_summary = (
        country_counts
        .groupby("WHO region", as_index=False)
        .agg(
            number_of_countries_represented=("country", "nunique"),
            number_of_study_country_records=("number of studies", "sum")
        )
        .sort_values(
            "number_of_study_country_records",
            ascending=False
        )
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        publication_table.to_excel(
            writer,
            sheet_name="Country counts",
            index=False
        )

        country_counts[
            ["country", "iso3", "number of studies", "WHO region"]
        ].to_excel(
            writer,
            sheet_name="Map table with ISO3",
            index=False
        )

        who_region_table.to_excel(
            writer,
            sheet_name="Country WHO region",
            index=False
        )

        who_region_summary.to_excel(
            writer,
            sheet_name="WHO region summary",
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

def repair_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Repair invalid geometries before dissolve, boundary extraction, or plotting.

    This helps avoid GEOS topology errors such as:
    TopologyException: side location conflict.
    """
    repaired_gdf = gdf.copy()

    repaired_gdf["geometry"] = repaired_gdf["geometry"].apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )

    repaired_gdf = repaired_gdf[
        repaired_gdf["geometry"].notna() &
        ~repaired_gdf["geometry"].is_empty
    ].copy()

    return repaired_gdf

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
    
    # Repair invalid country geometries before projection/dissolve
    world = repair_geometries(world)
        
    world["WHO region"] = (
        world["iso3"]
        .map(WHO_REGION_BY_ISO3)
        .fillna("Unclassified")
        )

    return world


# --------------------------------------------------
# Plotting
# --------------------------------------------------


def add_who_region_outlines(
    merged: gpd.GeoDataFrame,
    ax
) -> list[Line2D]:
    """
    Add WHO regional boundary outlines to the map.

    This does not change the country fill colours.
    It only overlays WHO-region outlines.

    If dissolving region geometries fails because of invalid geometry,
    the function falls back to plotting country-level boundaries by WHO region.
    """
    region_legend_handles = []

    region_data = merged[
        merged["WHO region"].isin(WHO_REGION_ORDER)
    ].copy()

    if region_data.empty:
        return region_legend_handles

    region_data = repair_geometries(region_data)

    try:
        # Try to create clean dissolved WHO-region polygons
        dissolved_regions = region_data.dissolve(by="WHO region")
        dissolved_regions = repair_geometries(dissolved_regions.reset_index())
        dissolved_regions = dissolved_regions.set_index("WHO region")

        for region_name in WHO_REGION_ORDER:
            if region_name not in dissolved_regions.index:
                continue

            style = WHO_REGION_OUTLINE_STYLES[region_name]

            region_boundary = gpd.GeoSeries(
                [dissolved_regions.loc[region_name].geometry.boundary],
                crs=merged.crs
            )

            region_boundary.plot(
                ax=ax,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=5
            )

            region_legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=region_name.replace("WHO ", "")
                )
            )

    except (GEOSException, ValueError, RuntimeError) as error:
        print(
            "Warning: WHO-region dissolve failed because of invalid geometry. "
            "Using country-level WHO-region outlines instead."
        )
        print(f"Original geometry error: {error}")

        # Fallback: plot country boundaries grouped by WHO region without dissolve
        for region_name in WHO_REGION_ORDER:
            subset = region_data[region_data["WHO region"] == region_name].copy()

            if subset.empty:
                continue

            style = WHO_REGION_OUTLINE_STYLES[region_name]

            subset.boundary.plot(
                ax=ax,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=5
            )

            region_legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    label=region_name.replace("WHO ", "")
                )
            )

    return region_legend_handles


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
        country_counts[["iso3", "country", "number of studies"]],
        on="iso3",
        how="left"
        )

    robinson_projection = (
        "+proj=robin +lon_0=0 +datum=WGS84 +units=m +no_defs"
    )

    merged = merged.to_crs(robinson_projection)

    # Repair again after projection because projection can expose geometry problems
    merged = repair_geometries(merged)

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
    
    # Overlay WHO region outlines without changing the country-count fill colours
    who_region_legend_handles = add_who_region_outlines(
        merged=merged,
        ax=ax
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
    
    count_legend = ax.legend(
        handles=legend_handles,
        title="Number of included studies",
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=10
        )
    
    ax.add_artist(count_legend)
    
    if who_region_legend_handles:
        ax.legend(
            handles=who_region_legend_handles,
            title="WHO region outline",
            loc="center left",
            bbox_to_anchor=(0.155, 0.145),  # move legend left/right and up/down manually
            frameon=True,
            fontsize=8,
            title_fontsize=9
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