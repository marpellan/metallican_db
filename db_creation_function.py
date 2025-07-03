import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from rapidfuzz import fuzz
import os


def populate_table_df(column_mapping, facility_df, dynamic_columns=None, source_dfs=None):
    """
    Populate a facility DataFrame based on a column mapping and optional dynamic column values.

    Parameters:
        column_mapping (dict): A dictionary where keys are DataFrame names (as strings) and values are mappings
                               of source column names to target column names.
        facility_df (pd.DataFrame): The target facility DataFrame to populate.
        dynamic_columns (dict, optional): A dictionary where keys are target column names and values are mappings
                                          of DataFrame names to specific values (e.g., facility type).
        source_dfs (dict): A dictionary where keys are DataFrame names (as strings) and values are the actual DataFrames.

    Returns:
        pd.DataFrame: The populated facility DataFrame.
    """
    # Debug: Ensure facility_df starts empty or with expected rows
    print(f"Initial facility_df rows: {len(facility_df)}")

    for source_name, mapping in column_mapping.items():
        print(f"Processing DataFrame: {source_name}")

        df = source_dfs.get(source_name)
        if df is None:
            print(f"Warning: DataFrame '{source_name}' not found.")
            continue

        # Create a temporary DataFrame for the current source
        temp_df = pd.DataFrame()

        for src_col, target_col in mapping.items():
            if target_col in facility_df.columns and src_col in df.columns:
                # Map the source column to the target column
                temp_df[target_col] = df[src_col]

        # Add dynamic columns if provided
        if dynamic_columns:
            for dynamic_col, source_values in dynamic_columns.items():
                if dynamic_col in facility_df.columns and source_name in source_values:
                    temp_df[dynamic_col] = source_values[source_name]

        # Add a 'source' column for provenance tracking
        #temp_df["source_df"] = source_name

        # Ensure temp_df aligns with facility_df
        missing_columns = set(facility_df.columns) - set(temp_df.columns)
        for col in missing_columns:
            temp_df[col] = pd.NA

        # Debug: Print temp_df shape before appending
        print(f"Temp DF rows to append: {len(temp_df)}")

        # Append temp_df to facility_df
        facility_df = pd.concat([facility_df, temp_df], ignore_index=True)

        # Debug: Print facility_df shape after appending
        print(f"Rows in facility_df after appending {source_name}: {len(facility_df)}")

    # Final debug: Ensure the final facility_df shape is correct
    print(f"Final facility_df rows: {len(facility_df)}")
    return facility_df


def check_duplicate_facilities(df, name_col="facility_name", lon_col="longitude", lat_col="latitude"):
    """
    Flags possible duplicate facilities based on name and coordinates.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        name_col (str): Column with facility names
        lon_col (str): Longitude column
        lat_col (str): Latitude column

    Returns:
        pd.DataFrame: Subset of rows with duplicated (name, lat, lon) pairs
    """
    dupes = df[df.duplicated(subset=[name_col, lon_col, lat_col], keep=False)]
    return dupes.sort_values(by=[name_col, lat_col, lon_col])


def convert_to_gdf(df):
    """
    Transform df to gdf with EPSG:4326

    Parameters:
        df (pd.DataFrame or gpd.GeoDataFrame): Data containing facilities/projects.

    Returns:
        gpd.GeoDataFrame
    """

    # Convert to GeoDataFrame if it's a regular DataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        if "latitude" in df.columns and "longitude" in df.columns:
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        else:
            raise ValueError("The input DataFrame must have 'latitude' and 'longitude' columns or a 'geometry' column.")

    return df


def assign_id(df, canada_provinces, id_column="main_id", prefix="OTH", geometry_col="geometry", name_col="facility_name"):
    """
    Assign deterministic IDs with inferred provinces from coordinates.
    Handles both **Point** and **Polygon** geometries.

    Parameters:
        df (pd.DataFrame or gpd.GeoDataFrame): Data containing facilities/projects.
        canada_provinces (gpd.GeoDataFrame): GeoDataFrame of Canada provinces with 'PRENAME' column.
        id_column (str): The column to store unique IDs.
        prefix (str): User-defined prefix (e.g., 'MIN' for mining).
        geometry_col (str): Name of geometry column (if using a GeoDataFrame).

    Returns:
        pd.DataFrame: Data with assigned IDs and inferred provinces.
    """

    # Ensure Canada provinces data is in EPSG:4326 (WGS 84)
    if canada_provinces.crs != "EPSG:4326":
        canada_provinces = canada_provinces.to_crs("EPSG:4326")

    # Dictionary mapping province names to their codes
    province_codes = {
        "Ontario": "ON", "Quebec": "QC", "British Columbia": "BC", "Alberta": "AB",
        "Manitoba": "MB", "Saskatchewan": "SK", "Newfoundland and Labrador": "NL",
        "New Brunswick": "NB", "Nova Scotia": "NS", "Prince Edward Island": "PE",
        "Northwest Territories": "NT", "Yukon": "YT", "Nunavut": "NU"
    }

    # Convert to GeoDataFrame if it's a regular DataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        if "latitude" in df.columns and "longitude" in df.columns:
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        else:
            raise ValueError("The input DataFrame must have 'latitude' and 'longitude' columns or a 'geometry' column.")

    def extract_coordinates(row):
        """Extracts latitude and longitude from the geometry."""
        geom = row.get(geometry_col)

        if isinstance(geom, Point):
            return geom.x, geom.y
        elif isinstance(geom, (Polygon, MultiPolygon)):
            centroid = geom.centroid
            return centroid.x, centroid.y
        return None, None

    def get_province(lon, lat):
        """Returns the province code based on coordinates."""
        if lon is None or lat is None:
            return "ZZ"
        point = Point(lon, lat)
        match = canada_provinces[canada_provinces.contains(point)]
        if not match.empty:
            province_name = match.iloc[0]["PRENAME"]
            return province_codes.get(province_name, "ZZ")
        return "ZZ"

    # Ensure the input is a GeoDataFrame
    if not isinstance(df, gpd.GeoDataFrame):
        raise ValueError("The input gdf must be a GeoDataFrame with a geometry column.")

    # Extract centroid for polygons & coordinates for points
    df["longitude"], df["latitude"] = zip(*df.apply(extract_coordinates, axis=1))

    def generate_id(row):
        """Generates a unique and deterministic ID."""
        lon, lat = row["longitude"], row["latitude"]
        province_code = get_province(lon, lat)  # Infer province
        facility_name = row.get(name_col, "Unknown")

        # Create a deterministic hash
        hash_input = f"{prefix}|{facility_name}|{lat}|{lon}"
        unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        # Ensure uniqueness by appending location_count for duplicate coordinates
        return f"{province_code}-{prefix}-{unique_hash}"

    # Apply the ID assignment
    df[id_column] = df.apply(generate_id, axis=1)

    # Ensure ID column is first
    cols = [id_column] + [col for col in df.columns if col != id_column]
    return df[cols]


def assign_deterministic_id(df, prefix, name_column, id_column):
    """
    Assign deterministic IDs based on a string column.

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        prefix (str): A short prefix for the ID (e.g., 'CMP-', 'GRP-').
        name_column (str): Column name containing the value to hash.
        id_column (str): Column name where the ID will be stored.

    Returns:
        pd.DataFrame: The updated DataFrame with assigned IDs.
    """
    def generate_id(name):
        if pd.isna(name):
            return None
        clean_name = str(name).strip().lower()
        return f"{prefix}{hashlib.md5(clean_name.encode()).hexdigest()[:8]}"

    df[id_column] = df[name_column].apply(generate_id)
    return df


def add_year(gdf, year, original_year_col=None):
    """
    Adds or moves the 'year' column to the first position in a GeoDataFrame.
    If an original column name is provided, it renames it to 'year' and moves it.

    Parameters:
    - gdf (gpd.GeoDataFrame): Input GeoDataFrame.
    - year (int): Year to assign if the 'year' column is not present.
    - original_year_col (str, optional): Name of the existing year column if different from 'year'.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame with 'year' as the first column.
    """

    # Check if 'year' column already exists or needs renaming
    if original_year_col and original_year_col in gdf.columns:
        # Rename original year column to 'year' if it is specified
        gdf = gdf.rename(columns={original_year_col: 'year'})

    if 'year' in gdf.columns:
        # Move 'year' to the first column
        gdf = gdf[['year'] + [col for col in gdf.columns if col != 'year']]
    else:
        # Add 'year' column with the specified value if it doesn't exist
        gdf.insert(0, 'year', year)

    return gdf


def assign_row_id(
    df,
    facility_id_col="main_id",
    row_id_col="row_id",
    prefix="ROW",
    year_col=None,
    scenario_col=None
):
    """
    Assigns a unique, stable row-level ID per entry linked to a facility.
    Reuses the hash part of the main_id directly (not a re-hash).

    Format:
      - With year & scenario:   PREFIX-<hash>-<year>-<scenario>-<index>
      - With year only:         PREFIX-<hash>-<year>-<index>
      - With scenario only:     PREFIX-<hash>-<scenario>-<index>
      - Without both:           PREFIX-<hash>-<index>

    Parameters:
        df (pd.DataFrame): Table with multiple rows per facility.
        facility_id_col (str): Column with facility-level ID (e.g., 'main_id').
        row_id_col (str): Name of the new row-level ID column.
        prefix (str): Table-specific ID prefix (e.g., 'POLL', 'GHG').
        year_col (str, optional): Column name for year (if applicable).
        scenario_col (str, optional): Column name for scenario (if applicable).

    Returns:
        pd.DataFrame: Same table with new row_id as the first column.
    """
    df = df.copy()

    def extract_hash(fac_id):
        s = str(fac_id)
        parts = s.split("-")
        if len(parts) >= 3:
            return parts[-1]  # Keep original logic for full main_id
        else:
            return s  # Use the full string (e.g., '10052')

    df["_hash"] = df[facility_id_col].apply(extract_hash)

    group_cols = [facility_id_col]
    if year_col:
        group_cols.append(year_col)
    if scenario_col:
        group_cols.append(scenario_col)

    df["_row_index"] = df.groupby(group_cols).cumcount() + 1

    def build_id(row):
        parts = [prefix, row["_hash"]]
        if year_col:
            parts.append(str(row[year_col]))
        if scenario_col:
            parts.append(str(row[scenario_col]))
        parts.append(str(row["_row_index"]))
        return "-".join(parts)

    df[row_id_col] = df.apply(build_id, axis=1)
    df = df.drop(columns=["_row_index", "_hash"])

    # Reorder to place row_id first
    cols = [row_id_col] + [col for col in df.columns if col != row_id_col]
    return df[cols]








def compute_similarity_score(df, name_col1="facility_name_df1", name_col2="facility_name_df2", threshold=80):
    """
    Compare two facility name columns and return a similarity score.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the two name columns.
        name_col1 (str): Column name for facility names in dataset 1.
        name_col2 (str): Column name for facility names in dataset 2.
        threshold (int): Minimum similarity score to consider as a match (0-100).

    Returns:
        pd.DataFrame: Original DataFrame with an added 'name_similarity' column.
    """
    df["similarity_score"] = df.apply(lambda row: fuzz.ratio(str(row[name_col1]), str(row[name_col2])), axis=1)
    #df["name_match"] = df["name_similarity"] >= threshold  # True if similarity is above the threshold
    return df


### POLYGONS
def add_surface_area_polygons(gdf, min_area_km2=None):
    """
    Add surface area in km² to polygons.
    Optionally filter out polygons below a given minimum area.
    """

    gdf = gdf.copy()
    gdf = gdf.to_crs(epsg=3978)  # Canada Albers Equal Area

    gdf["area_km2"] = gdf.geometry.area / 1e6  # m² to km²

    if min_area_km2 is not None:
        gdf = gdf[gdf["area_km2"] > min_area_km2]

    return gdf


def calculate_max_distance(land_df):
    """
    Computes the maximum distance from the centroid of a polygon to its edges.
    Returns the gdf with additionnal columns:
     - Geometry column storing primary spatial representation (e.g. full polygon shapes)
     - Centroid column storing the centroid of the polygon, e.g. point location
     - Max distance between the centroids and the edges in meters

    Args:
        land_df (gpd.GeoDataFrame): GeoDataFrame with land polygons.

    Returns:
        gpd.GeoDataFrame: The same DataFrame with an additional `max_distance` column, using meters as unit.
    """
    # Reproject to EPSG:3978 for accurate distance calculations
    land_df_proj = land_df.to_crs(epsg=3978)

    # Compute the centroid
    land_df_proj["centroid"] = land_df_proj.geometry.centroid

    # Compute max distance from centroid to polygon edges
    def max_distance_to_edges(row):
        """Calculate max distance from centroid to polygon edges, ensuring proper handling of floats."""
        if isinstance(row.geometry, Polygon):  # Ensure it's a polygon
            distances = row.geometry.boundary.distance(row.centroid)
            if isinstance(distances, float):  # Handle cases where only one distance is returned
                return distances
            return max(distances) if not distances.empty else 0
        return 0

    # Apply function
    land_df_proj["max_distance_centroid_to_edges_meters"] = land_df_proj.apply(max_distance_to_edges, axis=1)

    # Explicitly set the active geometry column back to the main 'geometry'
    land_df_proj = land_df_proj.set_geometry("geometry")

    # Drop the centroid column to clean up
    # land_df_proj = land_df_proj.drop(columns=["centroid"])

    # Convert back to EPSG:4326 for consistency
    return land_df_proj.to_crs(epsg=4326)


def analyze_and_compare_polygon_areas(gdf1, gdf2, dataset_name1, dataset_name2, save_path=None):
    """
    This function takes two GeoDataFrames, calculates polygon areas in square kilometers,
    and plots a comparative barplot with summary statistics.

    Parameters:
    - gdf1 (GeoDataFrame): First GeoDataFrame.
    - gdf2 (GeoDataFrame): Second GeoDataFrame.
    - dataset_name1 (str): Name of the first dataset.
    - dataset_name2 (str): Name of the second dataset.

    Returns:
    - A combined DataFrame with polygon count and area statistics.
    - A comparative barplot of polygon area distributions.
    """

    # Reproject to Canada Albers Equal Area (EPSG:3978) for accurate area calculations
    gdf1 = gdf1.to_crs(epsg=3978)
    gdf2 = gdf2.to_crs(epsg=3978)

    # Compute area in km²
    gdf1["area_km2"] = gdf1.geometry.area / 1e6  # Convert m² to km²
    gdf2["area_km2"] = gdf2.geometry.area / 1e6  # Convert m² to km²

    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        "Dataset": [dataset_name1] * len(gdf1) + [dataset_name2] * len(gdf2),
        "Area (km²)": list(gdf1["area_km2"]) + list(gdf2["area_km2"])
    })

    # Summary statistics
    stats1 = gdf1["area_km2"].describe()
    stats2 = gdf2["area_km2"].describe()

    summary_df = pd.DataFrame({
        "Dataset": [dataset_name1, dataset_name2],
        "Polygon Count": [len(gdf1), len(gdf2)],
        "Min Area (km²)": [stats1["min"], stats2["min"]],
        "Mean Area (km²)": [stats1["mean"], stats2["mean"]],
        "Median Area (km²)": [stats1["50%"], stats2["50%"]],
        "Max Area (km²)": [stats1["max"], stats2["max"]]
    })
    #print(summary_df)

    # Barplot with annotations
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="Dataset", y="Area (km²)", data=df_plot)

    # Adjust y-axis limits to ensure annotations fit
    ymax = max(stats1["max"], stats2["max"]) * 1.1  # Extend limit by 10%
    ax.set_ylim(0, ymax)

    # Add text annotations above the boxplots
    for i, dataset in enumerate([dataset_name1, dataset_name2]):
        count = summary_df.loc[summary_df["Dataset"] == dataset, "Polygon Count"].values[0]
        mean_area = summary_df.loc[summary_df["Dataset"] == dataset, "Mean Area (km²)"].values[0]
        max_area = summary_df.loc[summary_df["Dataset"] == dataset, "Max Area (km²)"].values[0]
        ax.text(i, max_area + (ymax * 0.02), f"Polygons count: {count}\nMean area: {mean_area:.2f} km²",
                horizontalalignment='center', verticalalignment='bottom', fontsize=8, color='black', fontweight='bold')

    plt.title("Comparison of Polygon Area Distributions")
    plt.ylabel("Area (km²)")
    plt.xlabel("")

    # Save the plot if a save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    plt.show()

    return summary_df


def create_substance_table(pollutant_gdf, env_df):
    """
    Combine and clean substance names from NPRI and manually collected datasets,
    apply harmonized naming, assign stable substance IDs, and return a master substance table.
    """

    # Step 1: Concatenate and tag provenance
    npri_sub = pollutant_gdf[['substance_name_npri']].copy()
    npri_sub['source'] = 'NPRI'
    npri_sub = npri_sub.rename(columns={'substance_name_npri': 'original_name'})

    manual_sub = env_df[['substance_name']].copy()
    manual_sub['source'] = 'Manual'
    manual_sub = manual_sub.rename(columns={'substance_name': 'original_name'})

    all_substances = pd.concat([npri_sub, manual_sub], ignore_index=True)

    # Step 2: Clean and exclude irrelevant entries
    all_substances['original_name'] = all_substances['original_name'].astype(str).str.strip()
    all_substances = all_substances[all_substances['original_name'] != '-']
    all_substances = all_substances[all_substances['original_name'] != 'nan']

    # Step 3: Harmonization dictionary
    harmonized_map = {
        "Ammonia (total)": "Ammonia",
        "Antimony (and its compounds)": "Antimony",
        "Arsenic (and its compounds)": "Arsenic",
        "Lead (and its compounds)": "Lead",
        "Mercury (and its compounds)": "Mercury",
        "PM10 - Particulate Matter <= 10 Micrometers": "PM10",
        "PM2.5 - Particulate Matter <= 2.5 Micrometers": "PM2.5",
        "Volatile Organic Compounds (Total)": "VOCs",
    }

    all_substances['harmonized_name'] = all_substances['original_name'].replace(harmonized_map)

    # Step 4: Drop duplicates
    substance_table = (
        all_substances
        .drop_duplicates(subset=['harmonized_name'])
        .sort_values('harmonized_name')
        .reset_index(drop=True)
    )

    # Step 5: Generate stable substance_id using SHA1 hash
    def make_id(row):
        raw = f"{row['harmonized_name']}"
        return "SUB" + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]

    substance_table['substance_id'] = substance_table.apply(make_id, axis=1)

    # Step 6: Reorder columns
    substance_table = substance_table[[
        'substance_id', 'harmonized_name', 'original_name', 'source'
    ]]

    return substance_table


def create_compartment_table(pollutant_gdf, env_df):
    """
    Build a harmonized compartment table with traceable raw labels for merging.
    """

    # --- Define mapping from emission_type to (compartment, pathway)
    compartment_mapping_clean = {
        "Air Emissions / Émissions à l'air": ("Air", "Unspecified"),
        "Water Releases / Rejets à l'eau": ("Water", "Unspecified"),
        "Land Releases /  Rejets au sol": ("Land", "Unspecified"),
        "On-Site Disposal / Élimination sur le site": ("Land", "On-site disposal"),
        "Off-Site Disposal / Élimination hors site": ("Land", "Off-site disposal"),
        "Total Releases / Rejets totaux": (None, "Aggregate"),
        "Transfers for Treatment / Transferts pour traitement": (None, "Transfer for treatment"),
        "Transfers for Recycling / Transferts pour recyclage": (None, "Transfer for recycling"),
        "Total On-Site, Off-Site and Treatment Disposal /\n Élimination sur le site, hors site et pour traitement totale": (None, "Aggregate"),
        "Grand Total": (None, "Grand total")
    }

    # --- NPRI data
    npri_comp = pollutant_gdf[['emission_type', 'emission_subtype']].dropna(subset=['emission_type']).copy()
    npri_comp['source'] = 'NPRI'
    npri_comp[['compartment', 'default_pathway']] = npri_comp['emission_type'].map(compartment_mapping_clean).apply(pd.Series)
    npri_comp['compartment_pathway'] = npri_comp['emission_subtype'].fillna('').str.strip()
    npri_comp.loc[npri_comp['compartment_pathway'] == '', 'compartment_pathway'] = npri_comp['default_pathway']
    npri_comp['raw_compartment_label'] = npri_comp['emission_type']
    npri_comp['raw_pathway_label'] = npri_comp['emission_subtype'].fillna('Unspecified')
    npri_comp = npri_comp.drop(columns=['default_pathway'])

    # --- Manual data
    env_comp = env_df[['compartment']].dropna().copy()
    env_comp = env_comp[env_comp['compartment'] != '-']
    env_comp['compartment_pathway'] = 'Unspecified'
    env_comp['raw_compartment_label'] = env_comp['compartment']
    env_comp['raw_pathway_label'] = 'Unspecified'
    env_comp['source'] = 'Manually collected data'

    # --- Align columns
    env_comp['compartment'] = env_comp['compartment']
    manual_cols = ['compartment', 'compartment_pathway', 'raw_compartment_label', 'raw_pathway_label', 'source']
    npri_cols = manual_cols

    # --- Combine all
    all_comps = pd.concat([
        npri_comp[npri_cols],
        env_comp[manual_cols]
    ], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # --- Create hashed ID
    def make_id(row):
        raw = f"{row['compartment']}_{row['compartment_pathway']}"
        return "CMP" + hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]

    all_comps['compartment_id'] = all_comps.apply(make_id, axis=1)

    # --- Final structure
    return all_comps[[
        'compartment_id', 'compartment', 'compartment_pathway',
        'raw_compartment_label', 'raw_pathway_label', 'source'
    ]].sort_values(['compartment', 'compartment_pathway'])


def add_source_id_to_collected_data(df, company_col="company", facility_col="facility", source_col="source", source_id_col="source_id"):
    """
    Add a human-readable source_id column to a DataFrame using company, facility, and source file name.

    Parameters:
    - df: pandas DataFrame
    - company_col: column name for the company
    - facility_col: column name for the facility
    - source_col: column name for the source file (e.g. PDF, Excel)
    - source_id_col: name of the output column to be added

    Returns:
    - df with a new 'source_id' column (if not already present)
    """
    import pandas as pd
    from pathlib import Path

    if source_id_col not in df.columns:
        def create_source_id(row):
            company = str(row[company_col]).strip().replace(" ", "")
            file_stem = Path(str(row[source_col])).stem.strip().replace(" ", "")
            return f"SRC_{company}_{file_stem}"

        df[source_id_col] = df.apply(create_source_id, axis=1)
    else:
        print(f"ℹ️ '{source_id_col}' already exists. No changes made.")

    return df


def create_source_table_from_datasets(dataset_dict, manually_collected_dfs, source_col="source", company_col="company", facility_col="facility"):
    """
    Create a consolidated source table from multiple GeoDataFrames and manually collected data.

    Parameters:
    - dataset_dict: dict mapping GeoDataFrame name (str) to actual df with a single source_id value
    - manually_collected_dfs: list of dataframes that use the 'add_source_id' function
    - source_col, company_col, facility_col: column names in manually collected dfs

    Returns:
    - A pandas DataFrame with columns: source_id, source_provenance, source_name
    """

    import pandas as pd
    from pathlib import Path

    # Step 1: Add known datasets
    dataset_sources = []
    for name, df in dataset_dict.items():
        unique_ids = df['source_id'].dropna().unique()
        for sid in unique_ids:
            dataset_sources.append({
                "source_id": sid.strip(),
                "source_provenance": "dataset",
                "source_name": name
            })

    # Step 2: Add manually collected sources
    manual_sources = []
    for df in manually_collected_dfs:
        if "source_id" not in df.columns:
            continue  # skip if not processed yet
        for _, row in df.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).iterrows():
            source_id = row["source_id"]
            company = str(row.get(company_col, "")).strip()
            facility = str(row.get(facility_col, "")).strip()
            file_path = Path(str(row.get(source_col, "")))
            file_name = file_path.name
            source_name = f"{company} – {facility} ({file_name})".strip(" –()")
            manual_sources.append({
                "source_id": source_id,
                "source_provenance": "report",
                "source_name": source_name
            })

    # Combine, deduplicate
    full_source_table = pd.DataFrame(dataset_sources + manual_sources)
    full_source_table = full_source_table.drop_duplicates(subset=["source_id"]).sort_values("source_provenance")

    return full_source_table


def assign_row_id_to_collected_data(
    df,
    facility_id_col="main_id",
    row_id_col="row_id",
    prefix="ROW",
    year_col=None,
    scenario_col=None,
    fallback_cols=["facility_group_id", "company_id"]
):
    df = df.copy()

    def is_valid(val):
        return pd.notna(val) and str(val).strip() not in ["", "-"]

    def resolve_facility_id(row):
        val = row.get(facility_id_col)
        if is_valid(val):
            return str(val).strip()
        for col in fallback_cols:
            fallback_val = row.get(col)
            if is_valid(fallback_val):
                return str(fallback_val).strip()
        return "UNKNOWN"

    def extract_hash(fac_id):
        s = str(fac_id)
        parts = s.split("-")
        if len(parts) >= 3:
            return parts[-1]
        else:
            return s

    df["_facility_resolved"] = df.apply(resolve_facility_id, axis=1)
    df["_hash"] = df["_facility_resolved"].apply(extract_hash)

    group_cols = ["_facility_resolved"]
    if year_col:
        group_cols.append(year_col)
    if scenario_col:
        group_cols.append(scenario_col)

    df["_row_index"] = df.groupby(group_cols).cumcount() + 1

    def build_id(row):
        parts = [prefix, row["_hash"]]
        if year_col:
            parts.append(str(row[year_col]))
        if scenario_col:
            parts.append(str(row[scenario_col]))
        parts.append(str(row["_row_index"]))
        return "-".join(parts)

    df[row_id_col] = df.apply(build_id, axis=1)
    df.drop(columns=["_row_index", "_hash", "_facility_resolved"], inplace=True)

    cols = [row_id_col] + [col for col in df.columns if col != row_id_col]
    return df[cols]