import pandas as pd
import numpy as np
import hashlib
import random

import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd
from geopandas.tools import clip
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
from geopandas.tools import clip
from shapely.ops import transform

from rapidfuzz import fuzz

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


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
        temp_df["source_df"] = source_name

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


def assign_id(df, canada_provinces, id_column="main_id", prefix="OTH", geometry_col="geometry"):
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
        raise ValueError("The input df must be a GeoDataFrame with a geometry column.")

    # Extract centroid for polygons & coordinates for points
    df["longitude"], df["latitude"] = zip(*df.apply(extract_coordinates, axis=1))

    # Add a counter for duplicate locations
    df["location_count"] = df.groupby(["longitude", "latitude"]).cumcount() + 1

    def generate_id(row):
        """Generates a unique and deterministic ID."""
        lon, lat = row["longitude"], row["latitude"]
        province_code = get_province(lon, lat)  # Infer province
        facility_name = row.get("facility_name", "Unknown")

        # Create a deterministic hash
        hash_input = f"{prefix}|{facility_name}|{lat}|{lon}"
        unique_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        # Ensure uniqueness by appending location_count for duplicate coordinates
        return f"{province_code}-{prefix}-{unique_hash}-{row['location_count']}"

    # Apply the ID assignment
    df[id_column] = df.apply(generate_id, axis=1)

    # Drop unnecessary columns
    df = df.drop(columns=['location_count'])

    # Ensure ID column is first
    cols = [id_column] + [col for col in df.columns if col != id_column]
    return df[cols]


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


def assign_row_id(df, id_column="id", prefix="MIN", num_digits=13):
    """
    Assigns unique IDs to each row of a DataFrame.

    - If `id_column` does not exist, it creates it as the first column.
    - If `id_column` exists but contains NaN values, it fills them with unique generated IDs.
    - Ensures uniqueness by avoiding duplicates within the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - id_column (str): The name of the ID column to create or fill (default is "id").
    - prefix (str): The prefix for the generated IDs (default is "MIN").
    - num_digits (int): The number of digits in the numeric part of the ID (default is 13).

    Returns:
    - pd.DataFrame: Updated DataFrame with unique IDs assigned.
    """
    if id_column not in df.columns:
        df.insert(0, id_column, None)  # Create the column as the first column

    # Ensure uniqueness by tracking existing IDs
    existing_ids = set(df[id_column].dropna().astype(str))  # Convert to str to match generated format
    new_ids = []

    while len(new_ids) < df[id_column].isna().sum():
        new_id = f"{prefix}-{random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1)}"
        if new_id not in existing_ids:
            existing_ids.add(new_id)
            new_ids.append(new_id)

    # Assign only to missing values
    df.loc[df[id_column].isna(), id_column] = new_ids

    return df


# def add_geospatial_info(facility_df, other_df, matching_columns, buffer_distance=10000, crs="EPSG:4326"):
#     """
#     Add information from another DataFrame to facility_df based on geospatial matching.
#
#     Parameters:
#         facility_df (pd.DataFrame): The main facility DataFrame.
#         other_df (pd.DataFrame): The secondary DataFrame with additional information.
#         matching_columns (dict): Columns to add from other_df. Format: {"source_column": "target_column"}.
#         buffer_distance (float): Buffer distance in meters for proximity matching.
#         crs (str): Coordinate Reference System, default is WGS 84 (EPSG:4326).
#
#     Returns:
#         pd.DataFrame: The updated facility_df with added information.
#     """
#     # Convert facility_df and other_df to GeoDataFrames
#     facility_gdf = gpd.GeoDataFrame(
#         facility_df,
#         geometry=gpd.points_from_xy(facility_df["longitude"], facility_df["latitude"]),
#         crs=crs,
#     )
#     other_gdf = gpd.GeoDataFrame(
#         other_df,
#         geometry=gpd.points_from_xy(other_df["longitude"], other_df["latitude"]),
#         crs=crs,
#     )
#
#     # Reproject to a projected CRS for accurate buffering
#     facility_gdf = facility_gdf.to_crs("EPSG:3978")
#     other_gdf = other_gdf.to_crs("EPSG:3978")
#
#     # Create a buffer around each facility
#     facility_gdf["geometry"] = facility_gdf["geometry"].buffer(buffer_distance)
#
#     # Perform a spatial join to find matches within the buffer
#     joined_gdf = gpd.sjoin(other_gdf, facility_gdf, how="inner", predicate="within")
#
#     # Drop duplicate matches and aggregate if necessary
#     joined_gdf = joined_gdf.groupby("index_right").first()
#
#     # Add the matching columns to facility_gdf
#     for source_col, target_col in matching_columns.items():
#         if source_col in other_gdf.columns:
#             facility_gdf[target_col] = joined_gdf[source_col]
#
#     # Reproject back to the original CRS
#     facility_gdf = facility_gdf.to_crs(crs)
#
#     # Drop buffer geometry for clean output
#     facility_gdf = facility_gdf.drop(columns="geometry")
#
#     return pd.DataFrame(facility_gdf)


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


# def merge_gdf(facility_df, other_df, buffer_distance=10000, crs="EPSG:4326"):
#     """
#     Add all columns from another DataFrame to facility_df based on geospatial proximity.
#
#     Works with either 'facility_id' or 'project_id' as the main identifier.
#
#     Parameters:
#         facility_df (pd.DataFrame): The main facility DataFrame.
#         other_df (pd.DataFrame): The secondary DataFrame with additional information.
#         buffer_distance (float): Buffer distance in meters for proximity matching.
#         crs (str): Coordinate Reference System, default is WGS 84 (EPSG:4326).
#
#     Returns:
#         pd.DataFrame: The updated facility_df with added information.
#     """
#
#     # Detect whether the primary key is 'facility_id' or 'project_id'
#     primary_id = "facility_id" if "facility_id" in facility_df.columns else "project_id" if "project_id" in facility_df.columns else None
#
#     if primary_id is None:
#         raise ValueError("Neither 'facility_id' nor 'project_id' found in facility_df.")
#
#     # Keep only relevant columns from facility_df
#     facility_df = facility_df[[primary_id, 'longitude', 'latitude']]
#
#     # Convert facility_df and other_df to GeoDataFrames
#     facility_gdf = gpd.GeoDataFrame(
#         facility_df,
#         geometry=gpd.points_from_xy(facility_df["longitude"], facility_df["latitude"]),
#         crs=crs,
#     )
#     other_gdf = gpd.GeoDataFrame(
#         other_df,
#         geometry=gpd.points_from_xy(other_df["longitude"], other_df["latitude"]),
#         crs=crs,
#     )
#
#     # Reproject to a projected CRS for accurate buffering
#     facility_gdf = facility_gdf.to_crs("EPSG:3978")
#     other_gdf = other_gdf.to_crs("EPSG:3978")
#
#     # Create a buffer around each facility **as a separate column**
#     facility_gdf["buffer_geom"] = facility_gdf.geometry.buffer(buffer_distance)
#
#     # Perform a spatial join using buffer geometry
#     joined_gdf = gpd.sjoin(other_gdf, facility_gdf.set_geometry("buffer_geom"), how="inner", predicate="within")
#
#     # Rename spatial join columns to standard names
#     joined_gdf.rename(columns={"geometry_right": "geometry"}, inplace=True)
#
#     # Drop duplicate matches and keep only the first match per facility
#     joined_gdf = joined_gdf.groupby("index_right").first()
#
#     # Merge all columns from other_df back into facility_gdf (fixing suffixes)
#     facility_gdf = facility_gdf.merge(
#         joined_gdf.drop(columns=["geometry"]),
#         left_index=True,
#         right_index=True,
#         how="left",
#         suffixes=("", "_matched")  # Prevent _x and _y mess
#     )
#
#     # Remove unnecessary columns created by spatial join
#     drop_cols = ["buffer_geom", "longitude_right", "latitude_right", "longitude_left", "latitude_left", "geometry_left",
#                  f"{primary_id}_matched"]
#     facility_gdf.drop(columns=[col for col in drop_cols if col in facility_gdf.columns], inplace=True)
#
#     # Fix ID duplication issue
#     if f"{primary_id}_x" in facility_gdf.columns:
#         facility_gdf.rename(columns={f"{primary_id}_x": primary_id}, inplace=True)
#
#     if f"{primary_id}_y" in facility_gdf.columns:
#         facility_gdf.drop(columns=[f"{primary_id}_y"], inplace=True)
#
#     # Reproject back to original CRS
#     facility_gdf = facility_gdf.to_crs(crs)
#
#     # Convert back to a DataFrame
#     return pd.DataFrame(facility_gdf)


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

