import pandas as pd
import geopandas as gpd
from geopandas.tools import clip
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
from geopandas.tools import clip
from shapely.ops import transform
from rapidfuzz.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


def match_facilities(df1, df2, id_col1="id_1", id_col2="id_2",
                     name_col1="facility_name_1", name_col2="facility_name_2",
                     buffer_km=10, geometry_col="geometry"):
    """
    Matches facilities from two datasets based on spatial proximity and name similarity.

    **Similarity metrics:**
        - `partial_ratio`: Measures if one name is a substring of another (e.g., "Detour Lake" vs "Detour Lake Project").
        - `token_set_ratio`: Compares sets of words, ignoring order and duplicates — best for messy or reordered names
                             (e.g., "Hemlo (Williams)" vs "Williams Mine").

    **Parameters:**
        df1, df2 (pd.DataFrame or gpd.GeoDataFrame): Datasets with facility coordinates and names.
        id_col1, id_col2 (str): Column names for unique IDs in df1 and df2.
        name_col1, name_col2 (str): Column names for facility names in df1 and df2.
        buffer_km (float): Radius (in kilometers) to consider spatially close facilities.
        geometry_col (str): Column for geometry if using pre-made GeoDataFrames.

    **Returns:**
        pd.DataFrame: Matched rows with distance and both similarity scores.
    """

    # Ensure GeoDataFrame format
    if not isinstance(df1, gpd.GeoDataFrame):
        df1 = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1["longitude"], df1["latitude"]), crs="EPSG:4326")
    if not isinstance(df2, gpd.GeoDataFrame):
        df2 = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2["longitude"], df2["latitude"]), crs="EPSG:4326")

    # Project to EPSG:3978 (meters-based) for buffer/distance
    df1 = df1.to_crs(epsg=3978)
    df2 = df2.to_crs(epsg=3978)

    buffer_m = buffer_km * 1000
    df1["buffer"] = df1.geometry.buffer(buffer_m)
    df2["buffer"] = df2.geometry.buffer(buffer_m)

    matches = []

    for _, row1 in df1.iterrows():
        possible_matches = df2[df2["buffer"].intersects(row1["buffer"])]
        if possible_matches.empty:
            matches.append({
                id_col1: row1[id_col1],
                id_col2: None,
                "distance_m": None,
                name_col1: row1.get(name_col1, None),
                name_col2: None,
                "similarity_partial_score": None,
                "similarity_token_set": None
            })
        else:
            for _, row2 in possible_matches.iterrows():
                distance_m = row1.geometry.distance(row2.geometry)

                name1 = str(row1.get(name_col1, ""))
                name2 = str(row2.get(name_col2, ""))
                similarity_partial_score = partial_ratio(name1, name2)
                similarity_token_set = token_set_ratio(name1, name2)

                matches.append({
                    id_col1: row1[id_col1],
                    id_col2: row2[id_col2],
                    "distance_m": round(distance_m, 2),
                    name_col1: name1,
                    name_col2: name2,
                    "similarity_partial_score": similarity_partial_score,
                    "similarity_token_set": similarity_token_set
                })

    unmatched_df2 = df2[~df2[id_col2].isin([m[id_col2] for m in matches if m[id_col2] is not None])]
    for _, row2 in unmatched_df2.iterrows():
        matches.append({
            id_col1: None,
            id_col2: row2[id_col2],
            "distance_m": None,
            name_col1: None,
            name_col2: row2.get(name_col2, None),
            "similarity_partial_score": None,
            "similarity_token_set": None
        })

    return pd.DataFrame(matches)


def assign_best_match(
    match_df,
    id_main_col,
    id_sat_col,
    name_main_col=None,
    name_sat_col=None,
    strategy="combined",  # Options: distance, similarity_partial, similarity_token_set, combined
    weight_similarity=0.7,
    weight_distance=0.3,
    similarity_metric="token_set"  # or "partial"
):
    """
    Assigns best matches from a match_facilities() result, using a strategy based on distance and/or name similarity.

    **Similarity options:**
        - "partial": uses RapidFuzz's partial_ratio (best when one name is a subset of the other)
        - "token_set": uses token_set_ratio (best for messy names with reordering or extra words)

    **Parameters:**
        match_df (pd.DataFrame): Output from match_facilities().
        id_main_col (str): ID column for the main facilities table.
        id_sat_col (str): ID column for the satellite table (e.g. GHG, waste).
        name_main_col (str): Optional, name column in main table (for inspection/debug).
        name_sat_col (str): Optional, name column in satellite table (for inspection/debug).
        strategy (str): Matching strategy. Options: "distance", "similarity_partial", "similarity_token_set", "combined".
        weight_similarity (float): Weight for similarity when using "combined".
        weight_distance (float): Weight for inverse distance when using "combined".
        similarity_metric (str): "partial" or "token_set" for use in "combined" strategy.

    **Returns:**
        pd.DataFrame: Columns [id_sat_col, id_main_col] for best match assignments.
    """
    df = match_df.dropna(subset=[id_main_col, id_sat_col]).copy()

    if strategy == "combined":
        if similarity_metric == "partial":
            sim_col = "similarity_partial_score"
        elif similarity_metric == "token_set":
            sim_col = "similarity_token_set"
        else:
            raise ValueError("similarity_metric must be 'partial' or 'token_set'")

        df["score"] = (
            weight_similarity * df[sim_col].fillna(0) +
            weight_distance * (1 / (1 + df["distance_m"]))
        )
        df = df.sort_values("score", ascending=False).drop_duplicates(subset=[id_sat_col])

    elif strategy == "distance":
        df = df.sort_values("distance_m", ascending=True).drop_duplicates(subset=[id_sat_col])

    elif strategy == "similarity_partial":
        df = df.sort_values("similarity_partial_score", ascending=False).drop_duplicates(subset=[id_sat_col])

    elif strategy == "similarity_token_set":
        df = df.sort_values("similarity_token_set", ascending=False).drop_duplicates(subset=[id_sat_col])

    else:
        raise ValueError("strategy must be one of: 'distance', 'similarity_partial', 'similarity_token_set', 'combined'")

    return df[[id_sat_col, id_main_col]]



### BUFFER
def match_facility_to_polygons_with_buffer(facility_df, polygon_gdf,
                                 facility_id_col="facility_id", polygon_id_col="polygon_id",
                                 buffer_km=10, crs="EPSG:3978"):
    """
    Matches facility (points) with polygons based on spatial proximity using a buffer.

    Parameters:
        facility_df (pd.DataFrame or gpd.GeoDataFrame): Facility dataset with lat/lon.
        polygon_gdf (gpd.GeoDataFrame): Polygons dataset.
        facility_id_col (str): Unique facility ID column in facility_df.
        polygon_id_col (str): Unique polygon ID column in polygon_gdf.
        buffer_km (float): Buffer radius in kilometers.
        crs (str): CRS for spatial analysis, default is EPSG:3978.

    Returns:
        pd.DataFrame: Matches with facility and polygon IDs, including unmatched facilities & polygons.
    """

    # Convert facility_df to GeoDataFrame if necessary
    if not isinstance(facility_df, gpd.GeoDataFrame):
        facility_gdf = gpd.GeoDataFrame(
            facility_df,
            geometry=gpd.points_from_xy(facility_df["longitude"], facility_df["latitude"]),
            crs="EPSG:4326"
        ).to_crs(crs)
    else:
        facility_gdf = facility_df.to_crs(crs)

    # Convert polygons to the same CRS
    polygon_gdf = polygon_gdf.to_crs(crs)

    # Remove Z-dimension from polygons if necessary
    polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(lambda geom: transform(lambda x, y, z=None: (x, y), geom))

    # Create buffer around facility points
    buffer_m = buffer_km * 1000  # Convert km to meters
    facility_gdf["buffer"] = facility_gdf.geometry.buffer(buffer_m)

    matches = []
    matched_facilities = set()
    matched_polygons = set()

    # Iterate over facility points to find matching polygons
    for _, facility in facility_gdf.iterrows():
        possible_matches = polygon_gdf[polygon_gdf.geometry.intersects(facility["buffer"])]

        if not possible_matches.empty:
            for _, polygon in possible_matches.iterrows():
                # Distance from facility point to polygon centroid
                distance_to_centroid_m = facility.geometry.distance(polygon.geometry.centroid)

                # Distance from facility point to polygon boundary (minimum distance)
                distance_to_edge_m = facility.geometry.distance(polygon.geometry.boundary)

                matches.append({
                    facility_id_col: facility[facility_id_col],
                    polygon_id_col: polygon[polygon_id_col],
                    "distance_to_centroid_m": round(distance_to_centroid_m, 2),
                    "distance_to_edge_m": round(distance_to_edge_m, 2)
                })

                # Track matched IDs
                matched_facilities.add(facility[facility_id_col])
                matched_polygons.add(polygon[polygon_id_col])
        else:
            # Facility has no matching polygon
            matches.append({
                facility_id_col: facility[facility_id_col],
                polygon_id_col: None,
                "distance_to_centroid_m": None,
                "distance_to_edge_m": None
            })

    # Add unmatched polygons (transparency: polygons with no linked facility)
    unmatched_polygons = polygon_gdf[~polygon_gdf[polygon_id_col].isin(matched_polygons)]
    for _, polygon in unmatched_polygons.iterrows():
        matches.append({
            facility_id_col: None,
            polygon_id_col: polygon[polygon_id_col],
            "distance_to_centroid_m": None,
            "distance_to_edge_m": None
        })

    # Convert results to DataFrame
    match_df = pd.DataFrame(matches)

    return match_df


### CLUSTERING
def cluster_sites_and_polygons(
    facility_gdf, polygon_gdf, tailing_gdf,
    facility_id_col="main_id",
    polygon_id_col="tang_id",
    tailing_id_col="tailing_id",
    eps_km=10, min_samples=2, crs="EPSG:3978", boundary_step=5):
    """
    Cluster facilities, tailings, and polygons spatially using DBSCAN, and assign shared cluster IDs.

    Each point or polygon will receive a 'cluster_id'.
    A 'check_manually' flag is added when multiple facilities or tailings are grouped together.

    Parameters:
    - facility_gdf (GeoDataFrame): Point data of facilities with a unique ID column.
    - polygon_gdf (GeoDataFrame): Polygon data to match, with a unique ID column.
    - tailing_gdf (GeoDataFrame): Point data of tailings, also with unique ID column.
    - facility_id_col (str): Column name for unique facility ID.
    - polygon_id_col (str): Column name for unique polygon ID.
    - tailing_id_col (str): Column name for unique tailing ID.
    - eps_km (float): Max distance (in km) to cluster together (DBSCAN's `eps`).
    - min_samples (int): Minimum number of samples to form a cluster (DBSCAN).
    - crs (str): CRS for accurate distance computation (default: EPSG:3978 = Canada Albers).
    - boundary_step (int): Sampling step for polygon boundary points (1 = no simplification).

    Returns:
    - facility_gdf (GeoDataFrame): With added 'cluster_id' and 'check_manually' columns.
    - polygon_gdf (GeoDataFrame): Same.
    - tailing_gdf (GeoDataFrame): Same.
    """

    import pandas as pd
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon
    from sklearn.cluster import DBSCAN

    # Ensure consistent CRS
    facility_gdf = facility_gdf.to_crs(crs)
    tailing_gdf = tailing_gdf.to_crs(crs)
    polygon_gdf = polygon_gdf.to_crs(crs).explode(index_parts=False)

    # Clean Z-dimension from polygons
    def to_2d(geom):
        if isinstance(geom, MultiPolygon):
            return MultiPolygon([Polygon([(x, y) for x, y, *_ in poly.exterior.coords]) for poly in geom.geoms])
        elif isinstance(geom, Polygon):
            return Polygon([(x, y) for x, y, *_ in geom.exterior.coords])
        return geom

    polygon_gdf["geometry"] = polygon_gdf["geometry"].apply(to_2d)

    # Extract polygon boundary points
    def boundary_points(geom, pid):
        coords = list(geom.exterior.coords)[::boundary_step]
        return [(pid, Point(c)) for c in coords]

    poly_points = []
    for _, row in polygon_gdf.iterrows():
        poly_points.extend(boundary_points(row["geometry"], row[polygon_id_col]))

    poly_gdf = gpd.GeoDataFrame(poly_points, columns=[polygon_id_col, "geometry"], crs=crs)
    poly_gdf[facility_id_col] = None
    poly_gdf[tailing_id_col] = None

    # Prepare facility and tailing points
    facility_tmp = facility_gdf[[facility_id_col, "geometry"]].copy()
    facility_tmp[polygon_id_col] = None
    facility_tmp[tailing_id_col] = None

    tailing_tmp = tailing_gdf[[tailing_id_col, "geometry"]].copy()
    tailing_tmp[polygon_id_col] = None
    tailing_tmp[facility_id_col] = None

    # Combine all points for clustering
    all_points = pd.concat([facility_tmp, tailing_tmp, poly_gdf], ignore_index=True)
    all_points = all_points[all_points.geometry.notna()].copy()


    # Get coordinates and run clustering
    coords = np.array([(geom.x, geom.y) for geom in all_points.geometry])
    db = DBSCAN(eps=eps_km * 1000, min_samples=min_samples).fit(coords)
    all_points["cluster_id"] = db.labels_

    # Cluster statistics
    cluster_stats = all_points.groupby("cluster_id").agg({
        facility_id_col: lambda x: x.notna().sum(),
        tailing_id_col: lambda x: x.notna().sum(),
    }).rename(columns={
        facility_id_col: "n_facilities",
        tailing_id_col: "n_tailings"
    }).reset_index()

    cluster_stats["check_manually"] = (cluster_stats["n_facilities"] > 1) | (cluster_stats["n_tailings"] > 1)

    # Merge stats back
    all_points = all_points.merge(cluster_stats[["cluster_id", "check_manually"]], on="cluster_id", how="left")

    # Assign back to original GeoDataFrames
    def assign_cluster(df, id_col):
        cluster_info = all_points[[id_col, "cluster_id", "check_manually"]].dropna(subset=[id_col])
        return df.merge(cluster_info, on=id_col, how="left")

    facility_gdf = assign_cluster(facility_gdf, facility_id_col)
    tailing_gdf = assign_cluster(tailing_gdf, tailing_id_col)
    polygon_gdf = assign_cluster(polygon_gdf, polygon_id_col)

    return facility_gdf, polygon_gdf, tailing_gdf