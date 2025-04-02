import pandas as pd
import geopandas as gpd
from geopandas.tools import clip
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
from geopandas.tools import clip
from shapely.ops import transform
from rapidfuzz import fuzz
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


def match_facilities(df1, df2, id_col1="id_1", id_col2="id_2",
                          name_col1="facility_name_1", name_col2="facility_name_2",
                          buffer_km=10, geometry_col="geometry"):
    """
    Matches facilities from two datasets based on two-sided geographical buffering and name similarity.
    Full and partial similarity scores are computed. Prioritization is **not included**.

    **Parameters:**
        df1 (pd.DataFrame or gpd.GeoDataFrame): First dataset with facility locations.
        df2 (pd.DataFrame or gpd.GeoDataFrame): Second dataset with facility locations.
        id_col1 (str): Column name for unique IDs in df1.
        id_col2 (str): Column name for unique IDs in df2.
        name_col1 (str): Column name for facility names in df1 (optional).
        name_col2 (str): Column name for facility names in df2 (optional).
        buffer_km (float): Search radius (in km) for geographical matching.
        geometry_col (str): Column name for geometry (if using GeoDataFrames).

    **Returns:**
        pd.DataFrame: All matches within the buffer, including distance and both similarity scores.
    """

    # Convert to GeoDataFrames if not already
    if not isinstance(df1, gpd.GeoDataFrame):
        df1 = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1["longitude"], df1["latitude"]), crs="EPSG:4326")
    if not isinstance(df2, gpd.GeoDataFrame):
        df2 = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2["longitude"], df2["latitude"]), crs="EPSG:4326")

    # Convert both datasets to EPSG:3978 (meters-based coordinate system for Canada)
    df1 = df1.to_crs(epsg=3978)
    df2 = df2.to_crs(epsg=3978)

    # Create a buffer of `buffer_km` converted to meters for both datasets
    buffer_m = buffer_km * 1000  # Convert km to meters
    df1["buffer"] = df1.geometry.buffer(buffer_m)
    df2["buffer"] = df2.geometry.buffer(buffer_m)

    matches = []

    # Iterate over df1 rows to find matches in df2
    for _, row1 in df1.iterrows():
        possible_matches = df2[df2["buffer"].intersects(row1["buffer"])]
        if possible_matches.empty:
            matches.append({
                id_col1: row1[id_col1],
                id_col2: None,
                "distance_m": None,
                name_col1: row1.get(name_col1, None),
                name_col2: None,
                "similarity_full_score": None,
                "similarity_partial_score": None
            })
        else:
            for _, row2 in possible_matches.iterrows():
                distance_m = row1.geometry.distance(row2.geometry)

                # Compute similarity scores
                similarity_full_score = None
                similarity_partial_score = None
                if name_col1 in df1.columns and name_col2 in df2.columns:
                    similarity_full_score = fuzz.ratio(str(row1[name_col1]), str(row2[name_col2]))
                    similarity_partial_score = fuzz.partial_ratio(str(row1[name_col1]), str(row2[name_col2]))

                matches.append({
                    id_col1: row1[id_col1],
                    id_col2: row2[id_col2],
                    "distance_m": round(distance_m, 2),
                    name_col1: row1.get(name_col1, None),
                    name_col2: row2.get(name_col2, None),
                    "similarity_full_score": similarity_full_score,
                    "similarity_partial_score": similarity_partial_score
                })

    # Find unmatched facilities in df2
    unmatched_df2 = df2[~df2[id_col2].isin([m[id_col2] for m in matches if m[id_col2] is not None])]
    for _, row2 in unmatched_df2.iterrows():
        matches.append({
            id_col1: None,
            id_col2: row2[id_col2],
            "distance_m": None,
            name_col1: None,
            name_col2: row2.get(name_col2, None),
            "similarity_full_score": None,
            "similarity_partial_score": None
        })

    # Convert matches to a DataFrame
    full_matches_df = pd.DataFrame(matches)

    return full_matches_df


def match_facilities_unique(df1, df2, id_col1="id_1", id_col2="id_2",
                            name_col1="facility_name_1", name_col2="facility_name_2",
                            buffer_km=10, geometry_col="geometry", similarity_threshold=80):
    """
    Matches facilities from two datasets based on geographical proximity and name similarity.
    Returns a SQL-friendly table where each row represents a facility, with matched facilities on the same row.

    Matches are prioritized = we store only the best match
     - Only facilities within the buffer (e.g. 10km) are considered as possible matches
     - Among the possible matches, the facility with the highest name similarity score is selected as match
     - If the similarity score is the same, the one with the shortest distance is selected

    Parameters:
        df1 (pd.DataFrame or gpd.GeoDataFrame): First dataset with facility locations.
        df2 (pd.DataFrame or gpd.GeoDataFrame): Second dataset with facility locations.
        id_col1 (str): Column name for unique IDs in df1.
        id_col2 (str): Column name for unique IDs in df2.
        name_col1 (str): Column name for facility names in df1 (optional).
        name_col2 (str): Column name for facility names in df2 (optional).
        buffer_km (float): Search radius (in km) for geographical matching.
        geometry_col (str): Column name for geometry (if using GeoDataFrames).
        similarity_threshold (int): Minimum similarity score (0-100) to consider as a name match.

    Returns:
        pd.DataFrame: A SQL-friendly table with matched and unmatched facilities.
    """

    # Convert to GeoDataFrames if not already
    if not isinstance(df1, gpd.GeoDataFrame):
        df1 = gpd.GeoDataFrame(df1, geometry=gpd.points_from_xy(df1["longitude"], df1["latitude"]), crs="EPSG:4326")
    if not isinstance(df2, gpd.GeoDataFrame):
        df2 = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2["longitude"], df2["latitude"]), crs="EPSG:4326")

    # Convert both datasets to EPSG:3978 (meters-based coordinate system for Canada)
    df1 = df1.to_crs(epsg=3978)
    df2 = df2.to_crs(epsg=3978)

    # Create a buffer of `buffer_km` converted to meters
    buffer_m = buffer_km * 1000  # Convert km to meters
    df1["buffer"] = df1.geometry.buffer(buffer_m)

    matches = []

    # Track matched IDs
    matched_ids_1 = set()
    matched_ids_2 = set()

    # Iterate over df1 rows to find matches in df2
    for _, row1 in df1.iterrows():
        possible_matches = df2[df2.geometry.within(row1["buffer"])]

        best_match = None
        best_score = -1
        best_distance = None

        for _, row2 in possible_matches.iterrows():
            distance_m = row1.geometry.distance(row2.geometry)

            similarity_score = None
            if name_col1 in df1.columns and name_col2 in df2.columns:
                similarity_score = fuzz.ratio(str(row1[name_col1]), str(row2[name_col2]))

            # Keep the best match (highest similarity score)
            if similarity_score is not None and similarity_score > best_score:
                best_match = row2
                best_score = similarity_score
                best_distance = distance_m

        # If a match is found
        if best_match is not None and best_score >= similarity_threshold:
            matches.append({
                id_col1: row1[id_col1],
                id_col2: best_match[id_col2],
                "distance_m": round(best_distance, 2),
                name_col1: row1[name_col1] if name_col1 in df1.columns else None,
                name_col2: best_match[name_col2] if name_col2 in df2.columns else None,
                "similarity_score": best_score
            })
            matched_ids_1.add(row1[id_col1])
            matched_ids_2.add(best_match[id_col2])
        else:
            # If no match is found, keep the unmatched facility from df1
            matches.append({
                id_col1: row1[id_col1],
                id_col2: None,
                "distance_m": None,
                name_col1: row1[name_col1] if name_col1 in df1.columns else None,
                name_col2: None,
                "similarity_score": None
            })

    # Find unmatched facilities in df2 (those not matched in df1)
    unmatched_df2 = df2[~df2[id_col2].isin(matched_ids_2)]
    for _, row2 in unmatched_df2.iterrows():
        matches.append({
            id_col1: None,
            id_col2: row2[id_col2],
            "distance_m": None,
            name_col1: None,
            name_col2: row2[name_col2] if name_col2 in df2.columns else None,
            "similarity_score": None
        })

    # Convert matches to a DataFrame
    match_df = pd.DataFrame(matches)

    return match_df


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