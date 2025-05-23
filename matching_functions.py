import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import re
from shapely.ops import transform, unary_union
from rapidfuzz.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.cluster import DBSCAN


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


def one_to_many_relationships(
    match_df,
    id_main_col,
    id_sat_col,
    distance_threshold_m=2000,
    similarity_threshold=80,
    similarity_metric="token_set"  # "partial" or "token_set"
):
    """
    Give me the single best match for each satellite entry — only if it passes the thresholds.

    Parameters:
        match_df (pd.DataFrame): Output from match_facilities().
        id_main_col (str): ID column for the main facilities table.
        id_sat_col (str): ID column for the satellite table.
        distance_threshold_m (float): Max distance in meters.
        similarity_threshold (float): Min similarity score (0–100).
        similarity_metric (str): 'partial' or 'token_set'.

    Returns:
        pd.DataFrame: Best matches [id_sat_col, id_main_col] or NaN if no good match found.
    """
    df = match_df.dropna(subset=[id_main_col, id_sat_col]).copy()

    # Select the similarity column
    sim_col = {
        "partial": "similarity_partial_score",
        "token_set": "similarity_token_set"
    }.get(similarity_metric, "similarity_token_set")

    # Apply threshold filtering
    df = df[
        (df["distance_m"] <= distance_threshold_m) &
        (df[sim_col] >= similarity_threshold)
    ]

    # Pick the best match per satellite ID
    df["score"] = df[sim_col] + (1 / (1 + df["distance_m"]))
    df = df.sort_values("score", ascending=False).drop_duplicates(subset=[id_sat_col])

    return df[[id_sat_col, id_main_col]]


def filter_matches(
    match_df,
    distance_threshold_m=None,
    similarity_threshold=None,
    similarity_metric="token_set",  # or "partial"
    strategy="and"
):
    """
    Give me all matches that meet the thresholds..

    Parameters:
        match_df (pd.DataFrame): Output from match_facilities().
        distance_threshold_m (float or None): Max distance allowed.
        similarity_threshold (float or None): Min similarity score allowed (0–100).
        similarity_metric (str): 'partial' or 'token_set'.
        strategy (str): 'and' (both conditions) or 'or' (either passes).

    Returns:
        pd.DataFrame: Filtered match candidates.
    """
    df = match_df.copy()

    sim_col = {
        "partial": "similarity_partial_score",
        "token_set": "similarity_token_set"
    }.get(similarity_metric, "similarity_token_set")

    conditions = []
    if distance_threshold_m is not None:
        conditions.append(df["distance_m"] <= distance_threshold_m)
    if similarity_threshold is not None:
        conditions.append(df[sim_col] >= similarity_threshold)

    if conditions:
        if strategy == "and":
            combined = conditions[0]
            for cond in conditions[1:]:
                combined &= cond
        elif strategy == "or":
            combined = conditions[0]
            for cond in conditions[1:]:
                combined |= cond
        else:
            raise ValueError("strategy must be 'and' or 'or'")
        df = df[combined]

    return df.reset_index(drop=True)


# def match_facility_to_polygons_with_buffer(
#     facility_df, polygon_gdf,
#     facility_id_col="facility_id", polygon_id_col="polygon_id",
#     buffer_km=10, crs="EPSG:3978"
# ):
#     """
#     Returns only matched facility–polygon pairs (excluding unmatched records).
#     """
#
#     # Convert to GeoDataFrame if needed
#     if not isinstance(facility_df, gpd.GeoDataFrame):
#         facility_gdf = gpd.GeoDataFrame(
#             facility_df,
#             geometry=gpd.points_from_xy(facility_df["longitude"], facility_df["latitude"]),
#             crs="EPSG:4326"
#         ).to_crs(crs)
#     else:
#         facility_gdf = facility_df.to_crs(crs)
#
#     polygon_gdf = polygon_gdf.to_crs(crs)
#
#     # Remove Z-dimension if needed
#     polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(lambda geom: transform(lambda x, y, z=None: (x, y), geom))
#
#     # Buffer around facilities
#     buffer_m = buffer_km * 1000
#     facility_gdf["buffer"] = facility_gdf.geometry.buffer(buffer_m)
#
#     matches = []
#
#     for _, facility in facility_gdf.iterrows():
#         nearby_polygons = polygon_gdf[polygon_gdf.geometry.intersects(facility["buffer"])]
#         for _, polygon in nearby_polygons.iterrows():
#             matches.append({
#                 facility_id_col: facility[facility_id_col],
#                 polygon_id_col: polygon[polygon_id_col],
#                 "distance_to_centroid_m": round(facility.geometry.distance(polygon.geometry.centroid), 2),
#                 "distance_to_edge_m": round(facility.geometry.distance(polygon.geometry.boundary), 2)
#             })
#
#     return pd.DataFrame(matches)


def associate_facilities_near_polygons(
    facility_gdf,
    polygon_gdf,
    facility_id_col="main_id",
    polygon_id_col="protected_id",
    buffer_km=50,
    crs="EPSG:3978"
):
    """
    Optimized association between facilities and polygons using buffer + spatial join.

    Returns:
        DataFrame with: main_id, protected_id, distance_km, relation_type
    """

    # Project to common CRS
    facility_gdf = facility_gdf[[facility_id_col, "geometry"]].copy().to_crs(crs)
    polygon_gdf = polygon_gdf[[polygon_id_col, "geometry"]].copy().to_crs(crs)

    # Step 1: Buffer facilities
    buffer_m = buffer_km * 1000
    facility_gdf["buffer"] = facility_gdf.geometry.buffer(buffer_m)

    # Step 2: Spatial join to find intersecting polygons
    buffer_gdf = gpd.GeoDataFrame(
        facility_gdf[[facility_id_col, "buffer"]],
        geometry="buffer",
        crs=crs
    )
    joined = gpd.sjoin(buffer_gdf, polygon_gdf, predicate="intersects", how="inner")

    # Step 3: For each match, compute actual distance from point to polygon
    results = []
    for _, row in joined.iterrows():
        fid = row[facility_id_col]
        pid = row[polygon_id_col]
        dist_km = facility_gdf.loc[facility_gdf[facility_id_col] == fid].geometry.values[0].distance(
            polygon_gdf.loc[polygon_gdf[polygon_id_col] == pid].geometry.values[0]
        ) / 1000

        relation = "within_polygon" if facility_gdf.loc[facility_gdf[facility_id_col] == fid].geometry.values[0].within(
            polygon_gdf.loc[polygon_gdf[polygon_id_col] == pid].geometry.values[0]
        ) else "within_buffer"

        results.append({
            facility_id_col: fid,
            polygon_id_col: pid,
            "distance_km": round(dist_km, 3),
            "relation_type": relation
        })

    return pd.DataFrame(results)


def assign_polygons_to_points(
        facility_gdf,
        tailing_gdf,
        polygon_gdf,
        facility_id_col="main_id",
        tailing_id_col="tailing_id",
        polygon_id_col="tang_id",
        max_dist_km=10,
        crs="EPSG:3978"
):
    """
    Assign each polygon to the closest facility or tailing site, preferring entities contained
    within the polygon when possible. Computes true minimum geometric distance (not centroid-based).

    Returns:
        DataFrame with: main_id, tailing_id, tang_id, distance_km, relation_type
    """

    # Project all geometries
    fac = facility_gdf[[facility_id_col, "geometry"]].copy().to_crs(crs)
    fac["entity_type"] = "facility"
    fac = fac.rename(columns={facility_id_col: "entity_id"})

    tail = tailing_gdf[[tailing_id_col, "geometry"]].copy().to_crs(crs)
    tail["entity_type"] = "tailing"
    tail = tail.rename(columns={tailing_id_col: "entity_id"})

    entities = pd.concat([fac, tail], ignore_index=True)
    polygons = polygon_gdf[[polygon_id_col, "geometry"]].copy().to_crs(crs)

    assignments = []

    for _, row in polygons.iterrows():
        tang_id = row[polygon_id_col]
        poly_geom = row.geometry

        # Containment-based assignment
        contained = entities[entities.geometry.within(poly_geom)]

        if not contained.empty:
            for _, ent in contained.iterrows():
                main_id = ent["entity_id"] if ent["entity_type"] == "facility" else None
                tailing_id = ent["entity_id"] if ent["entity_type"] == "tailing" else None
                dist_km = ent.geometry.distance(poly_geom) / 1000
                assignments.append({
                    "main_id": main_id,
                    "tailing_id": tailing_id,
                    polygon_id_col: tang_id,
                    "distance_km": dist_km
                })

        else:
            # Fallback: closest entity by true geometry
            distances = entities.geometry.apply(lambda g: g.distance(poly_geom))
            nearest_idx = distances.idxmin()
            dist_km = distances[nearest_idx] / 1000

            if dist_km <= max_dist_km:
                ent = entities.loc[nearest_idx]
                main_id = ent["entity_id"] if ent["entity_type"] == "facility" else None
                tailing_id = ent["entity_id"] if ent["entity_type"] == "tailing" else None
                assignments.append({
                    "main_id": main_id,
                    "tailing_id": tailing_id,
                    polygon_id_col: tang_id,
                    "distance_km": dist_km
                })

    assigned_df = pd.DataFrame(assignments)

    # Determine which entity was assigned
    assigned_df["entity_id"] = assigned_df["main_id"].combine_first(assigned_df["tailing_id"])

    # Count how many polygons per entity
    poly_per_entity = assigned_df.groupby("entity_id")[polygon_id_col].nunique().rename("n_polygons")

    # Count how many entities per polygon
    entity_per_poly = assigned_df.groupby(polygon_id_col)["entity_id"].nunique().rename("n_entities")

    # Merge back to assignment table
    assigned_df = assigned_df.merge(poly_per_entity, on="entity_id", how="left")
    assigned_df = assigned_df.merge(entity_per_poly, on=polygon_id_col, how="left")

    def classify(row):
        if row["n_entities"] > 1:
            return "many-to-one"
        elif row["n_polygons"] > 1:
            return "one-to-many"
        else:
            return "one-to-one"

    assigned_df["relation_type"] = assigned_df.apply(classify, axis=1)

    return assigned_df.drop(columns=["entity_id", "n_entities", "n_polygons"])


### CLUSTERING
# def clustering_dbcan_old(
#     facility_gdf, polygon_gdf, tailing_gdf,
#     facility_id_col="main_id",
#     polygon_id_col="tang_id",
#     tailing_id_col="tailing_id",
#     eps_km=10, min_samples=2, crs="EPSG:3978", boundary_step=5
# ):
#     """
#     Cluster facilities, tailings, and polygon boundaries using DBSCAN.
#
#     Returns:
#     - facility_gdf, tailing_gdf, polygon_gdf: each with one cluster_id per entity (replacing -1 with "No cluster")
#     - cluster_link_df: tidy table of (main_id, tailing_id, tang_id, cluster_id, check_manually)
#     """
#
#     import pandas as pd
#     import numpy as np
#     import geopandas as gpd
#     from shapely.geometry import Point, Polygon, MultiPolygon
#     from shapely.ops import transform
#     from sklearn.cluster import DBSCAN
#
#     # Preserve original polygon_gdf
#     polygon_gdf_original = polygon_gdf.copy()
#
#     # Project all layers
#     facility_gdf = facility_gdf.to_crs(crs)
#     tailing_gdf = tailing_gdf.to_crs(crs)
#     polygon_gdf_proj = polygon_gdf.to_crs(crs).explode(index_parts=False).reset_index(drop=True)
#
#     # Clean Z-dimension from polygon geometries
#     def to_2d(geom):
#         if isinstance(geom, MultiPolygon):
#             return MultiPolygon([Polygon([(x, y) for x, y, *_ in poly.exterior.coords]) for poly in geom.geoms])
#         elif isinstance(geom, Polygon):
#             return Polygon([(x, y) for x, y, *_ in geom.exterior.coords])
#         return geom
#
#     polygon_gdf_proj["geometry"] = polygon_gdf_proj["geometry"].apply(to_2d)
#
#     # Sample polygon boundary points
#     def boundary_points(geom, pid):
#         coords = list(geom.exterior.coords)[::boundary_step]
#         return [(pid, Point(c)) for c in coords]
#
#     poly_points = []
#     for _, row in polygon_gdf_proj.iterrows():
#         poly_points.extend(boundary_points(row["geometry"], row[polygon_id_col]))
#
#     poly_gdf = gpd.GeoDataFrame(poly_points, columns=[polygon_id_col, "geometry"], crs=crs)
#     poly_gdf[facility_id_col] = None
#     poly_gdf[tailing_id_col] = None
#
#     # Prepare facility and tailing points
#     facility_tmp = facility_gdf[[facility_id_col, "geometry"]].copy()
#     facility_tmp[polygon_id_col] = None
#     facility_tmp[tailing_id_col] = None
#
#     tailing_tmp = tailing_gdf[[tailing_id_col, "geometry"]].copy()
#     tailing_tmp[polygon_id_col] = None
#     tailing_tmp[facility_id_col] = None
#
#     # Combine all points
#     all_points = pd.concat([facility_tmp, tailing_tmp, poly_gdf], ignore_index=True)
#     all_points = all_points[all_points.geometry.notna()].copy()
#
#     # Apply DBSCAN clustering
#     coords = np.array([(geom.x, geom.y) for geom in all_points.geometry])
#     db = DBSCAN(eps=eps_km * 1000, min_samples=min_samples).fit(coords)
#     all_points["cluster_id"] = db.labels_
#
#     # Extract one cluster_id per unique entity
#     def extract_entity_clusters(col_name):
#         return (
#             all_points[[col_name, "cluster_id"]]
#             .dropna(subset=[col_name])
#             .drop_duplicates(subset=[col_name])
#         )
#
#     facility_clusters = extract_entity_clusters(facility_id_col)
#     tailing_clusters = extract_entity_clusters(tailing_id_col)
#     polygon_clusters = extract_entity_clusters(polygon_id_col)
#
#     # Merge cluster_id into original GeoDataFrames
#     facility_gdf = facility_gdf.merge(facility_clusters, on=facility_id_col, how="left")
#     tailing_gdf = tailing_gdf.merge(tailing_clusters, on=tailing_id_col, how="left")
#     polygon_gdf = polygon_gdf_original.merge(polygon_clusters, on=polygon_id_col, how="left")
#
#     # Replace cluster_id = -1 with "No cluster"
#     for gdf in (facility_gdf, polygon_gdf, tailing_gdf):
#         if "cluster_id" in gdf.columns:
#             gdf["cluster_id"] = gdf["cluster_id"].fillna(-1).replace(-1, "No cluster")
#
#     # Replace -1 with "No cluster" in all_points for consistency
#     all_points["cluster_id"] = all_points["cluster_id"].fillna(-1).replace(-1, "No cluster")
#
#     # Count #facilities and #tailings per cluster
#     cluster_stats = all_points.groupby("cluster_id").agg({
#         facility_id_col: lambda x: x.notna().sum(),
#         tailing_id_col: lambda x: x.notna().sum()
#     }).rename(columns={
#         facility_id_col: "n_facilities",
#         tailing_id_col: "n_tailings"
#     }).reset_index()
#
#     cluster_stats["check_manually"] = (cluster_stats["n_facilities"] > 1) | (cluster_stats["n_tailings"] > 1)
#
#     # Create cluster link table: one row per entity with cluster_id
#     cluster_link_df = all_points[
#         [facility_id_col, tailing_id_col, polygon_id_col, "cluster_id"]
#     ].dropna(subset=["cluster_id"]).copy()
#
#     cluster_link_df = (
#         pd.concat([
#             cluster_link_df[[facility_id_col, "cluster_id"]].dropna().drop_duplicates(),
#             cluster_link_df[[tailing_id_col, "cluster_id"]].dropna().drop_duplicates(),
#             cluster_link_df[[polygon_id_col, "cluster_id"]].dropna().drop_duplicates()
#         ])
#         .reset_index(drop=True)
#     )
#
#     # Merge check_manually flag into cluster_link_df
#     cluster_link_df = cluster_link_df.merge(cluster_stats[["cluster_id", "check_manually"]], on="cluster_id", how="left")
#
#     # Final tidy output
#     return facility_gdf, polygon_gdf, tailing_gdf, cluster_link_df


# def clustering_dbcan(
#     facility_gdf,
#     polygon_gdf,
#     tailing_gdf,
#     facility_id_col="main_id",
#     polygon_id_col="tang_id",
#     tailing_id_col="tailing_id",
#     eps_km=10,
#     min_samples=2,
#     crs="EPSG:3978",
#     boundary_step=5
# ):
#     """
#     Cluster facilities, tailings, and polygon boundaries using DBSCAN.
#
#     Returns:
#     - cluster_link_df: DataFrame with columns cluster_id, main_ids, tailing_ids, tang_ids
#     - cluster_table: GeoDataFrame with cluster summary including geometry and area
#     """
#
#     # 1. Project all layers
#     fac = facility_gdf.to_crs(crs).copy()
#     tail = tailing_gdf.to_crs(crs).copy()
#     poly = polygon_gdf.to_crs(crs).explode(index_parts=False).reset_index(drop=True).copy()
#
#     # 2. Ensure 2D geometries
#     def to_2d(g):
#         if isinstance(g, MultiPolygon):
#             return MultiPolygon([Polygon([(x, y) for x, y, *_ in p.exterior.coords]) for p in g.geoms])
#         elif isinstance(g, Polygon):
#             return Polygon([(x, y) for x, y, *_ in g.exterior.coords])
#         return g
#     poly["geometry"] = poly["geometry"].apply(to_2d)
#
#     # 3. Sample boundary points
#     boundary_pts = []
#     for _, row in poly.iterrows():
#         g = row.geometry
#         if not isinstance(g, (Polygon, MultiPolygon)):
#             continue
#         coords = list(g.exterior.coords)[::boundary_step]
#         for x, y in coords:
#             boundary_pts.append({
#                 polygon_id_col: row[polygon_id_col],
#                 "geometry": Point(x, y),
#                 facility_id_col: None,
#                 tailing_id_col: None
#             })
#     poly_pts = gpd.GeoDataFrame(boundary_pts, crs=crs)
#
#     # 4. Facility and tailing points
#     fac_pts = fac[[facility_id_col, "geometry"]].copy()
#     fac_pts[polygon_id_col] = None
#     fac_pts[tailing_id_col] = None
#
#     tail_pts = tail[[tailing_id_col, "geometry"]].copy()
#     tail_pts[polygon_id_col] = None
#     tail_pts[facility_id_col] = None
#
#     # 5. Combine and DBSCAN clustering
#     all_pts = pd.concat([fac_pts, tail_pts, poly_pts], ignore_index=True).dropna(subset=["geometry"])
#     coords = np.array([(geom.x, geom.y) for geom in all_pts.geometry])
#     all_pts["cluster_id"] = DBSCAN(eps=eps_km * 1000, min_samples=min_samples).fit_predict(coords)
#
#     # 6. Assign cluster_id back to polygons
#     poly_cls = (
#         all_pts[[polygon_id_col, "cluster_id"]]
#         .dropna(subset=[polygon_id_col])
#         .drop_duplicates(subset=[polygon_id_col])
#     )
#     poly = poly.merge(poly_cls, on=polygon_id_col, how="left")
#
#     # 7. Valid cluster IDs (excluding noise)
#     valid = sorted(all_pts.loc[all_pts["cluster_id"] != -1, "cluster_id"].unique())
#
#     # 8. Build cluster_link_df with lists of IDs
#     def collect_ids(colname):
#         df = all_pts[[colname, "cluster_id"]].dropna(subset=[colname])
#         df = df[df["cluster_id"].isin(valid)].drop_duplicates()
#         return df.groupby("cluster_id")[colname].agg(list)
#
#     main_ids = collect_ids(facility_id_col)
#     tailing_ids = collect_ids(tailing_id_col)
#     tang_ids = collect_ids(polygon_id_col)
#
#     cluster_link_df = pd.DataFrame({
#         "cluster_id": valid,
#         "main_ids": [main_ids.get(cid, []) for cid in valid],
#         "tailing_ids": [tailing_ids.get(cid, []) for cid in valid],
#         "tang_ids": [tang_ids.get(cid, []) for cid in valid],
#     })
#
#     cluster_link_df["n_facilities"] = cluster_link_df["main_ids"].apply(len)
#     cluster_link_df["n_tailings"] = cluster_link_df["tailing_ids"].apply(len)
#     cluster_link_df["n_polygons"] = cluster_link_df["tang_ids"].apply(len)
#
#     # 9. Build cluster_table with geometry union and classification
#     rows = []
#     for _, row in cluster_link_df.iterrows():
#         cid = row["cluster_id"]
#         nf, nt, npol = row["n_facilities"], row["n_tailings"], row["n_polygons"]
#         ne = nf + nt
#
#         if npol == 0:
#             relation = "no relation"
#         elif ne == 0:
#             relation = 'no relation'
#         elif ne == 1 and npol == 1:
#             relation = "one-to-one"
#         elif ne == 1 and npol > 1:
#             relation = "one-to-many"
#         elif ne > 1 and npol == 1:
#             relation = "many-to-one"
#         else:
#             relation = "many-to-many"
#
#         if npol > 0:
#             poly_subset = poly[poly["cluster_id"] == cid]
#             geom = unary_union(poly_subset.geometry.tolist())
#             area = geom.area / 1e6
#         else:
#             geom = None
#             area = 0.0
#
#         rows.append({
#             "cluster_id": cid,
#             "n_facilities": nf,
#             "n_tailings": nt,
#             "n_polygons": npol,
#             "relation": relation,
#             "geometry": geom,
#             "area_km2": area
#         })
#
#     cluster_table = gpd.GeoDataFrame(rows, crs=crs)
#
#     return cluster_link_df, cluster_table
#
#
# def assign_refined_clusters(cluster_link_df, cluster_table, facility_gdf, tailing_gdf, polygon_gdf,
#                              facility_id_col="main_id", tailing_id_col="tailing_id", polygon_id_col="tang_id"):
#     """
#     Assign final_cluster_id to each facility, tailing, and polygon based on cluster relationships.
#     - Uses nearest-entity polygon assignment for many-to-many
#     - Computes final geometry and area per subcluster
#
#     Returns:
#     - facility_gdf, tailing_gdf, polygon_gdf with final_cluster_id
#     - cluster_table_out: final_cluster_id, geometry, area_km2_subcluster, relation
#     """
#
#     facility_gdf = facility_gdf.copy()
#     tailing_gdf = tailing_gdf.copy()
#     polygon_gdf = polygon_gdf.copy()
#
#     facility_gdf["final_cluster_id"] = None
#     tailing_gdf["final_cluster_id"] = None
#     polygon_gdf["final_cluster_id"] = None
#
#     new_cluster_rows = []
#
#     for _, cluster in cluster_table.iterrows():
#         base_cid = str(cluster["cluster_id"])
#         relation = cluster["relation"]
#
#         row = cluster_link_df[cluster_link_df["cluster_id"] == cluster["cluster_id"]]
#         if row.empty:
#             continue
#
#         mains = row.iloc[0]["main_ids"] if isinstance(row.iloc[0]["main_ids"], list) else []
#         tails = row.iloc[0]["tailing_ids"] if isinstance(row.iloc[0]["tailing_ids"], list) else []
#         polys = row.iloc[0]["tang_ids"] if isinstance(row.iloc[0]["tang_ids"], list) else []
#
#         poly_subset = polygon_gdf[polygon_gdf[polygon_id_col].isin(polys)]
#
#         ## === ONE TO ONE ===
#         if relation == "one-to-one":
#             sub_cid = base_cid
#             geom = unary_union(poly_subset.geometry)
#             area = geom.area / 1e6
#
#             facility_gdf.loc[facility_gdf[facility_id_col].isin(mains), "final_cluster_id"] = sub_cid
#             tailing_gdf.loc[tailing_gdf[tailing_id_col].isin(tails), "final_cluster_id"] = sub_cid
#             polygon_gdf.loc[polygon_gdf[polygon_id_col].isin(polys), "final_cluster_id"] = sub_cid
#
#             new_cluster_rows.append({
#                 "final_cluster_id": sub_cid,
#                 "n_facilities": len(mains),
#                 "n_tailings": len(tails),
#                 "n_polygons": len(polys),
#                 "relation": relation,
#                 "geometry": geom,
#                 "area_km2_subcluster": area
#             })
#
#         ## === ONE TO MANY ===
#         elif relation == "one-to-many":
#             sub_cid = base_cid
#             geom = unary_union(poly_subset.geometry)
#             area = geom.area / 1e6
#
#             facility_gdf.loc[facility_gdf[facility_id_col].isin(mains), "final_cluster_id"] = sub_cid
#             tailing_gdf.loc[tailing_gdf[tailing_id_col].isin(tails), "final_cluster_id"] = sub_cid
#             polygon_gdf.loc[polygon_gdf[polygon_id_col].isin(polys), "final_cluster_id"] = sub_cid
#
#             new_cluster_rows.append({
#                 "final_cluster_id": sub_cid,
#                 "n_facilities": len(mains),
#                 "n_tailings": len(tails),
#                 "n_polygons": len(polys),
#                 "relation": relation,
#                 "geometry": geom,
#                 "area_km2_subcluster": area
#             })
#
#         ## === MANY TO ONE ===
#         elif relation == "many-to-one":
#             union_geom = unary_union(poly_subset.geometry)
#             total_area = union_geom.area / 1e6
#             entities = mains + tails
#             n = len(entities)
#             share_geom = union_geom
#             share_area = total_area / n if n > 0 else 0
#
#             for i, ent_id in enumerate(entities):
#                 sub_cid = f"{base_cid}.{chr(97 + i)}"
#                 if ent_id in mains:
#                     facility_gdf.loc[facility_gdf[facility_id_col] == ent_id, "final_cluster_id"] = sub_cid
#                 else:
#                     tailing_gdf.loc[tailing_gdf[tailing_id_col] == ent_id, "final_cluster_id"] = sub_cid
#                 polygon_gdf.loc[polygon_gdf[polygon_id_col].isin(polys), "final_cluster_id"] = sub_cid
#
#                 new_cluster_rows.append({
#                     "final_cluster_id": sub_cid,
#                     "n_facilities": 1 if ent_id in mains else 0,
#                     "n_tailings": 1 if ent_id in tails else 0,
#                     "n_polygons": len(polys),
#                     "relation": relation,
#                     "geometry": share_geom,
#                     "area_km2_subcluster": share_area
#                 })
#
#         ## === MANY TO MANY (Nearest polygon assignment) ===
#                 ## === MANY TO MANY (Greedy nearest, each entity gets at least one polygon) ===
#         elif relation == "many-to-many":
#             entities = [(ent, "facility") for ent in mains] + [(ent, "tailing") for ent in tails]
#             entity_geoms = {
#                 ent: facility_gdf.loc[facility_gdf[facility_id_col] == ent, "geometry"].values[0]
#                 if typ == "facility" else
#                 tailing_gdf.loc[tailing_gdf[tailing_id_col] == ent, "geometry"].values[0]
#                 for ent, typ in entities
#             }
#
#             # Prepare polygon pool
#             unassigned_poly_ids = set(polys)
#             poly_df = polygon_gdf[polygon_gdf[polygon_id_col].isin(unassigned_poly_ids)].copy()
#             poly_df["centroid"] = poly_df.geometry.centroid
#
#             # Ensure we have enough polygons
#             if len(unassigned_poly_ids) < len(entities):
#                 print(f"⚠️ Cluster {base_cid} has fewer polygons than entities: {len(unassigned_poly_ids)} < {len(entities)}")
#
#             assigned_poly_ids = set()
#             entity_to_polys = {ent: [] for ent, _ in entities}
#
#             # Step 1: one polygon per entity (nearest)
#             for ent_id, _ in entities:
#                 if not unassigned_poly_ids:
#                     break  # no more polygons
#
#                 ent_geom = entity_geoms[ent_id]
#                 subset = poly_df[poly_df[polygon_id_col].isin(unassigned_poly_ids)].copy()
#                 subset["dist"] = subset["centroid"].distance(ent_geom)
#                 nearest_idx = subset["dist"].idxmin()
#                 nearest_poly_id = subset.loc[nearest_idx, polygon_id_col]
#
#                 entity_to_polys[ent_id].append(nearest_poly_id)
#                 assigned_poly_ids.add(nearest_poly_id)
#                 unassigned_poly_ids.remove(nearest_poly_id)
#
#             # Step 2: distribute remaining polygons
#             remaining_polys = list(unassigned_poly_ids)
#             if remaining_polys:
#                 for i, poly_id in enumerate(remaining_polys):
#                     ent_id = entities[i % len(entities)][0]
#                     entity_to_polys[ent_id].append(poly_id)
#
#             # Final assignment
#             for i, (ent_id, poly_ids) in enumerate(entity_to_polys.items()):
#                 sub_cid = f"{base_cid}.{chr(97 + i)}"
#                 if ent_id in mains:
#                     facility_gdf.loc[facility_gdf[facility_id_col] == ent_id, "final_cluster_id"] = sub_cid
#                 else:
#                     tailing_gdf.loc[tailing_gdf[tailing_id_col] == ent_id, "final_cluster_id"] = sub_cid
#                 polygon_gdf.loc[polygon_gdf[polygon_id_col].isin(poly_ids), "final_cluster_id"] = sub_cid
#
#                 subgeom = unary_union(polygon_gdf.loc[polygon_gdf[polygon_id_col].isin(poly_ids)].geometry)
#                 area = subgeom.area / 1e6
#
#                 new_cluster_rows.append({
#                     "final_cluster_id": sub_cid,
#                     "n_facilities": 1 if ent_id in mains else 0,
#                     "n_tailings": 1 if ent_id in tails else 0,
#                     "n_polygons": len(poly_ids),
#                     "relation": relation,
#                     "geometry": subgeom,
#                     "area_km2_subcluster": area
#                 })
#
#
#         ## === Fallback ===
#         else:
#             continue  # skip no-relation or invalid clusters
#
#     cluster_table_out = gpd.GeoDataFrame(new_cluster_rows, crs=cluster_table.crs)
#     return facility_gdf, tailing_gdf, polygon_gdf, cluster_table_out


def export_sqlite_db(db_path, tables_dict, keep_geometry_tables=None, csv_dir=None):
    """
    Export multiple (Geo)DataFrames to both SQLite and CSV with optional geometry as WKT.

    Parameters:
        db_path (str): Path to SQLite database file.
        tables_dict (dict): {table_name: DataFrame or GeoDataFrame}.
        keep_geometry_tables (list): List of table names to keep geometry (as WKT).
        csv_dir (str, optional): Directory to export CSV files (default: same as db_path).
    """
    if keep_geometry_tables is None:
        keep_geometry_tables = []

    # Get folder for CSVs
    if csv_dir is None:
        csv_dir = os.path.dirname(db_path)

    os.makedirs(csv_dir, exist_ok=True)

    # Connect to SQLite
    conn = sqlite3.connect(db_path)

    for table_name, df in tables_dict.items():
        df_export = df.copy()

        # Handle geometry
        if "geometry" in df_export.columns:
            if table_name in keep_geometry_tables:
                df_export["geometry"] = df_export.geometry.to_wkt()
            else:
                df_export = df_export.drop(columns="geometry")

        # Export to SQLite
        df_export.to_sql(table_name, conn, if_exists="replace", index=False)

        # Export to CSV
        csv_path = os.path.join(csv_dir, f"{table_name}.csv")
        df_export.to_csv(csv_path, index=False)

        print(f"✅ Exported '{table_name}' → SQLite + CSV")

    conn.close()
    print(f"✅ All exports completed to SQLite and CSVs in: {csv_dir}")


def create_and_populate_database(
        db_path,
        schema_path,
        tables_dict,
        keep_geometry_tables=None
):
    """
    Creates a fresh SQLite database, applies schema, converts geometries,
    and inserts data from a dictionary of tables.

    Parameters:
    - db_path: str, path to the database file
    - schema_path: str, path to the .sql schema file
    - tables_dict: dict, {table_name: DataFrame or GeoDataFrame}
    - keep_geometry_tables: list of table names where geometry should be kept (default: None)
    """
    if keep_geometry_tables is None:
        keep_geometry_tables = []

    # --- 1. SAFE START ---
    try:
        conn.close()
    except:
        pass

    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Old database '{db_path}' deleted")
    else:
        print(f"ℹ️ No old database found at '{db_path}'")

    # --- 2. APPLY SCHEMA ---
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read()

    def insert_drops(schema_sql):
        return re.sub(
            r'(CREATE TABLE\s+"?([\w_]+)"?\s*\()',
            lambda m: f'DROP TABLE IF EXISTS \"{m.group(2)}\";\n{m.group(0)}',
            schema_sql,
            flags=re.IGNORECASE
        )

    schema = insert_drops(schema)

    conn_local = sqlite3.connect(db_path)
    conn_local.execute("PRAGMA foreign_keys = ON;")
    cursor = conn_local.cursor()
    statements = [s.strip() for s in schema.split(';') if s.strip()]
    for stmt in statements:
        cursor.execute(stmt + ";")
    conn_local.commit()
    conn_local.close()

    print(f"✅ Empty database structure created at '{db_path}'")

    # --- 3. CONVERT GEOMETRIES ---
    print("🔄 Converting geometries...")
    for table_name, df in tables_dict.items():
        if isinstance(df, gpd.GeoDataFrame) and "geometry" in df.columns:
            if table_name in keep_geometry_tables:
                df["geometry"] = df["geometry"].to_wkt()
            else:
                df.drop(columns=["geometry"], inplace=True)
    print("✅ Geometries handled (kept only where needed)")

    # --- 4. OPEN CONNECTION ---
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    print("✅ New connection opened")

    # --- 5. INSERT TABLES ---
    def safe_insert(df, table_name):
        try:
            df.to_sql(table_name, conn, if_exists="append", index=False)
            print(f"✅ Inserted {len(df)} rows into '{table_name}'")
        except Exception as e:
            print(f"❌ Error inserting '{table_name}': {e}")

    # Insert all tables
    for table_name, df in tables_dict.items():
        safe_insert(df, table_name)

    # --- 6. CHECK FOREIGN KEYS ---
    try:
        broken_foreign_keys = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if broken_foreign_keys:
            print("❌ Foreign key problems found:")
            for problem in broken_foreign_keys:
                print(problem)
        else:
            print("✅ No foreign key problems found!")
    except Exception as e:
        print(f"❌ Error checking foreign keys: {e}")

    # --- 7. CLOSE CONNECTION ---
    conn.close()
    print("✅ Connection closed properly")
