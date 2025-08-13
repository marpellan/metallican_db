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

    matches_df = pd.DataFrame(matches)
    matches_df = matches_df.drop_duplicates()

    return matches_df


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
