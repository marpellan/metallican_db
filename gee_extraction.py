import pandas as pd
import geopandas as gpd
import time
import ee


def extract_ghsl_population(main_gdf):

    start_time = time.time()
    ee.Initialize()

    if main_gdf.crs != "EPSG:4326":
        main_gdf = main_gdf.to_crs("EPSG:4326")

    ghsl = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
    years = ["2025", "2030"]
    buffer_sizes = {"10km": 10000, "50km": 50000}

    rows = []

    for idx, row in main_gdf.iterrows():
        point = row["geometry"]

        for year in years:
            image = ghsl.filter(ee.Filter.eq("system:index", year)).first()

            for label, radius in buffer_sizes.items():
                try:
                    ee_point = ee.Geometry.Point([point.x, point.y]).buffer(radius)

                    total_pop = image.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=ee_point,
                        scale=100,
                        maxPixels=1e9,
                        bestEffort=True
                    ).get("population_count").getInfo()

                except Exception as e:
                    print(f"❌ Error for {row['name']} ({label}, {year}): {e}")
                    total_pop = None

                rows.append({
                    "main_id": row["main_id"],
                    "name": row["name"],
                    "year": int(year),
                    "buffer_size": label,
                    "total_population": max(total_pop, 0) if total_pop is not None else None,
                    "geometry": point
                })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    print(f"✅ GHSL population (GeoDataFrame, long format) done in {time.time() - start_time:.2f}s")
    return gdf


def extract_npv(main_gdf):
    """
    Extracts biome type data from the OpenLandMap Potential Distribution of Biomes dataset
    for given facilities in main_gdf.

    Parameters:
    - main_gdf (gpd.GeoDataFrame): GeoDataFrame with columns ['main_id', 'name', 'geometry'].

    Returns:
    - biome_gdf (gpd.GeoDataFrame): GeoDataFrame with biome type data.
    """

    start_time = time.time()

    # Ensure required columns exist
    required_cols = {"main_id", "name", "geometry"}
    if not required_cols.issubset(main_gdf.columns):
        raise ValueError(f"Missing required columns in main_gdf: {required_cols - set(main_gdf.columns)}")

    # Ensure CRS is WGS 84 (EPSG:4326)
    if main_gdf.crs != "EPSG:4326":
        main_gdf = main_gdf.to_crs("EPSG:4326")

    # Initialize Google Earth Engine
    ee.Initialize()

    # Load the corrected OpenLandMap Biomes dataset
    biomes = ee.Image("OpenLandMap/PNV/PNV_BIOME-TYPE_BIOME00K_C/v01")

    # Store results
    results = []

    # Process each facility
    for index, row in main_gdf.iterrows():
        main_id = row["main_id"]
        name = row["name"]
        geom = row["geometry"]

        # Ensure geometry is a Point; if not, use centroid
        if geom.geom_type != "Point":
            centroid = geom.centroid  # Convert Polygon/MultiPolygon to Point
        else:
            centroid = geom

        lon, lat = centroid.x, centroid.y
        facility_location = ee.Geometry.Point([lon, lat])

        # Store extracted values
        facility_data = {
            "main_id": main_id,
            "name": name,
            "latitude": lat,
            "longitude": lon
        }

        try:
            # Use reduceRegion() instead of sample() for correct data retrieval
            biome_value = biomes.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=facility_location,
                scale=1000,  # Match dataset resolution (1 km²)
                bestEffort=True
            ).get("biome_type").getInfo()

            facility_data["biome_type"] = biome_value

        except Exception as e:
            facility_data["biome_type"] = None  # Handle missing data

        # Append results
        results.append(facility_data)

        # Progress tracking
        if index % 10 == 0:
            print(f"Processed {index + 1}/{len(main_gdf)} facilities...")

    # Compute total execution time
    total_time = time.time() - start_time
    print(f"✅ Extraction completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Convert results to GeoDataFrame
    biome_gdf = gpd.GeoDataFrame(pd.DataFrame(results), geometry=gpd.points_from_xy(
        pd.DataFrame(results)["longitude"], pd.DataFrame(results)["latitude"]
    ), crs="EPSG:4326")

    return biome_gdf


def extract_land_cover_type(main_gdf):
    """
    Extracts MODIS LC_Type1 and ESA WorldCover land cover for the year 2021 for each facility.

    Returns:
    - GeoDataFrame with columns: main_id, name, year, modis_land_cover, esa_land_cover, geometry
    """

    start_time = time.time()
    ee.Initialize()

    if main_gdf.crs != "EPSG:4326":
        main_gdf = main_gdf.to_crs("EPSG:4326")

    modis_dict = {
        1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest", 3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest", 5: "Mixed Forests", 6: "Closed Shrublands",
        7: "Open Shrublands", 8: "Woody Savannas", 9: "Savannas", 10: "Grasslands",
        11: "Permanent Wetlands", 12: "Croplands", 13: "Urban & Built-up", 14: "Cropland/Natural Vegetation Mosaic",
        15: "Snow & Ice", 16: "Barren", 17: "Water"
    }

    esa_dict = {
        10: "Tree Cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
        50: "Built-up", 60: "Bare/Sparse Vegetation", 70: "Snow/Ice",
        80: "Permanent Water Bodies", 90: "Herbaceous Wetland", 95: "Mangroves", 100: "Moss & Lichen"
    }

    modis_img = ee.ImageCollection("MODIS/061/MCD12Q1").filter(
        ee.Filter.calendarRange(2021, 2021, "year")
    ).first()

    esa_img = ee.Image("ESA/WorldCover/v200/2021")

    results = []

    for idx, row in main_gdf.iterrows():
        point = row["geometry"].centroid if row["geometry"].geom_type != "Point" else row["geometry"]
        ee_point = ee.Geometry.Point([point.x, point.y])

        record = {
            "main_id": row["main_id"],
            "name": row["name"],
            "year": 2021,
            "geometry": point
        }

        # MODIS extraction
        try:
            lc_modis = modis_img.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=ee_point,
                scale=1000,
                bestEffort=True
            ).get("LC_Type1").getInfo()
            record["modis_land_cover"] = modis_dict.get(lc_modis, "Unknown")
        except Exception as e:
            print(f"⚠️ MODIS error for {row['name']}: {e}")
            record["modis_land_cover"] = None

        # ESA WorldCover extraction
        try:
            lc_esa = esa_img.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=ee_point,
                scale=10,
                bestEffort=True
            ).get("Map").getInfo()
            record["esa_land_cover"] = esa_dict.get(lc_esa, "Unknown")
        except Exception as e:
            print(f"⚠️ ESA error for {row['name']}: {e}")
            record["esa_land_cover"] = None

        results.append(record)

    gdf = gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326")
    print(f"✅ Combined MODIS & ESA land cover (2021) completed in {time.time() - start_time:.2f} seconds")
    return gdf


def extract_aqueduct(main_gdf,
                                      baseline_variables=None,
                                      projection_indicators=None,
                                      projection_years=[30, 50, 80],
                                      projection_scenarios=["bau", "pes", "opt"]):
    """
    Extracts Aqueduct Water Risk baseline and future indicators in SQL-ready long format.

    Parameters:
    - main_gdf (GeoDataFrame): Must contain ['main_id', 'name', 'geometry']
    - baseline_variables (list): Baseline indicators (e.g. ['bws_label', 'bwd_label'])
    - projection_indicators (list): Core names like ['ws', 'wd', 'iv']
    - projection_years (list of int): [30, 50, 80] → years 2030, 2050, 2080
    - projection_scenarios (list of str): ['bau', 'pes', 'opt']

    Returns:
    - GeoDataFrame: Long format with columns: main_id, name, year, scenario, indicator, value, geometry
    """

    start_time = time.time()
    ee.Initialize()

    if main_gdf.crs != "EPSG:4326":
        main_gdf = main_gdf.to_crs("EPSG:4326")

    baseline_fc = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/baseline_annual")
    future_fc = ee.FeatureCollection("WRI/Aqueduct_Water_Risk/V4/future_annual")

    if baseline_variables is None:
        baseline_variables = ['bws_label', 'bwd_label', 'iav_label', 'gtd_label', 'cep_label']

    if projection_indicators is None:
        projection_indicators = ['ws', 'wd', 'iv']

    rows = []

    for idx, row in main_gdf.iterrows():
        point = row["geometry"].centroid if row["geometry"].geom_type != "Point" else row["geometry"]
        ee_point = ee.Geometry.Point([point.x, point.y])

        # Baseline
        try:
            baseline_feat = baseline_fc.filterBounds(ee_point).first()
            if baseline_feat:
                baseline_dict = baseline_feat.toDictionary().getInfo()
                for var in baseline_variables:
                    rows.append({
                        "main_id": row["main_id"],
                        "name": row["name"],
                        "indicator": var,
                        "value": baseline_dict.get(var),
                        "year": 2020,
                        "scenario": "baseline",
                        "geometry": point
                    })
        except Exception as e:
            print(f"❌ Baseline error for {row['name']}: {e}")

        # Future projections
        try:
            future_feat = future_fc.filterBounds(ee_point).first()
            if future_feat:
                future_dict = future_feat.toDictionary().getInfo()
                for year_suffix in projection_years:
                    year = 2000 + year_suffix
                    for scenario in projection_scenarios:
                        for indicator in projection_indicators:
                            key = f"{scenario}{year_suffix}_{indicator}_x_l"
                            value = future_dict.get(key)
                            rows.append({
                                "main_id": row["main_id"],
                                "name": row["name"],
                                "indicator": key,
                                "value": value,
                                "year": year,
                                "scenario": scenario,
                                "geometry": point
                            })
        except Exception as e:
            print(f"❌ Future error for {row['name']}: {e}")

        if idx % 10 == 0:
            print(f"Processed {idx + 1}/{len(main_gdf)} facilities...")

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    print(f"✅ Aqueduct extraction completed in {time.time() - start_time:.2f}s")
    return gdf