PRAGMA foreign_keys = OFF;

--- MAIN table ---
CREATE TABLE "main" (
"main_id" TEXT PRIMARY KEY,
  "facility_name" TEXT,
  "reported_company" TEXT,
  "longitude" REAL,
  "latitude" REAL,
  "city" TEXT,
  "province" TEXT,
  "status" TEXT,
  "activity_status" TEXT,
  "development_stage" TEXT,
  "facility_type" TEXT,
  "mining_processing_type" TEXT,
  "commodity_group" TEXT,
  "primary_commodity" TEXT,
  "commodities" TEXT,
  "source" TEXT,
  "source_df" TEXT,
  "geometry" TEXT,
  "cluster_id" TEXT
);

--- Satellite tables ---
CREATE TABLE "climate_categories" (
"row_id" TEXT PRIMARY KEY,
  "climate_category_id" TEXT,
  "main_id" TEXT,
  "name" TEXT,
  "year" TEXT,
  "scenario" TEXT,
  "category" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "ghg" (
"row_id" TEXT PRIMARY KEY,
  "year" INTEGER,
  "ghg_id" TEXT,
  "facility_name_ghg" TEXT,
  "longitude" REAL,
  "latitude" REAL,
  "city" TEXT,
  "province" TEXT,
  "sector" TEXT,
  "value" REAL,
  "unit" TEXT,
  "facility_url" TEXT,
  "source_df" TEXT,
  "main_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "indigenous_land" (
"indigenous_land_id" TEXT PRIMARY KEY,
  "Name" TEXT,
  "Category" TEXT,
  "Data_Src" TEXT,
  "Data_Date" TEXT,
  "longitude" REAL,
  "latitude" REAL
);


CREATE TABLE "land_cover" (
"row_id" TEXT PRIMARY KEY,
  "land_cover_id" TEXT,
  "main_id" TEXT,
  "name" TEXT,
  "year" INTEGER,
  "modis_land_cover" TEXT,
  "esa_land_cover" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "pollution" (
"row_id" TEXT PRIMARY KEY,
  "year" INTEGER,
  "pollutant_id" TEXT,
  "facility_name_npri" TEXT,
  "company_name_npri" TEXT,
  "facility_type" TEXT,
  "longitude" REAL,
  "latitude" REAL,
  "terrestrial_ecozone" TEXT,
  "watershed" TEXT,
  "substance_name_npri" TEXT,
  "substance_name_ecoinvent" TEXT,
  "substance_unit" TEXT,
  "emission_type" TEXT,
  "emission_subtype" TEXT,
  "value" TEXT,
  "source_df" TEXT,
  "main_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "land_occupation" (
"tang_id" TEXT PRIMARY KEY,
  "area_km2" REAL,
  "cluster_id" INTEGER
);


CREATE TABLE "mincan" (
"mincan_id" TEXT PRIMARY KEY,
  "namemine" TEXT,
  "town" TEXT,
  "province" TEXT,
  "latitude" REAL,
  "longitude" REAL,
  "commodityall" TEXT,
  "information" TEXT,
  "mine_status" TEXT,
  "operation_periods" TEXT,
  "main_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "natural_potential_vegetation" (
"npv_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "name" TEXT,
  "latitude" REAL,
  "longitude" REAL,
  "biome_type" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "peatland" (
"peatland_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "facility_name" TEXT,
  "longitude" REAL,
  "latitude" REAL,
  "peatland_presence" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "population" (
"row_id" TEXT PRIMARY KEY,
  "population_id" TEXT,
  "main_id" TEXT,
  "name" TEXT,
  "year" INTEGER,
  "buffer_size" TEXT,
  "total_population" REAL,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "protected_land" (
"wpda_id" TEXT PRIMARY KEY,
  "NAME" TEXT,
  "DESIG" TEXT,
  "OWN_TYPE" TEXT,
  "MANG_AUTH" TEXT,
  "STATUS_YR" INTEGER,
  "longitude" REAL,
  "latitude" REAL
);


CREATE TABLE "tailings" (
"row_id" TEXT PRIMARY KEY,
  "year" INTEGER,
  "tailing_id" TEXT,
  "tsf_name" TEXT,
  "related_mine" TEXT,
  "main_owner" TEXT,
  "full_ownership" TEXT,
  "operator" TEXT,
  "longitude" REAL,
  "latitude" REAL,
  "status" TEXT,
  "construction_year" REAL,
  "raise_type" TEXT,
  "current_maximum_height" REAL,
  "current_tailings_storage" TEXT,
  "planned_storage_5_years" TEXT,
  "hazard_categorization" TEXT,
  "classification_system" TEXT,
  "link" TEXT,
  "source" TEXT,
  "source_df" TEXT,
  "geometry" TEXT,
  "main_id" TEXT,
  "cluster_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "water_risk" (
"row_id" TEXT PRIMARY KEY,
  "water_risk_id" TEXT,
  "main_id" TEXT,
  "name" TEXT,
  "indicator" TEXT,
  "value" TEXT,
  "year" INTEGER,
  "scenario" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


CREATE TABLE "weather" (
"row_id" TEXT PRIMARY KEY,
  "weather_id" TEXT,
  "main_id" TEXT,
  "name" TEXT,
  "year" INTEGER,
  "variable" TEXT,
  "value" REAL,
  "unit" TEXT,
  "scenario" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id)
);


--- Linking tables ---
CREATE TABLE lt_clusters (
  main_id TEXT,
  cluster_id TEXT,
  tailing_id TEXT,
  tang_id TEXT,
  check_manually INTEGER,
FOREIGN KEY(main_id) REFERENCES main(main_id),
FOREIGN KEY(tailing_id) REFERENCES tailings(tailing_id),
FOREIGN KEY(tang_id) REFERENCES land_occupation(tang_id)
);


CREATE TABLE "lt_protected_land" (
"main_id" TEXT,
  "wpda_id" TEXT,
  "distance_to_centroid_m" REAL,
  "distance_to_edge_m" REAL,
FOREIGN KEY(main_id) REFERENCES main(main_id),
FOREIGN KEY(wpda_id) REFERENCES protected_land(wpda_id)
);


CREATE TABLE "lt_indigenous_land" (
"main_id" TEXT,
  "indigenous_land_id" TEXT,
  "distance_to_centroid_m" REAL,
  "distance_to_edge_m" REAL,
FOREIGN KEY(main_id) REFERENCES main(main_id),
FOREIGN KEY(indigenous_land_id) REFERENCES indigenous_land(indigenous_land_id)
);