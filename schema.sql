PRAGMA foreign_keys = ON;

--- MAIN table ---
CREATE TABLE "main" (
"main_id" TEXT PRIMARY KEY,
  "facility_group_id" TEXT,
  "company_id" TEXT,
  "facility_name" TEXT,
  "reported_company" TEXT,
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
  "source_id" TEXT,
  "geometry" TEXT,
  "cluster_id" TEXT,
FOREIGN KEY (source_id) REFERENCES source(source_id),
FOREIGN KEY (cluster_id) REFERENCES cluster(cluster_id)
);


--- Source table ---
CREATE TABLE "source" (
"source_id" TEXT PRIMARY KEY,
  "source_name" TEXT,
  "source_type" TEXT, #
  "year" REAL,
  "author" TEXT,
  "url_doi" TEXT
);



--- Manual data collection table ---
CREATE TABLE "ownership" (
    "main_id" TEXT,
    "facility_group_id" TEXT,
    "company_id" TEXT,
    "company_url" TEXT,
    "facility_url" TEXT,
    "mdo_url" TEXT,
    "recent_transaction" TEXT,
    "owners" TEXT,
    "operator" TEXT,
    "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (facility_group_id) REFERENCES main(facility_group_id),
FOREIGN KEY (company_id) REFERENCES main(company_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
);


CREATE TABLE "production" (
    "row_id" TEXT PRIMARY KEY,
    "main_id" TEXT,
    "facility_group_id" TEXT,
    "company_id" TEXT,
    "year" INTEGER,
    "commodity" TEXT,
    "reference_point" TEXT,
    "material_type" TEXT,
    "unit" TEXT,
    "value" REAL,
    "comment" TEXT,
    "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (facility_group_id) REFERENCES main(facility_group_id),
FOREIGN KEY (company_id) REFERENCES main(company_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
);


CREATE TABLE "reserves_resources" (
    "row_id" TEXT PRIMARY KEY,
    "main_id" TEXT,
    "facility_group_id" TEXT,
    "year" INTEGER,
    "commodity" TEXT,
    "reserves_resources" TEXT,
    "reserves_resources_type" TEXT,
    "ore" REAL,
    "ore_unit" TEXT,
    "grade" REAL,
    "grade_unit" TEXT,
    "metal_content" REAL,
    "metal_content_unit" TEXT,
    "recovery_rate" REAL,
    "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (facility_group_id) REFERENCES main(facility_group_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
);


CREATE TABLE "energy" (
    "row_id" TEXT PRIMARY KEY,
    "main_id" TEXT,
    "facility_group_id" TEXT,
    "company_id" TEXT,
    "year" INTEGER,
    "commodity" TEXT,
    "energy_type" TEXT,
    "unit" TEXT,
    "value" REAL,
    "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (facility_group_id) REFERENCES main(facility_group_id),
FOREIGN KEY (company_id) REFERENCES main(company_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
);


CREATE TABLE "environment" (
    "row_id" TEXT PRIMARY KEY,
    "main_id" TEXT,
    "facility_group_id" TEXT,
    "company_id" TEXT,
    "year" INTEGER,
    "commodity" TEXT,
    "type" TEXT,
    "elementary_flows" TEXT,
    "unit" TEXT,
    "value" REAL,
    "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (facility_group_id) REFERENCES main(facility_group_id),
FOREIGN KEY (company_id) REFERENCES main(company_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
);


CREATE TABLE "archetypes" (
    "main_id" TEXT,
    "facility_group_id" TEXT,
    "deposit_type" TEXT,
    "mining_depth" INTEGER,
    "mining_method" TEXT,
    "processing_method" TEXT,
    "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (facility_group_id) REFERENCES main(facility_group_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
);


--- Satellite tables with one-to-many relatioships ---
CREATE TABLE "tailings" (
"row_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "year" INTEGER,
  "tailing_id" TEXT UNIQUE,
  "tsf_name" TEXT,
  "status" TEXT,
  "construction_year" REAL,
  "raise_type" TEXT,
  "current_maximum_height" REAL,
  "current_tailings_storage" TEXT,
  "planned_storage_5_years" TEXT,
  "hazard_categorization" TEXT,
  "classification_system" TEXT,
  "geometry" TEXT,
  "source_id" TEXT,
  "cluster_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id),
FOREIGN KEY (cluster_id) REFERENCES cluster(cluster_id)
);


CREATE TABLE "mincan" (
"mincan_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "mine_status" TEXT,
  "operation_periods" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "conflict" (
"ej_atlas_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "case_name" TEXT,
  "start_date" TEXT,
  "end_date" TEXT,
  "conflict_description" TEXT,
  "conflict_details" TEXT,
  "population_affected" TEXT,
  "conflict_intensity" TEXT,
  "project_status" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "ghg" (
"row_id" TEXT PRIMARY KEY,
  "ghg_id" TEXT,
  "main_id" TEXT,
  "year" INTEGER,
  "sector" TEXT,
  "value" REAL,
  "unit" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "pollution" (
"row_id" TEXT PRIMARY KEY,
  "npri_id" TEXT,
  "main_id" TEXT,
  "year" INTEGER,
  "terrestrial_ecozone" TEXT,
  "watershed" TEXT,
  "substance_name_npri" TEXT,
  "substance_name_ecoinvent" TEXT,
  "substance_unit" TEXT,
  "emission_type" TEXT,
  "emission_subtype" TEXT,
  "value" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "climate_categories" (
"row_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "year" TEXT,
  "scenario" TEXT,
  "category" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "weather" (
"row_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "year" INTEGER,
  "variable" TEXT,
  "value" REAL,
  "unit" TEXT,
  "scenario" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "peatland" (
"peatland_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "peatland_presence" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "population" (
"row_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "year" INTEGER,
  "buffer_size" TEXT,
  "total_population" REAL,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "land_cover" (
"row_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "name" TEXT,
  "year" INTEGER,
  "modis_land_cover" TEXT,
  "esa_land_cover" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "npv" (
"npv_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "biome_type" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "water_risk" (
"row_id" TEXT PRIMARY KEY,
  "main_id" TEXT,
  "indicator" TEXT,
  "value" TEXT,
  "year" INTEGER,
  "scenario" TEXT,
  "source_id" TEXT,
FOREIGN KEY (main_id) REFERENCES main(main_id),
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


--- Satellite tables with many-to-many relationships ---
CREATE TABLE "land_occupation" (
"tang_id" TEXT PRIMARY KEY,
  "area_km2" REAL,
  "geometry" TEXT,
  "cluster_id" TEXT,
  "source_id" TEXT,
FOREIGN KEY (source_id) REFERENCES source(source_id),
FOREIGN KEY (cluster_id) REFERENCES cluster(cluster_id)
);


CREATE TABLE "protected_land" (
"wpda_id" TEXT PRIMARY KEY,
  "wpda_name" TEXT,
  "type" TEXT,
  "ownership" TEXT,
  "operator" TEXT,
  "status_year" INTEGER,
  "geometry" TEXT,
   "source_id" TEXT,
FOREIGN KEY (source_id) REFERENCES source(source_id)
);


CREATE TABLE "indigenous_land" (
"indigenous_land_id" TEXT PRIMARY KEY,
  "land_category" TEXT,
  "data_source" TEXT,
  "status_date" TEXT,
  "geometry" TEXT,
  "source_id" TEXT,
FOREIGN KEY (source_id) REFERENCES source(source_id)
);

