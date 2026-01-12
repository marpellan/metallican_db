PRAGMA foreign_keys = ON;

-- SOURCES
CREATE TABLE "Sources" (
    source_id TEXT, -- Primary key is not defined here for now
    source_provenance TEXT,
    source_name TEXT
);

-- MAIN TABLE
CREATE TABLE "Main" (
    main_id TEXT PRIMARY KEY,
    facility_name TEXT,
    facility_group_id TEXT,
    facility_group_name TEXT,
    company_id TEXT,
    company_name TEXT,
    city TEXT,
    province TEXT,
    status TEXT,
    activity_status TEXT,
    development_stage TEXT,
    facility_type TEXT,
    mining_processing_type TEXT,
    commodity_group TEXT,
    commodities TEXT,
    "owner(s)" TEXT,
    "operator(s)" TEXT,
    operation_periods TEXT,
    company_URL TEXT,
    facility_URL TEXT,
    MDO_URL TEXT,
    geometry TEXT,
    source_id TEXT
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- SUBSTANCES
CREATE TABLE "Substances" (
    substance_id TEXT PRIMARY KEY,
    substance_type TEXT,
    substance_name TEXT
);


-- RESERVES & RESOURCES
CREATE TABLE "Reserves_resources" (
    reserves_id TEXT PRIMARY KEY,
    year INTEGER,
    commodity TEXT,
    reserves_resources TEXT,
    reserves_resources_type TEXT,
    type TEXT,
    ore FLOAT,
    ore_unit TEXT,
    grade FLOAT,
    grade_unit TEXT,
    metal_content FLOAT,
    metal_content_unit TEXT,
    recovery_rate FLOAT,
    comment TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- PRODUCTION
CREATE TABLE "Production" (
    prod_id TEXT PRIMARY KEY, -- Primary key is not defined here, as it may not be unique
    year INTEGER,
    geography TEXT,
    commodity TEXT,
    reference_point TEXT,
    material_type TEXT,
    data_type TEXT,
    unit TEXT,
    value FLOAT,
    value_tonnes FLOAT,
    comment TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    company_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- TECHNICAL ATTRIBUTES
CREATE TABLE "Technical_attributes" (
    tech_attr_id TEXT PRIMARY KEY,
    year INTEGER,
    commodity TEXT,
    reference_point TEXT,
    material_type TEXT,
    method TEXT,
    unit TEXT,
    value FLOAT,
    comment TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    company_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- ENVIRONMENTAL FLOWS
CREATE TABLE "Environmental_flows" (
    env_id TEXT PRIMARY KEY,
    year INTEGER,
    compartment_name TEXT,
    substance_id TEXT,
    flow_direction TEXT,
    release_pathway TEXT,
    unit TEXT,
    value FLOAT,
    comment TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    company_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- ENVIRONMENTAL INTENSITY
CREATE TABLE "Intensity" (
    intensity_id TEXT PRIMARY KEY,
    year INTEGER,
    commodity TEXT,
    type TEXT,
    subtype TEXT,
    unit TEXT,
    value FLOAT,
    comment TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    company_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- MATERIALS AND ENERGY
CREATE TABLE "Materials and energy" (
    technosphere_id TEXT PRIMARY KEY,
    year INTEGER,
    flow_type TEXT,
    subflow_type TEXT,
    unit TEXT,
    value FLOAT,
    comment TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    company_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- TAILINGS
CREATE TABLE "Tailings" (
    tailing_id TEXT PRIMARY KEY,
    year INTEGER,
    tsf_name TEXT,
    status TEXT,
    construction_year INTEGER,
    raise_type TEXT,
    current_maximum_height FLOAT,
    current_storage FLOAT,
    hazard_categorization TEXT,
    classification_system TEXT,
    geometry TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- LAND OCCUPATION
CREATE TABLE "Land_occupation" (
    land_occupation_id TEXT PRIMARY KEY,
    area_km2 FLOAT,
    geometry TEXT,
    distance_km FLOAT,
    main_id TEXT,
    tailing_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id),
    FOREIGN KEY (tailing_id) REFERENCES "Tailings"(tailing_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- BY-PRODUCT RATIOS
CREATE TABLE "By_products" (
    host TEXT,
    by_product TEXT,
    ratio FLOAT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- ARCHETYPES
CREATE TABLE "Archetypes" (
    archetype_id TEXT PRIMARY KEY,
    deposit_cmmi TEXT,
    deposit_mdo TEXT,
    ore_type TEXT,
    mining_method TEXT,
    mining_submethod TEXT,
    main_id TEXT,
    facility_group_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- WATER RISK
CREATE TABLE "Water_risk" (
    water_risk_id TEXT PRIMARY KEY,
    indicator TEXT,
    value FLOAT,
    year INTEGER,
    scenario TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- CLIMATE CATEGORIES
CREATE TABLE "Climate_categories" (
    climate_category_id TEXT PRIMARY KEY,
    year INTEGER,
    scenario TEXT,
    category TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- WEATHER
CREATE TABLE "Weather" (
    weather_id TEXT PRIMARY KEY,
    year INTEGER,
    variable TEXT,
    value FLOAT,
    unit TEXT,
    scenario TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- LAND COVER
CREATE TABLE "Land_cover" (
    land_cover_id TEXT PRIMARY KEY,
    year INTEGER,
    modis_land_cover TEXT,
    esa_land_cover TEXT,
    npv_biome_type TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- CONFLICT
CREATE TABLE "Conflict" (
    ej_atlas_id TEXT PRIMARY KEY,
    case_name TEXT,
    start_date DATE,
    end_date DATE,
    conflict_description TEXT,
    conflict_details TEXT,
    population_affected TEXT,
    conflict_intensity TEXT,
    project_status TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    --FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- POPULATION
CREATE TABLE "Population" (
    population_id TEXT PRIMARY KEY,
    year INTEGER,
    buffer_size INTEGER,
    total_population INTEGER,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    -- FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- PEATLAND
CREATE TABLE "Peatland" (
    peatland_presence TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    -- FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- PRIORITIZATION CONSERVATION AREAS
CREATE TABLE "Prioritization_conservation_areas" (
    score_1_km FLOAT,
    mean_score_50_km FLOAT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
);

-- PROTECTED AND INDIGENOUS LANDS
CREATE TABLE "Protected_indigenous_lands" (
    protected_area_id TEXT,
    land_name TEXT,
    land_type TEXT,
    distance_km TEXT,
    geometry TEXT,
    main_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    -- FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

-- CARBON STOCK ECOSYSTEMS
CREATE TABLE "Carbon_stock_ecosystems" (
    carbon_stock_ecosystems_id TEXT,
    pool TEXT,
    variable TEXT,
    unit TEXT,
    value FLOAT,
    main_id TEXT,
    facility_group_id TEXT,
    source_id TEXT,
    FOREIGN KEY (main_id) REFERENCES "Main"(main_id)
    -- FOREIGN KEY (source_id) REFERENCES "Sources"(source_id)
);

