# We need to sort it by NAICS list, since they are not properly listed by "hierarchy", e.g. no parent nor code
# So we define classification lists based on NAICS subsectors, e.g. 5-6 digits code

metal_ore_mining_naics = [
    "Metal ore mining",
    "Iron ore mining",
    "Gold and silver ore mining",
    "Copper, nickel, lead and zinc ore mining",
    "Lead-zinc ore mining",
    "Nickel-copper ore mining",
    "Copper-zinc ore mining",
    "Other metal ore mining",
    "Uranium ore mining",
    "All other metal ore mining",
    "Non-metallic mineral mining and quarrying",
    "Stone mining and quarrying",
    "Granite mining and quarrying",
    "Limestone mining and quarrying",
    "Marble mining and quarrying",
    "Sandstone mining and quarrying",
    "Sand, gravel, clay, and ceramic and refractory minerals mining and quarrying",
    "Sand and gravel mining and quarrying",
    "Shale, clay and refractory mineral mining and quarrying",
    "Other non-metallic mineral mining and quarrying",
    "Diamond mining",
    "Salt mining",
    "Asbestos mining",
    "Gypsum mining",
    "Potash mining",
    "Peat extraction",
    "All other non-metallic mineral mining and quarrying",
    "Support activities for mining, and oil and gas extraction",
    "Oil and gas contract drilling",
    "Contract drilling (except oil and gas)",
    "Services to oil and gas extraction",
    "Other support activities for mining"
]

metal_manufacturing_naics = [
    "Primary metal manufacturing",
    "Iron and steel mills and ferro-alloy manufacturing",
    "Steel product manufacturing from purchased steel",
    "Iron and steel pipes and tubes manufacturing from purchased steel",
    "Rolling and drawing of purchased steel",
    "Cold-rolled steel shape manufacturing",
    "Steel wire drawing",
    "Alumina and aluminum production and processing",
    "Primary production of alumina and aluminum",
    "Aluminum rolling, drawing, extruding and alloying",
    "Non-ferrous metal (except aluminum) production and processing",
    "Non-ferrous metal (except aluminum) smelting and refining",
    "Copper rolling, drawing, extruding and alloying",
    "Non-ferrous metal (except copper and aluminum) rolling, drawing, extruding and alloying",
    "Foundries",
    "Ferrous metal foundries",
    "Iron foundries",
    "Steel foundries",
    "Non-ferrous metal foundries",
    "Non-ferrous metal die-casting foundries",
    "Non-ferrous metal foundries (except die-casting)"
]

koppen_dict = {
    1: "Af - Tropical, rainforest",
    2: "Am - Tropical, monsoon",
    3: "Aw - Tropical, savannah",
    4: "BWh - Arid, desert, hot",
    5: "BWk - Arid, desert, cold",
    6: "BSh - Arid, steppe, hot",
    7: "BSk - Arid, steppe, cold",
    8: "Csa - Temperate, dry summer, hot summer",
    9: "Csb - Temperate, dry summer, warm summer",
    10: "Csc - Temperate, dry summer, cold summer",
    11: "Cwa - Temperate, dry winter, hot summer",
    12: "Cwb - Temperate, dry winter, warm summer",
    13: "Cwc - Temperate, dry winter, cold summer",
    14: "Cfa - Temperate, no dry season, hot summer",
    15: "Cfb - Temperate, no dry season, warm summer",
    16: "Cfc - Temperate, no dry season, cold summer",
    17: "Dsa - Cold, dry summer, hot summer",
    18: "Dsb - Cold, dry summer, warm summer",
    19: "Dsc - Cold, dry summer, cold summer",
    20: "Dsd - Cold, dry summer, very cold winter",
    21: "Dwa - Cold, dry winter, hot summer",
    22: "Dwb - Cold, dry winter, warm summer",
    23: "Dwc - Cold, dry winter, cold summer",
    24: "Dwd - Cold, dry winter, very cold winter",
    25: "Dfa - Cold, no dry season, hot summer",
    26: "Dfb - Cold, no dry season, warm summer",
    27: "Dfc - Cold, no dry season, cold summer",
    28: "Dfd - Cold, no dry season, very cold winter",
    29: "ET - Polar, tundra",
    30: "EF - Polar, frost"
}

biome_dict = {
    1: "Tropical Evergreen Broadleaf Forest",
    2: "Tropical Semi-Evergreen Broadleaf Forest",
    3: "Tropical Deciduous Broadleaf Forest and Woodland",
    4: "Warm-Temperate Evergreen Broadleaf and Mixed Forest",
    7: "Cool-Temperate Rainforest",
    8: "Cool Evergreen Needleleaf Forest",
    9: "Cool Mixed Forest",
    13: "Temperate Deciduous Broadleaf Forest",
    14: "Cold Deciduous Forest",
    15: "Cold Evergreen Needleleaf Forest",
    16: "Temperate Sclerophyll Woodland and Shrubland",
    17: "Temperate Evergreen Needleleaf Open Woodland",
    18: "Tropical Savanna",
    20: "Xerophytic Woods/Scrub",
    22: "Steppe",
    27: "Desert",
    28: "Graminoid and Forb Tundra",
    30: "Erect Dwarf Shrub Tundra",
    31: "Low and High Shrub Tundra",
    32: "Prostrate Dwarf Shrub Tundra"
}

peatland_dict = {
    255: "No data",
    1: "peat dominated",
    2: "peat in soil mosaic"
}

water_risk_dict = {
    # Baseline indicators
    "bws_label": "Water Stress Label",
    "bwd_label": "Water Depletion Label",
    "iav_label": "Interannual Variability Label",
    "gtd_label": "Groundwater Table Decline Label",
    "cep_label": "Coastal Eutrophication Potential Label",

    # Future projections - Water Stress (ws)
    "bau30_ws_x_l": "Water Stress Label",
    "bau50_ws_x_l": "Water Stress Label",
    "bau80_ws_x_l": "Water Stress Label",
    "pes30_ws_x_l": "Water Stress Label",
    "pes50_ws_x_l": "Water Stress Label",
    "pes80_ws_x_l": "Water Stress Label",
    "opt30_ws_x_l": "Water Stress Label",
    "opt50_ws_x_l": "Water Stress Label",
    "opt80_ws_x_l": "Water Stress Label",

    # Future projections - Water Depletion (wd)
    "bau30_wd_x_l": "Water Depletion Label",
    "bau50_wd_x_l": "Water Depletion Label",
    "bau80_wd_x_l": "Water Depletion Label",
    "pes30_wd_x_l": "Water Depletion Label",
    "pes50_wd_x_l": "Water Depletion Label",
    "pes80_wd_x_l": "Water Depletion Label",
    "opt30_wd_x_l": "Water Depletion Label",
    "opt50_wd_x_l": "Water Depletion Label",
    "opt80_wd_x_l": "Water Depletion Label",

    # Future projections - Interannual Variability (iv)
    "bau30_iv_x_l": "Interannual Variability Label",
    "bau50_iv_x_l": "Interannual Variability Label",
    "bau80_iv_x_l": "Interannual Variability Label",
    "pes30_iv_x_l": "Interannual Variability Label",
    "pes50_iv_x_l": "Interannual Variability Label",
    "pes80_iv_x_l": "Interannual Variability Label",
    "opt30_iv_x_l": "Interannual Variability Label",
    "opt50_iv_x_l": "Interannual Variability Label",
    "opt80_iv_x_l": "Interannual Variability Label"
}

unit_conversion = {
    # Standard mass units
    "t": 1,                      # tonne
    "kt": 1_000,                 # kilotonne
    "mt": 1_000_000,             # megatonne
    "kmt": 1_000_000_000,        # 1000 megatonne
    "kg": 1e-3,                  # kilograms to tonnes
    "g": 1e-6,                   # grams to tonnes

    # Ounces — distinguish between **avoirdupois** and **troy**
    # Metals (e.g. gold, silver) use **troy ounces** (1 troy oz = 31.1035 g)
    # Regular goods (like lead, zinc) use **avoirdupois ounces** (1 avdp oz = 28.3495 g)
    #"oz": 0.0000283495,          # avoirdupois ounce to tonnes
    #"oz avdp": 0.0000283495,
    "oz": 0.0000311035,        # troy ounce to tonnes
    "oz au": 0.0000311035,       # assume gold uses troy ounce
    "oz au eq": 0.0000311035,    # gold equivalent, likely same

    "koz": 0.0311035,            # 1,000 troy ounces = 31.1035 kg = 0.0311035 t
    "moz": 31.1035,              # 1,000,000 troy ounces = 31.1035 t

    # Pounds
    "lb": 0.000453592,           # 1 lb = 0.453592 kg
    "klbs": 0.453592,            # 1,000 lb
    "mlbs": 453.592,             # 1,000,000 lb
    "million lbs": 453.592,      # same as mlbs

    # Wet metric tonne / dry metric tonne
    "wmt": 1,                    # often = tonne, but water content may vary
    "dmt": 1,                    # usually reported already corrected for moisture
    "mwmt": 1_000_000,           # Million wet metric tonnes
    "mdmt": 1_000_000,           # Million dry metric tonnes

    # Carats
    "ct": 2e-7,                  # 1 carat = 0.2 g = 0.0000002 t
    "kct": 2e-4,                 # 1,000 carats = 0.2 kg = 0.0002 t
    "mct": 0.2,                  # 1,000,000 carats = 200 kg = 0.2 t
    "mcts": 0.2,                 # alternative spelling (your value 2e-4 was too small)
}