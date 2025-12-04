# We need to sort it by NAICS list, since they are not properly listed by "hierarchy", e.g. no parent nor code
# So we define classification lists based on NAICS subsectors, e.g. 5-6 digits code

metal_ore_mining_naics = [
    "Metal Ore Mining",
    "Iron Ore Mining",
    "Gold and Silver Ore Mining",
    "Copper, Nickel, Lead and Zinc Ore Mining",
    "Lead-Zinc Ore Mining",
    "Nickel-Copper Ore Mining",
    "Copper-Zinc Ore Mining",
    "Other Metal Ore Mining",
    "Uranium Ore Mining",
    "All Other Metal Ore Mining",
    "Non-Metallic Mineral Mining And Quarrying",
    # "Stone Mining And Quarrying",
    # "Granite Mining And Quarrying",
    # "Limestone Mining And Quarrying",
    # "Marble Mining And Quarrying",
    # "Sandstone Mining And Quarrying",
    # "Sand, Gravel, Clay, And Ceramic And Refractory Minerals Mining And Quarrying",
    # "Sand And Gravel Mining And Quarrying",
    # "Shale, Clay And Refractory Mineral Mining And Quarrying",
    "Other Non-Metallic Mineral Mining And Quarrying",
    # "Diamond Mining",
    # "Salt Mining",
    # "Asbestos Mining",
    # "Gypsum Mining",
    # "Potash Mining",
    # "Peat Extraction",
    "All Other Non-Metallic Mineral Mining and Quarrying",
    "Support Activities For Mining, And Oil and Gas Extraction",
    # "Oil And Gas Contract Drilling",
    "Contract Drilling (except Oil And Gas)",
    # "Services To Oil And Gas Extraction",
    "Other Support Activities for Mining"
]


metal_manufacturing_naics = [
    "Non-Ferrous Metal (except Aluminum) Production and Processing",
    "Non-Ferrous Metal (except Aluminum) Smelting and Refining",
    "Primary Metal Manufacturing",
    "Iron and Steel Mills and Ferro-Alloy Manufacturing",
    "Steel Product Manufacturing from Purchased Steel",
    "Iron and Steel Pipes and Tubes Manufacturing from Purchased Steel",
    "Rolling and Drawing Of Purchased Steel",
    "Cold-Rolled Steel Shape Manufacturing",
    "Steel Wire Drawing",
    "Alumina and Aluminum Production and Processing",
    "Primary Production of Alumina and Aluminum",
    "Aluminum Rolling, Drawing, Extruding And Alloying",

    "Copper Rolling, Drawing, Extruding and Alloying",
    "Non-Ferrous Metal (except Copper and Aluminum) Rolling, Drawing, Extruding and Alloying",
    "Foundries",
    "Ferrous Metal Foundries",
    "Iron Foundries",
    "Steel Foundries",
    "Non-Ferrous Metal Foundries",
    "Non-Ferrous Metal Die-Casting Foundries",
    "Non-Ferrous Metal Foundries (except Die-Casting)"
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
    "ttpa": 1_000,                # thousands tonnes per annum to tonnes

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

    # US short tons
    "short tons": 0.90718474,
    "million short tons": 907_184.74
}