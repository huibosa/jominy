CORE_FEATURES = ["C", "Si", "Mn", "P", "S", "Cu", "Cr"]
EXTENDED_FEATURES = ["Ni", "V", "Ti", "W", "Al", "B"]
SPARSE_MISSING_FLAGS = ["V_missing", "Ti_missing", "W_missing", "Al_missing", "B_missing"]

FULL_FEATURES = CORE_FEATURES + EXTENDED_FEATURES + SPARSE_MISSING_FLAGS
CORE_PLUS_FLAGS = CORE_FEATURES + SPARSE_MISSING_FLAGS

EXCLUDED_PREDICTOR_COLUMNS = {
    "炉号",
    "base_heat_id",
    "hardness",
    "hardness_n",
    "hardness_std",
    "distance",
    "J9",
    "J15",
    "delta",
    "has_j9",
    "has_j15",
    "has_pair",
}
