from os import path

import pandas as pd
from joblib import dump, load

# file name is work/8-redo/output/metrics_big.pkl

file_name = "metrics_big.pkl"
directory = path.dirname(path.dirname(path.abspath(__file__)))

output_data = load(
    "work/8-redo/output/variable_input_data_big.pkl",
)

# add column called "altimetry_range" which has 90.0 for all rows
output_data["altimetry_range"] = 90.0
dump(
    output_data,
    "work/8-redo/output/metrics_big.pkl",
)
