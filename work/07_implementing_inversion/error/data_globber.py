import glob
import os
import pickle
from pathlib import Path

import pandas as pd


def consolidate_results(master_file="master_results.csv"):
    temp_dir = Path("inversion_results")
    master_path = Path(master_file)

    # 1. Load existing data if it exists
    if master_path.exists():
        print(
            f"Loading existing master file: {master_file}"
        )
        # Use Parquet if your data is large; it's much faster than CSV
        df_master = pd.read_csv(master_path)
    else:
        df_master = pd.DataFrame()

    # 2. Collect all new temporary pickle files
    new_data = []
    temp_files = list(temp_dir.glob("*.pkl"))

    if not temp_files:
        print("No new results to process.")
        return

    for f in temp_files:
        try:
            with open(f, "rb") as file:
                data = pickle.load(file)
                new_data.append(data)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # 3. Merge and Save
    if new_data:
        df_new = pd.DataFrame(new_data)
        df_combined = pd.concat(
            [df_master, df_new], ignore_index=True
        )

        # Save back to disk
        df_combined.to_csv(master_path, index=False)
        print(
            f"Success! Added {len(new_data)} new records. Total: {len(df_combined)}"
        )

        # 4. Cleanup: Delete the old temporary files
        for f in temp_files:
            os.remove(f)
        print("Temporary files cleared.")


# Run this after your Parallel block finishes
consolidate_results()
