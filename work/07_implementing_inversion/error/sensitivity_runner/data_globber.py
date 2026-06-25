import pickle
from pathlib import Path

import pandas as pd

OUTPUT_DIR = (
    Path(__file__).resolve().parent / "inversion_results"
)
MASTER_RESULTS_WIDE_PATH = (
    Path(__file__).resolve().parent
    / "master_results_wide.csv"
)

INDEX_COLUMNS = [
    "setup_index",
    "gmsl_true_mm",
    "altimetry_estimate_mm",
    "altimetry_std_mm",
    "altimetry_z",
    "truth_length_scale_m",
    "truth_gmsl_std_nd",
    "altimetry_degree_density",
    "altimetry_error_std_nd",
]

CASE_METRICS = [
    "prior_z",
    "prior_mean_mm",
    "prior_std_mm",
    "posterior_z",
    "posterior_mean_mm",
    "posterior_std_mm",
    "prior_bias_mm",
    "posterior_bias_mm",
    "cg_iterations",
    "runtime_s",
]

def _load_records_from_pickle(path: Path) -> list[dict]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(
            f"Unexpected payload format in {path.name}."
        )

    if not isinstance(records, list):
        raise ValueError(
            f"Invalid records payload in {path.name}."
        )

    return records

def _format_sweep_value(value: float) -> str:
    # Compact and stable float formatting for column names.
    return (
        format(float(value), ".12g")
        .replace("-", "m")
        .replace(".", "p")
    )

def _build_case_id_column(df: pd.DataFrame) -> pd.Series:
    return (
        df["sweep_type"].astype(str)
        + "__"
        + df["sweep_value"].map(_format_sweep_value)
    )

def _to_wide_format(df_long: pd.DataFrame) -> pd.DataFrame:
    required = set(
        INDEX_COLUMNS
        + ["sweep_type", "sweep_value"]
        + CASE_METRICS
    )
    missing = sorted(required.difference(df_long.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing required columns for wide output: {missing_text}"
        )

    df_cases = df_long.copy()
    df_cases["case_id"] = _build_case_id_column(df_cases)

    base = (
        df_cases[INDEX_COLUMNS]
        .drop_duplicates(subset=["setup_index"])
        .set_index("setup_index")
        .sort_index()
    )

    case_tables = []
    for metric in CASE_METRICS:
        metric_wide = df_cases.pivot_table(
            index="setup_index",
            columns="case_id",
            values=metric,
            aggfunc="first",
        )
        metric_wide = metric_wide.add_prefix(f"{metric}__")
        case_tables.append(metric_wide)

    cases = pd.concat(case_tables, axis=1).sort_index()
    return base.join(cases, how="left").reset_index()

def consolidate_results(
    *,
    output_dir: Path = OUTPUT_DIR,
    master_wide_path: Path = MASTER_RESULTS_WIDE_PATH,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(output_dir.glob("*.pkl"))
    if not pkl_files:
        if not master_wide_path.exists():
            print("No new pickle files to consolidate.")
        else:
            print(
                "No new pickle files. "
                "Wide table already up to date."
            )
        return

    all_records: list[dict] = []
    failed_files: list[Path] = []

    for pkl_path in pkl_files:
        try:
            all_records.extend(
                _load_records_from_pickle(pkl_path)
            )
        except Exception as exc:
            print(f"Failed to read {pkl_path.name}: {exc}")
            failed_files.append(pkl_path)

    if not all_records:
        print("No valid records were extracted.")
        return

    df_new = pd.DataFrame(all_records)
    df_wide_new = _to_wide_format(df_new)

    if master_wide_path.exists():
        df_wide_existing = pd.read_csv(master_wide_path)
        df_wide_combined = pd.concat(
            [df_wide_existing, df_wide_new],
            ignore_index=True,
        )
        df_wide_combined = (
            df_wide_combined.drop_duplicates(
                subset=["setup_index"],
                keep="last",
            )
            .sort_values("setup_index")
            .reset_index(drop=True)
        )
    else:
        df_wide_combined = df_wide_new.sort_values(
            "setup_index"
        ).reset_index(drop=True)

    df_wide_combined.to_csv(master_wide_path, index=False)
    print(
        f"Added {len(df_new)} long rows from pickles. "
        f"Wide table now has {len(df_wide_combined)} setup rows."
    )

    for pkl_path in pkl_files:
        if pkl_path not in failed_files:
            pkl_path.unlink()

    print(
        "Cleanup complete. "
        f"Kept {len(failed_files)} unreadable files for inspection."
    )

def main() -> None:
    consolidate_results()

if __name__ == "__main__":
    main()
