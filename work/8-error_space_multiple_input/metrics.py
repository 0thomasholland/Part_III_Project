import argparse
from os import path

import numpy as np
import pandas as pd
from joblib import dump, load

file = "variable_input_data_initial.pkl"


def kl(
    expectation_true: float,
    standard_deviation_true: float,
    expectation_approx: float,
    standard_deviation_approx: float,
) -> float:
    """Kullback-Leibler divergence between two univariate normal distributions."""
    KL: float = (
        np.log(standard_deviation_approx / standard_deviation_true)
        + (
            (
                standard_deviation_true**2
                + (expectation_true - expectation_approx) ** 2
            )
            / (2 * standard_deviation_approx**2)
        )
        - 0.5
    )
    return KL


def mse(
    expectation_true: float,
    expectation_approx: float,
) -> float:
    """Mean Squared Error between two univariate normal distributions."""
    MSE: float = (expectation_true - expectation_approx) ** 2
    return MSE


def cohens_d(
    expectation_true: float,
    standard_deviation_true: float,
    expectation_approx: float,
    standard_deviation_approx: float,
) -> float:
    """Cohen's d between two univariate normal distributions."""
    pooled_std: float = np.sqrt(
        (standard_deviation_true**2 + standard_deviation_approx**2)
        / 2,
    )
    d: float = (expectation_true - expectation_approx) / pooled_std
    return d


def wasserstein_distance(
    expectation_true: float,
    standard_deviation_true: float,
    expectation_approx: float,
    standard_deviation_approx: float,
) -> float:
    """Wasserstein distance between two univariate normal distributions."""
    WD: float = np.sqrt(
        (expectation_true - expectation_approx) ** 2
        + (standard_deviation_true - standard_deviation_approx) ** 2,
    )
    return WD


def load_data(data_path: str) -> pd.DataFrame:
    """Load the variable input data from a pickle file."""
    # data_path = path.join(
    #     path.dirname(path.abspath(__file__)),
    #     data_path,
    # )
    return pd.DataFrame(load(data_path))


def save_metrics(
    metrics: pd.DataFrame,
    original_data: pd.DataFrame,
    output_path: str,
) -> None:
    """Save the computed metrics to a pickle file."""
    # Combine metrics with original data
    combined_data = pd.concat(
        [
            original_data.reset_index(drop=True),
            metrics.reset_index(drop=True),
        ],
        axis=1,
    )
    # output_path = path.join(
    #     path.dirname(path.abspath(__file__)),
    #     output_path,
    # )
    dump(combined_data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save metrics between true and approximate distributions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input pickle file containing variable input data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output pickle file to save metrics.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="List of metrics to compute.",
        default=["kl", "mse", "cohens_d", "wasserstein_distance"],
    )
    parser.add_argument(
        "--true",
        type=str,
        help="Column prefix for true distribution parameters.",
    )
    parser.add_argument(
        "--approx",
        type=str,
        nargs="+",
        help="Column prefix for approximate distribution parameters.",
    )
    parser.add_argument(
        "--std_abbreviation",
        type=str,
        help="Suffix used to denote standard deviation columns.",
    )
    parser.add_argument(
        "--expectation_abbreviation",
        type=str,
        help="Suffix used to denote expectation columns.",
    )

    print("Parsing arguments...")

    try:
        args = parser.parse_args()
    except SystemExit:
        print(
            "Error parsing arguments. Please check the provided arguments.",
        )
        raise
    try:
        data = load_data(args.input)
    except Exception as e:
        print(f"Error loading data from {args.input}: {e}")
        exit(1)

    metrics_df = pd.DataFrame()

    if type(args.metrics) is str:
        args.metrics = [args.metrics]
    if type(args.approx) is str:
        args.approx = [args.approx]

    print("Computing metrics...")

    for metric in args.metrics:
        print(f"Computing {metric}...")
        for approx in args.approx:
            metric_values: list[float] = []
            for _, row in data.iterrows():
                expectation_true = row[
                    f"{args.true}{args.expectation_abbreviation}"
                ]
                standard_deviation_true = row[
                    f"{args.true}{args.std_abbreviation}"
                ]
                expectation_approx = row[
                    f"{approx}{args.expectation_abbreviation}"
                ]
                standard_deviation_approx = row[
                    f"{approx}{args.std_abbreviation}"
                ]

                if metric == "kl":
                    value = kl(
                        expectation_true,
                        standard_deviation_true,
                        expectation_approx,
                        standard_deviation_approx,
                    )
                elif metric == "mse":
                    value = mse(
                        expectation_true,
                        expectation_approx,
                    )
                elif metric == "cohens_d":
                    value = cohens_d(
                        expectation_true,
                        standard_deviation_true,
                        expectation_approx,
                        standard_deviation_approx,
                    )
                elif metric == "wasserstein_distance":
                    value = wasserstein_distance(
                        expectation_true,
                        standard_deviation_true,
                        expectation_approx,
                        standard_deviation_approx,
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                metric_values.append(value)
            metrics_df[f"{approx}_{metric}"] = metric_values
        print(f"Finished computing {metric} for: {args.approx}")
    print("Saving metrics...")
    save_metrics(metrics_df, data, args.output)
    print(f"Metrics saved successfully to {args.output}.")
    exit(0)
