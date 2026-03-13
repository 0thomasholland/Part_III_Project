# Auto-generated from notebook code cells.
# Source: notebooks/02 - Major Sources.ipynb

# ---- Notebook code cell 1 ----
from pathlib import Path

import numpy as np

np.random.seed(120102)
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from pyslfp import FingerPrint, IceModel, plot

from project import colors
from pygeoinf_extras.stats import expectation, standard_dev
from pyslfp_extras.ice_thickness import IceSheetChange

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.show = lambda *args, **kwargs: None
print = lambda *args, **kwargs: None


def _save_all_figures(prefix):
    for index, figure_number in enumerate(
        plt.get_fignums(), start=1
    ):
        fig = plt.figure(figure_number)
        fig.savefig(
            FIGURES_DIR / f"{prefix}_{index:02d}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close("all")


SOURCE_COLOURS = {
    "GIS": colors.gis,
    "WAIS": colors.wais,
    "EAIS": colors.eais,
}
TARGET_GMSL_MEAN_M = 0.010
TARGET_GMSL_STD_M = 0.003
ALTIMETRY_LATITUDE_RANGE = 66.0


def stats_in_mm(measure):
    return {
        "mean_mm": 1e3 * expectation(measure),
        "std_mm": 1e3 * standard_dev(measure),
    }


def normal_pdf(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (
        std * np.sqrt(2 * np.pi)
    )


def plot_scalar_distributions(measures, title, xlabel):
    stats = {
        name: stats_in_mm(measure)
        for name, measure in measures.items()
    }
    lower = min(
        value["mean_mm"] - 4 * value["std_mm"]
        for value in stats.values()
    )
    upper = max(
        value["mean_mm"] + 4 * value["std_mm"]
        for value in stats.values()
    )
    x = np.linspace(lower, upper, 600)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, value in stats.items():
        ax.plot(
            x,
            normal_pdf(
                x, value["mean_mm"], value["std_mm"]
            ),
            label=(
                f"{name}: mean = {value['mean_mm']:.2f} mm, "
                f"std = {value['std_mm']:.2f} mm"
            ),
            color=SOURCE_COLOURS[name],
            linewidth=2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.legend()
    ax.grid(alpha=0.3)
    sns.despine()
    plt.show()

    return pd.DataFrame.from_dict(stats, orient="index")


def resolve_scalar_data_path(filename):
    relative_paths = [
        Path("notebooks") / "data" / filename,
        Path("data") / filename,
        Path("work") / "01_major_source" / filename,
    ]
    search_roots = [Path.cwd(), *Path.cwd().parents]
    for root in search_roots:
        for relative_path in relative_paths:
            candidate = root / relative_path
            if candidate.exists():
                return candidate
    raise FileNotFoundError(filename)


# ---- Notebook code cell 2 ----
fp = FingerPrint(lmax=96)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)
length_scale = 0.2 * fp.mean_sea_floor_radius
pattern = IceSheetChange.UniformPattern()

source_builders = {
    "GIS": IceSheetChange.greenland,
    "WAIS": IceSheetChange.west_antarctic,
    "EAIS": IceSheetChange.east_antarctic,
}

ice_changes = {
    name: build(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=length_scale,
        pattern=pattern,
        ice_gmsl_std=TARGET_GMSL_STD_M,
        gmsl_target_mean=TARGET_GMSL_MEAN_M,
        altimetry_latitude_range=ALTIMETRY_LATITUDE_RANGE,
    )
    for name, build in source_builders.items()
}

true_gmsl_measures = {
    name: change.ice_thickness.affine_mapping(
        operator=change.ice_thickness_to_gmsl_operator
    )
    for name, change in ice_changes.items()
}

estimated_gmsl_measures = {
    name: change.ice_load.affine_mapping(
        operator=change.load_to_estimated_gmsl_operator
    )
    for name, change in ice_changes.items()
}

summary_df = pd.DataFrame(
    [
        {
            "Source": name,
            "True GMSL mean (mm)": stats_in_mm(
                true_gmsl_measures[name]
            )["mean_mm"],
            "True GMSL std (mm)": stats_in_mm(
                true_gmsl_measures[name]
            )["std_mm"],
            "Estimated GMSL mean (mm)": stats_in_mm(
                estimated_gmsl_measures[name]
            )["mean_mm"],
            "Estimated GMSL std (mm)": stats_in_mm(
                estimated_gmsl_measures[name]
            )["std_mm"],
        }
        for name in source_builders
    ]
)

summary_df

# ---- Notebook code cell 3 ----
thickness_scale_mm = max(
    1e3 * change.ice_thickness.expectation.data.max()
    for change in ice_changes.values()
)

for name, change in ice_changes.items():
    print(name)
    plot(
        1e3 * change.ice_thickness.expectation,
        colorbar_label="Mean ice-thickness change (mm)",
        vmin=0,
        vmax=thickness_scale_mm,
    )

# ---- Notebook code cell 4 ----
true_stats = plot_scalar_distributions(
    true_gmsl_measures,
    title="True GMSL distributions implied by SLC",
    xlabel="True GMSL change (mm)",
)

display(true_stats.round(3))

# ---- Notebook code cell 5 ----
# plot slc for each source
v_val = max(
    1e3
    * np.abs(
        (
            change.ice_slc.expectation
            * fp.ocean_projection(value=0.0)
        ).data
    ).max()
    for change in ice_changes.values()
)

for name, change in ice_changes.items():
    plot(
        1e3
        * change.ice_slc.expectation
        * fp.ocean_projection(),
        colorbar_label="Mean SLC (mm)",
        vmin=-v_val,
        vmax=v_val,
    )

# ---- Notebook code cell 6 ----
ssh_scale_mm = max(
    1e3 * np.abs(change.ice_ssh.expectation.data).max()
    for change in ice_changes.values()
)

for name, change in ice_changes.items():
    print(name)
    plot(
        1e3
        * change.ice_ssh.expectation
        * fp.ocean_projection(),
        colorbar_label="Mean SSHC (mm)",
        vmin=-ssh_scale_mm,
        vmax=ssh_scale_mm,
    )

# ---- Notebook code cell 7 ----
estimated_stats = plot_scalar_distributions(
    estimated_gmsl_measures,
    title=(
        "Altimetry-estimated GMSL distributions from SSHC "
        f"for +/- {ALTIMETRY_LATITUDE_RANGE:.0f} degrees"
    ),
    xlabel="Estimated GMSL change (mm)",
)

display(summary_df.round(3))
display(estimated_stats.round(3))

# ---- Notebook code cell 8 ----
scalar_data = np.load(
    resolve_scalar_data_path(
        "major_source_altimetry_errors_scalar.npz"
    )
)

latitudes = scalar_data["latitudes"]
plot_df = pd.DataFrame(
    {
        "Latitude": np.concatenate(
            [latitudes, latitudes, latitudes]
        ),
        "Relative Error (%)": 100
        * np.concatenate(
            [
                scalar_data["gis_errors"],
                scalar_data["wais_errors"],
                scalar_data["eais_errors"],
            ]
        ),
        "Source": np.concatenate(
            [
                ["GIS"] * len(latitudes),
                ["WAIS"] * len(latitudes),
                ["EAIS"] * len(latitudes),
            ]
        ),
    }
)

plt.figure(figsize=(6.5, 4.0))
sns.lineplot(
    data=plot_df,
    x="Latitude",
    y="Relative Error (%)",
    hue="Source",
    palette=SOURCE_COLOURS,
)
plt.axhline(0, color="black", linewidth=1)
plt.axvline(
    ALTIMETRY_LATITUDE_RANGE,
    color=colors.primary_error,
    linestyle="--",
    linewidth=1,
    label="Typical altimetry range",
)
plt.fill_between(
    x=[60, 75],
    y1=-10,
    y2=10,
    color=colors.primary_error,
    alpha=0.1,
)
plt.ylim(-10, 4)
plt.xlabel("Latitude limit (degrees)")
plt.ylabel("Relative error (%)")
plt.title(
    "Scalar-field GMSL estimation error across latitude coverage"
)
plt.grid(alpha=0.3)
plt.legend()
sns.despine()
plt.show()

_save_all_figures("02_major_sources")
