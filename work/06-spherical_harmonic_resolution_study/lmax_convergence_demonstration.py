# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import matplotlib.pyplot as plt
from pygeoinf import GaussianMeasure, LinearOperator
from pyslfp import (
    EarthModelParameters,
    FingerPrint,
    averaging_operator,
    plot,
)

print("FOR LMAX = 256")
fp = FingerPrint(
    lmax=256,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)

fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

# %%
length_scale = 0.1 * fp.mean_sea_floor_radius
sd = 50.0 / fp.length_scale  # in meters, non-dimensionalized

measure = (
    fingerprint_operator.domain.point_value_scaled_sobolev_kernel_gaussian_measure(
        scale=length_scale,
        order=2,
        amplitude=sd,
    )
)

# %%
(
    fig2,
    ax2,
    im2,
) = plot(measure.sample() * fp.length_scale)
fig2.colorbar(
    im2,
    ax=ax2,
    label="Measure sample (m)",
    orientation="horizontal",
)

# %%
average_measure_operator = averaging_operator(
    measure.domain,
    [
        fp.altimetry_projection(
            latitude_max=90,
            latitude_min=-90,
            value=0,
        )
        / (
            fp.integrate(
                fp.altimetry_projection(
                    latitude_max=90,
                    latitude_min=-90,
                    value=0,
                ),
            )
        ),
    ],
)
average_measure = measure.affine_mapping(
    operator=average_measure_operator,
)

# %%
variance = (
    average_measure.covariance.matrix(dense=True)[
        0,
        0,
    ]
    * fp.length_scale**2
)

print("For averaging over whole earth:")
print("Expected ice standard deviation:", sd * fp.length_scale)
standard_deviation_whole_256 = variance**0.5
print(
    "Ice thickness standard deviation (m):",
    standard_deviation_whole_256,
)
print("Ice thickness variance (m):", variance)

# print("Expected ice thicknesss expecatation:" shift)
expectation = average_measure.expectation[0] * fp.length_scale
print("Ice thickness expectation (m):", expectation)

# %%
average_measure_operator = averaging_operator(
    measure.domain,
    [
        fp.ice_projection(
            value=0,
        )
        / (
            fp.integrate(
                fp.ice_projection(
                    value=0,
                ),
            )
        ),
    ],
)
average_measure = measure.affine_mapping(
    operator=average_measure_operator,
)

# %%
variance = (
    average_measure.covariance.matrix(dense=True)[
        0,
        0,
    ]
    * fp.length_scale**2
)

print("For averaging over ice sheets:")
print("Expected ice standard deviation:", sd * fp.length_scale)
standard_deviation_ice_256 = variance**0.5
print(
    "Ice thickness standard deviation (m):",
    standard_deviation_ice_256,
)
print("Ice thickness variance (m):", variance)

# print("Expected ice thicknesss expecatation:" shift)
expectation = average_measure.expectation[0] * fp.length_scale
print("Ice thickness expectation (m):", expectation)

print("FOR LMAX = 128")

fp = FingerPrint(
    lmax=128,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)

fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

# %%
length_scale = 0.1 * fp.mean_sea_floor_radius
sd = 50.0 / fp.length_scale  # in meters, non-dimensionalized

measure = (
    fingerprint_operator.domain.point_value_scaled_sobolev_kernel_gaussian_measure(
        scale=length_scale,
        order=2,
        amplitude=sd,
    )
)

# %%
(
    fig2,
    ax2,
    im2,
) = plot(measure.sample() * fp.length_scale)
fig2.colorbar(
    im2,
    ax=ax2,
    label="Measure sample (m)",
    orientation="horizontal",
)

# %%
average_measure_operator = averaging_operator(
    measure.domain,
    [
        fp.altimetry_projection(
            latitude_max=90,
            latitude_min=-90,
            value=0,
        )
        / (
            fp.integrate(
                fp.altimetry_projection(
                    latitude_max=90,
                    latitude_min=-90,
                    value=0,
                ),
            )
        ),
    ],
)
average_measure = measure.affine_mapping(
    operator=average_measure_operator,
)

# %%
variance = (
    average_measure.covariance.matrix(dense=True)[
        0,
        0,
    ]
    * fp.length_scale**2
)

print("For averaging over whole earth:")
print("Expected ice standard deviation:", sd * fp.length_scale)
standard_deviation_whole_128 = variance**0.5
print(
    "Ice thickness standard deviation (m):",
    standard_deviation_whole_128,
)
print("Ice thickness variance (m):", variance)

# print("Expected ice thicknesss expecatation:" shift)
expectation = average_measure.expectation[0] * fp.length_scale
print("Ice thickness expectation (m):", expectation)

# %%
average_measure_operator = averaging_operator(
    measure.domain,
    [
        fp.ice_projection(
            value=0,
        )
        / (
            fp.integrate(
                fp.ice_projection(
                    value=0,
                ),
            )
        ),
    ],
)
average_measure = measure.affine_mapping(
    operator=average_measure_operator,
)

# %%
variance = (
    average_measure.covariance.matrix(dense=True)[
        0,
        0,
    ]
    * fp.length_scale**2
)

print("For averaging over ice sheets:")
print("Expected ice standard deviation:", sd * fp.length_scale)
standard_deviation_ice_128 = variance**0.5
print(
    "Ice thickness standard deviation (m):",
    standard_deviation_ice_128,
)
print("Ice thickness variance (m):", variance)

# print("Expected ice thicknesss expecatation:" shift)
expectation = average_measure.expectation[0] * fp.length_scale
print("Ice thickness expectation (m):", expectation)

# %%
print("FOR LMAX = 512")

fp = FingerPrint(
    lmax=512,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)

fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

# %%
length_scale = 0.1 * fp.mean_sea_floor_radius
sd = 50.0 / fp.length_scale  # in meters, non-dimensionalized

measure = (
    fingerprint_operator.domain.point_value_scaled_sobolev_kernel_gaussian_measure(
        scale=length_scale,
        order=2,
        amplitude=sd,
    )
)

# %%
(
    fig2,
    ax2,
    im2,
) = plot(measure.sample() * fp.length_scale)
fig2.colorbar(
    im2,
    ax=ax2,
    label="Measure sample (m)",
    orientation="horizontal",
)

# %%
average_measure_operator = averaging_operator(
    measure.domain,
    [
        fp.altimetry_projection(
            latitude_max=90,
            latitude_min=-90,
            value=0,
        )
        / (
            fp.integrate(
                fp.altimetry_projection(
                    latitude_max=90,
                    latitude_min=-90,
                    value=0,
                ),
            )
        ),
    ],
)
average_measure = measure.affine_mapping(
    operator=average_measure_operator,
)

# %%
variance = (
    average_measure.covariance.matrix(dense=True)[
        0,
        0,
    ]
    * fp.length_scale**2
)

print("For averaging over whole earth:")
print("Expected ice standard deviation:", sd * fp.length_scale)
standard_deviation_whole_512 = variance**0.5
print(
    "Ice thickness standard deviation (m):",
    standard_deviation_whole_512,
)
print("Ice thickness variance (m):", variance)

# print("Expected ice thicknesss expecatation:" shift)
expectation = average_measure.expectation[0] * fp.length_scale
print("Ice thickness expectation (m):", expectation)

# %%
average_measure_operator = averaging_operator(
    measure.domain,
    [
        fp.ice_projection(
            value=0,
        )
        / (
            fp.integrate(
                fp.ice_projection(
                    value=0,
                ),
            )
        ),
    ],
)
average_measure = measure.affine_mapping(
    operator=average_measure_operator,
)

# %%
variance = (
    average_measure.covariance.matrix(dense=True)[
        0,
        0,
    ]
    * fp.length_scale**2
)

print("For averaging over ice sheets:")
print("Expected ice standard deviation:", sd * fp.length_scale)
standard_deviation_ice_512 = variance**0.5
print(
    "Ice thickness standard deviation (m):",
    standard_deviation_ice_512,
)
print("Ice thickness variance (m):", variance)

# print("Expected ice thicknesss expecatation:" shift)
expectation = average_measure.expectation[0] * fp.length_scale
print("Ice thickness expectation (m):", expectation)

# %%
print("Summary:")

# as table format of lmax vs standard deviations
print("lmax\t\tStd whole (m)\tStd ice (m)")
print(
    f"input\t\t{sd * fp.length_scale:.4f}\t\t{sd * fp.length_scale:.4f}",
)
print(
    f"128\t\t{standard_deviation_whole_128:.4f}\t\t{standard_deviation_ice_128:.4f}",
)
print(
    f"256\t\t{standard_deviation_whole_256:.4f}\t\t{standard_deviation_ice_256:.4f}",
)
print(
    f"512\t\t{standard_deviation_whole_512:.4f}\t\t{standard_deviation_ice_512:.4f}",
)
