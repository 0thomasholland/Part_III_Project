from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray, ArrayLike
from pygeoinf import GaussianMeasure, LinearOperator 
from pyslfp import FingerPrint, sea_level_change_to_load_operator, ice_thickness_change_to_load_operator

from Part_III_Project.measure_space.operators import 



def ice_thickness_change_measures(fingerprint: FingerPrint = None, fingerprint_operator: LinearOperator = None, length_scale: float = 60, thickness_95_range: float = 100, net_thickness_change: float = 0.) -> tuple[GaussianMeasure, GaussianMeasure]:
    """
    -> ice_thickness_measure, ice_thickness_load_measure
    Takes a length scale for the ice thickness changes and either a target GMSL std or a 95% range for the ice thickness changes to set the amplitude of the ocean dynamic topography measure.
    """

    ice_measure = fingerprint_operator.domain.point_value_scaled_heat_kernel_gaussian_measure(
        scale=length_scale / fingerprint.length_scale,  # controls correlation length between nearby points
        amplitude=thickness_95_range / 3.92 / fingerprint.length_scale  # the standard deviation of melt at each point
    )
    if net_thickness_change != 0.:
        shift_vector = np.zeros(fingerprint_operator.domain.dim)
        shift_vector[0] = net_thickness_change
        shift_vector = fingerprint_operator.domain.from_components(shift_vector)
        ice_measure = ice_measure.affine_mapping(
            operator=fingerprint_operator.domain.identity_operator(),
            translation=shift_vector
        )
    ice_load_measure = ice_measure.affine_mapping(
        operator=ice_thickness_change_to_load_operator(
            finger_print=fingerprint, load_space=fingerprint_operator.domain
        )
    )
    return ice_measure, ice_load_measure

def ocean_dynamic_topography_measures(
        fingerprint: FingerPrint = None, fingerprint_operator: LinearOperator = None, lengthscale: float = 60, amplitude: float = 0.001
        ) -> tuple[GaussianMeasure, GaussianMeasure]:
    """
    -> ODT_measure, ODT_load_measure
    """
    initial_odt_measure = fingerprint_operator.domain.point_value_scaled_sobolev_kernel_gaussian_measure(
        order=1.5,
        scale=lengthscale / fingerprint.length_scale,
        amplitude=amplitude / fingerprint.length_scale,
    )
    pass

def load_measure(
    ice_thickness_load_measure: GaussianMeasure | tuple[GaussianMeasure, GaussianMeasure], odt_load_measure: GaussianMeasure | tuple[GaussianMeasure, GaussianMeasure]
) -> GaussianMeasure:
    """
    -> total_load_measure
    """
    if isinstance(ice_thickness_load_measure, tuple):
        ice_thickness_load_measure = ice_thickness_load_measure[1]
    if isinstance(odt_load_measure, tuple):
        odt_load_measure = odt_load_measure[1]
    pass

def sea_level_change_measure():
    pass

def sensor_noise_measure(
        *, noise_scale: float = 0.01, noise_lengthscale: float = 1.0
        ) -> GaussianMeasure:
    """
    -> noise_measure
    """
    pass

def sea_surface_height_measure(slc_measure: GaussianMeasure, odt_measure: GaussianMeasure | tuple[GaussianMeasure, GaussianMeasure], noise_measure: GaussianMeasure) -> tuple[GaussianMeasure, GaussianMeasure, GaussianMeasure]:
    """
    -> SSH, SSH+ODT, SSH+ODT+NOISE
    takes in slc_measure and odt_measure (nb odt_measures[0] is the ODT measure)
    """
    if isinstance(odt_measure, tuple):
        odt_measure = odt_measure[0]
    pass

def altimetry_measurements_measure(
    ssh_measure: tuple[GaussianMeasure, GaussianMeasure, GaussianMeasure], altimetry_range: float = 66
) -> tuple[GaussianMeasure, GaussianMeasure, GaussianMeasure]:
    """
    -> SSH_alt range, SSH+ODT_alt range, SSH+ODT+NOISE_alt range
    """
    pass