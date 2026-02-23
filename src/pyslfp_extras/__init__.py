"""pyslfp_extras

Extensions to pyslfp.
"""

from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.measures import (
    altimetry_error_gaussian_measure,
    ice_thickness_gaussian_measure,
    non_ice_ssh_variability_field,
    non_ice_ssh_variability_fingerprint_ssh_measure,
    non_ice_ssh_variability_gaussian_measure,
    non_ice_ssh_variability_total_ssh_measure,
)
