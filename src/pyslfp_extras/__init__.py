"""pyslfp_extras

Extensions to pyslfp.
"""

from pyslfp_extras.altimetry import (
    altimetry_error_gaussian_measure,
)
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
    IceSheetChangeSample,
)
from pyslfp_extras.ocean_dynamics import (
    non_ice_ssh_variability_field,
    non_ice_ssh_variability_fingerprint_ssh_measure,
    non_ice_ssh_variability_gaussian_measure,
    non_ice_ssh_variability_total_ssh_measure,
)
