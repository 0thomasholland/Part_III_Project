"""pyslfp_extras

Extensions to pyslfp.
"""

from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.measures import (
    ice_thickness_gaussian_measure,
    altimetry_error_gaussian_measure,
    odt_variability_field,
    odt_gaussian_measure,
    odt_fingerprint_ssh_measure,
    odt_total_ssh_measure,
)
