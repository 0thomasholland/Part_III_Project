"""pyslfp_extras

Extensions to pyslfp.
"""

from pyslfp_extras.altimetry import (
    GridPoints,
    altimetry_error_gaussian_measure,
)
from pyslfp_extras.gmsl import (
    GMSLOperatorBase,
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
    IceSheetChangeSample,
    IceThicknessGMSLOperators,
)
from pyslfp_extras.ocean_dynamics import (
    OceanDynamics,
)
