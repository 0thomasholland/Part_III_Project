# =============================================================================
# Factored forward operator (eq 23)
# =============================================================================
#
# The original 3×3 block operator (eq 22):
#
#   [[P_S·S·F·L_I,  P_S·S·F·L_F,  P_S·(S·F·L_W + 1)],
#    [P_T·F·L_I,    P_T·F·L_I,    P_T·(F·L_W + 1)  ],
#    [P_I,           P_I,           0                  ]]
#
# is factored as  P_left @ F_middle @ L_right  where the expensive
# fingerprint F appears only once in the block-diagonal F_middle.
#
#   P_left  (3×4):  point evaluation / SSH / SLC extraction
#   F_middle(4×4):  block_diag(F, I, I, I)
#   L_right (4×3):  load operators + routing permutation
# =============================================================================

from pygeoinf import (
    BlockDiagonalLinearOperator,
    BlockLinearOperator,
)
from pyslfp.linear_operators import (
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)


def build_factored_forward_operator(
    fp,
    fp_op,
    ice,
    odt,
    ssh_altimetry,
    ice_altimetry,
    tide_gauge_points,
):
    """
    Build the forward operator in the factored form:

        forward_operator = P_left @ F_middle @ L_right

    Parameters
    ----------
    fp : FingerPrint
    fp_op : LinearOperator
        Fingerprint operator (load -> response), from
        fp.as_sobolev_linear_operator(...).
    ice : IceSheetChange
        Must have include_firn=True.
    odt : OceanDynamics
    ssh_altimetry : GridPoints
        SSH altimetry observation locations.
    ice_altimetry : GridPoints
        Ice altimetry observation locations.
    tide_gauge_points : list of (lat, lon)
        Tide gauge locations.

    Returns
    -------
    forward_operator : LinearOperator
    """

    # -- Spaces --
    load_space = fp_op.domain
    response_space = fp_op.codomain
    ice_space = ice.ice_thickness.domain
    firn_space = ice.firn_thickness.domain
    odt_space = odt.height_measure.domain

    # -- Component operators --
    F = fp_op
    S = sea_surface_height_operator(fp, response_space)
    slc_proj = response_space.subspace_projection(0)
    slc_space = slc_proj.codomain

    L_I = ice.ice_thickness_to_load_operator
    L_F = ice.firn_thickness_to_load_operator
    L_W = sea_level_change_to_load_operator(
        fp, odt_space, load_space
    )

    # -- Point evaluation operators --
    P_S_ssh = ssh_altimetry.point_evaluation_operator(
        S.codomain
    )
    P_S_odt = ssh_altimetry.point_evaluation_operator(
        odt_space
    )

    # Tide gauge sampling — use point evaluation on SLC and ODT spaces
    # directly, rather than pyslfp.tide_gauge_operator which requires
    # a HilbertSpaceDirectSum (the full response space).
    P_T_slc = slc_space.point_evaluation_operator(
        tide_gauge_points
    )
    P_T_odt = odt_space.point_evaluation_operator(
        tide_gauge_points
    )

    P_I_ice = ice_altimetry.point_evaluation_operator(
        ice_space
    )
    P_I_firn = ice_altimetry.point_evaluation_operator(
        firn_space
    )

    # -- Identities --
    id_odt = odt_space.identity_operator()
    id_ice = ice_space.identity_operator()
    id_firn = firn_space.identity_operator()

    # -- Observation spaces --
    ssh_obs = P_S_ssh.codomain
    tg_obs = P_T_slc.codomain
    ice_obs = P_I_ice.codomain

    # == L_right (4×3) ==
    # [[L_I,  L_F,  L_W],     -> load
    #  [0,    0,    I  ],     -> odt
    #  [I,    0,    0  ],     -> ice
    #  [0,    I,    0  ]]     -> firn
    L_right = BlockLinearOperator(
        [
            [L_I, L_F, L_W],
            [
                ice_space.zero_operator(codomain=odt_space),
                firn_space.zero_operator(
                    codomain=odt_space
                ),
                id_odt,
            ],
            [
                id_ice,
                firn_space.zero_operator(
                    codomain=ice_space
                ),
                odt_space.zero_operator(codomain=ice_space),
            ],
            [
                ice_space.zero_operator(
                    codomain=firn_space
                ),
                id_firn,
                odt_space.zero_operator(
                    codomain=firn_space
                ),
            ],
        ]
    )

    # == F_middle (4×4 block diagonal) ==
    # diag(F, I_odt, I_ice, I_firn)
    F_middle = BlockDiagonalLinearOperator(
        [F, id_odt, id_ice, id_firn]
    )

    # == P_left (3×4) ==
    # [[P_S·S,        P_S_odt,  0,       0      ],
    #  [P_T·slc_proj, P_T_odt,  0,       0      ],
    #  [0,            0,         P_I_ice, P_I_firn]]
    P_left = BlockLinearOperator(
        [
            [
                P_S_ssh @ S,
                P_S_odt,
                ice_space.zero_operator(codomain=ssh_obs),
                firn_space.zero_operator(codomain=ssh_obs),
            ],
            [
                P_T_slc @ slc_proj,
                P_T_odt,
                ice_space.zero_operator(codomain=tg_obs),
                firn_space.zero_operator(codomain=tg_obs),
            ],
            [
                response_space.zero_operator(
                    codomain=ice_obs
                ),
                odt_space.zero_operator(codomain=ice_obs),
                P_I_ice,
                P_I_firn,
            ],
        ]
    )

    return P_left @ F_middle @ L_right


def build_factored_forward_operator_precon(
    precon_fp,
    precon_fp_op,
    precon_ice,
    precon_odt,
    ssh_altimetry_coords,
    ice_altimetry_coords,
    tide_gauge_points,
):
    """
    Build the preconditioner forward operator in the same factored form,
    using low-resolution operators but sampling at full-resolution points.

    Parameters
    ----------
    precon_fp : FingerPrint
        Low-resolution fingerprint.
    precon_fp_op : LinearOperator
        Low-resolution fingerprint operator.
    precon_ice : IceSheetChange
        Low-resolution ice model.
    precon_odt : OceanDynamics
        Low-resolution ocean dynamics.
    ssh_altimetry_coords : list of (lat, lon)
        Full-resolution SSH altimetry coordinates.
    ice_altimetry_coords : list of (lat, lon)
        Full-resolution ice altimetry coordinates.
    tide_gauge_points : list of (lat, lon)
        Tide gauge locations.

    Returns
    -------
    forward_operator : LinearOperator
    """

    # -- Spaces --
    load_space = precon_fp_op.domain
    response_space = precon_fp_op.codomain
    ice_space = precon_ice.ice_thickness.domain
    firn_space = precon_ice.firn_thickness.domain
    odt_space = precon_odt.height_measure.domain

    # -- Component operators --
    F = precon_fp_op
    S = sea_surface_height_operator(
        precon_fp, response_space
    )
    slc_proj = response_space.subspace_projection(0)
    slc_space = slc_proj.codomain

    L_I = precon_ice.ice_thickness_to_load_operator
    L_F = precon_ice.firn_thickness_to_load_operator
    L_W = sea_level_change_to_load_operator(
        precon_fp, odt_space, load_space
    )

    # -- Point evaluation at FULL-RESOLUTION coordinates --
    # These use the precon space's point_evaluation_operator method
    # but evaluate at the full-resolution observation coords.
    P_S_ssh = S.codomain.point_evaluation_operator(
        ssh_altimetry_coords
    )
    P_S_odt = odt_space.point_evaluation_operator(
        ssh_altimetry_coords
    )

    # Tide gauge sampling — point evaluation on SLC and ODT spaces
    P_T_slc = slc_space.point_evaluation_operator(
        tide_gauge_points
    )
    P_T_odt = odt_space.point_evaluation_operator(
        tide_gauge_points
    )

    P_I_ice = ice_space.point_evaluation_operator(
        ice_altimetry_coords
    )
    P_I_firn = firn_space.point_evaluation_operator(
        ice_altimetry_coords
    )

    # -- Identities --
    id_odt = odt_space.identity_operator()
    id_ice = ice_space.identity_operator()
    id_firn = firn_space.identity_operator()

    # -- Observation spaces --
    ssh_obs = P_S_ssh.codomain
    tg_obs = P_T_slc.codomain
    ice_obs = P_I_ice.codomain

    # == L_right (4×3) ==
    L_right = BlockLinearOperator(
        [
            [L_I, L_F, L_W],
            [
                ice_space.zero_operator(codomain=odt_space),
                firn_space.zero_operator(
                    codomain=odt_space
                ),
                id_odt,
            ],
            [
                id_ice,
                firn_space.zero_operator(
                    codomain=ice_space
                ),
                odt_space.zero_operator(codomain=ice_space),
            ],
            [
                ice_space.zero_operator(
                    codomain=firn_space
                ),
                id_firn,
                odt_space.zero_operator(
                    codomain=firn_space
                ),
            ],
        ]
    )

    # == F_middle (4×4 block diagonal) ==
    F_middle = BlockDiagonalLinearOperator(
        [F, id_odt, id_ice, id_firn]
    )

    # == P_left (3×4) ==
    P_left = BlockLinearOperator(
        [
            [
                P_S_ssh @ S,
                P_S_odt,
                ice_space.zero_operator(codomain=ssh_obs),
                firn_space.zero_operator(codomain=ssh_obs),
            ],
            [
                P_T_slc @ slc_proj,
                P_T_odt,
                ice_space.zero_operator(codomain=tg_obs),
                firn_space.zero_operator(codomain=tg_obs),
            ],
            [
                response_space.zero_operator(
                    codomain=ice_obs
                ),
                odt_space.zero_operator(codomain=ice_obs),
                P_I_ice,
                P_I_firn,
            ],
        ]
    )

    return P_left @ F_middle @ L_right
