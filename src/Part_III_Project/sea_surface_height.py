def dSL_to_dSSH(delta_sea_level, delta_gravity_potential):
    delta_sea_surface = delta_sea_level + (delta_gravity_potential / 9.81)
    return delta_sea_surface
