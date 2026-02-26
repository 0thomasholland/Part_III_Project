# %%
from __future__ import annotations

import pathlib

import numpy as np
from pyshtools import SHGrid
from pyslfp import FingerPrint, plot

LMAX = 256

fp = FingerPrint(lmax=LMAX)
fp.set_state_from_ice_ng()


def _find_project_root(start: pathlib.Path) -> pathlib.Path:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "pyslfp_extras").exists():
            return candidate
    return start


try:
    _start_path = pathlib.Path(__file__).resolve()
except NameError:
    _start_path = pathlib.Path.cwd()

_base_path = _find_project_root(_start_path)

DATA_PATH = (
    _base_path
    / "src"
    / "pyslfp_extras"
    / "data"
    / "altimetry"
    / f"sla_diff_std_lmax{LMAX}.npz"
)

shgrid = np.load(DATA_PATH)["sla_diff_std"]
print(shgrid)
shgrid = SHGrid.from_array(shgrid)
print(shgrid.lmax)
fig, ax, im = plot(shgrid * fp.altimetry_projection())
ax.set_title("SLA diff std (lmax=256)")

fig.show()
