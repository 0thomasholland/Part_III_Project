import importlib
import os
import sys
import types
import unittest
from typing import Any, Optional
from unittest import mock

import numpy as np

# Ensure the src directory is on sys.path for module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class FakeGrid:
    """A minimal SHGrid stand-in supporting + and / operations."""

    def __init__(self, data: Any):
        self.data = np.asarray(data)

    def __add__(self, other: "FakeGrid") -> "FakeGrid":
        return FakeGrid(self.data + np.asarray(other.data))

    def __truediv__(self, scalar: float) -> "FakeGrid":
        return FakeGrid(self.data / scalar)

    def asarray(self):
        return np.asarray(self.data)

    # Helpful for assertion messages
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"FakeGrid(data={self.data!r})"


class DummyFingerPrint:
    def __init__(self, centrifugal_effect: FakeGrid):
        self._centrifugal_effect = centrifugal_effect
        self.last_angular_velocity_change: Optional[np.ndarray] = None

    def calculate_centrifugal_effect(self, *, angular_velocity_change: np.ndarray):
        self.last_angular_velocity_change = angular_velocity_change
        return self._centrifugal_effect


def import_module_with_fakes(gravity_value: float = 10.0):
    """Import the module under test while faking external deps."""
    # Create fake pyshtools with SHGrid
    fake_pyshtools = types.ModuleType("pyshtools")
    fake_pyshtools.SHGrid = FakeGrid  # type: ignore[attr-defined]

    # Create fake pyslfp and pyslfp.physical_parameters
    fake_pyslfp = types.ModuleType("pyslfp")
    # Provide a FingerPrint symbol for typing; not used functionally
    class _FingerPrint:  # noqa: N801 - mimic external class name
        pass

    fake_pyslfp.FingerPrint = _FingerPrint  # type: ignore[attr-defined]

    fake_physical_parameters = types.ModuleType("pyslfp.physical_parameters")
    fake_physical_parameters.GRAVITATIONAL_ACCELERATION = gravity_value  # type: ignore[attr-defined]

    # Patch sys.modules so the import inside the module succeeds
    patches = {
        "pyshtools": fake_pyshtools,
        "pyslfp": fake_pyslfp,
        "pyslfp.physical_parameters": fake_physical_parameters,
    }

    return patches


class SeaSurfaceHeightChangeTests(unittest.TestCase):
    def setUp(self):
        # Clear any previously imported module to ensure clean import per test
        sys.modules.pop("Part_III_Project.sea_surface_height_change", None)

    def test_scalar_inputs(self):
        patches = import_module_with_fakes(gravity_value=4.0)
        with mock.patch.dict(sys.modules, patches):
            mod = importlib.import_module("Part_III_Project.sea_surface_height_change")

        sea_level = FakeGrid(10.0)
        displacement = FakeGrid(2.0)
        centrifugal = FakeGrid(8.0)
        fp = DummyFingerPrint(centrifugal)

        result = mod.sea_surface_height_change(
            finger_print=fp,
            sea_level_change=sea_level,
            displacement=displacement,
            angular_velocity_change=np.array([1.0, 0.0, 0.0]),
        )

        # Expected: 10 + 2 + 8/4 = 14
        np.testing.assert_allclose(result.asarray(), 14.0)
        # Ensure we passed through the angular velocity change
        self.assertIsNotNone(fp.last_angular_velocity_change)

    def test_array_inputs(self):
        patches = import_module_with_fakes(gravity_value=2.0)
        with mock.patch.dict(sys.modules, patches):
            mod = importlib.import_module("Part_III_Project.sea_surface_height_change")

        sea_level = FakeGrid([1.0, 2.0, 3.0])
        displacement = FakeGrid([0.5, 0.5, 0.5])
        centrifugal = FakeGrid([0.0, 2.0, 4.0])
        fp = DummyFingerPrint(centrifugal)

        result = mod.sea_surface_height_change(
            finger_print=fp,
            sea_level_change=sea_level,
            displacement=displacement,
            angular_velocity_change=np.array([0.0, 1.0, 0.0]),
        )

        # Expected elementwise: [1+0.5+0/2, 2+0.5+2/2, 3+0.5+4/2] = [1.5, 3.5, 5.5]
        np.testing.assert_allclose(result.asarray(), np.array([1.5, 3.5, 5.5]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
