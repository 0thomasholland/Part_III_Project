import unittest

import numpy as np

from pyslfp_extras.ocean_dynamics import OceanDynamics


class DummyGrid:
    def __init__(self, data: np.ndarray):
        self.data = np.array(data, dtype=float)

    def to_array(self) -> np.ndarray:
        return np.array(self.data, dtype=float)

    def copy(self) -> "DummyGrid":
        return DummyGrid(self.data.copy())


class DummyFingerPrint:
    def __init__(
        self,
        ocean_mask: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ):
        self._ocean_mask = np.array(ocean_mask, dtype=float)
        self._lats = np.array(lats, dtype=float)
        self._lons = np.array(lons, dtype=float)

    def ocean_projection(
        self, value: float = 0
    ) -> DummyGrid:
        _ = value
        return DummyGrid(self._ocean_mask)

    def zero_grid(self) -> DummyGrid:
        return DummyGrid(np.zeros_like(self._ocean_mask))

    def lats(self) -> np.ndarray:
        return self._lats

    def lons(self) -> np.ndarray:
        return self._lons


class TestOceanDynamicsPatterns(unittest.TestCase):
    def setUp(self) -> None:
        lats = np.array([-10.0, 10.0])
        lons = np.array([0.0, 90.0])
        ocean_mask = np.array([[1.0, 0.0], [1.0, 1.0]])
        self.fp = DummyFingerPrint(ocean_mask, lats, lons)

    def test_default_pattern_is_uniform(self) -> None:
        odt = OceanDynamics(self.fp, object())
        self.assertIsInstance(
            odt._pattern, OceanDynamics.UniformPattern
        )

    def test_uniform_pattern_matches_ocean_mask(
        self,
    ) -> None:
        pattern = OceanDynamics.UniformPattern()
        grid = pattern.spatial_field(self.fp)

        self.assertTrue(
            np.allclose(grid.data, self.fp._ocean_mask)
        )

    def test_synthetic_pattern_weights_in_range_and_masked(
        self,
    ) -> None:
        pattern = OceanDynamics.SyntheticPattern(
            point_multiplier=5.0
        )
        grid = pattern.spatial_field(self.fp)
        data = grid.data
        ocean = self.fp._ocean_mask.astype(bool)

        self.assertTrue(np.all(data[~ocean] == 0.0))
        self.assertTrue(np.all(data[ocean] >= 0.0))
        self.assertTrue(np.all(data[ocean] <= 1.0))
        self.assertTrue(np.isclose(data[ocean].max(), 1.0))

    def test_data_pattern_normalises_to_unit_range(
        self,
    ) -> None:
        pattern = OceanDynamics.DataPattern()
        grid = pattern.spatial_field(self.fp)
        data = grid.data
        ocean = self.fp._ocean_mask.astype(bool)

        self.assertTrue(np.all(data[~ocean] == 0.0))
        self.assertTrue(np.isclose(data[ocean].min(), 0.0))
        self.assertTrue(np.isclose(data[ocean].max(), 1.0))


if __name__ == "__main__":
    unittest.main()
