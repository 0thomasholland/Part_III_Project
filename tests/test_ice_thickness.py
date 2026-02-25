import unittest

import numpy as np

from pyslfp_extras.ice_thickness import IceSheetChange


class DummyGrid:
    def __init__(self, data: np.ndarray):
        self.data = np.array(data, dtype=float)

    def copy(self) -> "DummyGrid":
        return DummyGrid(self.data.copy())

    def __mul__(self, other: "DummyGrid") -> "DummyGrid":
        return DummyGrid(self.data * other.data)

    def __add__(self, other: "DummyGrid") -> "DummyGrid":
        return DummyGrid(self.data + other.data)


class DummyFingerPrint:
    def __init__(
        self,
        ice_thickness: np.ndarray,
        ice_mask: np.ndarray,
    ):
        self.ice_thickness = DummyGrid(ice_thickness)
        self._ice_mask = np.array(ice_mask, dtype=float)

    def ice_projection(self, value: float = 0) -> DummyGrid:
        # value is ignored for this dummy; we return a mask grid
        return DummyGrid(self._ice_mask)


class FakeMeasure:
    def __init__(self, sample_value: DummyGrid):
        self._sample_value = sample_value

    def sample(self) -> DummyGrid:
        return self._sample_value


class DummyIceSheetChange(IceSheetChange):
    pass


class TestIceThicknessPatterns(unittest.TestCase):
    def test_thickness_weighted_pattern_standardise_range(
        self,
    ) -> None:
        pattern = IceSheetChange.ThicknessWeightedPattern()
        data = np.array([2.0, 4.0, 6.0], dtype=float)
        standardised = pattern._standardise(data)

        self.assertTrue(np.isclose(standardised.min(), 0.0))
        self.assertTrue(np.isclose(standardised.max(), 1.0))

    def test_thickness_weighted_pattern_spatial_weights_masked(
        self,
    ) -> None:
        thickness = np.array([[0.0, 1.0], [2.0, 3.0]])
        ice_mask = np.array([[1.0, 0.0], [1.0, 0.0]])
        fp = DummyFingerPrint(thickness, ice_mask)
        pattern = IceSheetChange.ThicknessWeightedPattern()

        weights = pattern.spatial_weights(fp)

        # Outside ice extent should be zero after masking
        self.assertTrue(
            np.all(weights.data[ice_mask == 0] == 0.0)
        )

    def test_thickness_weighted_pattern_firn_complement_within_mask(
        self,
    ) -> None:
        thickness = np.array([[0.0, 1.0], [2.0, 3.0]])
        ice_mask = np.array([[1.0, 0.0], [1.0, 0.0]])
        fp = DummyFingerPrint(thickness, ice_mask)
        pattern = IceSheetChange.ThicknessWeightedPattern()

        ice_weights = pattern.spatial_weights(fp)
        firn_weights = pattern.firn_weights(fp)

        combined = ice_weights.data + firn_weights.data
        # Within mask, complements should sum to 1
        self.assertTrue(
            np.allclose(combined[ice_mask == 1], 1.0)
        )
        # Outside mask, both are zero
        self.assertTrue(
            np.all(combined[ice_mask == 0] == 0.0)
        )

    def test_uniform_pattern_returns_mask(self) -> None:
        thickness = np.array([[5.0, 6.0], [7.0, 8.0]])
        ice_mask = np.array([[1.0, 0.0], [1.0, 1.0]])
        fp = DummyFingerPrint(thickness, ice_mask)
        pattern = IceSheetChange.UniformPattern()

        weights = pattern.spatial_weights(fp)

        self.assertTrue(np.allclose(weights.data, ice_mask))

    def test_activator_within_asymptotes(self) -> None:
        pattern = IceSheetChange.ThicknessWeightedPattern(
            lower_asymptote=0.2, upper_asymptote=0.9
        )
        x = np.linspace(0.0, 1.0, 5)
        y = pattern._activator(x)

        self.assertTrue(np.all(y >= 0.2))
        self.assertTrue(np.all(y <= 0.9))

    def test_standardise_preserves_shape(self) -> None:
        pattern = IceSheetChange.ThicknessWeightedPattern()
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        standardised = pattern._standardise(data)

        self.assertEqual(standardised.shape, data.shape)


class TestIceSheetChangeSampling(unittest.TestCase):
    def _make_instance(
        self,
        include_firn: bool,
        ice_sample: DummyGrid,
        firn_sample: DummyGrid | None,
    ) -> IceSheetChange:
        instance = object.__new__(DummyIceSheetChange)
        instance._include_firn = include_firn
        instance._thickness_to_load_op = lambda grid: (
            DummyGrid(grid.data * 2.0)
        )
        instance.ice_thickness_measure = FakeMeasure(
            ice_sample
        )
        if include_firn:
            instance.firn_thickness_measure = FakeMeasure(
                firn_sample
            )
        return instance

    def test_sample_without_firn(self) -> None:
        ice_sample = DummyGrid(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        instance = self._make_instance(
            False, ice_sample, None
        )

        sample = IceSheetChange.sample(instance)

        self.assertIsNone(sample.firn_thickness)
        self.assertIsNone(sample.firn_load)
        self.assertTrue(
            np.allclose(
                sample.ice_thickness.data, ice_sample.data
            )
        )
        self.assertTrue(
            np.allclose(
                sample.total_thickness.data, ice_sample.data
            )
        )
        self.assertTrue(
            np.allclose(
                sample.ice_load.data, ice_sample.data * 2.0
            )
        )
        self.assertTrue(
            np.allclose(
                sample.total_load.data,
                ice_sample.data * 2.0,
            )
        )

    def test_sample_with_firn(self) -> None:
        ice_sample = DummyGrid(
            np.array([[1.0, 2.0], [3.0, 4.0]])
        )
        firn_sample = DummyGrid(
            np.array([[0.5, 0.0], [0.5, 1.0]])
        )
        instance = self._make_instance(
            True, ice_sample, firn_sample
        )

        sample = IceSheetChange.sample(instance)

        self.assertIsNotNone(sample.firn_thickness)
        self.assertIsNotNone(sample.firn_load)
        self.assertTrue(
            np.allclose(
                sample.total_thickness.data,
                ice_sample.data + firn_sample.data,
            )
        )
        self.assertTrue(
            np.allclose(
                sample.total_load.data,
                2.0 * (ice_sample.data + firn_sample.data),
            )
        )


if __name__ == "__main__":
    unittest.main()
