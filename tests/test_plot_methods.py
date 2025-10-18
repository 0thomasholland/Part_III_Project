import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

# Ensure the src directory is on sys.path for module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from Part_III_Project.plot_methods import (
    _add_colorbar_to_axis,
    _configure_ternary_axis,
    _extract_segment_data,
    _get_global_error_range,
    _plot_tripcolor,
)


class TestExtractSegmentData(unittest.TestCase):
    """Tests for _extract_segment_data helper function."""

    def setUp(self):
        """Create sample DataFrame for testing."""
        self.df = pd.DataFrame(
            {
                "segment": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0],
                "error": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "source_a": [0.5, 0.3, 0.2, 0.6, 0.4, 0.7],
                "source_b": [0.3, 0.4, 0.5, 0.2, 0.3, 0.1],
                "source_c": [0.2, 0.3, 0.3, 0.2, 0.3, 0.2],
            }
        )
        self.sources = ["source_a", "source_b", "source_c"]

    def test_extract_valid_segment(self):
        """Test extraction of data for a segment that exists."""
        result = _extract_segment_data(self.df, 1.0, self.sources)

        self.assertIsNotNone(result)
        top, left, right, errors = result

        # Check types
        self.assertIsInstance(top, np.ndarray)
        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)
        self.assertIsInstance(errors, np.ndarray)

        # Check values
        np.testing.assert_array_equal(top, [0.5, 0.3, 0.2])
        np.testing.assert_array_equal(left, [0.3, 0.4, 0.5])
        np.testing.assert_array_equal(right, [0.2, 0.3, 0.3])
        np.testing.assert_array_equal(errors, [0.1, 0.2, 0.3])

        # Check all arrays have the same length
        self.assertEqual(len(top), len(left))
        self.assertEqual(len(left), len(right))
        self.assertEqual(len(right), len(errors))

    def test_extract_nonexistent_segment(self):
        """Test extraction of data for a segment that doesn't exist."""
        result = _extract_segment_data(self.df, 99.0, self.sources)
        self.assertIsNone(result)

    def test_extract_single_row_segment(self):
        """Test extraction of data for a segment with only one row."""
        result = _extract_segment_data(self.df, 3.0, self.sources)

        self.assertIsNotNone(result)
        top, left, right, errors = result

        self.assertEqual(len(top), 1)
        np.testing.assert_array_equal(top, [0.7])
        np.testing.assert_array_equal(left, [0.1])
        np.testing.assert_array_equal(right, [0.2])
        np.testing.assert_array_equal(errors, [0.6])

    def test_extract_preserves_column_order(self):
        """Test that sources list order determines the returned order."""
        # Reverse the sources order
        reversed_sources = ["source_c", "source_b", "source_a"]
        result = _extract_segment_data(self.df, 1.0, reversed_sources)

        self.assertIsNotNone(result)
        top, left, right, errors = result

        # With reversed sources, top should now be source_c values
        np.testing.assert_array_equal(top, [0.2, 0.3, 0.3])
        np.testing.assert_array_equal(left, [0.3, 0.4, 0.5])
        np.testing.assert_array_equal(right, [0.5, 0.3, 0.2])

    def test_extract_with_float_segment(self):
        """Test extraction works with float segment values."""
        result = _extract_segment_data(self.df, 2.0, self.sources)

        self.assertIsNotNone(result)
        top, left, right, errors = result
        self.assertEqual(len(top), 2)
        np.testing.assert_array_equal(errors, [0.4, 0.5])


class TestGetGlobalErrorRange(unittest.TestCase):
    """Tests for _get_global_error_range helper function."""

    def setUp(self):
        """Create sample DataFrame for testing."""
        self.df = pd.DataFrame(
            {
                "segment": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0],
                "error": [0.1, 0.5, 0.2, 0.8, 0.3, 0.4, 1.0],
            }
        )

    def test_global_range_single_segment(self):
        """Test getting error range for a single segment."""
        vmin, vmax = _get_global_error_range(self.df, [1.0])

        self.assertEqual(vmin, 0.1)
        self.assertEqual(vmax, 0.5)

    def test_global_range_multiple_segments(self):
        """Test getting error range across multiple segments."""
        vmin, vmax = _get_global_error_range(self.df, [1.0, 2.0, 3.0])

        self.assertEqual(vmin, 0.1)
        self.assertEqual(vmax, 0.8)

    def test_global_range_all_segments(self):
        """Test getting error range across all segments."""
        vmin, vmax = _get_global_error_range(self.df, [1.0, 2.0, 3.0, 4.0])

        self.assertEqual(vmin, 0.1)
        self.assertEqual(vmax, 1.0)

    def test_global_range_empty_segment_list(self):
        """Test getting error range with empty segment list."""
        vmin, vmax = _get_global_error_range(self.df, [])

        # Should return NaN for empty selection
        self.assertTrue(np.isnan(vmin))
        self.assertTrue(np.isnan(vmax))

    def test_global_range_nonexistent_segment(self):
        """Test getting error range for segments that don't exist."""
        vmin, vmax = _get_global_error_range(self.df, [99.0])

        # Should return NaN for nonexistent segments
        self.assertTrue(np.isnan(vmin))
        self.assertTrue(np.isnan(vmax))

    def test_global_range_mixed_existing_nonexisting(self):
        """Test getting error range with mix of existing and nonexistent segments."""
        vmin, vmax = _get_global_error_range(self.df, [1.0, 99.0, 2.0])

        # Should only consider existing segments
        self.assertEqual(vmin, 0.1)
        self.assertEqual(vmax, 0.8)

    def test_global_range_preserves_float_precision(self):
        """Test that float values are preserved correctly."""
        df = pd.DataFrame(
            {
                "segment": [1.0, 1.0],
                "error": [0.123456789, 0.987654321],
            }
        )
        vmin, vmax = _get_global_error_range(df, [1.0])

        self.assertAlmostEqual(vmin, 0.123456789)
        self.assertAlmostEqual(vmax, 0.987654321)


class TestConfigureTernaryAxis(unittest.TestCase):
    """Tests for _configure_ternary_axis helper function."""

    def setUp(self):
        """Create mock axis for testing."""
        self.mock_ax = Mock()
        self.mock_ax.taxis = Mock()
        self.mock_ax.laxis = Mock()
        self.mock_ax.raxis = Mock()
        self.labels = ["Top Label", "Left Label", "Right Label"]

    def test_configure_sets_labels(self):
        """Test that axis labels are set correctly."""
        _configure_ternary_axis(self.mock_ax, self.labels, "Test Title")

        # Check that set_tlabel, set_llabel, set_rlabel were called
        self.mock_ax.set_tlabel.assert_called_once_with(
            "Top Label", fontsize=13, fontweight="bold"
        )
        self.mock_ax.set_llabel.assert_called_once_with(
            "Left Label", fontsize=13, fontweight="bold"
        )
        self.mock_ax.set_rlabel.assert_called_once_with(
            "Right Label", fontsize=13, fontweight="bold"
        )

    def test_configure_sets_label_positions(self):
        """Test that label positions are set to corner."""
        _configure_ternary_axis(self.mock_ax, self.labels, "Test Title")

        self.mock_ax.taxis.set_label_position.assert_called_once_with("corner")
        self.mock_ax.laxis.set_label_position.assert_called_once_with("corner")
        self.mock_ax.raxis.set_label_position.assert_called_once_with("corner")

    def test_configure_sets_title(self):
        """Test that title is set correctly."""
        _configure_ternary_axis(self.mock_ax, self.labels, "Test Title")

        self.mock_ax.set_title.assert_called_once_with(
            "Test Title", fontsize=14, fontweight="bold", pad=20
        )

    def test_configure_with_custom_fontsizes(self):
        """Test configuration with custom font sizes."""
        _configure_ternary_axis(
            self.mock_ax,
            self.labels,
            "Test Title",
            fontsize_labels=15,
            fontsize_title=16,
        )

        self.mock_ax.set_tlabel.assert_called_once_with(
            "Top Label", fontsize=15, fontweight="bold"
        )
        self.mock_ax.set_title.assert_called_once_with(
            "Test Title", fontsize=16, fontweight="bold", pad=20
        )

    def test_configure_title_padding_small_font(self):
        """Test that smaller title font size uses smaller padding."""
        _configure_ternary_axis(
            self.mock_ax, self.labels, "Test Title", fontsize_title=12
        )

        # When fontsize < 14, pad should be 15
        self.mock_ax.set_title.assert_called_once_with(
            "Test Title", fontsize=12, fontweight="bold", pad=15
        )

    def test_configure_title_padding_large_font(self):
        """Test that larger title font size uses larger padding."""
        _configure_ternary_axis(
            self.mock_ax, self.labels, "Test Title", fontsize_title=14
        )

        # When fontsize >= 14, pad should be 20
        self.mock_ax.set_title.assert_called_once_with(
            "Test Title", fontsize=14, fontweight="bold", pad=20
        )

    def test_configure_with_different_labels(self):
        """Test configuration with different label texts."""
        custom_labels = ["Custom A", "Custom B", "Custom C"]
        _configure_ternary_axis(self.mock_ax, custom_labels, "Custom Title")

        self.mock_ax.set_tlabel.assert_called_once_with(
            "Custom A", fontsize=13, fontweight="bold"
        )
        self.mock_ax.set_llabel.assert_called_once_with(
            "Custom B", fontsize=13, fontweight="bold"
        )
        self.mock_ax.set_rlabel.assert_called_once_with(
            "Custom C", fontsize=13, fontweight="bold"
        )


class TestAddColorbarToAxis(unittest.TestCase):
    """Tests for _add_colorbar_to_axis helper function."""

    def setUp(self):
        """Create mock axis and colorbar artist for testing."""
        self.mock_ax = Mock()
        self.mock_cs = Mock()
        self.mock_fig = Mock()
        self.mock_cax = Mock()
        self.mock_colorbar = Mock()

        # Setup return values
        self.mock_ax.get_figure.return_value = self.mock_fig
        self.mock_ax.inset_axes.return_value = self.mock_cax
        self.mock_fig.colorbar.return_value = self.mock_colorbar
        self.mock_ax.transAxes = "mock_transform"

    def test_colorbar_creation(self):
        """Test that colorbar is created with correct parameters."""
        result = _add_colorbar_to_axis(self.mock_ax, self.mock_cs)

        # Check inset_axes was called with correct parameters
        self.mock_ax.inset_axes.assert_called_once_with(
            [1.05, 0.1, 0.05, 0.8], transform=self.mock_ax.transAxes
        )

        # Check colorbar was created
        self.mock_fig.colorbar.assert_called_once_with(self.mock_cs, cax=self.mock_cax)

        # Check colorbar label was set
        self.mock_colorbar.set_label.assert_called_once_with(
            "Error", rotation=270, va="baseline", fontsize=12
        )

        # Check that the colorbar is returned
        self.assertEqual(result, self.mock_colorbar)

    def test_colorbar_with_custom_fontsize(self):
        """Test colorbar creation with custom font size."""
        _add_colorbar_to_axis(self.mock_ax, self.mock_cs, fontsize=15)

        self.mock_colorbar.set_label.assert_called_once_with(
            "Error", rotation=270, va="baseline", fontsize=15
        )

    def test_colorbar_gets_figure_from_axis(self):
        """Test that the figure is retrieved from the axis."""
        _add_colorbar_to_axis(self.mock_ax, self.mock_cs)

        self.mock_ax.get_figure.assert_called_once()

    def test_colorbar_returns_colorbar_object(self):
        """Test that the function returns the colorbar object."""
        result = _add_colorbar_to_axis(self.mock_ax, self.mock_cs)

        self.assertIsNotNone(result)
        self.assertEqual(result, self.mock_colorbar)


class TestPlotTripcolor(unittest.TestCase):
    """Tests for _plot_tripcolor helper function."""

    def setUp(self):
        """Create mock axis and test data."""
        self.mock_ax = Mock()
        self.mock_artist = Mock()
        self.mock_ax.tripcolor.return_value = self.mock_artist

        # Create test data
        self.top = np.array([0.5, 0.3, 0.2])
        self.left = np.array([0.3, 0.4, 0.5])
        self.right = np.array([0.2, 0.3, 0.3])
        self.errors = np.array([0.1, 0.2, 0.3])

    def test_tripcolor_called_with_correct_params(self):
        """Test that tripcolor is called with correct parameters."""
        result = _plot_tripcolor(
            self.mock_ax, self.top, self.left, self.right, self.errors, vmin=0.0, vmax=1.0
        )

        self.mock_ax.tripcolor.assert_called_once()
        call_args = self.mock_ax.tripcolor.call_args

        # Check positional arguments
        np.testing.assert_array_equal(call_args[0][0], self.top)
        np.testing.assert_array_equal(call_args[0][1], self.left)
        np.testing.assert_array_equal(call_args[0][2], self.right)
        np.testing.assert_array_equal(call_args[0][3], self.errors)

        # Check keyword arguments
        self.assertEqual(call_args[1]["shading"], "gouraud")
        self.assertEqual(call_args[1]["cmap"], "RdYlBu_r")
        self.assertEqual(call_args[1]["vmin"], 0.0)
        self.assertEqual(call_args[1]["vmax"], 1.0)
        self.assertTrue(call_args[1]["rasterized"])

    def test_tripcolor_returns_artist(self):
        """Test that the function returns the tripcolor artist."""
        result = _plot_tripcolor(
            self.mock_ax, self.top, self.left, self.right, self.errors, vmin=0.0, vmax=1.0
        )

        self.assertEqual(result, self.mock_artist)

    def test_tripcolor_with_different_vmin_vmax(self):
        """Test tripcolor with different color scale limits."""
        _plot_tripcolor(
            self.mock_ax,
            self.top,
            self.left,
            self.right,
            self.errors,
            vmin=-5.0,
            vmax=10.0,
        )

        call_args = self.mock_ax.tripcolor.call_args
        self.assertEqual(call_args[1]["vmin"], -5.0)
        self.assertEqual(call_args[1]["vmax"], 10.0)

    def test_tripcolor_uses_correct_colormap(self):
        """Test that the correct colormap is used."""
        _plot_tripcolor(
            self.mock_ax, self.top, self.left, self.right, self.errors, vmin=0.0, vmax=1.0
        )

        call_args = self.mock_ax.tripcolor.call_args
        self.assertEqual(call_args[1]["cmap"], "RdYlBu_r")

    def test_tripcolor_uses_gouraud_shading(self):
        """Test that gouraud shading is used."""
        _plot_tripcolor(
            self.mock_ax, self.top, self.left, self.right, self.errors, vmin=0.0, vmax=1.0
        )

        call_args = self.mock_ax.tripcolor.call_args
        self.assertEqual(call_args[1]["shading"], "gouraud")

    def test_tripcolor_rasterized(self):
        """Test that rasterized is set to True."""
        _plot_tripcolor(
            self.mock_ax, self.top, self.left, self.right, self.errors, vmin=0.0, vmax=1.0
        )

        call_args = self.mock_ax.tripcolor.call_args
        self.assertTrue(call_args[1]["rasterized"])


if __name__ == "__main__":
    unittest.main()
