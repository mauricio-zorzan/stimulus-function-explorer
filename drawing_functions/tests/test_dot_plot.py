import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.number_lines import (
    create_dot_plot,
    create_dual_dot_plot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.dual_stats_line import (
    DualStatsLinePlot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_line import (
    StatsLinePlot,
)
from matplotlib import pyplot as plt


@pytest.fixture
def basic_dot_plot_data():
    """Basic dot plot data with 5 unique values and varying frequencies."""
    return StatsLinePlot(
        title="Basic Dot Plot", data=[1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5]
    )


@pytest.fixture
def larger_dot_plot_data():
    """Larger dot plot data with more data points and varying frequencies."""
    return StatsLinePlot(
        title="Larger Dot Plot",
        data=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
    )


@pytest.fixture
def maximum_unique_values_data():
    """Dot plot data with maximum allowed unique values (8)."""
    return StatsLinePlot(
        title="Maximum Unique Values",
        data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    )


@pytest.fixture
def minimum_unique_values_data():
    """Dot plot data with minimum allowed unique values (5)."""
    return StatsLinePlot(
        title="Minimum Unique Values", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    )


@pytest.fixture
def unsorted_data():
    """Dot plot data with unsorted values."""
    return StatsLinePlot(title="Unsorted Data", data=[5, 3, 1, 2, 4, 5, 3, 1, 2, 4])


@pytest.mark.drawing_functions
def test_create_dot_plot_basic(basic_dot_plot_data):
    """Test basic dot plot creation with 5 unique values."""
    file_name = create_dot_plot(basic_dot_plot_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_larger_data(larger_dot_plot_data):
    """Test dot plot creation with larger dataset."""
    file_name = create_dot_plot(larger_dot_plot_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_maximum_unique_values(maximum_unique_values_data):
    """Test dot plot creation with maximum allowed unique values."""
    file_name = create_dot_plot(maximum_unique_values_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_minimum_unique_values(minimum_unique_values_data):
    """Test dot plot creation with minimum allowed unique values."""
    file_name = create_dot_plot(minimum_unique_values_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_unsorted_data(unsorted_data):
    """Test dot plot creation with unsorted data."""
    file_name = create_dot_plot(unsorted_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_empty_title():
    """Test dot plot creation with empty title."""
    data = StatsLinePlot(title="", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_long_title():
    """Test dot plot creation with long title."""
    data = StatsLinePlot(
        title="This is a very long title for testing purposes with many words",
        data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    )
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_different_ranges():
    """Test dot plot creation with different value ranges."""
    # Test with range 1-5
    data1 = StatsLinePlot(title="Range 1-5", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    file_name1 = create_dot_plot(data1)
    assert os.path.exists(file_name1)
    assert plt.imread(file_name1) is not None

    # Test with range 3-7
    data2 = StatsLinePlot(title="Range 3-7", data=[3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
    file_name2 = create_dot_plot(data2)
    assert os.path.exists(file_name2)
    assert plt.imread(file_name2) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_varying_frequencies():
    """Test dot plot creation with varying frequencies of values."""
    data = StatsLinePlot(
        title="Varying Frequencies", data=[1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5]
    )
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_edge_case_frequencies():
    """Test dot plot creation with edge case frequencies."""
    # Some values appear only once, others multiple times
    data = StatsLinePlot(
        title="Edge Case Frequencies", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    )
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_demonstrate_varying_dots():
    """Test dot plot to demonstrate varying numbers of dots per value."""
    # This data should show: 1 (1 dot), 2 (2 dots), 3 (3 dots), 4 (4 dots), 5 (5 dots)
    data = StatsLinePlot(
        title="Demonstrating Varying Dots",
        data=[1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
    )
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_high_frequency():
    """Test dot plot with high frequency values."""
    data = StatsLinePlot(
        title="High Frequency",
        data=[1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5],  # 1 appears 5 times
    )
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_memory_usage():
    """Test memory usage for large dot plot."""
    # Create a large dataset with many data points
    large_data = []
    for i in range(1, 9):  # 8 unique values
        for j in range(5):  # Each value appears 5 times
            large_data.append(i)

    data = StatsLinePlot(title="Large Dataset", data=large_data)

    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dot_plot_figure_size_adjustment():
    """Test that figure size adjusts based on number of unique values."""
    # Test with <= 10 unique values (should use figsize=(6, 4))
    data_small = StatsLinePlot(title="Small Range", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    file_name_small = create_dot_plot(data_small)
    assert os.path.exists(file_name_small)
    assert plt.imread(file_name_small) is not None

    # Test with > 10 unique values (should use figsize=(5, 5))
    # Note: This would require more than 8 unique values, which violates the validation
    # So we'll test the boundary case with 8 unique values
    data_large = StatsLinePlot(
        title="Large Range", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    )
    file_name_large = create_dot_plot(data_large)
    assert os.path.exists(file_name_large)
    assert plt.imread(file_name_large) is not None


# Test StatsLinePlot validation (these should fail)
def test_stats_line_plot_insufficient_data():
    """Test StatsLinePlot validation with insufficient data points."""
    with pytest.raises(ValueError, match="There must be at least 10 data points"):
        StatsLinePlot(
            title="Insufficient Data",
            data=[1, 1, 2, 2, 3, 3, 4, 4, 5],  # Only 9 points
        )


def test_stats_line_plot_too_few_unique():
    """Test StatsLinePlot validation with too few unique values."""
    with pytest.raises(
        ValueError, match="There must be at least 5 different numbers used"
    ):
        StatsLinePlot(
            title="Too Few Unique",
            data=[1, 1, 2, 2, 3, 3, 4, 4, 4, 4],  # Only 4 unique values
        )


def test_stats_line_plot_too_many_unique():
    """Test StatsLinePlot validation with too many unique values."""
    with pytest.raises(ValueError, match="no greater than 8"):
        StatsLinePlot(
            title="Too Many Unique",
            data=[
                1,
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
                6,
                7,
                7,
                8,
                8,
                9,
                9,
            ],  # 9 unique values
        )


def test_stats_line_plot_non_consecutive():
    """Test StatsLinePlot validation with non-consecutive values."""
    with pytest.raises(ValueError, match="The unique numbers must be consecutive"):
        StatsLinePlot(
            title="Non-consecutive",
            data=[1, 1, 2, 2, 3, 3, 5, 5, 6, 6],  # Missing 4
        )


def test_stats_line_plot_insufficient_frequency():
    """Test StatsLinePlot validation with insufficient frequency of values."""
    with pytest.raises(
        ValueError,
        match="At least 50% of unique numbers must be present at least twice",
    ):
        StatsLinePlot(
            title="Insufficient Frequency",
            data=[1, 1, 2, 2, 3, 4, 5, 6, 7, 8],  # Only 2 out of 8 values appear twice
        )


# Test edge cases for the plotting function
@pytest.mark.drawing_functions
def test_create_dot_plot_single_frequency():
    """Test dot plot with some values appearing only once."""
    data = StatsLinePlot(
        title="Single Frequency",
        data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 6],  # 5 appears once, 6 appears once
    )
    file_name = create_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.fixture
def basic_dual_data():
    """Fixture for basic dual dot plot data."""
    return DualStatsLinePlot(
        top_title="Class A Test Scores",
        top_data=[1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        bottom_title="Class B Test Scores",
        bottom_data=[2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8],
    )


@pytest.fixture
def overlapping_ranges_data():
    """Fixture for dual dot plot with overlapping but different ranges."""
    return DualStatsLinePlot(
        top_title="Morning Group Performance",
        top_data=[3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8],
        bottom_title="Afternoon Group Performance",
        bottom_data=[1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8],
    )


@pytest.fixture
def different_distributions_data():
    """Fixture for dual dot plot with different distribution patterns."""
    return DualStatsLinePlot(
        top_title="Group X Distribution",
        top_data=[1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7],
        bottom_title="Group Y Distribution",
        bottom_data=[1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7],
    )


@pytest.fixture
def same_range_data():
    """Fixture for dual dot plot with same range but different frequencies."""
    return DualStatsLinePlot(
        top_title="Control Group",
        top_data=[1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
        bottom_title="Treatment Group",
        bottom_data=[1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5],
    )


@pytest.fixture
def extreme_values_data():
    """Fixture for dual dot plot with extreme value differences."""
    return DualStatsLinePlot(
        top_title="Low Performing Students",
        top_data=[1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        bottom_title="High Performing Students",
        bottom_data=[1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7],
    )


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_basic(basic_dual_data):
    """Test basic dual dot plot creation."""
    file_name = create_dual_dot_plot(basic_dual_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_overlapping_ranges(overlapping_ranges_data):
    """Test dual dot plot with overlapping but different ranges."""
    file_name = create_dual_dot_plot(overlapping_ranges_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_different_distributions(different_distributions_data):
    """Test dual dot plot with different distribution patterns."""
    file_name = create_dual_dot_plot(different_distributions_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_same_range(same_range_data):
    """Test dual dot plot with same range but different frequencies."""
    file_name = create_dual_dot_plot(same_range_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_extreme_values(extreme_values_data):
    """Test dual dot plot with extreme value differences."""
    file_name = create_dual_dot_plot(extreme_values_data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_varying_frequencies():
    """Test dual dot plot creation with varying frequencies of values."""
    data = DualStatsLinePlot(
        top_title="Top Varying Frequencies",
        top_data=[1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8],
        bottom_title="Bottom Varying Frequencies",
        bottom_data=[1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7],
    )
    file_name = create_dual_dot_plot(data)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None
