import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_line import (
    StatsLinePlot,
)


def test_valid_stats_line_plot():
    valid_data = StatsLinePlot(title="Test Data", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    assert valid_data.title == "Test Data"
    assert valid_data.data == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_insufficient_total_length():
    with pytest.raises(ValueError, match="There must be at least 10 data points"):
        StatsLinePlot(title="Short Data", data=[1, 1, 2, 2, 3, 3, 4, 4, 5])


def test_less_than_five_unique_numbers():
    with pytest.raises(
        ValueError,
        match="There must be at least 5 different numbers used, but no greater than 8",
    ):
        StatsLinePlot(title="Too Few Unique", data=[1, 1, 2, 2, 3, 3, 4, 4, 4, 4])


def test_more_than_eight_unique_numbers():
    with pytest.raises(
        ValueError,
        match="There must be at least 5 different numbers used, but no greater than 8",
    ):
        StatsLinePlot(
            title="Too Many Unique",
            data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
        )


def test_non_consecutive_numbers():
    with pytest.raises(ValueError, match="The unique numbers must be consecutive"):
        StatsLinePlot(title="Non-consecutive Data", data=[1, 1, 2, 2, 3, 3, 5, 5, 6, 6])


def test_valid_unsorted_data():
    unsorted_data = StatsLinePlot(
        title="Unsorted Data", data=[5, 3, 1, 2, 4, 5, 3, 1, 2, 4]
    )
    assert set(unsorted_data.data) == {1, 2, 3, 4, 5}
    assert len(unsorted_data.data) == 10


def test_minimum_valid_length():
    min_valid_data = StatsLinePlot(
        title="Minimum Data", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    )
    assert len(min_valid_data.data) == 10


def test_longer_valid_data():
    longer_data = StatsLinePlot(
        title="Longer Data", data=[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    )
    assert len(longer_data.data) == 15


def test_maximum_unique_numbers():
    max_unique = StatsLinePlot(
        title="Max Unique", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    )
    assert len(set(max_unique.data)) == 8


def test_most_numbers_present_twice():
    valid_data = StatsLinePlot(
        title="Most Present Twice", data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 6]
    )
    assert len(set(valid_data.data)) == 6
    assert len([n for n in set(valid_data.data) if valid_data.data.count(n) >= 2]) >= 4


def test_insufficient_numbers_present_twice():
    with pytest.raises(
        ValueError,
        match="At least 50% of unique numbers must be present at least twice",
    ):
        StatsLinePlot(
            title="Not Enough Present Twice", data=[1, 1, 2, 2, 3, 4, 5, 6, 7, 8]
        )
