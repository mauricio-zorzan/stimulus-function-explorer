import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_diagram import (
    StatsBarDiagram,
)
from pydantic import ValidationError


def test_valid_stats_bar_diagram():
    valid_data = {
        "x_axis_label": "Categories",
        "y_axis_label": "Frequency",
        "x_axis_data": [1, 2, 3, 4, 5],
        "y_axis_data": [10, 20, 15, 25, 30],
    }
    diagram = StatsBarDiagram(**valid_data)
    assert diagram.x_axis_label == "Categories"
    assert diagram.y_axis_label == "Frequency"
    assert diagram.x_axis_data == [1, 2, 3, 4, 5]
    assert diagram.y_axis_data == [10, 20, 15, 25, 30]


def test_invalid_non_integer_data():
    invalid_data = {
        "x_axis_label": "Months",
        "y_axis_label": "Frequency",
        "x_axis_data": ["Jan", "feb", "mar", "apr", "may"],
        "y_axis_data": [10, 20, 15.5, 25, 30],  # 15.5 is not an integer
    }
    with pytest.raises(ValueError, match="All values must be integers"):
        StatsBarDiagram(**invalid_data)


def test_invalid_x_axis_data_not_sequential():
    invalid_data = {
        "x_axis_label": "Categories",
        "y_axis_label": "Frequency",
        "x_axis_data": [1, 3, 2, 4, 5],
        "y_axis_data": [10, 20, 15, 25, 30],
    }
    with pytest.raises(ValueError, match="x_axis_data must be in sequential order"):
        StatsBarDiagram(**invalid_data)


def test_invalid_data_length_mismatch():
    invalid_data = {
        "x_axis_label": "Categories",
        "y_axis_label": "Frequency",
        "x_axis_data": [1, 2, 3, 4, 5],  # Changed to meet min_length requirement
        "y_axis_data": [10, 20, 15, 25, 30, 35],  # Added one more item
    }
    with pytest.raises(
        ValueError, match="x_axis_data and y_axis_data must have the same length"
    ):
        StatsBarDiagram(**invalid_data)


def test_invalid_x_axis_data_length():
    invalid_data = {
        "x_axis_label": "Categories",
        "y_axis_label": "Frequency",
        "x_axis_data": [1, 2, 3],
        "y_axis_data": [10, 20, 15],
    }
    with pytest.raises(ValidationError):
        StatsBarDiagram(**invalid_data)


def test_invalid_y_axis_data_length():
    invalid_data = {
        "x_axis_label": "Categories",
        "y_axis_label": "Frequency",
        "x_axis_data": [1, 2, 3, 4, 5],
        "y_axis_data": [10, 20, 15],
    }
    with pytest.raises(ValidationError):
        StatsBarDiagram(**invalid_data)
