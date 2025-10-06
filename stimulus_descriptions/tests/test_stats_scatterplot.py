import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_scatterplot import (
    StatsScatterplot,
)


def test_valid_stats_scatterplot():
    valid_data = {
        "points": [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 5},
            {"x": 4, "y": 7},
            {"x": 5, "y": 8},
        ],
        "line_of_best_fit": {"slope": 1.5, "intercept": 0.5},
    }
    scatterplot = StatsScatterplot(**valid_data)
    assert len(scatterplot.points) == 5
    assert scatterplot.line_of_best_fit.slope == 1.5
    assert scatterplot.line_of_best_fit.intercept == 0.5


def test_invalid_line_of_best_fit():
    invalid_data = {
        "points": [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 6},
            {"x": 4, "y": 8},
            {"x": 5, "y": 10},
        ],
        "line_of_best_fit": {"slope": 1, "intercept": 0},
    }
    with pytest.raises(
        ValueError,
        match="The provided line of best fit does not match the actual line of best fit",
    ):
        StatsScatterplot(**invalid_data)


def test_point_range():
    invalid_data = {
        "points": [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 6},
            {"x": 4, "y": 8},
            {"x": 16, "y": 10},
        ],
        "line_of_best_fit": {"slope": 2, "intercept": 0},
    }
    with pytest.raises(
        ValueError, match="All point coordinates must be within the range of -15 to 15"
    ):
        StatsScatterplot(**invalid_data)


def test_non_unique_points():
    invalid_data = {
        "points": [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 6},
            {"x": 3, "y": 6},
            {"x": 5, "y": 10},
        ],
        "line_of_best_fit": {"slope": 2, "intercept": 0},
    }
    with pytest.raises(ValueError, match="All points must be unique"):
        StatsScatterplot(**invalid_data)


def test_integer_coordinates():
    valid_data = {
        "points": [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 5},
            {"x": 4, "y": 7},
            {"x": 5, "y": 8},
        ],
        "line_of_best_fit": {"slope": 1.5, "intercept": 0.5},
    }
    scatterplot = StatsScatterplot(**valid_data)
    assert all(
        isinstance(point.x, float) and point.x.is_integer()
        for point in scatterplot.points
    )
    assert all(
        isinstance(point.y, float) and point.y.is_integer()
        for point in scatterplot.points
    )


def test_float_coordinates():
    valid_data = {
        "points": [
            {"x": 1.5, "y": 2.5},
            {"x": 2.5, "y": 4.5},
            {"x": 3.5, "y": 5.5},
            {"x": 4.5, "y": 7.5},
            {"x": 5.5, "y": 8.5},
        ],
        "line_of_best_fit": {"slope": 1.5, "intercept": 0.75},
    }
    scatterplot = StatsScatterplot(**valid_data)
    assert all(isinstance(point.x, float) for point in scatterplot.points)
    assert all(isinstance(point.y, float) for point in scatterplot.points)
