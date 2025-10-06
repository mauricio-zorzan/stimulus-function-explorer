import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing import (
    draw_stats_scatterplot,
)
from content_generators.additional_content.stimulus_image.drawing_functions.lines_of_best_fit import (
    draw_lines_of_best_fit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.lines_of_best_fit_model import (
    Line,
    LinesOfBestFit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_scatterplot import (
    StatsScatterplot,
)


@pytest.mark.drawing_functions
def test_draw_lines_of_best_fit():
    stimulus_description = LinesOfBestFit(
        lines=[
            Line(slope=0.5, y_intercept=2.5, label="Line A", best_fit=False),
            Line(slope=0.8, y_intercept=0, label="Line B", best_fit=False),
            Line(slope=0.25, y_intercept=4, label="Line C", best_fit=True),
            Line(slope=0.6, y_intercept=2, label="Line D", best_fit=False),
        ]
    )
    file_name = draw_lines_of_best_fit(stimulus_description)
    assert os.path.exists(file_name)


def test_insufficient_points_on_line():
    """Test that validation fails when fewer than 2 points are on the line of best fit"""
    # Create a dataset where the line of best fit is approximately y = 1.5x + 0.5
    # but only 1 point is exactly on the line y = 1.5x + 0.5
    invalid_data = {
        "points": [
            {"x": 1, "y": 2},  # On line y = 1.5x + 0.5
            {"x": 2, "y": 4.2},  # Off the line (should be 3.5, difference = 0.7)
            {"x": 3, "y": 5.8},  # Off the line (should be 5.0, difference = 0.8)
            {"x": 4, "y": 7.3},  # Off the line (should be 6.5, difference = 0.8)
            {"x": 5, "y": 9.2},  # Off the line (should be 8.0, difference = 1.2)
        ],
        "line_of_best_fit": {"slope": 1.5, "intercept": 0.5},
    }
    with pytest.raises(
        ValueError,
        match="At least two points must exist on the provided line of best fit",
    ):
        StatsScatterplot(**invalid_data)


def test_exactly_two_points_on_line():
    """Test that validation passes when exactly 2 points are on the line of best fit"""
    # Create a dataset where the line of best fit is approximately y = 1.5x + 0.5
    # but only 2 points are exactly on the line y = 1.5x + 0.5
    valid_data = {
        "points": [
            {"x": 1, "y": 2},  # On line y = 1.5x + 0.5
            {"x": 2, "y": 3.5},  # On line y = 1.5x + 0.5
            {"x": 3, "y": 5.8},  # Off the line (should be 5.0, difference = 0.8)
            {"x": 4, "y": 7.3},  # Off the line (should be 6.5, difference = 0.8)
            {"x": 5, "y": 9.2},  # Off the line (should be 8.0, difference = 1.2)
        ],
        "line_of_best_fit": {"slope": 1.5, "intercept": 0.5},
    }
    # This should pass validation since the line matches the calculated best fit
    # and we have exactly 2 points on the line
    scatterplot = StatsScatterplot(**valid_data)
    assert len(scatterplot.points) == 5
    assert scatterplot.line_of_best_fit.slope == 1.5
    assert scatterplot.line_of_best_fit.intercept == 0.5


def test_more_than_two_points_on_line():
    """Test that validation passes when more than 2 points are on the line of best fit"""
    # All points are exactly on the line y = 1.5x + 0.5
    valid_data = {
        "points": [
            {"x": 1, "y": 2.0},  # 1.5*1 + 0.5 = 2.0
            {"x": 2, "y": 3.5},  # 1.5*2 + 0.5 = 3.5
            {"x": 3, "y": 5.0},  # 1.5*3 + 0.5 = 5.0
            {"x": 4, "y": 6.5},  # 1.5*4 + 0.5 = 6.5
            {"x": 5, "y": 8.0},  # 1.5*5 + 0.5 = 8.0
        ],
        "line_of_best_fit": {"slope": 1.5, "intercept": 0.5},
    }
    # This should pass validation since the line matches the calculated best fit
    # and we have more than 2 points on the line
    scatterplot = StatsScatterplot(**valid_data)
    assert len(scatterplot.points) == 5
    assert scatterplot.line_of_best_fit.slope == 1.5
    assert scatterplot.line_of_best_fit.intercept == 0.5


@pytest.mark.drawing_functions
def test_draw_stats_scatterplot_basic():
    """Test basic stats scatterplot with single line of best fit"""
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
    stimulus_description = StatsScatterplot(**valid_data)
    file_name = draw_stats_scatterplot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_stats_scatterplot_negative_slope():
    """Test stats scatterplot with negative slope line of best fit"""
    valid_data = {
        "points": [
            {"x": 1, "y": 8},
            {"x": 2, "y": 6},
            {"x": 3, "y": 5},
            {"x": 4, "y": 3},
            {"x": 5, "y": 1},
        ],
        "line_of_best_fit": {"slope": -1.5, "intercept": 9.5},
    }
    stimulus_description = StatsScatterplot(**valid_data)
    file_name = draw_stats_scatterplot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_stats_scatterplot_zero_slope():
    """Test stats scatterplot with zero slope (horizontal line)"""
    valid_data = {
        "points": [
            {"x": 1, "y": 3},
            {"x": 2, "y": 3.2},
            {"x": 3, "y": 2.8},
            {"x": 4, "y": 3.1},
            {"x": 5, "y": 2.9},
        ],
        "line_of_best_fit": {"slope": 0.0, "intercept": 3.0},
    }
    stimulus_description = StatsScatterplot(**valid_data)
    file_name = draw_stats_scatterplot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_stats_scatterplot_more_points():
    """Test stats scatterplot with more data points"""
    valid_data = {
        "points": [
            {"x": 1, "y": 1.5},
            {"x": 2, "y": 3.2},
            {"x": 3, "y": 4.8},
            {"x": 4, "y": 6.1},
            {"x": 5, "y": 7.5},
            {"x": 6, "y": 9.2},
            {"x": 7, "y": 10.8},
            {"x": 8, "y": 12.1},
        ],
        "line_of_best_fit": {"slope": 1.6, "intercept": -0.1},
    }
    stimulus_description = StatsScatterplot(**valid_data)
    file_name = draw_stats_scatterplot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_stats_scatterplot_float_coordinates():
    """Test stats scatterplot with float coordinates"""
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
    stimulus_description = StatsScatterplot(**valid_data)
    file_name = draw_stats_scatterplot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_stats_scatterplot_highlighted_points():
    """Test that points on the line of best fit are highlighted with different colors"""
    # Create data where some points are exactly on the line y = 2x
    valid_data = {
        "points": [
            {"x": 1, "y": 2},  # On line y = 2x
            {"x": 2, "y": 4},  # On line y = 2x
            {"x": 3, "y": 7},  # NOT on line y = 2x (should be y=6)
            {"x": 4, "y": 8},  # On line y = 2x
            {"x": 5, "y": 11},  # NOT on line y = 2x (should be y=10)
        ],
        "line_of_best_fit": {"slope": 2, "intercept": 0},  # Line y = 2x
    }
    stimulus_description = StatsScatterplot(**valid_data)
    file_name = draw_stats_scatterplot(stimulus_description)
    assert os.path.exists(file_name)

    # Verify that the file was created successfully
    # The visual verification would need to be done manually or with image analysis tools
