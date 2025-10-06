import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing import (
    plot_lines,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_lines import (
    Line,
    PlotLines,
)


@pytest.mark.drawing_functions
def test_plot_single_line():
    line_list = PlotLines([Line(slope=1, y_intercept=0, label="y = x")])
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_multiple_lines():
    line_list = PlotLines(
        [
            Line(slope=1, y_intercept=0, label="y = x"),
            Line(slope=-1, y_intercept=0, label="y = -x"),
            Line(slope=0, y_intercept=5, label="y = 5"),
        ]
    )
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_parallel_lines():
    line_list = PlotLines(
        [
            Line(slope=2, y_intercept=0, label="y = 2x"),
            Line(slope=2, y_intercept=3, label="y = 2x + 3"),
            Line(slope=2, y_intercept=-3, label="y = 2x - 3"),
        ]
    )
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_perpendicular_lines():
    line_list = PlotLines(
        [
            Line(slope=2, y_intercept=0, label="y = 2x"),
            Line(slope=-0.5, y_intercept=5, label="y = -0.5x + 5"),
        ]
    )
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_horizontal_and_vertical_lines():
    line_list = PlotLines(
        [
            Line(slope=0, y_intercept=3, label="y = 3"),
            Line(slope=9999999, y_intercept=-2, label="x = -2"),
        ]
    )
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_lines_with_large_slopes():
    line_list = PlotLines(
        [
            Line(slope=10, y_intercept=0, label="y = 10x"),
            Line(slope=-10, y_intercept=0, label="y = -10x"),
        ]
    )
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_lines_with_small_slopes():
    line_list = PlotLines(
        [
            Line(slope=0.1, y_intercept=0, label="y = 0.1x"),
            Line(slope=-0.1, y_intercept=0, label="y = -0.1x"),
        ]
    )
    file_name = plot_lines(line_list)
    assert os.path.exists(file_name)
