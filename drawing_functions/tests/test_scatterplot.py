import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing import (
    create_scatterplot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.scatter_plot import (
    Axis,
    Point,
    ScatterPlot,
)


@pytest.mark.drawing_functions
def test_create_scatterplot_basic():
    data = ScatterPlot(
        title="Basic Scatter Plot",
        x_axis=Axis(label="X-Axis", min_value=0, max_value=10),
        y_axis=Axis(label="Y-Axis", min_value=0, max_value=10),
        points=[
            Point(x=1, y=2),
            Point(x=3, y=4),
            Point(x=5, y=6),
        ],
    )
    file_name = create_scatterplot(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_scatterplot_with_edge_case():
    data = ScatterPlot(
        title="Edge Case Scatter Plot",
        x_axis=Axis(label="X-Axis", min_value=0, max_value=5),
        y_axis=Axis(label="Y-Axis", min_value=0, max_value=5),
        points=[
            Point(x=0, y=0),
            Point(x=5, y=5),
        ],
    )
    file_name = create_scatterplot(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_scatterplot_with_large_range():
    data = ScatterPlot(
        title="Large Range Scatter Plot",
        x_axis=Axis(label="X-Axis", min_value=0, max_value=100),
        y_axis=Axis(label="Y-Axis", min_value=0, max_value=100),
        points=[
            Point(x=10, y=20),
            Point(x=30, y=40),
            Point(x=50, y=60),
        ],
    )
    file_name = create_scatterplot(data)
    assert os.path.exists(file_name)
