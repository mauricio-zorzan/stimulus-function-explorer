import os

import pytest

# Imports from the newly created line graph files
from content_generators.additional_content.stimulus_image.drawing_functions.line_graph import (
    create_line_graph,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.line_graph_description import (
    LineDataPoint,
    LineGraph,
    LineGraphList,
    LineGraphSeries,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_create_single_line_graph():
    """
    Tests the creation of a basic line graph with a single data series.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Monthly Temperature",
                x_axis_label="Month Index",
                y_axis_label="Temperature (°C)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=5),
                            LineDataPoint(x=2, y=8),
                            LineDataPoint(x=3, y=12),
                            LineDataPoint(x=4, y=15),
                        ],
                        label="2024",
                        color="green",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_2():
    """
    Tests the creation of a graph with two distinct lines (multi-series).
    This confirms the loop in the drawing function works as expected.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Product Sales Comparison",
                x_axis_label="Quarter",
                y_axis_label="Units Sold (in thousands)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=20),
                            LineDataPoint(x=2, y=25),
                            LineDataPoint(x=3, y=35),
                            LineDataPoint(x=4, y=30),
                        ],
                        label="Product A",
                        color="blue",
                    ),
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=15),
                            LineDataPoint(x=2, y=30),
                            LineDataPoint(x=3, y=28),
                            LineDataPoint(x=4, y=40),
                        ],
                        label="Product B",
                        color="red",
                    ),
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_minimal_config():
    """
    Tests graph creation with optional parameters (label, color) omitted
    to ensure the function falls back to default values gracefully.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Data with Defaults",
                x_axis_label="Time",
                y_axis_label="Value",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=0, y=10),
                            LineDataPoint(x=5, y=15),
                            LineDataPoint(x=10, y=5),
                        ],
                        # No label or color provided
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_with_negative_values():
    """
    Tests an edge case where data points include negative coordinates,
    ensuring the plot adapts and displays them correctly.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Profit and Loss",
                x_axis_label="Fiscal Month",
                y_axis_label="Profit ($)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=100),
                            LineDataPoint(x=2, y=50),
                            LineDataPoint(x=3, y=-25),  # Negative value
                            LineDataPoint(x=4, y=-75),  # Negative value
                            LineDataPoint(x=5, y=120),
                        ],
                        label="Net Profit",
                        color="#FF5733",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_invalid_data_structure():
    """
    Tests that Pydantic validation catches invalid data structures before
    the drawing function is called. Here, we violate the 'min_length=1'
    constraint for data_series.
    """
    with pytest.raises(ValidationError):
        LineGraphList(
            [
                LineGraph(
                    title="Empty Graph",
                    x_axis_label="X",
                    y_axis_label="Y",
                    data_series=[],  # Invalid: must contain at least one series
                )
            ]
        )


@pytest.mark.drawing_functions
def test_line_graph_single_data_point():
    """
    Edge Case 1: Tests if the graph can be created when a series contains only a single data point.
    Matplotlib should render this as a single marker.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Single Event",
                x_axis_label="Time",
                y_axis_label="Measurement",
                data_series=[
                    LineGraphSeries(
                        data_points=[LineDataPoint(x=10, y=25)],
                        label="Event A",
                        color="purple",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_unordered_x_values():
    """
    Edge Case 2: Tests behavior when x-values are not sequential.
    The line should connect the points in the order they are provided, potentially crossing over itself.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Erratic Measurement Data",
                x_axis_label="Sample ID",
                y_axis_label="Value",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=10),
                            LineDataPoint(x=5, y=30),  # Jumps forward
                            LineDataPoint(x=3, y=15),  # Goes backward
                            LineDataPoint(x=4, y=25),  # Forward again
                        ],
                        label="Unordered Samples",
                        color="orange",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_horizontal_line():
    """
    Edge Case 3: Tests a series where all y-values are identical, resulting in a perfectly horizontal line.
    This checks if axis auto-scaling handles a zero-height data range.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Constant Speed",
                x_axis_label="Time (s)",
                y_axis_label="Speed (m/s)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=0, y=50),
                            LineDataPoint(x=10, y=50),
                            LineDataPoint(x=20, y=50),
                            LineDataPoint(x=30, y=50),
                        ],
                        label="Cruise Control",
                        color="cyan",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_mismatched_data_lengths():
    """
    Edge Case 4: Tests plotting two lines where one has significantly more data points than the other.
    This ensures the plot renders correctly with different data densities.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Sensor Data Comparison",
                x_axis_label="Time",
                y_axis_label="Reading",
                data_series=[
                    LineGraphSeries(
                        data_points=[  # High-frequency data
                            LineDataPoint(x=i, y=i % 10) for i in range(20)
                        ],
                        label="Sensor A (High-Res)",
                        color="magenta",
                    ),
                    LineGraphSeries(
                        data_points=[  # Low-frequency data
                            LineDataPoint(x=0, y=5),
                            LineDataPoint(x=10, y=15),
                            LineDataPoint(x=20, y=0),
                        ],
                        label="Sensor B (Low-Res)",
                        color="black",
                    ),
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_invalid_color_name():
    """
    Edge Case 5: Tests the error handling when an invalid color name is provided.
    This should raise a ValueError from the underlying Matplotlib library.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Invalid Config Test",
                x_axis_label="X",
                y_axis_label="Y",
                data_series=[
                    LineGraphSeries(
                        data_points=[LineDataPoint(x=1, y=1)],
                        label="Bad Color",
                        color="not_a_real_color",  # Invalid color string
                    )
                ],
            )
        ]
    )
    # Expect an error because the color name is invalid
    with pytest.raises(ValueError):
        create_line_graph(data)


@pytest.mark.drawing_functions
def test_line_graph_very_large_numbers():
    """
    Edge Case 6: Tests graph creation with very large numbers to ensure
    matplotlib handles scientific notation and scaling properly.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Scientific Data",
                x_axis_label="Time (milliseconds)",
                y_axis_label="Value (millions)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1000000, y=2500000),
                            LineDataPoint(x=2000000, y=3500000),
                            LineDataPoint(x=3000000, y=1500000),
                            LineDataPoint(x=4000000, y=4500000),
                        ],
                        label="Large Scale Data",
                        color="navy",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_very_small_numbers():
    """
    Edge Case 7: Tests graph creation with very small decimal numbers
    to ensure matplotlib handles precision and scaling correctly.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Microscopic Measurements",
                x_axis_label="Distance (micrometers)",
                y_axis_label="Intensity (arbitrary units)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=0.0001, y=0.0005),
                            LineDataPoint(x=0.0002, y=0.0008),
                            LineDataPoint(x=0.0003, y=0.0003),
                            LineDataPoint(x=0.0004, y=0.0009),
                        ],
                        label="Micro Data",
                        color="darkgreen",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_identical_data_points():
    """
    Edge Case 8: Tests behavior when multiple data points have identical coordinates.
    This should render as overlapping markers on the same position.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Overlapping Measurements",
                x_axis_label="Sample",
                y_axis_label="Value",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=10),
                            LineDataPoint(x=2, y=10),  # Same y as first point
                            LineDataPoint(x=2, y=10),  # Identical to previous
                            LineDataPoint(x=3, y=15),
                            LineDataPoint(x=3, y=15),  # Identical to previous
                        ],
                        label="Overlapping Data",
                        color="crimson",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_extreme_axis_range():
    """
    Edge Case 9: Tests graph with extreme differences between x and y axis ranges.
    One axis has very small values while the other has very large values.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="Extreme Range Data",
                x_axis_label="Time (seconds)",
                y_axis_label="Population (billions)",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=0.1, y=1000000000),
                            LineDataPoint(x=0.2, y=2000000000),
                            LineDataPoint(x=0.3, y=1500000000),
                            LineDataPoint(x=0.4, y=3000000000),
                        ],
                        label="Population Growth",
                        color="darkorange",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_line_graph_very_long_title_and_labels():
    """
    Edge Case 10: Tests graph creation with extremely long title and axis labels
    to ensure proper text wrapping and layout handling.
    """
    data = LineGraphList(
        [
            LineGraph(
                title="This is an extremely long title that should test the layout and text wrapping capabilities of the matplotlib rendering system when dealing with very long text strings",
                x_axis_label="This is a very long x-axis label that contains multiple words and should test the text rendering and layout capabilities of the graph system",
                y_axis_label="This is an equally long y-axis label that should also test the text rendering and layout capabilities when dealing with very long descriptive text",
                data_series=[
                    LineGraphSeries(
                        data_points=[
                            LineDataPoint(x=1, y=5),
                            LineDataPoint(x=2, y=8),
                            LineDataPoint(x=3, y=12),
                            LineDataPoint(x=4, y=15),
                        ],
                        label="This is a very long series label that should test the legend text rendering capabilities",
                        color="purple",
                    )
                ],
            )
        ]
    )
    file_name = create_line_graph(data)
    assert os.path.exists(file_name)
