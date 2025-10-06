import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.line_plots import (
    generate_double_line_plot,
)


@pytest.mark.drawing_functions
def test_double_line_plot_example():
    stimulus_description = {
        "x_axis_label": "Height in inches",
        "datasets": [
            {
                "title": "Class A Heights",
                "data_points": [
                    {"value": 45, "frequency": 3},
                    {"value": 47, "frequency": 5},
                    {"value": 49, "frequency": 6},
                    {"value": 51, "frequency": 2},
                    {"value": 53, "frequency": 1},
                ],
            },
            {
                "title": "Class B Heights",
                "data_points": [
                    {"value": 46, "frequency": 4},
                    {"value": 48, "frequency": 7},
                    {"value": 50, "frequency": 3},
                    {"value": 52, "frequency": 4},
                    {"value": 54, "frequency": 2},
                ],
            },
        ],
    }
    file_name = generate_double_line_plot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_double_line_plot_long_x_axis_label():
    stimulus_description = {
        "x_axis_label": "Height in inches with some additional text",
        "datasets": [
            {
                "title": "Class A Heights",
                "data_points": [
                    {"value": 45, "frequency": 3},
                    {"value": 47, "frequency": 5},
                    {"value": 49, "frequency": 6},
                    {"value": 51, "frequency": 2},
                    {"value": 53, "frequency": 1},
                ],
            },
            {
                "title": "Class B Heights",
                "data_points": [
                    {"value": 46, "frequency": 4},
                    {"value": 48, "frequency": 7},
                    {"value": 50, "frequency": 3},
                    {"value": 52, "frequency": 4},
                    {"value": 54, "frequency": 2},
                ],
            },
        ],
    }
    file_name = generate_double_line_plot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_double_line_plot_value_spacing_double_digits():
    stimulus_description = {
        "x_axis_label": "Height in inches",
        "datasets": [
            {
                "title": "Class A Heights",
                "data_points": [
                    {"value": 45, "frequency": 3},
                    {"value": 47, "frequency": 5},
                    {"value": 49, "frequency": 6},
                    {"value": 51, "frequency": 2},
                    {"value": 53, "frequency": 1},
                    {"value": 55, "frequency": 4},
                    {"value": 57, "frequency": 3},
                ],
            },
            {
                "title": "Class B Heights",
                "data_points": [
                    {"value": 46, "frequency": 4},
                    {"value": 48, "frequency": 7},
                    {"value": 50, "frequency": 3},
                    {"value": 52, "frequency": 4},
                    {"value": 54, "frequency": 2},
                    {"value": 56, "frequency": 6},
                ],
            },
        ],
    }
    file_name = generate_double_line_plot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_double_line_plot_value_spacing_triple_digits():
    stimulus_description = {
        "x_axis_label": "Height in inches",
        "datasets": [
            {
                "title": "Class A Heights",
                "data_points": [
                    {"value": 145, "frequency": 3},
                    {"value": 147, "frequency": 5},
                    {"value": 149, "frequency": 6},
                    {"value": 151, "frequency": 2},
                    {"value": 153, "frequency": 1},
                    {"value": 155, "frequency": 4},
                ],
            },
            {
                "title": "Class B Heights",
                "data_points": [
                    {"value": 146, "frequency": 4},
                    {"value": 148, "frequency": 7},
                    {"value": 150, "frequency": 3},
                    {"value": 152, "frequency": 4},
                    {"value": 154, "frequency": 2},
                    {"value": 156, "frequency": 6},
                ],
            },
        ],
    }
    file_name = generate_double_line_plot(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_double_line_plot_sparse_data():
    stimulus_description = {
        "x_axis_label": "Scores",
        "datasets": [
            {
                "title": "Section A",
                "data_points": [
                    {"value": 31, "frequency": 1},
                    {"value": 32, "frequency": 1},
                    {"value": 33, "frequency": 1},
                    {"value": 42, "frequency": 1},
                    {"value": 47, "frequency": 1},
                    {"value": 49, "frequency": 1},
                ],
            },
            {
                "title": "Section B",
                "data_points": [
                    {"value": 31, "frequency": 1},
                    {"value": 38, "frequency": 1},
                    {"value": 40, "frequency": 1},
                    {"value": 42, "frequency": 1},
                    {"value": 43, "frequency": 1},
                    {"value": 46, "frequency": 1},
                ],
            },
        ],
    }
    with pytest.raises(ValueError) as excinfo:
        generate_double_line_plot(stimulus_description)
    assert (
        "The range of the x-axis is greater than 12 with double or triple digit values."
        in str(excinfo.value)
    )
