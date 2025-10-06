import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.line_plots import (
    generate_stacked_line_plots,
)


@pytest.mark.drawing_functions
def test_line_plot_stimulus_generator_3_plots():
    stimulus_description = [
        {
            "title": "Figure 1",
            "x_axis_label": "Number of Candies",
            "data_points": [
                {"value": "0", "frequency": 4},
                {"value": "1", "frequency": 5},
                {"value": "2", "frequency": 4},
                {"value": "3", "frequency": 1},
            ],
        },
        {
            "title": "Figure 2",
            "x_axis_label": "Number of Candies",
            "data_points": [
                {"value": "1", "frequency": 5},
                {"value": "2", "frequency": 2},
                {"value": "3", "frequency": 3},
                {"value": "4", "frequency": 5},
            ],
        },
        {
            "title": "Figure 3",
            "x_axis_label": "Number of Candies",
            "data_points": [
                {"value": "0", "frequency": 2},
                {"value": "1", "frequency": 5},
                {"value": "2", "frequency": 7},
                {"value": "3", "frequency": 1},
            ],
        },
    ]

    file_name = generate_stacked_line_plots(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_proportional_spacing_between_values():
    stimulus_description = [
        {
            "title": "Ribbon Piece Lengths",
            "x_axis_label": "Length (inches)",
            "data_points": [
                {"value": "1 1/4", "frequency": 1},
                {"value": "2", "frequency": 2},
                {"value": "2 1/2", "frequency": 1},
                {"value": "3", "frequency": 1},
            ],
        }
    ]
    file_name = generate_stacked_line_plots(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_basic_plot():
    stimulus_description = [
        {
            "title": "Number of Pets in Households",
            "x_axis_label": "Number of Pets",
            "data_points": [
                {"value": "0", "frequency": 2},
                {"value": "1", "frequency": 5},
                {"value": "2", "frequency": 8},
                {"value": "3", "frequency": 3},
                {"value": "4", "frequency": 2},
            ],
        }
    ]
    file_name = generate_stacked_line_plots(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_basic_plot_with_fractions():
    stimulus_description = [
        {
            "title": "Line Plot A",
            "x_axis_label": "Measurements in Fractions of a Unit",
            "data_points": [
                {"value": "2 1/2", "frequency": 5},
                {"value": "2 3/4", "frequency": 2},
                {"value": "3", "frequency": 2},
                {"value": "3 1/4", "frequency": 4},
                {"value": "3 1/2", "frequency": 2},
            ],
        }
    ]
    file_name = generate_stacked_line_plots(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_with_category_labels():
    stimuli_descriptions = [
        {
            "title": "Line Plot 1",
            "x_axis_label": "Number of Fruits",
            "data_points": [
                {"value": "Apples", "frequency": 4},
                {"value": "Bananas", "frequency": 6},
                {"value": "Oranges", "frequency": 2},
                {"value": "Pears", "frequency": 3},
                {"value": "Pineapples", "frequency": 1},
            ],
        },
        {
            "title": "Line Plot 2",
            "x_axis_label": "Number of Fruits",
            "data_points": [
                {"value": "Apples", "frequency": 6},
                {"value": "Bananas", "frequency": 4},
                {"value": "Oranges", "frequency": 5},
                {"value": "Pears", "frequency": 1},
                {"value": "Pineapples", "frequency": 0},
            ],
        },
        {
            "title": "Line Plot 3",
            "x_axis_label": "Number of Fruits",
            "data_points": [
                {"value": "Apples", "frequency": 1},
                {"value": "Bananas", "frequency": 1},
                {"value": "Oranges", "frequency": 1},
                {"value": "Pears", "frequency": 1},
                {"value": "Pineapples", "frequency": 1},
            ],
        },
        {
            "title": "Line Plot 4",
            "x_axis_label": "Number of Fruits",
            "data_points": [
                {"value": "Apples", "frequency": 3},
                {"value": "Bananas", "frequency": 2},
                {"value": "Oranges", "frequency": 3},
                {"value": "Pears", "frequency": 8},
                {"value": "Pineapples", "frequency": 0},
            ],
        },
    ]
    file_name = generate_stacked_line_plots(stimuli_descriptions)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_with_small_fractions():
    stimuli_descriptions = [
        {
            "title": "Weight of Apples",
            "x_axis_label": "Weight (pounds)",
            "data_points": [{"value": "1/4", "frequency": 2}],
        },
        {
            "title": "Weight of Bananas",
            "x_axis_label": "Weight (pounds)",
            "data_points": [{"value": "1/2", "frequency": 4}],
        },
        {
            "title": "Weight of Oranges",
            "x_axis_label": "Weight (pounds)",
            "data_points": [{"value": "3/4", "frequency": 3}],
        },
    ]
    file_name = generate_stacked_line_plots(stimuli_descriptions)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_with_really_small_fractions():
    stimuli_descriptions = [
        {
            "title": "Weight of Apples",
            "x_axis_label": "Weight (pounds)",
            "data_points": [{"value": "1/12", "frequency": 2}],
        },
        {
            "title": "Weight of Bananas",
            "x_axis_label": "Weight (pounds)",
            "data_points": [{"value": "1/2", "frequency": 4}],
        },
        {
            "title": "Weight of Oranges",
            "x_axis_label": "Weight (pounds)",
            "data_points": [{"value": "3/4", "frequency": 3}],
        },
    ]
    file_name = generate_stacked_line_plots(stimuli_descriptions)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_with_lots_of_really_small_fractions():
    stimulus_descriptions = [
        {
            "title": "Fractions Part One",
            "x_axis_label": "Fraction Value",
            "data_points": [
                {"value": "1/2", "frequency": 2},
                {"value": "1/3", "frequency": 1},
                {"value": "2/3", "frequency": 2},
                {"value": "1/4", "frequency": 1},
            ],
        },
        {
            "title": "Fractions Part Two",
            "x_axis_label": "Fraction Value",
            "data_points": [
                {"value": "1/5", "frequency": 2},
                {"value": "1/2", "frequency": 1},
                {"value": "2/5", "frequency": 1},
                {"value": "3/4", "frequency": 1},
            ],
        },
    ]
    file_name = generate_stacked_line_plots(stimulus_descriptions)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_with_improper_fractions():
    # Should be converted to proper fractions
    stimulus_descriptions = [
        {
            "title": "Figure 1",
            "x_axis_label": "Oranges consumed per day",
            "data_points": [
                {"value": "1/2", "frequency": 5},
                {"value": "3/4", "frequency": 1},
                {"value": "1/1", "frequency": 4},
                {"value": "3/2", "frequency": 7},
            ],
        },
        {
            "title": "Figure 2",
            "x_axis_label": "Oranges consumed per day",
            "data_points": [
                {"value": "1/2", "frequency": 1},
                {"value": "3/4", "frequency": 3},
                {"value": "1/1", "frequency": 4},
                {"value": "3/2", "frequency": 2},
            ],
        },
        {
            "title": "Figure 3",
            "x_axis_label": "Oranges consumed per day",
            "data_points": [
                {"value": "1/2", "frequency": 1},
                {"value": "3/4", "frequency": 4},
                {"value": "1/1", "frequency": 3},
                {"value": "3/2", "frequency": 2},
            ],
        },
        {
            "title": "Figure 4",
            "x_axis_label": "Oranges consumed per day",
            "data_points": [
                {"value": "1/2", "frequency": 1},
                {"value": "3/4", "frequency": 2},
                {"value": "1/1", "frequency": 4},
                {"value": "3/2", "frequency": 3},
            ],
        },
    ]
    file_name = generate_stacked_line_plots(stimulus_descriptions)
    assert os.path.exists(file_name)
