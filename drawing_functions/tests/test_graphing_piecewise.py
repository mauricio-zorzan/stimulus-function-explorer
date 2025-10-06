import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing_piecewise import (
    generate_piecewise_graph,
)


@pytest.mark.drawing_functions
def test_basic_piecewise_function():  # CCSS.MATH.CONTENT.8.F.B.5+1 Example Stimulus
    stimulus_dict = {
        "segments": [
            {
                "start_coordinate": (-7, -4),
                "end_coordinate": (-4, 2),
                "linear": False,
            },
            {
                "start_coordinate": (-4, 2),
                "end_coordinate": (1, 2),
                "linear": True,
            },
            {
                "start_coordinate": (1, 2),
                "end_coordinate": (4, 9),
                "linear": False,
            },
            {
                "start_coordinate": (4, 9),
                "end_coordinate": (10, 1),
                "linear": False,
            },
        ]
    }
    file_name = generate_piecewise_graph(stimulus_dict)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_piecewise_function_with_labels():  # CCSS.MATH.CONTENT.8.F.B.5+2 Example Stimulus
    stimulus_dict = {
        "x_axis_label": "Time",
        "y_axis_label": "Distance",
        "segments": [
            {
                "start_coordinate": (0, 0),
                "end_coordinate": (3, 1),
                "linear": True,
            },
            {
                "start_coordinate": (3, 1),
                "end_coordinate": (5, 1),
                "linear": True,
            },
            {
                "start_coordinate": (5, 1),
                "end_coordinate": (6, 5),
                "linear": True,
            },
            {
                "start_coordinate": (6, 5),
                "end_coordinate": (8, 6),
                "linear": True,
            },
        ],
    }
    file_name = generate_piecewise_graph(stimulus_dict)
    assert os.path.exists(file_name)
