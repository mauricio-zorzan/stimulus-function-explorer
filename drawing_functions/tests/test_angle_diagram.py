import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.angles import (
    generate_angle_diagram_360,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_with_expressions():
    stimulus_description = {
        "diagram": {
            "angles": [
                {
                    "measure": "1",
                    "positioning_measure": 30.0,
                    "points": ["A", "B", "C"],
                },
                {
                    "measure": "2",
                    "positioning_measure": 61.0,
                    "points": ["C", "B", "D"],
                },
                {
                    "measure": "3",
                    "positioning_measure": 89.0,
                    "points": ["D", "B", "E"],
                },
                {
                    "measure": "4",
                    "positioning_measure": 30.0,
                    "points": ["E", "B", "F"],
                },
                {
                    "measure": "5",
                    "positioning_measure": 61.0,
                    "points": ["F", "B", "G"],
                },
                {
                    "measure": "6",
                    "positioning_measure": 89.0,
                    "points": ["G", "B", "A"],
                },
            ]
        }
    }
    file_name = generate_angle_diagram_360(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_with_90_degrees_expressions():
    stimulus_description = {
        "diagram": {
            "angles": [
                {
                    "measure": "1",
                    "positioning_measure": 30.0,
                    "points": ["A", "B", "C"],
                },
                {
                    "measure": "2",
                    "positioning_measure": 60.0,
                    "points": ["C", "B", "D"],
                },
                {
                    "measure": "3",
                    "positioning_measure": 90.0,
                    "points": ["D", "B", "E"],
                },
                {
                    "measure": "4",
                    "positioning_measure": 30.0,
                    "points": ["E", "B", "F"],
                },
                {
                    "measure": "5",
                    "positioning_measure": 60.0,
                    "points": ["F", "B", "G"],
                },
                {
                    "measure": "6",
                    "positioning_measure": 90.0,
                    "points": ["G", "B", "A"],
                },
            ]
        }
    }
    file_name = generate_angle_diagram_360(stimulus_description)
    assert os.path.exists(file_name)


def test_generate_angle_diagram_360_with_90_degrees_non_standard():
    stimulus_description = {
        "diagram": {
            "angles": [
                {
                    "measure": "1",
                    "positioning_measure": 30.0,
                    "points": ["A", "B", "C"],
                },
                {
                    "measure": "2",
                    "positioning_measure": 90.0,
                    "points": ["C", "B", "D"],
                },
                {
                    "measure": "3",
                    "positioning_measure": 60.0,
                    "points": ["D", "B", "E"],
                },
                {
                    "measure": "4",
                    "positioning_measure": 30.0,
                    "points": ["E", "B", "F"],
                },
                {
                    "measure": "5",
                    "positioning_measure": 90.0,
                    "points": ["F", "B", "G"],
                },
                {
                    "measure": "6",
                    "positioning_measure": 60.0,
                    "points": ["G", "B", "A"],
                },
            ]
        }
    }
    file_name = generate_angle_diagram_360(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_with_expressions_v2():
    stimulus_description = {
        "diagram": {
            "angles": [
                {
                    "measure": "1",
                    "positioning_measure": 30.0,
                    "points": ["A", "B", "C"],
                },
                {
                    "measure": "2",
                    "positioning_measure": 61.0,
                    "points": ["C", "B", "D"],
                },
                {
                    "measure": "3",
                    "positioning_measure": 89.0,
                    "points": ["D", "B", "E"],
                },
                {
                    "measure": "4",
                    "positioning_measure": 91.0,
                    "points": ["E", "B", "G"],
                },
                {
                    "measure": "6",
                    "positioning_measure": 89.0,
                    "points": ["G", "B", "A"],
                },
            ]
        }
    }
    file_name = generate_angle_diagram_360(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_total_not_correct():
    with pytest.raises(ValidationError):
        stimulus_description = {
            "diagram": {
                "angles": [
                    {
                        "measure": "1",
                        "positioning_measure": 120.0,
                        "points": ["A", "B", "C"],
                    },
                    {
                        "measure": "2",
                        "positioning_measure": 90.0,
                        "points": ["C", "B", "D"],
                    },
                    {
                        "measure": "3",
                        "positioning_measure": 90.0,
                        "points": ["D", "B", "A"],
                    },
                ]
            }
        }
        generate_angle_diagram_360(stimulus_description)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_letters_are_not_matching():
    with pytest.raises(ValidationError):
        stimulus_description = {
            "diagram": {
                "angles": [
                    {
                        "measure": "1",
                        "positioning_measure": 120.0,
                        "points": ["A", "B", "C"],
                    },
                    {
                        "measure": "2",
                        "positioning_measure": 120.0,
                        "points": ["C", "B", "D"],
                    },
                    {
                        "measure": "3",
                        "positioning_measure": 120.0,
                        "points": ["D", "B", "E"],
                    },
                ]
            }
        }
        generate_angle_diagram_360(stimulus_description)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_with_expressions_v3():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"points": ["A", "O", "C"], "measure": 90},
                {"points": ["C", "O", "E"], "measure": 45},
                {"points": ["E", "O", "B"], "measure": 45},
                {"points": ["B", "O", "D"], "measure": 90},
                {"points": ["D", "O", "A"], "measure": 90},
            ]
        }
    }

    file_name = generate_angle_diagram_360(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_360_with_expressions_v4():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 20, "points": ["A", "B", "C"]},
                {"measure": 70, "points": ["C", "B", "D"]},
                {"measure": 25, "points": ["D", "B", "E"]},
                {"measure": 65, "points": ["E", "B", "F"]},
                {"measure": 30, "points": ["F", "B", "G"]},
                {"measure": 60, "points": ["G", "B", "H"]},
                {"measure": 40, "points": ["H", "B", "I"]},
                {"measure": 50, "points": ["I", "B", "A"]},
            ]
        }
    }

    file_name = generate_angle_diagram_360(stimulus_description)
    assert os.path.exists(file_name)
