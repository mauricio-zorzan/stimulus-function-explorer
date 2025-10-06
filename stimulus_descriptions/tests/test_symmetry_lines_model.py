import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.symmetry_lines_model import (
    Line,
    LinesOfSymmetry,
)
from pydantic import ValidationError


def test_valid_lines_of_symmetry():
    valid_data = {
        "shape_coordinates": [(0, 0), (2, 0), (2, 2), (0, 2)],
        "lines": [
            {"slope": 0, "intercept": 1.5, "label": "A"},
            {"slope": None, "intercept": 1, "label": "B"},
        ],
    }

    lines_of_symmetry = LinesOfSymmetry(**valid_data)
    assert len(lines_of_symmetry.shape_coordinates) == 4
    assert len(lines_of_symmetry.lines) == 2
    assert isinstance(lines_of_symmetry.lines[0], Line)
    assert lines_of_symmetry.lines[0].slope == 0
    assert lines_of_symmetry.lines[0].label == "A"
    assert lines_of_symmetry.lines[1].intercept == 1
    assert lines_of_symmetry.lines[1].label == "B"


def test_invalid_shape_coordinates():
    invalid_data = {
        "shape_coordinates": [(0, 0), (2, 0)],  # Only 2 coordinates
        "lines": [
            {"slope": 0, "intercept": 1, "label": "A"},
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        LinesOfSymmetry(**invalid_data)

    assert "Shape must have at least 3 vertices" in str(exc_info.value)


def test_invalid_label():
    invalid_data = {
        "shape_coordinates": [(0, 0), (2, 0), (2, 2), (0, 2)],
        "lines": [
            {"slope": 0, "intercept": 1, "label": "E"},  # Invalid label
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        LinesOfSymmetry(**invalid_data)

    assert "Input should be 'A', 'B', 'C' or 'D'" in str(exc_info.value)


def test_too_many_lines():
    invalid_data = {
        "shape_coordinates": [(0, 0), (2, 0), (2, 2), (0, 2)],
        "lines": [
            {"slope": 0, "intercept": 1, "label": "A"},
            {"slope": None, "intercept": 1, "label": "B"},
            {"slope": 1, "intercept": 0, "label": "C"},
            {"slope": -1, "intercept": 2, "label": "D"},
            {"slope": 0.5, "intercept": 1, "label": "A"},  # This makes it 5 lines
        ],
    }

    with pytest.raises(ValidationError) as exc_info:
        LinesOfSymmetry(**invalid_data)

    assert "There can be no more than 4 lines" in str(exc_info.value)
