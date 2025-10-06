import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.probability_diagrams import (
    ProbabilityDiagram,
)
from pydantic import ValidationError


@pytest.fixture
def valid_probability_diagram_data() -> dict:
    return {
        "rows_title": [
            {"label_1": "Square"},
            {"label_2": "Pentagon"},
            {"label_3": "Total"},
        ],
        "columns_title": [
            {"label_1": "Red"},
            {"label_2": "Blue"},
            {"label_3": "Yellow"},
            {"label_4": "Total"},
        ],
        "data": [
            {"1": 10, "2": 20, "3": 25, "4": 55},
            {"1": 20, "2": 10, "3": 15, "4": 45},
            {"1": 30, "2": 30, "3": 40, "4": 100},
        ],
    }


def test_valid_probability_diagram(valid_probability_diagram_data):
    diagram = ProbabilityDiagram(**valid_probability_diagram_data)
    assert isinstance(diagram, ProbabilityDiagram)


def test_invalid_row_total(valid_probability_diagram_data):
    invalid_data = valid_probability_diagram_data.copy()
    invalid_data["data"][0]["4"] = 56  # Incorrect total
    with pytest.raises(ValueError, match="Row 1 total is incorrect"):
        ProbabilityDiagram(**invalid_data)


def test_invalid_column_total(valid_probability_diagram_data):
    invalid_data = valid_probability_diagram_data.copy()
    invalid_data["data"][2]["1"] = 31  # Incorrect column total
    with pytest.raises(ValueError, match="Column 1 total is incorrect"):
        ProbabilityDiagram(**invalid_data)


def test_invalid_data_type(valid_probability_diagram_data):
    invalid_data = valid_probability_diagram_data.copy()
    invalid_data["data"][0]["1"] = "ten"  # Invalid data type
    with pytest.raises(ValidationError):
        ProbabilityDiagram(**invalid_data)


def test_invalid_row_label(valid_probability_diagram_data):
    invalid_data = valid_probability_diagram_data.copy()
    invalid_data["rows_title"][1] = {"label_4": "Invalid"}
    with pytest.raises(ValueError, match="Row 2 label is incorrect"):
        ProbabilityDiagram(**invalid_data)


def test_invalid_column_label(valid_probability_diagram_data):
    invalid_data = valid_probability_diagram_data.copy()
    invalid_data["columns_title"][2] = {"label_5": "Invalid"}
    with pytest.raises(ValueError, match="Column 3 label is incorrect"):
        ProbabilityDiagram(**invalid_data)
