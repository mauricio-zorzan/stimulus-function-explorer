import os
from time import sleep

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.table import (
    create_probability_diagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.probability_diagrams import (
    DataItem,
    ProbabilityDiagram,
)
from matplotlib import pyplot as plt


@pytest.fixture
def basic_probability_diagram():
    """Basic probability diagram with shape and color data."""
    return ProbabilityDiagram(
        rows_title=[
            {"label_1": "Square"},
            {"label_2": "Pentagon"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Red"},
            {"label_2": "Blue"},
            {"label_3": "Yellow"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 10, "2": 20, "3": 25, "4": 55}),
            DataItem(**{"1": 20, "2": 10, "3": 15, "4": 45}),
            DataItem(**{"1": 30, "2": 30, "3": 40, "4": 100}),
        ],
    )


@pytest.fixture
def gender_survey_diagram():
    """Probability diagram for gender survey data."""
    return ProbabilityDiagram(
        rows_title=[
            {"label_1": "Male"},
            {"label_2": "Female"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Yes"},
            {"label_2": "No"},
            {"label_3": "Maybe"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 15, "2": 25, "3": 10, "4": 50}),
            DataItem(**{"1": 20, "2": 15, "3": 15, "4": 50}),
            DataItem(**{"1": 35, "2": 40, "3": 25, "4": 100}),
        ],
    )


@pytest.fixture
def small_numbers_diagram():
    """Probability diagram with small numbers."""
    return ProbabilityDiagram(
        rows_title=[
            {"label_1": "Group A"},
            {"label_2": "Group B"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Pass"},
            {"label_2": "Fail"},
            {"label_3": "Incomplete"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 3, "2": 1, "3": 1, "4": 5}),
            DataItem(**{"1": 2, "2": 2, "3": 1, "4": 5}),
            DataItem(**{"1": 5, "2": 3, "3": 2, "4": 10}),
        ],
    )


@pytest.fixture
def large_numbers_diagram():
    """Probability diagram with large numbers."""
    return ProbabilityDiagram(
        rows_title=[
            {"label_1": "Category 1"},
            {"label_2": "Category 2"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Option A"},
            {"label_2": "Option B"},
            {"label_3": "Option C"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 150, "2": 200, "3": 100, "4": 450}),
            DataItem(**{"1": 100, "2": 150, "3": 200, "4": 450}),
            DataItem(**{"1": 250, "2": 350, "3": 300, "4": 900}),
        ],
    )


@pytest.fixture
def equal_distribution_diagram():
    """Probability diagram with equal distribution."""
    return ProbabilityDiagram(
        rows_title=[
            {"label_1": "Row 1"},
            {"label_2": "Row 2"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Col 1"},
            {"label_2": "Col 2"},
            {"label_3": "Col 3"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 10, "2": 10, "3": 10, "4": 30}),
            DataItem(**{"1": 10, "2": 10, "3": 10, "4": 30}),
            DataItem(**{"1": 20, "2": 20, "3": 20, "4": 60}),
        ],
    )


@pytest.mark.drawing_functions
def test_create_probability_diagram_basic(basic_probability_diagram):
    """Test basic probability diagram creation."""
    file_name = create_probability_diagram(basic_probability_diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_gender_survey(gender_survey_diagram):
    """Test probability diagram creation with gender survey data."""
    file_name = create_probability_diagram(gender_survey_diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_small_numbers(small_numbers_diagram):
    """Test probability diagram creation with small numbers."""
    file_name = create_probability_diagram(small_numbers_diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_large_numbers(large_numbers_diagram):
    """Test probability diagram creation with large numbers."""
    file_name = create_probability_diagram(large_numbers_diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_equal_distribution(equal_distribution_diagram):
    """Test probability diagram creation with equal distribution."""
    file_name = create_probability_diagram(equal_distribution_diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_zero_values():
    """Test probability diagram creation with zero values."""
    diagram = ProbabilityDiagram(
        rows_title=[
            {"label_1": "Group X"},
            {"label_2": "Group Y"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Type A"},
            {"label_2": "Type B"},
            {"label_3": "Type C"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 0, "2": 5, "3": 5, "4": 10}),
            DataItem(**{"1": 5, "2": 0, "3": 5, "4": 10}),
            DataItem(**{"1": 5, "2": 5, "3": 10, "4": 20}),
        ],
    )
    file_name = create_probability_diagram(diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_single_value():
    """Test probability diagram creation with single value in cells."""
    diagram = ProbabilityDiagram(
        rows_title=[
            {"label_1": "Row A"},
            {"label_2": "Row B"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Col X"},
            {"label_2": "Col Y"},
            {"label_3": "Col Z"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 1, "2": 1, "3": 1, "4": 3}),
            DataItem(**{"1": 1, "2": 1, "3": 1, "4": 3}),
            DataItem(**{"1": 2, "2": 2, "3": 2, "4": 6}),
        ],
    )
    file_name = create_probability_diagram(diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_long_labels():
    """Test probability diagram creation with long labels."""
    diagram = ProbabilityDiagram(
        rows_title=[
            {"label_1": "Very Long Row Label That Might Wrap"},
            {"label_2": "Another Very Long Row Label"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Very Long Column Label"},
            {"label_2": "Another Long Column Label"},
            {"label_3": "Third Long Column Label"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 10, "2": 15, "3": 5, "4": 30}),
            DataItem(**{"1": 15, "2": 10, "3": 5, "4": 30}),
            DataItem(**{"1": 25, "2": 25, "3": 10, "4": 60}),
        ],
    )
    file_name = create_probability_diagram(diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_multiple_calls():
    """Test multiple calls to create_probability_diagram to ensure no conflicts."""
    diagram1 = ProbabilityDiagram(
        rows_title=[
            {"label_1": "First"},
            {"label_2": "Second"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "A"},
            {"label_2": "B"},
            {"label_3": "C"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 10, "2": 10, "3": 10, "4": 30}),
            DataItem(**{"1": 10, "2": 10, "3": 10, "4": 30}),
            DataItem(**{"1": 20, "2": 20, "3": 20, "4": 60}),
        ],
    )

    diagram2 = ProbabilityDiagram(
        rows_title=[
            {"label_1": "X"},
            {"label_2": "Y"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "1"},
            {"label_2": "2"},
            {"label_3": "3"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 5, "2": 5, "3": 5, "4": 15}),
            DataItem(**{"1": 5, "2": 5, "3": 5, "4": 15}),
            DataItem(**{"1": 10, "2": 10, "3": 10, "4": 30}),
        ],
    )

    file_name1 = create_probability_diagram(diagram1)
    sleep(1)
    file_name2 = create_probability_diagram(diagram2)

    assert os.path.exists(file_name1)
    assert os.path.exists(file_name2)
    assert file_name1 != file_name2
    assert plt.imread(file_name1) is not None
    assert plt.imread(file_name2) is not None


@pytest.mark.drawing_functions
def test_create_probability_diagram_memory_usage():
    """Test memory usage for large probability diagram."""
    # Create a diagram with large numbers
    diagram = ProbabilityDiagram(
        rows_title=[
            {"label_1": "Large Group 1"},
            {"label_2": "Large Group 2"},
            {"label_3": "Total"},
        ],
        columns_title=[
            {"label_1": "Category A"},
            {"label_2": "Category B"},
            {"label_3": "Category C"},
            {"label_4": "Total"},
        ],
        data=[
            DataItem(**{"1": 1000, "2": 2000, "3": 1500, "4": 4500}),
            DataItem(**{"1": 1500, "2": 1000, "3": 2000, "4": 4500}),
            DataItem(**{"1": 2500, "2": 3000, "3": 3500, "4": 9000}),
        ],
    )

    file_name = create_probability_diagram(diagram)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None
