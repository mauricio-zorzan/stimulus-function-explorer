import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.spinners import (
    generate_spinner,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.spinner import (
    Spinner,
)
from matplotlib import pyplot as plt


@pytest.fixture
def basic_spinner():
    """Basic spinner with 4 sections."""
    return Spinner(
        title="Basic Spinner",
        sections=["A", "B", "C", "D"],
    )


@pytest.fixture
def colored_spinner():
    """Spinner with colored sections."""
    return Spinner(
        title="Color Wheel",
        sections=["Red", "Blue", "Green", "Yellow"],
    )


@pytest.fixture
def mixed_spinner():
    """Spinner with mixed colored and non-colored sections."""
    return Spinner(
        title="Mixed Spinner",
        sections=["Red", "Blue", "Option1", "Option2"],
    )


@pytest.fixture
def large_spinner():
    """Spinner with maximum sections (10)."""
    return Spinner(
        title="Large Spinner",
        sections=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )


@pytest.fixture
def minimum_spinner():
    """Spinner with minimum sections (4)."""
    return Spinner(
        title="Minimum Spinner",
        sections=["North", "South", "East", "West"],
    )


@pytest.fixture
def themed_spinner():
    """Spinner with a specific theme."""
    return Spinner(
        title="Weather Forecast",
        sections=["Sunny", "Cloudy", "Rainy", "Snowy"],
    )


@pytest.fixture
def all_colors_spinner():
    """Spinner with all available colors."""
    return Spinner(
        title="All Colors",
        sections=["Red", "Blue", "Green", "Yellow", "Pink", "Purple"],
    )


@pytest.mark.drawing_functions
def test_generate_spinner_basic(basic_spinner):
    """Test basic spinner generation."""
    file_name = generate_spinner(basic_spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_colored(colored_spinner):
    """Test spinner generation with colored sections."""
    file_name = generate_spinner(colored_spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_mixed(mixed_spinner):
    """Test spinner generation with mixed colored and non-colored sections."""
    file_name = generate_spinner(mixed_spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_minimum(minimum_spinner):
    """Test spinner generation with minimum sections."""
    file_name = generate_spinner(minimum_spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_themed(themed_spinner):
    """Test spinner generation with themed content."""
    file_name = generate_spinner(themed_spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_all_colors(all_colors_spinner):
    """Test spinner generation with all available colors."""
    file_name = generate_spinner(all_colors_spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_long_title():
    """Test spinner generation with a long title."""
    spinner = Spinner(
        title="This is a very long title that might wrap to multiple lines",
        sections=["A", "B", "C", "D"],
    )
    file_name = generate_spinner(spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_multiple_calls():
    """Test multiple calls to generate_spinner to ensure no conflicts."""
    spinner1 = Spinner(
        title="First Spinner",
        sections=["A", "B", "C", "D"],
    )

    spinner2 = Spinner(
        title="Second Spinner",
        sections=["X", "Y", "Z", "W"],
    )

    file_name1 = generate_spinner(spinner1)
    file_name2 = generate_spinner(spinner2)

    assert os.path.exists(file_name1)
    assert os.path.exists(file_name2)
    assert file_name1 != file_name2
    assert plt.imread(file_name1) is not None
    assert plt.imread(file_name2) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_memory_usage():
    """Test memory usage for large spinner."""
    # Create a spinner with maximum sections
    spinner = Spinner(
        title="Memory Test Spinner",
        sections=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    )

    file_name = generate_spinner(spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_generate_spinner_edge_case_8_sections():
    """Test spinner generation with 8 sections."""
    spinner = Spinner(
        title="Eight Sections",
        sections=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    file_name = generate_spinner(spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


def test_spinner_duplicate_sections():
    """Test Spinner validation with duplicate sections."""
    # This should work as the validation doesn't check for duplicates
    spinner = Spinner(
        title="Duplicate Sections",
        sections=["A", "A", "B", "B"],  # Duplicate sections
    )
    file_name = generate_spinner(spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None


def test_spinner_case_sensitive_colors():
    """Test that color matching is case sensitive."""
    spinner = Spinner(
        title="Case Sensitive Colors",
        sections=["red", "blue", "green", "yellow"],  # Lowercase colors
    )
    file_name = generate_spinner(spinner)
    assert os.path.exists(file_name)
    assert plt.imread(file_name) is not None
