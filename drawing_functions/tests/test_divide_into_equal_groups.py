import os
import time

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.divide_into_equal_groups import (
    draw_divide_into_equal_groups,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.divide_into_equal_groups import (
    DivideIntoEqualGroups,
)


@pytest.mark.drawing_functions
def test_generate_divide_into_equal_groups_basic():
    """Test five dots per group."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=5, number_of_groups=2)

    time.sleep(1)
    file_path = draw_divide_into_equal_groups(stimulus)

    # Verify file was created and exists
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_divide_into_equal_groups_various_cases():
    """Test four groups."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=3, number_of_groups=4)

    time.sleep(1)
    file_path = draw_divide_into_equal_groups(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_four_dots_per_group():
    """Test four dots per group case."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=4, number_of_groups=3)

    time.sleep(1)
    file_path = draw_divide_into_equal_groups(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_single_group():
    """Test single group case."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=1, number_of_groups=1)

    time.sleep(1)
    file_path = draw_divide_into_equal_groups(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_many_small_groups():
    """Test five groups."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=2, number_of_groups=5)

    time.sleep(1)
    file_path = draw_divide_into_equal_groups(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_ten_groups_seven_dots():
    """Test 10 groups with 7 dots per group - generates and saves image file."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=7, number_of_groups=10)
    
    file_path = draw_divide_into_equal_groups(stimulus)
    
    assert os.path.exists(file_path)
    print(f"Generated 10 groups, 7 dots per group image: {file_path}")


@pytest.mark.drawing_functions
def test_generate_seven_groups_ten_dots():
    """Test 7 groups with 10 dots per group - generates and saves image file."""
    stimulus = DivideIntoEqualGroups(number_of_dots_per_group=10, number_of_groups=7)
    
    file_path = draw_divide_into_equal_groups(stimulus)
    
    assert os.path.exists(file_path)
    print(f"Generated 7 groups, 10 dots per group image: {file_path}")
