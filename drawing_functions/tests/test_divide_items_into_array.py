import os
import time

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.divide_items_into_array import (
    draw_divide_items_into_array,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.divide_items_into_array import (
    DivideItemsIntoArray,
)


@pytest.mark.drawing_functions
def test_generate_divide_items_into_array_basic():
    """Test 2 rows, 3 columns array."""
    stimulus = DivideItemsIntoArray(num_rows=2, num_columns=3)

    time.sleep(1)
    file_path = draw_divide_items_into_array(stimulus)

    # Verify file was created and exists
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_divide_items_into_array_square():
    """Test 3x3 square array."""
    stimulus = DivideItemsIntoArray(num_rows=3, num_columns=3)

    time.sleep(1)
    file_path = draw_divide_items_into_array(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_single_row():
    """Test single row case."""
    stimulus = DivideItemsIntoArray(num_rows=1, num_columns=5)

    time.sleep(1)
    file_path = draw_divide_items_into_array(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_single_column():
    """Test single column case."""
    stimulus = DivideItemsIntoArray(num_rows=4, num_columns=1)

    time.sleep(1)
    file_path = draw_divide_items_into_array(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_rectangular_array():
    """Test rectangular array."""
    stimulus = DivideItemsIntoArray(num_rows=4, num_columns=6)

    time.sleep(1)
    file_path = draw_divide_items_into_array(stimulus)

    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_tall_array():
    """Test 5 rows, 2 columns - generates and saves image file."""
    stimulus = DivideItemsIntoArray(num_rows=5, num_columns=2)
    
    file_path = draw_divide_items_into_array(stimulus)
    
    assert os.path.exists(file_path)
    print(f"Generated 5 rows, 2 columns array image: {file_path}")


@pytest.mark.drawing_functions
def test_generate_wide_array():
    """Test 2 rows, 8 columns - generates and saves image file."""
    stimulus = DivideItemsIntoArray(num_rows=2, num_columns=8)
    
    file_path = draw_divide_items_into_array(stimulus)
    
    assert os.path.exists(file_path)
    print(f"Generated 2 rows, 8 columns array image: {file_path}")


@pytest.mark.drawing_functions
def test_generate_maximum_array():
    """Test maximum size array 10x10."""
    stimulus = DivideItemsIntoArray(num_rows=10, num_columns=10)
    
    file_path = draw_divide_items_into_array(stimulus)
    
    assert os.path.exists(file_path)
    print(f"Generated 10x10 array image: {file_path}") 