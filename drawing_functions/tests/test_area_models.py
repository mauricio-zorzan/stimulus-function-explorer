import json
import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.area_models import (
    create_area_model,
    unit_square_decomposition,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.unit_squares import (
    UnitSquareDecomposition,
)


@pytest.mark.drawing_functions
def testcreate_area_model_1():
    stimulus_description_str = '{"dimensions": {"columns": 3, "rows": 1}, "headers": { "columns": [200, 10, "S"], "rows": [15]}, "data": [[3000, "Q", 45]]}'
    stimulus_description = json.loads(stimulus_description_str)
    file_name = create_area_model(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def testcreate_area_model_2():
    stimulus_description_str = '{"dimensions": {"columns": 4, "rows": 1}, "headers": { "columns": [2000, 200, 10, "S"], "rows": [15]}, "data": [[30000, "Q", 150, 45]]}'
    stimulus_description = json.loads(stimulus_description_str)
    file_name = create_area_model(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def testcreate_area_model_3():
    stimulus_description_str = '{"dimensions": {"columns": 3, "rows": 2}, "headers": { "columns": [200, 80, 8], "rows": [10, 2]}, "data": [[2000, "T", 80], [400, 160, 16]]}'
    stimulus_description = json.loads(stimulus_description_str)
    file_name = create_area_model(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def testcreate_area_model_4():
    stimulus_description_str = '{"dimensions": {"columns": 2, "rows": 2}, "headers": { "columns": [40, 3], "rows": [20, 4]}, "data": [[800, 60], [160, 12]]}'
    stimulus_description = json.loads(stimulus_description_str)
    file_name = create_area_model(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def testcreate_area_model_5():
    stimulus_description_str = '{"dimensions": {"columns": 3, "rows": 1}, "headers": {"columns": [300, 70, 8], "rows": [6]}, "data": [[1800, 420, 48]]}'
    stimulus_description = json.loads(stimulus_description_str)
    file_name = create_area_model(stimulus_description)
    print(file_name)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_unit_square_decomposition():
    stimulus_description = UnitSquareDecomposition(size=6, filled_count=12)
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_unit_square_decomposition_edge_cases():
    # Test with minimal grid size=5, max fillable area = 9
    stimulus_description = UnitSquareDecomposition(size=5, filled_count=4)
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)

    # Test with no filled squares
    stimulus_description = UnitSquareDecomposition(size=6, filled_count=0)
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)

    # Test with maximum fillable area for size=6 (16 squares)
    stimulus_description = UnitSquareDecomposition(size=6, filled_count=16)
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_unit_square_decomposition_rectangular_tiles():
    """Test rectangular tiles functionality with fixed 1.5:1 aspect ratio."""
    # Test rectangular tiles with various grid sizes
    stimulus_description = UnitSquareDecomposition(
        size=9,
        height=8,
        filled_count=21,
        rectangle_tiles=True,
    )
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_unit_square_decomposition_rectangular_edge_cases():
    """Test edge cases for rectangular tiles."""
    # Test with minimal rectangular grid (4x4 with 2x2 fillable area = 4 max)
    stimulus_description = UnitSquareDecomposition(
        size=4,
        height=4,
        filled_count=2,
        rectangle_tiles=True,
    )
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)

    # Test with maximum fillable area for 7x5 grid ((7-2)*(5-2) = 15 squares)
    stimulus_description = UnitSquareDecomposition(
        size=7,
        height=5,
        filled_count=15,
        rectangle_tiles=True,
    )
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)

    # Test tall narrow grid
    stimulus_description = UnitSquareDecomposition(
        size=4,
        height=8,
        filled_count=4,
        rectangle_tiles=True,
    )
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)

    # Test wide short grid
    stimulus_description = UnitSquareDecomposition(
        size=10,
        height=3,
        filled_count=8,
        rectangle_tiles=True,
    )
    file_name = unit_square_decomposition(stimulus_description)
    assert os.path.exists(file_name)
