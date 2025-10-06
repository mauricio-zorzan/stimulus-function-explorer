import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_fraction_strips,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionStrips,
)


@pytest.mark.drawing_functions
def test_two_strips_basic():
    """Test 2 strips with basic division (1 whole, then 1/4 units)"""
    model_data = FractionStrips(splits=2, first_division=4)
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_small_division():
    """Test 2 strips with small division (1 whole, then 1/2 units)"""
    model_data = FractionStrips(splits=2, first_division=2)
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_large_division():
    """Test 2 strips with large division (1 whole, then 1/10 units)"""
    model_data = FractionStrips(splits=2, first_division=10)
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_three_strips_basic():
    """Test 3 strips basic (1 whole, 1/3, then 1/15 units)"""
    model_data = FractionStrips(
        splits=3,
        first_division=3,
        second_division=5,
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_three_strips_small_divisions():
    """Test 3 strips with small divisions (1 whole, 1/2, then 1/6 units)"""
    model_data = FractionStrips(
        splits=3,
        first_division=2,
        second_division=3,
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_three_strips_large_divisions():
    """Test 3 strips with larger divisions (1 whole, 1/5, then 1/20 units)"""
    model_data = FractionStrips(
        splits=3,
        first_division=5,
        second_division=4,
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_medium_division():
    """Test 2 strips with medium division (1 whole, then 3/6 units)"""
    model_data = FractionStrips(
        splits=2, first_division=6, target_numerator=3, target_denominator=6
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_odd_division():
    """Test 2 strips with odd division (1 whole, then 4/7 units)"""
    model_data = FractionStrips(
        splits=2, first_division=7, target_numerator=4, target_denominator=7
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_large_odd_division():
    """Test 2 strips with large odd division (1 whole, then 5/9 units)"""
    model_data = FractionStrips(
        splits=2, first_division=9, target_numerator=5, target_denominator=9
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_denominator_2():
    """Test 2 strips with denominator 2 (1 whole, then 1/2 units)"""
    model_data = FractionStrips(
        splits=2, first_division=2, target_numerator=1, target_denominator=2
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_strips_denominator_12():
    """Test 2 strips with denominator 12 (1 whole, then 7/12 units)"""
    model_data = FractionStrips(
        splits=2, first_division=12, target_numerator=7, target_denominator=12
    )
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_whole_number_two_grid_two_splits():
    """Test whole_number=2 with 2 splits in 2x2 grid layout"""
    model_data = FractionStrips(whole_number=2, splits=2, first_division=3)
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_whole_number_three_grid():
    """Test whole_number=3 with modular 2x2 grid layout"""
    model_data = FractionStrips(whole_number=3, splits=2, first_division=4)
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_whole_number_four_grid():
    """Test whole_number=4 with modular 2x2 grid layout (full grid)"""
    model_data = FractionStrips(whole_number=4, splits=2, first_division=10)
    file_name = draw_fraction_strips(model_data)
    assert os.path.exists(file_name)
