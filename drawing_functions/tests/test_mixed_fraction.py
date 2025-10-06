import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_mixed_fractional_models,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionShape,
    MixedFraction,
    MixedFractionList,
)


@pytest.mark.drawing_functions
def test_mixed_fraction_proper_fractions():
    """Test proper fractions (less than 1)"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="3/4"),
            MixedFraction(shape=FractionShape.CIRCLE, fraction="1/2"),
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_improper_fractions():
    """Test improper fractions (greater than 1)"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="5/4"),  # 1 1/4
            MixedFraction(shape=FractionShape.CIRCLE, fraction="7/3"),  # 2 1/3
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_mixed_numbers():
    """Test mixed numbers with whole numbers"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="8/4"),  # 2
            MixedFraction(shape=FractionShape.CIRCLE, fraction="10/3"),  # 3 1/3
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="11/5"),  # 2 1/5
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_single_fraction():
    """Test single fraction"""
    model_data = MixedFractionList(
        [MixedFraction(shape=FractionShape.RECTANGLE, fraction="3/2")]  # 1 1/2
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_maximum_fractions():
    """Test maximum 4 fractions"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="1/2"),
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="3/2"),
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="5/2"),
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="7/2"),
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_equal_to_one():
    """Test fractions equal to 1"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="4/4"),  # 1
            MixedFraction(shape=FractionShape.CIRCLE, fraction="3/3"),  # 1
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_all_rectangles():
    """Test all rectangles with same shape"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="1/3"),
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="4/3"),
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="7/3"),
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_all_circles():
    """Test all circles with same shape"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.CIRCLE, fraction="2/5"),
            MixedFraction(shape=FractionShape.CIRCLE, fraction="7/5"),
            MixedFraction(shape=FractionShape.CIRCLE, fraction="12/5"),
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_fraction_edge_case_zero():
    """Test edge case with fraction close to zero"""
    model_data = MixedFractionList(
        [
            MixedFraction(shape=FractionShape.RECTANGLE, fraction="1/10"),
            MixedFraction(shape=FractionShape.CIRCLE, fraction="1/8"),
        ]
    )
    file_name = draw_mixed_fractional_models(model_data)
    assert os.path.exists(file_name)
