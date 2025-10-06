"""Test cases for draw_fraction_addition_model function."""

import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_addition import (
    draw_fraction_addition_model,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionAdditionModel,
    FractionBar,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_vertical_basic():
    """Test basic vertical layout with 1/3 + 3/6"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=1, denominator=3, color="lavender"),
        fraction2=FractionBar(numerator=3, denominator=6, color="lightcyan"),
        layout="vertical",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_horizontal_basic():
    """Test basic horizontal layout with 1/4 + 4/8"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=1, denominator=4, color="lavender"),
        fraction2=FractionBar(numerator=4, denominator=8, color="lightgreen"),
        layout="horizontal",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_without_plus_sign():
    """Test vertical layout without plus sign"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=2, denominator=3, color="lightblue"),
        fraction2=FractionBar(numerator=1, denominator=6, color="pink"),
        layout="vertical",
        show_plus_sign=False,
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_without_divider():
    """Test vertical layout without divider line"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=1, denominator=2, color="lavender"),
        fraction2=FractionBar(numerator=2, denominator=4, color="lightcyan"),
        layout="vertical",
        show_divider_line=False,
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_simple_case():
    """Test simple case 1/6 + 2/3"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=1, denominator=6, color="lightcyan"),
        fraction2=FractionBar(numerator=2, denominator=3, color="lavender"),
        layout="vertical",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_with_larger_numerators():
    """Test with larger numerators: 7/10 + 4/5"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=7, denominator=10, color="lightblue"),
        fraction2=FractionBar(numerator=4, denominator=5, color="lightgreen"),
        layout="vertical",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_max_denominator():
    """Test with maximum denominator (12)"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=1, denominator=12, color="lavender"),
        fraction2=FractionBar(numerator=2, denominator=6, color="lightcyan"),
        layout="vertical",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_all_allowed_denominators():
    """Test with various allowed denominators: 2,3,4,5,6,8,9,10,12"""
    # Test 3/9 + 1/3
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=3, denominator=9, color="pink"),
        fraction2=FractionBar(numerator=1, denominator=3, color="lavender"),
        layout="horizontal",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_validation_errors():
    """Test validation error handling for invalid inputs"""
    
    # Test invalid denominator (not in allowed list: 2,3,4,5,6,8,9,10,12)
    # Using denominator=11 which is within range but not in allowed list
    with pytest.raises(ValidationError, match="Denominators must be from"):
        FractionAdditionModel(
            fraction1=FractionBar(numerator=1, denominator=11, color="blue"),
            fraction2=FractionBar(numerator=1, denominator=11, color="green"),
            layout="vertical",
        )
    
    # Test denominators not multiples of each other
    with pytest.raises(ValueError, match="One denominator must be a multiple of the other"):
        FractionAdditionModel(
            fraction1=FractionBar(numerator=1, denominator=3, color="blue"),
            fraction2=FractionBar(numerator=1, denominator=5, color="green"),
            layout="vertical",
        )


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_schema_validation():
    """Test Pydantic schema validation for stimulus description"""
    
    # Test numerator exceeds max
    with pytest.raises(ValidationError, match="less than or equal to 12"):
        FractionBar(numerator=13, denominator=12, color="blue")
    
    # Test numerator below min
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        FractionBar(numerator=0, denominator=6, color="blue")
    
    # Test denominator exceeds max
    with pytest.raises(ValidationError, match="less than or equal to 12"):
        FractionBar(numerator=1, denominator=15, color="blue")
    
    # Test denominator below min
    with pytest.raises(ValidationError, match="greater than or equal to 2"):
        FractionBar(numerator=1, denominator=1, color="blue")


@pytest.mark.drawing_functions
def test_draw_fraction_addition_model_edge_case_all_shaded():
    """Test edge case where all cells are shaded (improper fractions)"""
    model_data = FractionAdditionModel(
        fraction1=FractionBar(numerator=12, denominator=12, color="lightblue"),
        fraction2=FractionBar(numerator=2, denominator=2, color="lightgreen"),
        layout="vertical",
    )
    file_name = draw_fraction_addition_model(model_data)
    assert os.path.exists(file_name)
