import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_division_model,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    DivisionModel,
    FractionNumber,
)


@pytest.mark.drawing_functions
def test_division_model_fraction_dividend_whole_divisor():
    """Test division with fraction dividend and whole number divisor: 6/7 ÷ 3"""
    model_data = DivisionModel(
        dividend=FractionNumber(numerator=6, denominator=7), divisor=3
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_whole_dividend_fraction_divisor():
    """Test division with whole number dividend and fraction divisor: 3 ÷ 3/4"""
    model_data = DivisionModel(
        dividend=3, divisor=FractionNumber(numerator=3, denominator=4)
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_fraction_dividend_fraction_divisor():
    """Test division with both fraction dividend and divisor: 2/3 ÷ 1/4"""
    model_data = DivisionModel(
        dividend=FractionNumber(numerator=2, denominator=3),
        divisor=FractionNumber(numerator=1, denominator=4),
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_unit_fraction_dividend():
    """Test division with unit fraction dividend: 1/5 ÷ 2"""
    model_data = DivisionModel(
        dividend=FractionNumber(numerator=1, denominator=5), divisor=2
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_large_whole_dividend():
    """Test division with larger whole number dividend: 6 ÷ 2/3"""
    model_data = DivisionModel(
        dividend=6, divisor=FractionNumber(numerator=2, denominator=3)
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_improper_fraction_dividend():
    """Test division with improper fraction dividend: 5/3 ÷ 2"""
    model_data = DivisionModel(
        dividend=FractionNumber(numerator=5, denominator=3), divisor=2
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_unit_fraction_divisor():
    """Test division with unit fraction divisor: 4 ÷ 1/6"""
    model_data = DivisionModel(
        dividend=4, divisor=FractionNumber(numerator=1, denominator=6)
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_complex_fractions():
    """Test division with complex fractions: 7/8 ÷ 3/5"""
    model_data = DivisionModel(
        dividend=FractionNumber(numerator=7, denominator=8),
        divisor=FractionNumber(numerator=3, denominator=5),
    )
    file_name = draw_division_model(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_dictionary_input_fraction_dividend():
    """Test division using dictionary input with fraction dividend"""
    stimulus_description = {
        "dividend": {"numerator": 6, "denominator": 7},
        "divisor": 3,
    }
    file_name = draw_division_model(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_dictionary_input_fraction_divisor():
    """Test division using dictionary input with fraction divisor"""
    stimulus_description = {
        "dividend": 3,
        "divisor": {"numerator": 3, "denominator": 4},
    }
    file_name = draw_division_model(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_division_model_dictionary_input_both_fractions():
    """Test division using dictionary input with both operands as fractions"""
    stimulus_description = {
        "dividend": {"numerator": 2, "denominator": 3},
        "divisor": {"numerator": 1, "denominator": 4},
    }
    file_name = draw_division_model(stimulus_description)
    assert os.path.exists(file_name)


def test_division_model_validation_zero_divisor():
    """Test that zero divisor raises ValueError"""
    with pytest.raises(ValueError, match="Divisor cannot be zero"):
        DivisionModel(dividend=FractionNumber(numerator=1, denominator=2), divisor=0)


def test_division_model_validation_zero_fraction_divisor():
    """Test that zero fraction divisor raises ValidationError (caught by Pydantic at FractionNumber level)"""
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError, match="Input should be greater than or equal to 1"
    ):
        DivisionModel(
            dividend=FractionNumber(numerator=1, denominator=2),
            divisor=FractionNumber(numerator=0, denominator=5),
        )


def test_division_model_validation_whole_number_only():
    """Test that whole number only division raises ValueError"""
    with pytest.raises(ValueError, match="At least one operand must be a fraction"):
        DivisionModel(dividend=6, divisor=3)


def test_division_model_validation_large_dividend():
    """Test that large whole number dividend raises ValueError"""
    with pytest.raises(
        ValueError, match="Whole number dividend must be between 1 and 20"
    ):
        DivisionModel(dividend=25, divisor=FractionNumber(numerator=1, denominator=2))


def test_division_model_validation_large_divisor():
    """Test that large whole number divisor raises ValueError"""
    with pytest.raises(
        ValueError, match="Whole number divisor must be between 1 and 20"
    ):
        DivisionModel(dividend=FractionNumber(numerator=1, denominator=2), divisor=25)


def test_division_model_validation_zero_dividend():
    """Test that zero dividend raises ValueError"""
    with pytest.raises(
        ValueError, match="Whole number dividend must be between 1 and 20"
    ):
        DivisionModel(dividend=0, divisor=FractionNumber(numerator=1, denominator=2))


def test_division_model_validation_large_denominator():
    """Test that large denominator raises ValidationError (caught by Pydantic at FractionNumber level)"""
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError, match="Input should be less than or equal to 25"
    ):
        DivisionModel(dividend=FractionNumber(numerator=1, denominator=30), divisor=2)


def test_fraction_number_validation_zero_numerator():
    """Test that zero numerator in FractionNumber raises ValueError"""
    with pytest.raises(ValueError):
        FractionNumber(numerator=0, denominator=5)


def test_fraction_number_validation_zero_denominator():
    """Test that zero denominator in FractionNumber raises ValueError"""
    with pytest.raises(ValueError):
        FractionNumber(numerator=1, denominator=0)
