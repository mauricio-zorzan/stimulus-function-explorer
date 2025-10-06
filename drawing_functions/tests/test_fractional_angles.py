import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.angles import (
    draw_fractional_angle_circle,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fractional_angle import (
    FractionalAngle,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_draw_fractional_angle_circle_basic_unit_fractions():
    """Test drawing unit fractions (numerator = 1) with common denominators."""
    test_cases = [
        (1, 2),  # 1/2 = 180°
        (1, 3),  # 1/3 = 120°
        (1, 4),  # 1/4 = 90°
        (1, 6),  # 1/6 = 60°
        (1, 8),  # 1/8 = 45°
    ]

    for numerator, denominator in test_cases:
        stimulus = FractionalAngle(
            numerator=numerator, denominator=denominator, sector_color="lightblue"
        )

        file_path = draw_fractional_angle_circle(stimulus)
        assert os.path.exists(file_path)
        assert stimulus.angle_measure == 360 / denominator
        assert stimulus.is_unit_fraction is True


@pytest.mark.drawing_functions
def test_draw_fractional_angle_circle_non_unit_fractions():
    """Test drawing non-unit fractions with various denominators."""
    test_cases = [
        (2, 3),  # 2/3 = 240°
        (3, 4),  # 3/4 = 270°
        (2, 5),  # 2/5 = 144°
        (3, 8),  # 3/8 = 135°
        (5, 6),  # 5/6 = 300°
        (4, 9),  # 4/9 = 160°
    ]

    for numerator, denominator in test_cases:
        stimulus = FractionalAngle(
            numerator=numerator, denominator=denominator, sector_color="lightgreen"
        )

        file_path = draw_fractional_angle_circle(stimulus)
        assert os.path.exists(file_path)
        assert stimulus.angle_measure == (numerator / denominator) * 360
        assert stimulus.is_unit_fraction is False


@pytest.mark.drawing_functions
def test_draw_fractional_angle_circle_all_factors_of_360():
    """Test with all single-digit factors of 360."""
    factors_of_360 = [1, 2, 3, 4, 5, 6, 8, 9]  # Single-digit factors only

    for denominator in factors_of_360:
        if denominator == 1:
            continue  # Skip 1/1 as it's not a proper fraction

        # Test with numerator = 1 for each factor
        stimulus = FractionalAngle(
            numerator=1, denominator=denominator, sector_color="orange"
        )

        file_path = draw_fractional_angle_circle(stimulus)
        assert os.path.exists(file_path)
        assert 360 % denominator == 0  # Verify it's actually a factor


@pytest.mark.drawing_functions
def test_draw_fractional_angle_circle_with_labels():
    """Test drawing with different label configurations."""
    stimulus_with_fraction = FractionalAngle(
        numerator=2, denominator=5, show_fraction_label=True, show_angle_measure=False
    )

    stimulus_with_angle = FractionalAngle(
        numerator=3, denominator=8, show_fraction_label=False, show_angle_measure=True
    )

    stimulus_with_both = FractionalAngle(
        numerator=1, denominator=6, show_fraction_label=True, show_angle_measure=True
    )

    stimulus_with_neither = FractionalAngle(
        numerator=2, denominator=9, show_fraction_label=False, show_angle_measure=False
    )

    for stimulus in [
        stimulus_with_fraction,
        stimulus_with_angle,
        stimulus_with_both,
        stimulus_with_neither,
    ]:
        file_path = draw_fractional_angle_circle(stimulus)
        assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_fractional_angle_circle_different_colors():
    """Test drawing with different sector colors."""
    colors = ["red", "blue", "green", "yellow", "purple", "pink"]

    for i, color in enumerate(colors):
        stimulus = FractionalAngle(numerator=1, denominator=4, sector_color=color)

        file_path = draw_fractional_angle_circle(stimulus)
        assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_fractional_angle_circle_angle_calculations():
    """Test that angle calculations are correct for various fractions."""
    test_cases = [
        (1, 2, 180.0),  # Half circle
        (1, 4, 90.0),  # Quarter circle
        (3, 4, 270.0),  # Three quarters
        (1, 6, 60.0),  # One sixth
        (2, 9, 80.0),  # Two ninths
        (5, 8, 225.0),  # Five eighths
    ]

    for numerator, denominator, expected_angle in test_cases:
        stimulus = FractionalAngle(numerator=numerator, denominator=denominator)

        assert stimulus.angle_measure == expected_angle

        file_path = draw_fractional_angle_circle(stimulus)
        assert os.path.exists(file_path)


# Validation tests
def test_fractional_angle_validation_single_digit_constraint():
    """Test validation for single-digit numerator and denominator."""
    # Valid single digits
    valid_stimulus = FractionalAngle(numerator=3, denominator=4)
    assert valid_stimulus.numerator == 3
    assert valid_stimulus.denominator == 4

    # Invalid numerator (double digit)
    with pytest.raises(ValueError, match="Values must be single-digit numbers"):
        FractionalAngle(numerator=10, denominator=4)

    # Invalid denominator (double digit)
    with pytest.raises(ValueError, match="Values must be single-digit numbers"):
        FractionalAngle(numerator=2, denominator=12)

    # Invalid zero values - expect ValidationError from Pydantic, not ValueError
    with pytest.raises(ValidationError):
        FractionalAngle(numerator=0, denominator=4)


def test_fractional_angle_validation_factor_of_360():
    """Test validation that denominator must be a factor of 360."""
    # Valid factors of 360
    valid_factors = [1, 2, 3, 4, 5, 6, 8, 9]
    for factor in valid_factors:
        if factor > 1:  # Skip 1 to avoid proper fraction validation
            stimulus = FractionalAngle(numerator=1, denominator=factor)
            assert stimulus.denominator == factor

    # Invalid factor (7 is not a factor of 360)
    with pytest.raises(ValueError, match="Denominator must be a factor of 360"):
        FractionalAngle(numerator=1, denominator=7)


def test_fractional_angle_validation_proper_fraction():
    """Test validation that numerator must be less than denominator."""
    # Valid proper fraction
    valid_stimulus = FractionalAngle(numerator=2, denominator=3)
    assert valid_stimulus.numerator < valid_stimulus.denominator

    # Invalid: numerator equals denominator
    with pytest.raises(ValueError, match="Numerator must be less than denominator"):
        FractionalAngle(numerator=4, denominator=4)

    # Invalid: numerator greater than denominator
    with pytest.raises(ValueError, match="Numerator must be less than denominator"):
        FractionalAngle(numerator=5, denominator=3)


def test_fractional_angle_properties():
    """Test the calculated properties of FractionalAngle."""
    stimulus = FractionalAngle(numerator=3, denominator=8)

    # Test fraction property
    assert stimulus.fraction.numerator == 3
    assert stimulus.fraction.denominator == 8

    # Test angle measure calculation
    assert stimulus.angle_measure == 135.0  # (3/8) * 360 = 135

    # Test unit fraction detection
    assert stimulus.is_unit_fraction is False

    # Test unit fraction
    unit_stimulus = FractionalAngle(numerator=1, denominator=6)
    assert unit_stimulus.is_unit_fraction is True
    assert unit_stimulus.angle_measure == 60.0


def test_fractional_angle_edge_cases():
    """Test edge cases for fractional angles."""
    # Smallest possible fraction (1/9)
    small_stimulus = FractionalAngle(numerator=1, denominator=9)
    assert small_stimulus.angle_measure == 40.0

    # Largest proper fraction with small denominator (1/2)
    large_stimulus = FractionalAngle(numerator=1, denominator=2)
    assert large_stimulus.angle_measure == 180.0

    # Near-complete circle (8/9)
    near_complete = FractionalAngle(numerator=8, denominator=9)
    assert near_complete.angle_measure == 320.0
