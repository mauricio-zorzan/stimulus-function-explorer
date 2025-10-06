import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_fractional_models,
    draw_fractional_models_labeled,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    Fraction,
    FractionList,
    FractionShape,
)


@pytest.mark.drawing_functions
def test_draw_fractional_models_basic():
    """Test basic functionality with one rectangle and one circle."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="3/4"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_triangle_halves():
    """Test triangle divided into halves."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.TRIANGLE, fraction="1/2"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_triangle_thirds():
    """Test triangle divided into thirds."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.TRIANGLE, fraction="2/3"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_triangle_sixths():
    """Test triangle divided into sixths."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.TRIANGLE, fraction="5/6"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_single_shape():
    """Test with a single rectangle shape."""
    model_data = FractionList([Fraction(shape=FractionShape.RECTANGLE, fraction="1/2")])
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_single_circle():
    """Test with a single circle shape."""
    model_data = FractionList([Fraction(shape=FractionShape.CIRCLE, fraction="2/3")])
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_multiple_rectangles():
    """Test with multiple rectangle shapes to verify different division types."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="1/4"),
            Fraction(shape=FractionShape.RECTANGLE, fraction="3/5"),
            Fraction(shape=FractionShape.RECTANGLE, fraction="2/3"),
            Fraction(shape=FractionShape.RECTANGLE, fraction="4/7"),
            Fraction(shape=FractionShape.RECTANGLE, fraction="5/6"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_grid_division_candidates():
    """Test rectangles with denominators that can support grid division."""
    model_data = FractionList(
        [
            Fraction(
                shape=FractionShape.RECTANGLE, fraction="3/4"
            ),  # 2x2 grid possible
            Fraction(
                shape=FractionShape.RECTANGLE, fraction="2/6"
            ),  # 2x3 or 3x2 grid possible
            Fraction(
                shape=FractionShape.RECTANGLE, fraction="5/8"
            ),  # 2x4 or 4x2 grid possible
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_prime_denominators():
    """Test rectangles with prime denominators (only vertical/horizontal division)."""
    model_data = FractionList(
        [
            Fraction(
                shape=FractionShape.RECTANGLE, fraction="1/5"
            ),  # Prime denominator
            Fraction(
                shape=FractionShape.RECTANGLE, fraction="2/7"
            ),  # Prime denominator
            Fraction(
                shape=FractionShape.RECTANGLE, fraction="4/11"
            ),  # Prime denominator
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_whole_numbers():
    """Test with whole number fractions (improper fractions)."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="4/4"),
            Fraction(shape=FractionShape.CIRCLE, fraction="3/3"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_zero_numerators():
    """Test with zero numerators."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="0/5"),
            Fraction(shape=FractionShape.CIRCLE, fraction="0/7"),
        ]
    )
    file_name = draw_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_labeled_basic():
    """Test basic functionality with rectangle and circle shapes."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="3/4"),
            Fraction(shape=FractionShape.CIRCLE, fraction="1/2"),
        ]
    )
    file_name = draw_fractional_models_labeled(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_labeled_single_shape():
    """Test with a single shape to ensure single model handling works."""
    model_data = FractionList([Fraction(shape=FractionShape.RECTANGLE, fraction="1/2")])
    file_name = draw_fractional_models_labeled(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_labeled_many_shapes():
    """Test with multiple shapes to ensure layout handles many models."""
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="1/4"),
            Fraction(shape=FractionShape.CIRCLE, fraction="3/5"),
            Fraction(shape=FractionShape.RECTANGLE, fraction="2/3"),
            Fraction(shape=FractionShape.CIRCLE, fraction="4/7"),
            Fraction(shape=FractionShape.RECTANGLE, fraction="5/6"),
        ]
    )
    file_name = draw_fractional_models_labeled(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_labeled_edge_cases():
    """Test edge cases with whole numbers and zero numerators."""
    # Test whole number (improper fraction)
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="4/4"),
            Fraction(shape=FractionShape.CIRCLE, fraction="3/3"),
        ]
    )
    file_name = draw_fractional_models_labeled(model_data)
    assert os.path.exists(file_name)

    # Test zero numerator
    model_data = FractionList(
        [
            Fraction(shape=FractionShape.RECTANGLE, fraction="0/5"),
            Fraction(shape=FractionShape.CIRCLE, fraction="0/7"),
        ]
    )
    file_name = draw_fractional_models_labeled(model_data)
    assert os.path.exists(file_name)
