import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_whole_fractional_models,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionShape,
    WholeFractionalShapes,
)


@pytest.mark.drawing_functions
def test_single_rectangle_basic():
    """Test single whole rectangle (4/4 = 1)"""
    model_data = WholeFractionalShapes(
        count=1, shape=FractionShape.RECTANGLE, divisions=4
    )
    file_name = draw_whole_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_single_circle_basic():
    """Test single whole circle (6/6 = 1)"""
    model_data = WholeFractionalShapes(count=1, shape=FractionShape.CIRCLE, divisions=6)
    file_name = draw_whole_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_two_rectangles():
    """Test two whole rectangles (16/8 = 2) in single row"""
    model_data = WholeFractionalShapes(
        count=2, shape=FractionShape.RECTANGLE, divisions=8
    )
    file_name = draw_whole_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_three_circles():
    """Test three whole circles (30/10 = 3) in single row"""
    model_data = WholeFractionalShapes(
        count=3, shape=FractionShape.CIRCLE, divisions=10
    )
    file_name = draw_whole_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_four_rectangles():
    """Test four whole rectangles (20/5 = 4) in single row"""
    model_data = WholeFractionalShapes(
        count=4, shape=FractionShape.RECTANGLE, divisions=5
    )
    file_name = draw_whole_fractional_models(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_five_circles():
    """Test five whole circles (60/12 = 5) in single row"""
    model_data = WholeFractionalShapes(
        count=5, shape=FractionShape.CIRCLE, divisions=12
    )
    file_name = draw_whole_fractional_models(model_data)
    assert os.path.exists(file_name)
