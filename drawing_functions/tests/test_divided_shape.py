import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_fractional_models_full_shade,
    draw_fractional_models_no_shade,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    DividedShape,
    DividedShapeList,
    FractionShape,
)
from pydantic import ValidationError


@pytest.fixture
def sample_divided_shape_list():
    return DividedShapeList(
        [
            DividedShape(shape=FractionShape.RECTANGLE, denominator=4),
            DividedShape(shape=FractionShape.CIRCLE, denominator=3),
        ]
    )


@pytest.mark.drawing_functions
def test_draw_fractional_models_no_shade(sample_divided_shape_list):
    file_name = draw_fractional_models_no_shade(sample_divided_shape_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_no_shade_single_model():
    model_data = DividedShapeList(
        [DividedShape(shape=FractionShape.RECTANGLE, denominator=5)]
    )
    file_name = draw_fractional_models_no_shade(model_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_no_shade_invalid_shape():
    with pytest.raises(ValidationError):
        DividedShapeList([DividedShape(shape="square", denominator=4)])


@pytest.mark.drawing_functions
def test_draw_fractional_models_full_shade(sample_divided_shape_list):
    file_name = draw_fractional_models_full_shade(sample_divided_shape_list)
    assert os.path.exists(file_name)
