import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_fractional_models_unequal,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionShape,
    UnequalFraction,
    UnequalFractionList,
)
from pydantic import ValidationError


@pytest.fixture
def sample_unequal_fraction_list():
    return UnequalFractionList(
        [
            UnequalFraction(
                shape=FractionShape.RECTANGLE, divided_parts=8, equally_divided=True
            ),
            UnequalFraction(
                shape=FractionShape.CIRCLE, divided_parts=4, equally_divided=False
            ),
        ]
    )


@pytest.fixture
def sample_unequal_fraction_list_three_models():
    return UnequalFractionList(
        [
            UnequalFraction(
                shape=FractionShape.RECTANGLE, divided_parts=8, equally_divided=True
            ),
            UnequalFraction(
                shape=FractionShape.CIRCLE, divided_parts=6, equally_divided=False
            ),
            UnequalFraction(
                shape=FractionShape.RECTANGLE, divided_parts=4, equally_divided=False
            ),
        ]
    )


@pytest.fixture
def sample_unequal_fraction_list_four_models():
    return UnequalFractionList(
        [
            UnequalFraction(
                shape=FractionShape.TRIANGLE, divided_parts=3, equally_divided=True
            ),
            UnequalFraction(
                shape=FractionShape.CIRCLE, divided_parts=6, equally_divided=False
            ),
            UnequalFraction(
                shape=FractionShape.RECTANGLE, divided_parts=4, equally_divided=False
            ),
            UnequalFraction(
                shape=FractionShape.CIRCLE, divided_parts=10, equally_divided=True
            ),
        ]
    )


@pytest.mark.drawing_functions
def test_draw_fractional_models_unequal_two_models(sample_unequal_fraction_list):
    file_name = draw_fractional_models_unequal(sample_unequal_fraction_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_unequal_three_models(
    sample_unequal_fraction_list_three_models,
):
    file_name = draw_fractional_models_unequal(
        sample_unequal_fraction_list_three_models
    )
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_unequal_four_models(
    sample_unequal_fraction_list_four_models,
):
    file_name = draw_fractional_models_unequal(sample_unequal_fraction_list_four_models)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_unequal_invalid_shape():
    with pytest.raises(ValidationError):
        UnequalFractionList(
            [UnequalFraction(shape="square", divided_parts=8, equally_divided=True)]
        )


@pytest.mark.drawing_functions
def test_draw_fractional_models_unequal_invalid_divided_parts():
    with pytest.raises(ValidationError):
        UnequalFractionList(
            [
                UnequalFraction(
                    shape=FractionShape.RECTANGLE, divided_parts=1, equally_divided=True
                )
            ]
        )
