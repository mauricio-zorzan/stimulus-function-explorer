import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.fraction_models import (
    draw_fractional_models_multiplication_units,
    draw_fractional_pair_models,
    draw_fractional_sets_models,
)


@pytest.mark.drawing_functions
def test_draw_fractional_pair_models():
    stimulus_description = [
        {"shape": "circle", "fractions": ["3/6", "1/6"]},
        {"shape": "circle", "fractions": ["3/7", "3/7"]},
        {"shape": "circle", "fractions": ["5/8", "3/8"]},
        {"shape": "circle", "fractions": ["5/9", "3/9"]},
    ]
    file_name = draw_fractional_pair_models(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_pair_models_2():
    stimulus_description = [
        {"shape": "circle", "fractions": ["3/6", "1/6"]},
        {"shape": "circle", "fractions": ["3/7", "3/7"]},
        {"shape": "circle", "fractions": ["5/8", "3/8"]},
        {"shape": "circle", "fractions": ["5/9", "3/9"]},
    ]
    file_name = draw_fractional_pair_models(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_pair_models_addition():
    stimulus_description = [
        {"shape": "circle", "fractions": ["3/6", "1/6", "2/6"]},
        {"shape": "circle", "fractions": ["3/7", "3/7"]},
        {"shape": "circle", "fractions": ["2/8", "3/8", "3/8"]},
        {"shape": "circle", "fractions": ["5/9", "3/9"]},
    ]
    file_name = draw_fractional_pair_models(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_sets_models():
    stimulus_description = [
        {
            "shape": "rectangle",
            "fractions": ["2/5", "2/5", "2/5", "2/5"],
            "color": "red",
        },
        {
            "shape": "rectangle",
            "fractions": ["3/5", "3/5", "3/5", "3/5"],
            "color": "green",
        },
        {"shape": "rectangle", "fractions": ["2/5", "2/5", "2/5"], "color": "yellow"},
        {
            "shape": "rectangle",
            "fractions": ["2/5", "2/5", "2/5", "2/5", "2/5"],
            "color": "blue",
        },
    ]
    file_name = draw_fractional_sets_models(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_multiplication_units():
    stimulus_description = {"fractions": ["2/5", "3/7"]}
    file_name = draw_fractional_models_multiplication_units(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_multiplication_units_2():
    stimulus_description = {"fractions": ["4/5", "1/8"]}
    file_name = draw_fractional_models_multiplication_units(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_fractional_models_multiplication_units_3():
    stimulus_description = {"fractions": ["7/8", "9/9"]}
    file_name = draw_fractional_models_multiplication_units(stimulus_description)
    assert os.path.exists(file_name)
