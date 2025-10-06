import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.measurements import (
    draw_measurement,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.measurements_model import (
    Measurements,
)


@pytest.mark.drawing_functions
def test_draw_measurement_with_color2():
    measurement = Measurements(measurement=1000, units="milliliters", color="purple")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_measurement_with_color3_3_and_half_liters():
    measurement = Measurements(measurement=3.5, units="liters", color="green")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)

@pytest.mark.drawing_functions
def test_draw_measurement_with_color3_8_and_three_quarters_liters():
    measurement = Measurements(measurement=8.75, units="liters", color="green")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)

@pytest.mark.drawing_functions
def test_draw_measurement_with_6_liter():
    measurement = Measurements(measurement=6, units="liters", color="green")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_measurement_with_color5():
    measurement = Measurements(measurement=90, units="grams", color="green")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_measurement_with_color():
    measurement = Measurements(measurement=575, units="milliliters", color="yellow")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_measurement_with_color4():
    measurement = Measurements(measurement=5075, units="milliliters", color="yellow")
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_measurement_without_color():
    measurement = Measurements(measurement=73, units="grams", color=None)
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_measurement_without_color2():
    measurement = Measurements(measurement=7, units="kilograms", color=None)
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)


def test_draw_measurement_without_color3():
    measurement = Measurements(measurement=700, units="kilograms", color=None)
    file_name = draw_measurement(measurement)

    assert os.path.exists(file_name)
