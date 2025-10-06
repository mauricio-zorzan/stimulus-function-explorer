from content_generators.additional_content.stimulus_image.stimulus_descriptions.measurements_model import (
    Measurements,
)


def test_valid_measurements_with_color():
    measurement = Measurements(measurement=575, units="milliliters", color="yellow")
    assert measurement.measurement == 575
    assert measurement.units == "milliliters"
    assert measurement.color == "yellow"


def test_valid_measurements_without_color():
    measurement = Measurements(measurement=73, units="grams", color=None)
    assert measurement.measurement == 73
    assert measurement.units == "grams"
    assert measurement.color is None
