import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.ruler import (
    MeasuredItem,
    MeasuredItemName,
    MeasurementUnit,
    Ruler,
    RulerStimulus,
)
from pydantic import ValidationError


def test_valid_ruler_stimulus_with_individual_rulers():
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=7.5,
                ruler=Ruler(unit=MeasurementUnit.INCHES),
            ),
            MeasuredItem(
                name=MeasuredItemName.STRAW,
                length=11.0,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    )
    assert len(stimulus.items) == 2
    assert stimulus.items[0].ruler.unit == MeasurementUnit.INCHES
    assert stimulus.items[1].ruler.unit == MeasurementUnit.CENTIMETERS


def test_invalid_ruler_stimulus_without_any_ruler():
    with pytest.raises(ValidationError):
        RulerStimulus(
            items=[
                MeasuredItem(name=MeasuredItemName.PENCIL, length=7.5),  # type: ignore
            ],
        )


def test_individual_ruler_length_adjustment():
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=7.5,
                ruler=Ruler(unit=MeasurementUnit.INCHES, length=10),
            ),
            MeasuredItem(
                name=MeasuredItemName.ARROW,
                length=12.0,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS, length=12),
            ),
        ],
    )
    assert (
        stimulus.items[0].ruler.length == 10
    )  # Adjusted to accommodate the longest item ceil(30 cm converted to inches + 1)
    assert (
        stimulus.items[1].ruler.length == 12.0
    )  # Adjusted to accommodate the longest item ceil(30 cm + 1)


def test_conversion_methods():
    assert Ruler.convert_to_unit(
        10, MeasurementUnit.INCHES, MeasurementUnit.CENTIMETERS
    ) == pytest.approx(25.4)
    assert Ruler.convert_to_unit(
        25.4, MeasurementUnit.CENTIMETERS, MeasurementUnit.INCHES
    ) == pytest.approx(10)


def test_measured_item_name_enum():
    assert MeasuredItemName.PENCIL == "pencil"
    assert MeasuredItemName.ARROW == "arrow"
    assert MeasuredItemName.STRAW == "straw"


def test_measurement_unit_enum():
    assert MeasurementUnit.INCHES == "inches"
    assert MeasurementUnit.CENTIMETERS == "centimeters"
