import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.rulers import (
    draw_ruler_measured_objects,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.ruler import (
    MeasuredItem,
    MeasuredItemName,
    MeasurementUnit,
    Ruler,
    RulerStimulus,
)


@pytest.fixture
def sample_ruler_stimulus():
    return RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=11.8,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            )
        ],
    ).model_dump()


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_basic(sample_ruler_stimulus):
    file_path = draw_ruler_measured_objects(sample_ruler_stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_max_items():
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=7.5,
                ruler=Ruler(unit=MeasurementUnit.INCHES),
            ),
            MeasuredItem(
                name=MeasuredItemName.STRAW,
                length=12,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
            MeasuredItem(
                name=MeasuredItemName.ARROW,
                length=12,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)  # type: ignore
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_with_labels():
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=1.5,
                ruler=Ruler(unit=MeasurementUnit.INCHES),
                label="Figure 1",
            ),
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=11.0,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
                label="Figure 2",
            ),
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=6.4,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
                label="Figure 3",
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_mixed_units():
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=7.5,
                ruler=Ruler(unit=MeasurementUnit.INCHES),
            ),
            MeasuredItem(
                name=MeasuredItemName.STRAW,
                length=12.0,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
            MeasuredItem(
                name=MeasuredItemName.ARROW,
                length=10.0,
                ruler=Ruler(unit=MeasurementUnit.INCHES),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_long_item():
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=3.5,
                ruler=Ruler(unit=MeasurementUnit.INCHES),
            ),
            MeasuredItem(
                name=MeasuredItemName.ARROW,
                length=3.0,
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_object_starts_at_zero():
    """Test objects that start at the zero mark on the ruler"""
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=7.5,
                start_position=0.0,  # Object starts at zero mark
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_object_starts_at_non_zero():
    """Test objects that start at a non-zero position on the ruler"""
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.STRAW,
                length=5.0,
                start_position=3.0,  # Object starts at position 3, not at zero
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_with_start_position():
    """Test that objects can be positioned with different start positions"""
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=5.0,
                start_position=2.0,  # Object starts at position 2, not at zero
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
            MeasuredItem(
                name=MeasuredItemName.STRAW,
                length=3.0,
                start_position=0.0,  # Object starts at position 0
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_inches_with_start_position():
    """Test start_position functionality with inches"""
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.ARROW,
                length=2.5,
                start_position=1.0,  # Starts at 1 inch mark
                ruler=Ruler(unit=MeasurementUnit.INCHES),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_fractional_start_position():
    """Test start_position with fractional values"""
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.PENCIL,
                length=4.5,
                start_position=1.5,  # Starts at 1.5 cm mark
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_ruler_measured_objects_ruler_length_adjustment():
    """Test that ruler length is properly adjusted for objects with start_position"""
    stimulus = RulerStimulus(
        items=[
            MeasuredItem(
                name=MeasuredItemName.STRAW,
                length=6.0,
                start_position=4.0,  # Object from 4cm to 10cm, so ruler needs to be at least 10cm
                ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
            ),
        ],
    ).model_dump()
    file_path = draw_ruler_measured_objects(stimulus)
    assert os.path.exists(file_path)

    # Verify the ruler was adjusted properly by checking the stimulus object
    ruler_stimulus = RulerStimulus(**stimulus)
    ruler_length = ruler_stimulus.items[0].ruler.length
    assert (
        ruler_length is not None and ruler_length >= 10
    )  # Should be at least 10 to accommodate the object
