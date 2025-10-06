import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.measurement_comparison import (
    draw_measurement_comparison,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.measurement_comparison import (
    MeasuredItemName,
    MeasuredObject,
    MeasurementComparison,
    MeasurementUnitImage,
    UnitDisplayError,
)


@pytest.fixture
def sample_stimulus_with_unit_errors():
    return MeasurementComparison(
        [
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=8,
                label="Pencil A",
                unit=MeasurementUnitImage.UNIT_SQUARES,
            ),
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=8,
                label="Pencil B",
                unit=MeasurementUnitImage.UNIT_SQUARES,
                unit_display_error=UnitDisplayError.GAP,
            ),
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=8,
                label="Pencil C",
                unit=MeasurementUnitImage.UNIT_SQUARES,
                unit_display_error=UnitDisplayError.OVERLAP,
            ),
        ],
    )


@pytest.mark.drawing_functions
def test_draw_measurement_comparison_with_unit_errors(sample_stimulus_with_unit_errors):
    # Call the function with the sample stimulus with unit errors
    file_path = draw_measurement_comparison(sample_stimulus_with_unit_errors)

    # Check if the file was created
    assert os.path.exists(file_path), f"Image file was not created at {file_path}"


@pytest.fixture
def sample_stimulus_comparing_object_lengths():
    return MeasurementComparison(
        [
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=3,
                label="Pencil A",
            ),
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=5,
                label="Pencil B",
            ),
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=12,
                label="Pencil C",
            ),
        ],
    )


@pytest.mark.drawing_functions
def test_draw_measurement_comparison_comparing_object_lengths(
    sample_stimulus_comparing_object_lengths,
):
    # Call the function with the current sample stimulus
    file_path = draw_measurement_comparison(sample_stimulus_comparing_object_lengths)

    # Check if the file was created
    assert os.path.exists(file_path), f"Image file was not created at {file_path}"


@pytest.fixture
def sample_stimulus_comparing_object_lengths_using_common_units():
    return MeasurementComparison(
        [
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=3,
                label="Pencil",
                unit=MeasurementUnitImage.BUTTON,
            ),
            MeasuredObject(
                object_name=MeasuredItemName.ARROW,
                length=9,
                label="Arrow",
                unit=MeasurementUnitImage.BUTTON,
            ),
        ],
    )


@pytest.mark.drawing_functions
def test_draw_measurement_comparison_comparing_object_lengths_using_common_units(
    sample_stimulus_comparing_object_lengths_using_common_units,
):
    # Call the function with the current sample stimulus
    file_path = draw_measurement_comparison(
        sample_stimulus_comparing_object_lengths_using_common_units
    )

    # Check if the file was created
    assert os.path.exists(file_path), f"Image file was not created at {file_path}"


@pytest.fixture
def sample_stimulus_comparing_object_lengths_long_labels():
    return MeasurementComparison(
        [
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=5,
                label="John's Pencil",
            ),
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=12,
                label="Alice's Pencil",
            ),
        ],
    )


@pytest.mark.drawing_functions
def test_draw_measurement_comparison_comparing_object_lengths_long_labels(
    sample_stimulus_comparing_object_lengths_long_labels,
):
    # Call the function with the current sample stimulus
    file_path = draw_measurement_comparison(
        sample_stimulus_comparing_object_lengths_long_labels
    )

    # Check if the file was created
    assert os.path.exists(file_path), f"Image file was not created at {file_path}"


@pytest.fixture
def sample_stimulus_comparing_object_lengths_short_lengths():
    return MeasurementComparison(
        [
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=1,
                label="John's Pencil",
            ),
            MeasuredObject(
                object_name=MeasuredItemName.PENCIL,
                length=2,
                label="Alice's Pencil",
            ),
        ],
    )


@pytest.mark.drawing_functions
def test_draw_measurement_comparison_comparing_object_lengths_short_lengths(
    sample_stimulus_comparing_object_lengths_short_lengths,
):
    # Call the function with the current sample stimulus
    file_path = draw_measurement_comparison(
        sample_stimulus_comparing_object_lengths_short_lengths
    )

    # Check if the file was created
    assert os.path.exists(file_path), f"Image file was not created at {file_path}"
