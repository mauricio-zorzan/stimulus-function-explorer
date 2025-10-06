import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
    generate_stimulus_with_grid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    EAbbreviatedMeasurementUnit,
    RectangularGrid,
)


@pytest.fixture
def rectangular_grid():
    return RectangularGrid(
        length=5, width=3, unit=EAbbreviatedMeasurementUnit.CENTIMETERS
    )


@pytest.mark.drawing_functions
def test_generate_stimulus_with_grid(rectangular_grid):
    file_name = generate_stimulus_with_grid(rectangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_stimulus_with_grid_large_rectangle():
    data = RectangularGrid(
        length=10, width=8, unit=EAbbreviatedMeasurementUnit.METERS
    ).model_dump(by_alias=True)
    file_name = generate_stimulus_with_grid(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_stimulus_with_grid_small_rectangle():
    data = RectangularGrid(
        length=1, width=1, unit=EAbbreviatedMeasurementUnit.CENTIMETERS
    ).model_dump(by_alias=True)
    file_name = generate_stimulus_with_grid(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_stimulus_with_grid_non_square():
    data = RectangularGrid(
        length=2, width=9, unit=EAbbreviatedMeasurementUnit.INCHES
    ).model_dump(by_alias=True)
    file_name = generate_stimulus_with_grid(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_stimulus_with_grid_square():
    data = RectangularGrid(
        length=5, width=5, unit=EAbbreviatedMeasurementUnit.CENTIMETERS
    ).model_dump(by_alias=True)
    file_name = generate_stimulus_with_grid(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_stimulus_with_grid_irregular_with_extra_squares():
    data = RectangularGrid(
        length=5, width=5, unit=EAbbreviatedMeasurementUnit.CENTIMETERS, irregular=True, extra_unit_squares=5
    ).model_dump(by_alias=True)
    file_name = generate_stimulus_with_grid(data)
    assert os.path.exists(file_name)

