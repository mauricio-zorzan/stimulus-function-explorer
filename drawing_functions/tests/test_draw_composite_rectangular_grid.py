import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
    draw_composite_rectangular_grid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.complex_figure import (
    CompositeRectangularGrid,
    RectangleSpec,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.complex_figure import (
    EAbbreviatedMeasurementUnit as ComplexFigureUnit,
)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_l_shape():
    l_shape = CompositeRectangularGrid(
        rectangles=[
            RectangleSpec(
                unit=ComplexFigureUnit.KILOMETERS, length=3, width=8, x=0, y=3
            ),
            RectangleSpec(
                unit=ComplexFigureUnit.KILOMETERS, length=4, width=5, x=3, y=0
            ),
        ]
    )
    file_name = draw_composite_rectangular_grid(l_shape)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_overlap():
    overlap = CompositeRectangularGrid(
        rectangles=[
            RectangleSpec(unit=ComplexFigureUnit.METERS, length=4, width=6, x=0, y=0),
            RectangleSpec(unit=ComplexFigureUnit.METERS, length=4, width=6, x=2, y=2),
        ]
    )
    file_name = draw_composite_rectangular_grid(overlap)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_horizontal_l():
    # Horizontal L-shape: two rectangles, one to the right of the other
    l_shape = CompositeRectangularGrid(
        rectangles=[
            RectangleSpec(
                unit=ComplexFigureUnit.METERS, length=2, width=6, x=0, y=2
            ),  # Top horizontal
            RectangleSpec(
                unit=ComplexFigureUnit.METERS, length=4, width=2, x=4, y=0
            ),  # Bottom right
        ]
    )
    file_name = draw_composite_rectangular_grid(l_shape)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_vertical_l():
    # Vertical L-shape: tall bar with a horizontal bar at the bottom
    l_shape = CompositeRectangularGrid(
        rectangles=[
            RectangleSpec(
                unit=ComplexFigureUnit.METERS, length=6, width=2, x=0, y=0
            ),  # Vertical bar
            RectangleSpec(
                unit=ComplexFigureUnit.METERS, length=2, width=4, x=0, y=0
            ),  # Bottom bar
        ]
    )
    file_name = draw_composite_rectangular_grid(l_shape)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_disjoint():
    with pytest.raises(ValueError, match="touch or overlap"):
        CompositeRectangularGrid(
            rectangles=[
                RectangleSpec(unit=ComplexFigureUnit.FEET, length=2, width=3, x=0, y=0),
                RectangleSpec(unit=ComplexFigureUnit.FEET, length=2, width=3, x=5, y=5),
            ]
        )


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_json_payload():
    # Test case from JSON payload: {"rectangles": [{"unit": "Units", "length": 6, "width": 7, "x": 0, "y": 3}, {"unit": "Units", "length": 3, "width": 5, "x": 0, "y": 0}]}
    payload = CompositeRectangularGrid(
        rectangles=[
            RectangleSpec(unit=ComplexFigureUnit.UNITS, length=6, width=7, x=0, y=3),
            RectangleSpec(unit=ComplexFigureUnit.UNITS, length=3, width=5, x=0, y=0),
        ]
    )
    file_name = draw_composite_rectangular_grid(payload)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_grid_side_by_side():
    # Test case from JSON payload: {"rectangles": [{"unit": "Units", "length": 4, "width": 7, "x": 0, "y": 0}, {"unit": "Units", "length": 3, "width": 8, "x": 7, "y": 0}]}
    payload = CompositeRectangularGrid(
        rectangles=[
            RectangleSpec(unit=ComplexFigureUnit.UNITS, length=4, width=7, x=0, y=0),
            RectangleSpec(unit=ComplexFigureUnit.UNITS, length=3, width=8, x=7, y=0),
        ]
    )
    file_name = draw_composite_rectangular_grid(payload)
    assert os.path.exists(file_name)
