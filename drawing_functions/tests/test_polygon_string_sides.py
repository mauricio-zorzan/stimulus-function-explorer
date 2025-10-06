import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
    draw_polygon_with_string_sides,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_string_sides import (
    PolygonStringSides,
)


@pytest.mark.drawing_functions
def test_triangle_with_string_sides():
    """Test drawing a triangle with string side lengths."""
    side_data = PolygonStringSides(
        side_lengths=["3", "4", "5"],
        side_labels=["9", "(4g + 12)", "(5g + 15)"],
    )
    file_name = draw_polygon_with_string_sides(side_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_quadrilateral_with_string_sides():
    """Test drawing a quadrilateral with string side lengths."""
    side_data = PolygonStringSides(
        side_lengths=["(6g + 6)", "(3g + 9)", "(6g + 6)", "(3g + 9)"],
    )
    file_name = draw_polygon_with_string_sides(side_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_pentagon_with_string_sides():
    """Test drawing a pentagon with string side lengths."""
    side_data = PolygonStringSides(
        side_lengths=["(6g + 6)", "(3g + 9)", "(6g + 6)", "(3g + 9)", "(6g + 6)"],
    )
    file_name = draw_polygon_with_string_sides(side_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_hexagon_with_string_sides():
    """Test drawing a hexagon with string side lengths."""
    side_data = PolygonStringSides(
        side_lengths=[
            "(6g + 6)",
            "(3g + 9)",
            "(6g + 6)",
            "(3g + 9)",
            "(6g + 6)",
            "(3g + 9)",
        ],
    )
    file_name = draw_polygon_with_string_sides(side_data)
    assert os.path.exists(file_name)
