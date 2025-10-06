import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
    draw_polygon_perimeter,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_perimeter import (
    PolygonPerimeter,
)

# =============================================================================
# Test Cases for Polygon Perimeter Problems (3-10 sides, one unknown)       #
# =============================================================================


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_triangle():
    """Test perimeter problem with regular triangle - unknown side at index 1."""
    polygon_data = PolygonPerimeter(
        side_lengths=[5, 5, 5],  # Regular triangle with equal sides
        unknown_side_indices=[1, 2],
        unit="cm",
        shape_type="regular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_quadrilateral():
    """Test perimeter problem with regular quadrilateral - unknown side at index 2."""
    polygon_data = PolygonPerimeter(
        side_lengths=[7, 7, 7, 7],  # Regular quadrilateral with equal sides
        unknown_side_indices=[0, 1, 2],
        unit="m",
        shape_type="regular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_pentagon():
    """Test perimeter problem with regular pentagon - unknown side at index 0."""
    polygon_data = PolygonPerimeter(
        side_lengths=[6, 6, 6, 6, 6],  # Regular pentagon with equal sides
        unknown_side_indices=[0, 1, 2, 3],
        unit="ft",
        shape_type="regular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_hexagon():
    """Test perimeter problem with regular hexagon - unknown side at index 3."""
    polygon_data = PolygonPerimeter(
        side_lengths=[5, 5, 5, 5, 5, 5],  # Regular hexagon with equal sides
        unknown_side_indices=[0, 2, 3, 4, 5],
        unit="inches",
        shape_type="regular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_decagon():
    """Test perimeter problem with regular decagon - unknown side at index 3."""
    polygon_data = PolygonPerimeter(
        side_lengths=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8],  # Regular decagon with equal sides
        unknown_side_indices=[0, 1, 2, 3, 5, 6, 7, 8, 9],
        unit="inches",
        shape_type="regular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_l_shape():
    """Test perimeter problem with L-shaped polygon - 6 sides, unknown side at index 2."""
    polygon_data = PolygonPerimeter(
        side_lengths=[20, 10, 15, 5, 5, 15],  # 6 sides for L-shape
        unknown_side_indices=[2],  # Inner horizontal side unknown
        unit="cm",
        shape_type="L-shape",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_t_shape():
    """Test perimeter problem with T-shaped polygon - 8 sides, unknown side at index 4."""
    polygon_data = PolygonPerimeter(
        side_lengths=[4, 2, 3, 3, 10, 3, 3, 2],  # 8 sides for T-shape
        unknown_side_indices=[4],  # Top horizontal side unknown
        unit="ft",
        shape_type="T-shape",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_default_shape_type():
    """Test that regular polygons work with default shape_type (backward compatibility)."""
    polygon_data = PolygonPerimeter(
        side_lengths=[5, 5, 5, 5],  # Square
        unknown_side_indices=[1, 2, 3],
        unit="cm",
        # shape_type defaults to "regular"
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_iiregular_triangle_with_unknown_side():
    """Test irregular triangle - should scale to actual side lengths."""
    polygon_data = PolygonPerimeter(
        side_lengths=[10, 4, 8],  # Very different side lengths
        unknown_side_indices=[0],
        unit="m",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_irregular_quadrilateral():
    """Test irregular quadrilateral - should scale to actual side lengths."""
    polygon_data = PolygonPerimeter(
        side_lengths=[10, 4, 8, 6],  # Very different side lengths
        unknown_side_indices=[2],
        unit="m",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_irregular_pentagon():
    """Test irregular pentagon - should show variation in side lengths."""
    polygon_data = PolygonPerimeter(
        side_lengths=[12, 5, 8, 15, 3],  # Highly irregular pentagon
        unknown_side_indices=[0],
        unit="ft",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_irregular_hexagon():
    """Test irregular octagon - should show variation in side lengths."""
    polygon_data = PolygonPerimeter(
        side_lengths=[8, 3, 12, 15, 15, 10],  # Highly irregular hexagon
        unknown_side_indices=[0],
        unit="ft",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_irregular_octagon():
    """Test irregular octagon - should show variation in side lengths."""
    polygon_data = PolygonPerimeter(
        side_lengths=[5, 6, 7, 8, 9, 8, 7, 6],
        unknown_side_indices=[0],
        unit="ft",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_irregular_rectangle_with_unknown_sides():
    """Test irregular rectangle - quadrilateral with opposite sides equal and unknown sides."""
    polygon_data = PolygonPerimeter(
        side_lengths=[
            8,
            5,
            8,
            5,
        ],  # Rectangle: opposite sides equal (length=8, width=5)
        unknown_side_indices=[0, 2],  # Unknown side should be 8 (same as first side)
        unit="ft",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_multiple_unknown_sides():
    """Test polygon with multiple unknown sides - students need to find two missing measurements."""
    polygon_data = PolygonPerimeter(
        side_lengths=[12, 8, 15, 6, 10],  # Pentagon with different side lengths
        unknown_side_indices=[1, 3],  # Two unknown sides at indices 1 and 3
        unit="m",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_perimeter_all_sides_labeled():
    """Test polygon with empty unknown_side_indices - all sides show measurements."""
    polygon_data = PolygonPerimeter(
        side_lengths=[6, 9, 4, 7],  # Irregular quadrilateral
        unknown_side_indices=[],  # Empty list - all sides show measurements
        unit="cm",
        shape_type="irregular",
    )
    file_name = draw_polygon_perimeter(polygon_data)
    assert os.path.exists(file_name)
