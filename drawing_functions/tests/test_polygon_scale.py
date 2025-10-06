import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.polygon_scales import (
    draw_polygon_scale,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_scale import (
    PolygonScale,
)


@pytest.mark.drawing_functions
def test_draw_polygon_scale_triangle():
    """Test drawing scaled triangles with scale factor less than 1."""
    triangle_data = PolygonScale(
        polygon_type="triangle",
        original_polygon_label="Triangle E",
        scaled_polygon_label="Triangle F",
        scale_factor=0.7,
        original_vertex_labels=["A", "B", "C"],
        scaled_vertex_labels=["A'", "B'", "C'"],
        original_measurements=[30, 26, 16],  # AB, BC, CA
        scaled_measurements=[21, 18.2, 11.2],  # Scaled measurements
        original_visible_sides=["AB", "BC", "CA"],
        scaled_visible_sides=["A'B'", "B'C'"],
        measurement_unit="cm"
    )
    
    file_name = draw_polygon_scale(triangle_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_scale_quadrilateral():
    """Test drawing scaled quadrilaterals with measurement scale factor."""
    quad_data = PolygonScale(
        polygon_type="quadrilateral",
        original_polygon_label="Quadrilateral N",
        scaled_polygon_label="Quadrilateral O",
        scale_factor=1.6,
        original_vertex_labels=["A", "B", "C", "D"],
        scaled_vertex_labels=["A'", "B'", "C'", "D'"],
        original_measurements=[20, 28, 26, 22],  # AB, BC, CD, DA
        scaled_measurements=[32, 44.8, 41.6, 35.2],  # Scaled measurements
        original_visible_sides=["AB", "BC", "CD"],
        scaled_visible_sides=["A'B'", "B'C'"],
        measurement_unit="feet"
    )
    
    file_name = draw_polygon_scale(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_scale_trapezoid():
    """Test drawing scaled trapezoids like in the example image with measurement scale factor."""
    trapezoid_data = PolygonScale(
        polygon_type="quadrilateral",
        original_polygon_label="Trapezoid E",
        scaled_polygon_label="Trapezoid F",
        scale_factor=1.2,
        original_vertex_labels=["A", "B", "C", "D"],
        scaled_vertex_labels=["A'", "B'", "C'", "D'"],
        original_measurements=[40, 30, 20, 30],  # AB, BC, CD, DA
        scaled_measurements=[48, 36, 24, 36],  # Scaled measurements
        original_visible_sides=["AB", "CD"],
        scaled_visible_sides=["A'B'", "C'D'"],
        measurement_unit="meters"
    )
    
    file_name = draw_polygon_scale(trapezoid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_scale_pentagon():
    """Test drawing scaled pentagons."""
    pentagon_data = PolygonScale(
        polygon_type="pentagon",
        original_polygon_label="Pentagon P",
        scaled_polygon_label="Pentagon Q",
        scale_factor=2.0,
        original_vertex_labels=["A", "B", "C", "D", "E"],
        scaled_vertex_labels=["A'", "B'", "C'", "D'", "E'"],
        original_measurements=[20, 15, 25, 20, 18],  # AB, BC, CD, DE, EA
        scaled_measurements=[40, 30, 50, 40, 36],  # Scaled measurements
        original_visible_sides=["AB", "BC", "CD"],
        scaled_visible_sides=["A'B'", "B'C'", "C'D'", "D'E'"],
        measurement_unit="inches"
    )
    
    file_name = draw_polygon_scale(pentagon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_scale_irregular_shape():
    """Test drawing scaled irregular polygons like in the first example image with measurement scale factor."""
    irregular_data = PolygonScale(
        polygon_type="irregular",
        original_polygon_label="Polygon P",
        scaled_polygon_label="Polygon Q",
        scale_factor=2.0,
        original_vertex_labels=["A", "B", "C", "D"],
        scaled_vertex_labels=["A'", "B'", "C'", "D'"],
        original_measurements=[31, 4, 2, 62],  # AB, BC, CD, DE
        scaled_measurements=[62, 8, 4, 124],  # Scaled measurements
        original_visible_sides=["AB", "BC", "CD"],
        scaled_visible_sides=["A'B'", "B'C'"],
        measurement_unit="mm"
    )
    
    file_name = draw_polygon_scale(irregular_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_scale_with_decimal_measurements():
    """Test drawing scaled polygons with scale factor less than 1 and measurement scale factor that results in decimal values."""
    triangle_data = PolygonScale(
        polygon_type="triangle",
        original_polygon_label="Triangle A",
        scaled_polygon_label="Triangle B",
        scale_factor=0.8,
        original_vertex_labels=["P", "Q", "R"],
        scaled_vertex_labels=["P'", "Q'", "R'"],
        original_measurements=[23, 19, 17],  # PQ, QR, RP
        scaled_measurements=[18.4, 15.2, 13.6],  # Scaled measurements
        original_visible_sides=["PQ", "QR"],
        scaled_visible_sides=["P'Q'", "Q'R'", "R'P'"],
        measurement_unit="cm"
    )
    
    file_name = draw_polygon_scale(triangle_data)
    assert os.path.exists(file_name)



@pytest.mark.drawing_functions
def test_draw_polygon_scale_hexagon():
    """Test drawing scaled hexagons with different visible sides."""
    hexagon_data = PolygonScale(
        polygon_type="hexagon",
        original_polygon_label="Hexagon H",
        scaled_polygon_label="Hexagon I",
        scale_factor=0.6,
        original_vertex_labels=["X", "Y", "Z", "W", "V", "U"],
        scaled_vertex_labels=["X'", "Y'", "Z'", "W'", "V'", "U'"],
        original_measurements=[17, 17, 17, 17, 17, 17],  # Regular hexagon
        scaled_measurements=[10.2, 10.2, 10.2, 10.2, 10.2, 10.2],  # Scaled measurements
        original_visible_sides=["XY", "YZ", "ZW"],
        scaled_visible_sides=["X'Y'", "Y'Z'"],
        measurement_unit="inches"
    )
    
    file_name = draw_polygon_scale(hexagon_data)
    assert os.path.exists(file_name)
