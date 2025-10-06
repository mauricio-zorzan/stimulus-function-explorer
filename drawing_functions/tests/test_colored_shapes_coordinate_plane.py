import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.colored_shapes_coordinate_plane import (
    draw_colored_shapes_coordinate_plane,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.colored_shapes_coordinate_plane import (
    ColoredShape,
    ColoredShapesCoordinatePlane,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_basic():
    """Test basic functionality with simple shapes at integer coordinates."""
    shapes = [
        ColoredShape(label="Star", x=2, y=3, shape_type="star", color="pink"),
        ColoredShape(label="Circle", x=5, y=6, shape_type="circle", color="green"),
        ColoredShape(label="Pentagon", x=9, y=9, shape_type="pentagon", color="blue"),
        ColoredShape(label="Heart", x=4, y=7, shape_type="heart", color="purple"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_fractional_coordinates():
    """Test with fractional coordinates to verify decimal positioning works."""
    shapes = [
        ColoredShape(label="Letter A", x=2.5, y=5, shape_type="letter", color="blue", letter="A"),
        ColoredShape(label="Letter B", x=4, y=1.5, shape_type="letter", color="red", letter="B"),
        ColoredShape(label="Letter C", x=8.5, y=6.5, shape_type="letter", color="green", letter="C"),
        ColoredShape(label="Square", x=1, y=9, shape_type="square", color="yellow"),
        ColoredShape(label="Triangle", x=8, y=4, shape_type="triangle", color="red"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_mixed_shapes_and_letters():
    """Test combination of shapes and letters on the same coordinate plane."""
    shapes = [
        ColoredShape(label="Star", x=2, y=3, shape_type="star", color="pink"),
        ColoredShape(label="Letter D", x=1, y=8, shape_type="letter", color="orange", letter="D"),
        ColoredShape(label="Circle", x=5, y=6, shape_type="circle", color="green"),
        ColoredShape(label="Letter E", x=7, y=2, shape_type="letter", color="purple", letter="E"),
        ColoredShape(label="Heart", x=4, y=7, shape_type="heart", color="purple"),
        ColoredShape(label="Letter F", x=9, y=9, shape_type="letter", color="brown", letter="F"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_edge_cases():
    """Test shapes at boundary coordinates (0,0) and (10,10)."""
    shapes = [
        ColoredShape(label="Corner1", x=0, y=0, shape_type="square", color="red"),
        ColoredShape(label="Corner2", x=10, y=10, shape_type="circle", color="blue"),
        ColoredShape(label="Edge1", x=0, y=5, shape_type="triangle", color="green"),
        ColoredShape(label="Edge2", x=5, y=0, shape_type="pentagon", color="purple"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_multiple_colors():
    """Test different color combinations to verify color rendering."""
    shapes = [
        ColoredShape(label="Red", x=1, y=1, shape_type="circle", color="red"),
        ColoredShape(label="Blue", x=3, y=3, shape_type="square", color="blue"),
        ColoredShape(label="Green", x=5, y=5, shape_type="triangle", color="green"),
        ColoredShape(label="Purple", x=7, y=7, shape_type="pentagon", color="purple"),
        ColoredShape(label="Orange", x=9, y=9, shape_type="star", color="orange"),
        ColoredShape(label="Pink", x=2, y=8, shape_type="heart", color="pink"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_maximum_shapes():
    """Test with maximum number of shapes (10) to verify layout handling."""
    shapes = [
        ColoredShape(label="Shape1", x=1, y=1, shape_type="circle", color="red"),
        ColoredShape(label="Shape2", x=3, y=2, shape_type="square", color="blue"),
        ColoredShape(label="Shape3", x=5, y=3, shape_type="triangle", color="green"),
        ColoredShape(label="Shape4", x=7, y=4, shape_type="pentagon", color="purple"),
        ColoredShape(label="Shape5", x=9, y=5, shape_type="star", color="orange"),
        ColoredShape(label="Shape6", x=2, y=6, shape_type="heart", color="pink"),
        ColoredShape(label="Shape7", x=4, y=7, shape_type="circle", color="yellow"),
        ColoredShape(label="Shape8", x=6, y=8, shape_type="square", color="brown"),
        ColoredShape(label="Shape9", x=8, y=9, shape_type="triangle", color="gray"),
        ColoredShape(label="Shape10", x=1, y=9, shape_type="pentagon", color="cyan"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_minimum_shapes():
    """Test with minimum number of shapes (2) to verify minimal input handling."""
    shapes = [
        ColoredShape(label="Shape1", x=2, y=3, shape_type="circle", color="red"),
        ColoredShape(label="Shape2", x=7, y=8, shape_type="square", color="blue"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_letters_only():
    """Test with only letter shapes to verify letter rendering."""
    shapes = [
        ColoredShape(label="Letter A", x=2, y=3, shape_type="letter", color="blue", letter="A"),
        ColoredShape(label="Letter B", x=5, y=6, shape_type="letter", color="red", letter="B"),
        ColoredShape(label="Letter C", x=8, y=9, shape_type="letter", color="green", letter="C"),
        ColoredShape(label="Letter D", x=1, y=8, shape_type="letter", color="orange", letter="D"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_shapes_only():
    """Test with only geometric shapes (no letters) to verify shape rendering."""
    shapes = [
        ColoredShape(label="Star", x=2, y=3, shape_type="star", color="pink"),
        ColoredShape(label="Circle", x=5, y=6, shape_type="circle", color="green"),
        ColoredShape(label="Pentagon", x=9, y=9, shape_type="pentagon", color="blue"),
        ColoredShape(label="Heart", x=4, y=7, shape_type="heart", color="purple"),
        ColoredShape(label="Square", x=1, y=9, shape_type="square", color="yellow"),
        ColoredShape(label="Triangle", x=8, y=4, shape_type="triangle", color="red"),
    ]
    
    stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
    file_name = draw_colored_shapes_coordinate_plane(stimulus)
    
    assert os.path.exists(file_name)
    assert file_name.endswith(".webp")


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_validation_errors():
    """Test validation error handling for invalid inputs."""
    
    # Test invalid coordinates (outside 0-10 range)
    with pytest.raises(ValidationError, match="less than or equal to 10"):
        shapes = [
            ColoredShape(label="Invalid", x=11, y=5, shape_type="circle", color="red"),
        ]
        stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
        draw_colored_shapes_coordinate_plane(stimulus)
    
    # Test duplicate coordinates
    with pytest.raises(ValueError, match="Duplicate coordinates found"):
        shapes = [
            ColoredShape(label="Shape1", x=5, y=5, shape_type="circle", color="red"),
            ColoredShape(label="Shape2", x=5, y=5, shape_type="square", color="blue"),
        ]
        stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
        draw_colored_shapes_coordinate_plane(stimulus)
    
    # Test letter shape without letter field
    with pytest.raises(ValueError, match="must have a 'letter' field specified"):
        shapes = [
            ColoredShape(label="Circle", x=3, y=3, shape_type="circle", color="blue"),
            ColoredShape(label="Letter", x=5, y=5, shape_type="letter", color="red", letter=None),
        ]
        stimulus = ColoredShapesCoordinatePlane(shapes=shapes)
        draw_colored_shapes_coordinate_plane(stimulus)


@pytest.mark.drawing_functions
def test_draw_colored_shapes_coordinate_plane_schema_validation():
    """Test Pydantic schema validation for stimulus description."""
    
    # Test minimum shapes requirement (less than 2)
    with pytest.raises(ValidationError):
        ColoredShapesCoordinatePlane(shapes=[
            ColoredShape(label="Only", x=5, y=5, shape_type="circle", color="red"),
        ])
    
    # Test maximum shapes requirement (more than 10)
    with pytest.raises(ValidationError):
        shapes = []
        for i in range(11):  # 11 shapes, exceeding max of 10
            shapes.append(ColoredShape(
                label=f"Shape{i}", 
                x=i % 10, 
                y=i % 10, 
                shape_type="circle", 
                color="red"
            ))
        ColoredShapesCoordinatePlane(shapes=shapes)
    
    # Test invalid shape type
    with pytest.raises(ValidationError):
        ColoredShape(label="Invalid", x=5, y=5, shape_type="invalid_shape", color="red")
    
    # Test invalid coordinates in schema
    with pytest.raises(ValidationError):
        ColoredShape(label="Invalid", x=-1, y=5, shape_type="circle", color="red")
