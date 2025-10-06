import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.symmetry_lines import (
    generate_lines_of_symmetry,
    generate_symmetry_identification_task,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.symmetry_identification_model import (
    IdentificationLine,
    SymmetryIdentification,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.symmetry_lines_model import (
    Line,
    LinesOfSymmetry,
)


# Tests for LinesOfSymmetry (using A,B,C,D labels and requiring exactly one true symmetry line)
@pytest.mark.drawing_functions
def test_generate_lines_of_symmetry_triangle_with_four_lines():
    # Isosceles triangle with one true line of symmetry and three non-symmetry lines
    symmetry_lines = LinesOfSymmetry(
        shape_coordinates=[[0, 0], [4, 0], [2, 3]],
        lines=[
            Line(slope=0, intercept=1, label="A"),  # Not symmetry
            Line(
                slope=None, intercept=2, label="B"
            ),  # TRUE symmetry line (vertical through apex)
            Line(slope=-1, intercept=4, label="C"),  # Not symmetry
            Line(slope=0.5, intercept=-1, label="D"),  # Not symmetry
        ],
    )
    file_name = generate_lines_of_symmetry(symmetry_lines)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_lines_of_symmetry_rectangle_no_lines():
    symmetry_lines = LinesOfSymmetry(
        shape_coordinates=[[0, 0], [4, 0], [4, 2], [0, 2]],
        lines=[],
    )
    file_name = generate_lines_of_symmetry(symmetry_lines)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_lines_of_symmetry_square_with_one_line():
    # Square with one true line of symmetry
    symmetry_lines = LinesOfSymmetry(
        shape_coordinates=[[0, 0], [4, 0], [4, 4], [0, 4]],
        lines=[
            Line(
                slope=None, intercept=2, label="A"
            ),  # TRUE vertical symmetry line through center
        ],
    )
    file_name = generate_lines_of_symmetry(symmetry_lines)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_lines_of_symmetry_pentagon():
    """Test regular pentagon with one line of symmetry."""
    # Create a regular pentagon with one vertex at the top (has vertical symmetry)
    import math

    # Start with top vertex and go clockwise to ensure vertical symmetry
    angles = [math.pi / 2 + 2 * math.pi * i / 5 for i in range(5)]  # Start at top (π/2)
    pentagon_coords = [[math.cos(a), math.sin(a)] for a in angles]

    symmetry_lines = LinesOfSymmetry(
        shape_coordinates=pentagon_coords,
        lines=[
            Line(
                slope=None, intercept=0, label="A"
            ),  # Vertical line through center - TRUE symmetry
        ],
    )
    file_name = generate_lines_of_symmetry(symmetry_lines)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_lines_of_symmetry_single_line():
    """Test isosceles triangle with single line of symmetry."""
    symmetry_lines = LinesOfSymmetry(
        shape_coordinates=[[0, 0], [4, 0], [2, 3]],  # Isosceles triangle
        lines=[
            Line(
                slope=None, intercept=2, label="B"
            ),  # Vertical line through apex - TRUE symmetry
        ],
    )
    file_name = generate_lines_of_symmetry(symmetry_lines)
    assert os.path.exists(file_name)


# Tests for SymmetryIdentification (using vertical, horizontal, diagonal, not_symmetry labels)
@pytest.mark.drawing_functions
def test_generate_symmetry_identification_flower_with_vertical_line():
    """Test flower shape with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="flower",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_flower_with_horizontal_line():
    """Test flower shape with horizontal line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="flower",
        lines=[IdentificationLine(slope=0, intercept=0, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_flower_with_diagonal_line():
    """Test flower shape with diagonal line."""
    symmetry_task = SymmetryIdentification(
        shape_type="flower",
        lines=[IdentificationLine(slope=1, intercept=0, label="diagonal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_sun_with_vertical_line():
    """Test sun shape with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="sun",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_sun_with_horizontal_line():
    """Test sun shape with horizontal line."""
    symmetry_task = SymmetryIdentification(
        shape_type="sun",
        lines=[IdentificationLine(slope=0, intercept=0, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_sun_with_off_center_line():
    """Test sun shape with off-center line."""
    symmetry_task = SymmetryIdentification(
        shape_type="sun",
        lines=[IdentificationLine(slope=None, intercept=0.4, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_diamond_with_vertical_line():
    """Test diamond shape with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="diamond",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_diamond_with_horizontal_line():
    """Test diamond shape with horizontal line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="diamond",
        lines=[IdentificationLine(slope=0, intercept=0, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_diamond_with_wrong_diagonal():
    """Test diamond shape with diagonal line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="diamond",
        lines=[IdentificationLine(slope=0.5, intercept=0.2, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_heart_with_vertical_line():
    """Test heart shape with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="heart",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_heart_with_horizontal_line():
    """Test heart shape with horizontal line."""
    symmetry_task = SymmetryIdentification(
        shape_type="heart",
        lines=[IdentificationLine(slope=0, intercept=0, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_house_with_vertical_line():
    """Test house shape with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="house",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_house_with_horizontal_line():
    """Test house shape with horizontal line."""
    symmetry_task = SymmetryIdentification(
        shape_type="house",
        lines=[IdentificationLine(slope=0, intercept=0.1, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_wheel_with_diagonal_line():
    """Test ship wheel shape with diagonal line - like the image shown."""
    symmetry_task = SymmetryIdentification(
        shape_type="wheel",
        lines=[IdentificationLine(slope=1, intercept=0, label="diagonal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_wheel_with_off_diagonal():
    """Test ship wheel with diagonal line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="wheel",
        lines=[IdentificationLine(slope=0.5, intercept=0.2, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_wheel_with_vertical_line():
    """Test ship wheel with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="wheel",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_wheel_with_horizontal_line():
    """Test ship wheel with horizontal line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="wheel",
        lines=[IdentificationLine(slope=0, intercept=0, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_multiple_lines():
    """Test with multiple lines for identification."""
    symmetry_task = SymmetryIdentification(
        shape_type="diamond",
        lines=[
            IdentificationLine(slope=None, intercept=0, label="vertical"),
            IdentificationLine(slope=0, intercept=0, label="horizontal"),
        ],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_polygon_shapes():
    """Test custom polygon shapes."""
    # Isosceles triangle with vertical symmetry
    symmetry_task = SymmetryIdentification(
        shape_type="polygon",
        shape_coordinates=[[0, 0], [4, 0], [2, 3]],
        lines=[IdentificationLine(slope=None, intercept=2, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_scalene_triangle():
    """Test scalene triangle with line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="polygon",
        shape_coordinates=[[0, 0], [5, 0], [1, 3]],
        lines=[IdentificationLine(slope=None, intercept=2.5, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_irregular_polygon():
    """Test irregular polygon with line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="polygon",
        shape_coordinates=[[0, 0], [3, 1], [2, 4], [-1, 3], [-2, 1]],
        lines=[IdentificationLine(slope=None, intercept=1, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_letter_shapes():
    """Test letter-like shapes for symmetry identification."""
    # Letter A shape (has vertical symmetry)
    letter_a = SymmetryIdentification(
        shape_type="polygon",
        shape_coordinates=[
            [-1, 0],
            [-0.5, 2],
            [0.5, 2],
            [1, 0],
            [0.6, 0],
            [0.2, 1],
            [-0.2, 1],
            [-0.6, 0],
        ],
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(letter_a)
    assert os.path.exists(file_name)

    # Letter L shape (no symmetry)
    letter_l = SymmetryIdentification(
        shape_type="polygon",
        shape_coordinates=[[0, 0], [0, 3], [0.5, 3], [0.5, 0.5], [2, 0.5], [2, 0]],
        lines=[IdentificationLine(slope=None, intercept=1, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(letter_l)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_football_with_vertical_line():
    """Test football shape with vertical line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="football",
        lines=[IdentificationLine(slope=None, intercept=0, label="vertical")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_football_with_horizontal_line():
    """Test football shape with horizontal line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="football",
        lines=[IdentificationLine(slope=0, intercept=0, label="horizontal")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_football_with_diagonal_line():
    """Test football shape with diagonal line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="football",
        lines=[IdentificationLine(slope=1, intercept=0.2, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_various_shapes_no_lines():
    """Test various shapes with no lines (edge case)."""
    shape_types = ["flower", "sun", "diamond", "heart", "house", "wheel", "football"]

    for shape_type in shape_types:
        symmetry_task = SymmetryIdentification(
            shape_type=shape_type,
            lines=[],  # No lines
        )
        file_name = generate_symmetry_identification_task(symmetry_task)
        assert os.path.exists(file_name)


# Additional tests with fixed validation issues
@pytest.mark.drawing_functions
def test_generate_symmetry_identification_flower_with_diagonal_line_not_symmetry():
    """Test flower shape with diagonal line that is NOT a line of symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="flower",
        lines=[IdentificationLine(slope=2, intercept=0.3, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_sun_with_off_center_vertical_line():
    """Test sun shape with off-center vertical line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="sun",
        lines=[IdentificationLine(slope=None, intercept=0.4, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_heart_with_horizontal_line_not_symmetry():
    """Test heart shape with horizontal line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="heart",
        lines=[IdentificationLine(slope=0, intercept=0, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_symmetry_identification_house_with_horizontal_line_not_symmetry():
    """Test house shape with horizontal line that is NOT symmetry."""
    symmetry_task = SymmetryIdentification(
        shape_type="house",
        lines=[IdentificationLine(slope=0, intercept=0.1, label="not_symmetry")],
    )
    file_name = generate_symmetry_identification_task(symmetry_task)
    assert os.path.exists(file_name)
