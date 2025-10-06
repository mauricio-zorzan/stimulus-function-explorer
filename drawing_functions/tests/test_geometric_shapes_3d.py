import os
from typing import List

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes_3d import (
    draw_cross_section_question,
    draw_multiple_3d_objects,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.three_dimensional_objects import (
    Cone,
    CrossSectionQuestion,
    Cube,
    Cylinder,
    Pyramid,
    RectangularPrism,
    Sphere,
    ThreeDimensionalObjectsList,
)


@pytest.fixture
def sample_3d_shapes():
    return ThreeDimensionalObjectsList(
        shapes=[
            Sphere(shape="sphere", label="Sphere", radius=3),
            Pyramid(shape="pyramid", label="Pyramid", side=5, height=5),
            RectangularPrism(
                shape="rectangular prism",
                label="Rectangular Prism",
                height=5,
                width=5,
                length=5,
            ),
            Cone(shape="cone", label="Cone", height=5, radius=3),
            Cylinder(shape="cylinder", label="Cylinder", height=5, radius=3),
            Cube(shape="cube", label="Cube", height=5, width=5, length=5),
        ],
        units="cm",
    )


@pytest.fixture
def single_rectangular_prism():
    return ThreeDimensionalObjectsList(
        shapes=[
            RectangularPrism(
                shape="rectangular prism",
                label="Figure 1",
                height=7,
                width=8,
                length=10,
            )
        ],
        units="units",
    )


@pytest.fixture
def mixed_units_shapes():
    return ThreeDimensionalObjectsList(
        shapes=[
            Sphere(shape="sphere", label="A", radius=4),
            Cylinder(shape="cylinder", label="B", height=6, radius=3),
        ],
        units="m",
    )


@pytest.mark.drawing_functions
def test_draw_multiple_3d_objects(sample_3d_shapes):
    """Test drawing multiple 3D objects with dimensional labels."""
    file_name = draw_multiple_3d_objects(sample_3d_shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_single_rectangular_prism_with_labels(single_rectangular_prism):
    """Test drawing a single rectangular prism with specific dimensions."""
    file_name = draw_multiple_3d_objects(single_rectangular_prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_mixed_units_shapes(mixed_units_shapes):
    """Test drawing shapes with different unit specifications."""
    file_name = draw_multiple_3d_objects(mixed_units_shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_sphere_only():
    """Test drawing only a sphere with radius label."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[Sphere(shape="sphere", label="Sphere A", radius=8)], units="inches"
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_cylinder_only():
    """Test drawing only a cylinder with height and radius labels."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[Cylinder(shape="cylinder", label="Cylinder B", height=10, radius=4)],
        units="ft",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_cone_only():
    """Test drawing only a cone with height and radius labels."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[Cone(shape="cone", label="Cone C", height=8, radius=5)], units="cm"
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_pyramid_only():
    """Test drawing only a pyramid with side and height labels."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[Pyramid(shape="pyramid", label="Pyramid D", side=6, height=9)],
        units="m",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_cube_only():
    """Test drawing only a cube with all dimension labels."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[Cube(shape="cube", label="Cube E", height=7, width=7, length=7)],
        units="units",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_minimum_dimensions():
    """Test drawing shapes with minimum allowed dimensions."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            Sphere(shape="sphere", label="Min Sphere", radius=3),
            RectangularPrism(
                shape="rectangular prism",
                label="Min Prism",
                height=3,
                width=3,
                length=3,
            ),
        ],
        units="mm",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_maximum_dimensions():
    """Test drawing shapes with maximum allowed dimensions."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            Sphere(shape="sphere", label="Max Sphere", radius=10),
            Cylinder(shape="cylinder", label="Max Cylinder", height=10, radius=10),
        ],
        units="km",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_nine_shapes_maximum():
    """Test drawing the maximum number of shapes (9)."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            Sphere(shape="sphere", label="S1", radius=3),
            Sphere(shape="sphere", label="S2", radius=4),
            Sphere(shape="sphere", label="S3", radius=5),
            Cone(shape="cone", label="C1", height=5, radius=3),
            Cone(shape="cone", label="C2", height=6, radius=4),
            Cone(shape="cone", label="C3", height=7, radius=5),
            Cube(shape="cube", label="Cube1", height=4, width=4, length=4),
            Cube(shape="cube", label="Cube2", height=5, width=5, length=5),
            Cube(shape="cube", label="Cube3", height=6, width=6, length=6),
        ],
        units="units",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_rectangular_prisms_different_ratios():
    """Test drawing rectangular prisms with different aspect ratios."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            RectangularPrism(
                shape="rectangular prism",
                label="Tall",
                height=10,
                width=3,
                length=3,
            ),
            RectangularPrism(
                shape="rectangular prism",
                label="Wide",
                height=3,
                width=10,
                length=3,
            ),
            RectangularPrism(
                shape="rectangular prism",
                label="Long",
                height=3,
                width=3,
                length=10,
            ),
        ],
        units="cm",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_default_units():
    """Test drawing shapes with default units."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            RectangularPrism(
                shape="rectangular prism",
                label="Default Units",
                height=5,
                width=6,
                length=7,
            )
        ]
        # No units specified, should use default
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_shapes_with_partial_dimensions():
    """Test drawing shapes where some dimensions are None (should use defaults)."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            Sphere(
                shape="sphere", label="Sphere Default", radius=None
            ),  # Should use default
            Pyramid(
                shape="pyramid", label="Pyramid Partial", side=7, height=None
            ),  # Should use default height
        ],
        units="mm",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_custom_labels():
    """Test drawing shapes with custom labels."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            RectangularPrism(
                shape="rectangular prism",
                label="Building Block",
                height=4,
                width=8,
                length=6,
            ),
            Sphere(shape="sphere", label="Ball", radius=3),
            Cylinder(shape="cylinder", label="Can", height=5, radius=3),
        ],
        units="inches",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_edge_case_single_shape():
    """Test edge case with single shape minimum configuration."""
    shapes = ThreeDimensionalObjectsList(
        shapes=[
            RectangularPrism(
                shape="rectangular prism",
                label="Solo",
                height=3,
                width=3,
                length=3,
            )
        ],
        units="units",
    )
    file_name = draw_multiple_3d_objects(shapes)
    assert os.path.exists(file_name)


# Cross-section question tests
@pytest.mark.drawing_functions
def test_cross_section_sphere():
    """Test cross-section question with a sphere."""
    question = CrossSectionQuestion(
        shape=Sphere(shape="sphere", label="Test Sphere", radius=5),
        correct_cross_section="circle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_cylinder():
    """Test cross-section question with a cylinder."""
    question = CrossSectionQuestion(
        shape=Cylinder(shape="cylinder", label="Test Cylinder", height=8, radius=4),
        correct_cross_section="circle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_cone():
    """Test cross-section question with a cone."""
    question = CrossSectionQuestion(
        shape=Cone(shape="cone", label="Test Cone", height=10, radius=6),
        correct_cross_section="circle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_pyramid():
    """Test cross-section question with a pyramid."""
    question = CrossSectionQuestion(
        shape=Pyramid(shape="pyramid", label="Test Pyramid", side=7, height=9),
        correct_cross_section="square",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_rectangular_prism():
    """Test cross-section question with a rectangular prism."""
    question = CrossSectionQuestion(
        shape=RectangularPrism(
            shape="rectangular prism", label="Test Prism", height=5, width=6, length=4
        ),
        correct_cross_section="rectangle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_cube():
    """Test cross-section question with a cube."""
    question = CrossSectionQuestion(
        shape=Cube(shape="cube", label="Test Cube", height=6, width=6, length=6),
        correct_cross_section="square",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_with_default_units():
    """Test cross-section question with default units."""
    question = CrossSectionQuestion(
        shape=Sphere(shape="sphere", label="Default Units Sphere", radius=4),
        correct_cross_section="circle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_minimum_dimensions():
    """Test cross-section question with minimum allowed dimensions."""
    question = CrossSectionQuestion(
        shape=Sphere(shape="sphere", label="Min Sphere", radius=3),
        correct_cross_section="circle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_maximum_dimensions():
    """Test cross-section question with maximum allowed dimensions."""
    question = CrossSectionQuestion(
        shape=Cylinder(shape="cylinder", label="Max Cylinder", height=10, radius=10),
        correct_cross_section="circle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_multiple_shapes():
    """Test multiple cross-section questions with different shapes."""
    shapes_and_cross_sections = [
        (Sphere(shape="sphere", label="Sphere A", radius=5), "circle"),
        (Cylinder(shape="cylinder", label="Cylinder B", height=8, radius=4), "circle"),
        (Cone(shape="cone", label="Cone C", height=10, radius=6), "circle"),
        (Pyramid(shape="pyramid", label="Pyramid D", side=7, height=9), "square"),
    ]

    results: List[str] = []
    for shape, cross_section in shapes_and_cross_sections:
        question = CrossSectionQuestion(
            shape=shape, correct_cross_section=cross_section, correct_letter="a"
        )
        result = draw_cross_section_question(question)
        results.append(result)
        assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_different_units():
    """Test cross-section questions with different unit specifications."""
    units_tests = [
        ("a", Sphere(shape="sphere", label="CM Sphere", radius=5), "circle"),
        (
            "a",
            Cylinder(shape="cylinder", label="Inch Cylinder", height=8, radius=4),
            "circle",
        ),
        ("a", Cone(shape="cone", label="Foot Cone", height=10, radius=6), "circle"),
        (
            "a",
            Pyramid(shape="pyramid", label="Meter Pyramid", side=7, height=9),
            "square",
        ),
        (
            "a",
            Cube(shape="cube", label="Unit Cube", height=6, width=6, length=6),
            "square",
        ),
    ]

    results: List[str] = []
    for correct_letter, shape, cross_section in units_tests:
        question = CrossSectionQuestion(
            shape=shape,
            correct_cross_section=cross_section,
            correct_letter=correct_letter,
        )
        result = draw_cross_section_question(question)
        results.append(result)
        assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_different_positions():
    """Test cross-section questions with different answer positions."""
    question = CrossSectionQuestion(
        shape=Sphere(shape="sphere", label="Test Sphere", radius=5),
        correct_cross_section="circle",
        correct_letter="c",  # Place correct answer in position c
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_rectangular_prism_vertical():
    """Test cross-section question with a rectangular prism and vertical cut (square cross-section)."""
    question = CrossSectionQuestion(
        shape=RectangularPrism(
            shape="rectangular prism",
            label="Test Prism Vertical",
            height=5,
            width=6,
            length=4,
        ),
        correct_cross_section="square",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_cylinder_vertical():
    """Test cross-section question with a cylinder and vertical cut (rectangle cross-section)."""
    question = CrossSectionQuestion(
        shape=Cylinder(
            shape="cylinder", label="Test Cylinder Vertical", height=8, radius=4
        ),
        correct_cross_section="rectangle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)


@pytest.mark.drawing_functions
def test_cross_section_cone_vertical():
    """Test cross-section question with a cone and vertical cut (triangle cross-section)."""
    question = CrossSectionQuestion(
        shape=Cone(shape="cone", label="Test Cone Vertical", height=10, radius=6),
        correct_cross_section="triangle",
        correct_letter="a",
    )
    result: str = draw_cross_section_question(question)
    assert os.path.exists(result)
