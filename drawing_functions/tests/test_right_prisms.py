import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes_3d import (
    draw_right_prisms,
)
from content_generators.additional_content.stimulus_image.drawing_functions.rectangular_prisms import (
    draw_multiple_rectangular_prisms,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.right_prisms import (
    CubeRight,
    HexagonalPrism,
    IrregularPrism,
    OctagonalPrism,
    PentagonalPrism,
    RectangularPrismRight,
    RightPrismsList,
    TrapezoidalPrism,
    TriangularPrism,
)


@pytest.mark.drawing_functions
def test_plot_with_tall_prisms():
    stimulus_description = [
        {"title": "Prism A", "height": 9, "width": 3, "length": 6, "fill": "empty"},
        {"title": "Prism B", "height": 9, "width": 4, "length": 5, "fill": "empty"},
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_cube():
    """Test drawing a single cube right prism"""
    prisms = RightPrismsList(
        prisms=[
            CubeRight(
                label="Test Cube",
                height=5.0,
                side_length=4.0,
                base_area=None,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_rectangular():
    """Test drawing a single rectangular right prism"""
    prisms = RightPrismsList(
        prisms=[
            RectangularPrismRight(
                label="Test Rectangular Prism",
                height=6.0,
                width=4.0,
                length=5.0,
                base_area=None,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_rectangular_base_area():
    """Test drawing a rectangular prism using base area instead of dimensions"""
    prisms = RightPrismsList(
        prisms=[
            RectangularPrismRight(
                label="Test Rectangular Prism (Base Area)",
                height=6.0,
                width=None,
                length=None,
                base_area=20.0,  # Will calculate width=length=√20 ≈ 4.47
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_cube_base_area():
    """Test drawing a cube using base area instead of side length"""
    prisms = RightPrismsList(
        prisms=[
            CubeRight(
                label="Test Cube (Base Area)",
                height=5.0,
                side_length=None,
                base_area=16.0,  # Will calculate side_length=√16=4
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular():
    """Test drawing a single triangular right prism"""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Test Triangular Prism",
                height=5.0,
                side_a=3.0,
                side_b=4.0,
                side_c=5.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_not_right():
    """Test drawing a single triangular right prism"""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Test Triangular Prism",
                height=5.0,
                side_a=3.0,
                side_b=6.0,
                side_c=6.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_right_small():
    """Test drawing a small right triangular prism (3-4-5 triangle)"""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Small Right Triangle Prism",
                height=3.0,
                side_a=3.0,
                side_b=4.0,
                side_c=5.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_right_large():
    """Test drawing a large right triangular prism (4.8-6.4-8 triangle)"""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Large Right Triangle Prism",
                height=10.0,
                side_a=4.8,
                side_b=6.4,
                side_c=8.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_isosceles_small():
    """Test drawing a small isosceles triangular prism"""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Small Isosceles Triangle Prism",
                height=3.0,
                side_a=3.0,
                side_b=2.0,
                side_c=2.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_isosceles_large():
    """Test drawing a large isosceles triangular prism"""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Large Isosceles Triangle Prism",
                height=10.0,
                side_a=7.0,
                side_b=7.0,
                side_c=8.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_medium():
    """Test drawing a triangular prism with medium difficulty (dimensions < 20)."""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Medium Triangular Prism",
                height=15,
                side_a=12,
                side_b=16,
                side_c=19,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)
    print(f"\nGenerated medium triangular prism image at: {filename}")


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_hard():
    """Test drawing a triangular prism with hard difficulty (20 ≤ dimensions ≤ 40)."""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Hard Triangular Prism",
                height=36,
                side_a=21,
                side_b=28,
                side_c=35,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)
    print(f"\nGenerated hard triangular prism image at: {filename}")


@pytest.mark.drawing_functions
def test_draw_right_prisms_triangular_right_issue_case():
    """Regression: right-angle markers must stay inside for 21-28-35 triangle, height 26."""
    prisms = RightPrismsList(
        prisms=[
            TriangularPrism(
                label="Issue Case Right Triangle Prism",
                height=26.0,
                side_a=21.0,
                side_b=28.0,
                side_c=35.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_hexagonal():
    """Test drawing a single hexagonal right prism"""
    prisms = RightPrismsList(
        prisms=[
            HexagonalPrism(
                label="Test Hexagonal Prism",
                height=5.0,
                side_length=3.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_pentagonal():
    """Test drawing a single pentagonal right prism"""
    prisms = RightPrismsList(
        prisms=[
            PentagonalPrism(
                label="Test Pentagonal Prism",
                height=5.0,
                side_length=3.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_octagonal():
    """Test drawing a single octagonal right prism"""
    prisms = RightPrismsList(
        prisms=[
            OctagonalPrism(
                label="Test Octagonal Prism",
                height=5.0,
                side_length=3.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_trapezoidal():
    """Test drawing a single trapezoidal right prism"""
    prisms = RightPrismsList(
        prisms=[
            TrapezoidalPrism(
                label="Test Trapezoidal Prism",
                height=5.0,
                top_base=3.0,
                bottom_base=6.0,
                trapezoid_height=4.0,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_irregular():
    """Test drawing a single irregular right prism"""
    prisms = RightPrismsList(
        prisms=[
            IrregularPrism(
                label="Test Irregular Prism",
                height=5.0,
                base_vertices=[[0, 0], [4, 0], [3, 3], [1, 4]],
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_multiple():
    """Test drawing multiple different types of right prisms"""
    prisms = RightPrismsList(
        prisms=[
            CubeRight(label="Cube", height=4.0, side_length=4.0, base_area=None),
            RectangularPrismRight(
                label="Rectangular Prism",
                height=5.0,
                width=3.0,
                length=6.0,
                base_area=None,
            ),
            TriangularPrism(
                label="Triangular Prism",
                height=4.0,
                side_a=3.0,
                side_b=4.0,
                side_c=5.0,
            ),
            HexagonalPrism(
                label="Hexagonal Prism",
                height=4.0,
                side_length=2.5,
            ),
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_all_types():
    """Test drawing all types of right prisms in one grid"""
    prisms = RightPrismsList(
        prisms=[
            CubeRight(label="Cube", height=4.0, side_length=4.0, base_area=None),
            RectangularPrismRight(
                label="Rectangular",
                height=5.0,
                width=3.0,
                length=6.0,
                base_area=None,
            ),
            TriangularPrism(
                label="Triangular",
                height=4.0,
                side_a=3.0,
                side_b=4.0,
                side_c=5.0,
            ),
            HexagonalPrism(label="Hexagonal", height=4.0, side_length=2.5),
            PentagonalPrism(
                label="Pentagonal",
                height=4.0,
                side_length=2.5,
            ),
            OctagonalPrism(label="Octagonal", height=4.0, side_length=2.0),
            TrapezoidalPrism(
                label="Trapezoidal",
                height=4.0,
                top_base=2.0,
                bottom_base=4.0,
                trapezoid_height=3.0,
            ),
            IrregularPrism(
                label="Irregular",
                height=4.0,
                base_vertices=[[0, 0], [3, 0], [2.5, 2], [0.5, 2.5]],
            ),
            TriangularPrism(
                label="Triangular 2",
                height=9.0,
                side_a=6.0,
                side_b=7.0,
                side_c=8.0,
            ),
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_right_prisms_cube_3cm_labels():
    prisms = RightPrismsList(
        prisms=[
            CubeRight(
                label="Test Cube",
                height=3.0,
                side_length=3.0,  # cube side
                base_area=None,
            )
        ],
        units="cm",
    )
    filename = draw_right_prisms(prisms)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_multiple_rectangular_prisms_cube_like():
    stimulus_description = [
        {
            "title": "Cube",
            "height": 3,
            "width": 3,
            "length": 3,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
        {
            "title": "Rect",
            "height": 3,
            "width": 4,
            "length": 5,
            "fill": "empty",
            "prism_unit_label": "cm",
        },
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)


@pytest.mark.drawing_functions
def test_draw_multiple_rectangular_prisms_single_cube_7cm():
    stimulus_description = [
        {
            "title": "Cube",
            "height": 7,
            "width": 7,
            "length": 7,
            "fill": "empty",
            "prism_unit_label": "cm",
            "unit_cube_unit_size_and_label": None,
            "show_length": True,
            "show_width": True,
            "show_height": True,
        }
    ]
    filename = draw_multiple_rectangular_prisms(stimulus_description)
    assert os.path.exists(filename)
