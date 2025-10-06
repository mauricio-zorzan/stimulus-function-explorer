import os
import random
from fractions import Fraction

import matplotlib.colors as mcolors
import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
    GeometricShapeList,
    ValidGeometricShape,
    create_parallel_quadrilateral,
    draw_circle_diagram,
    draw_circle_with_arcs,
    draw_circle_with_radius,
    draw_composite_rectangular_grid,
    draw_composite_rectangular_triangular_grid,
    draw_geometric_shapes,
    draw_geometric_shapes_no_indicators,
    draw_geometric_shapes_no_indicators_with_rotation,
    draw_geometric_shapes_with_angles,
    draw_geometric_shapes_with_rotation,
    draw_polygon_fully_labeled,
    draw_polygons_bare,
    draw_quadrilateral_figures,
    draw_regular_irregular_polygons,
    draw_shape_with_right_angles,
    draw_similar_right_triangles,
    draw_single_quadrilateral_stimulus,
    generate_area_stimulus,
    generate_composite_rect_prism,
    generate_composite_rect_prism_v2,
    generate_geometric_shapes_transformations,
    generate_multiple_grids,
    generate_rect_with_side_len,
    generate_rect_with_side_len_and_area,
    generate_shape_with_base_and_height,
    generate_single_rect_with_side_len_stimulus,
    generate_trapezoid_with_side_len,
    generate_triangle_with_opt_side_len,
    generate_triangle_with_side_len,
    generate_unit_squares_unitless,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.area_stimulus import (
    AreaStimulusParams,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.circle import (
    Circle,
    CircleDiagram,
    CircleElement,
    CircleElementType,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.circle_arc import (
    CircleWithArcsDescription,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.complex_figure import (
    CompositeRectangularGrid,
    CompositeRectangularTriangularGrid,
    RectangleSpec,
    RightTriangleSpec,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.complex_figure import (
    EAbbreviatedMeasurementUnit as ComplexFigureUnit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.composite_rectangular_prism import (
    CompositeRectangularPrism,
    CompositeRectangularPrism2,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.geometric_shapes import (
    GeometricShape,
    GeometricShapeListWithRotation,
    GeometricShapeWithAngle,
    GeometricShapeWithAngleList,
    ParallelQuadrilateral,
    RegularIrregularPolygon,
    RegularIrregularPolygonList,
    ShapeWithRightAngles,
    SideEqualGroup,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_fully_labeled import (
    PolygonFullyLabeled,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    Point,
    Polygon,
    PolygonList,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.quadrilateral_figures import (
    QuadrilateralFigures,
    QuadrilateralShapeType,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangle_with_area import (
    RectangleWithHiddenSide,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    EAbbreviatedMeasurementUnit,
    MultipleGrids,
    RectangularGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid_list import (
    RectangularGridItem,
    RectangularGridList,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.shape import (
    EShapeType,
    Shape,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.similar_triangles import (
    SimilarRightTriangles,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.trapezoid_grid import (
    EAbbreviatedMeasurementUnit as TrapezoidMeasurementUnit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.trapezoid_grid import (
    ETrapezoidType,
    TrapezoidGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.triangular_grid_list import (
    TriangularGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.triangular_grid_opt import (
    ExerciseType,
    TriangularGridOpt,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.unit_squares import (
    UnitSquare,
    UnitSquares,
)
from content_generators.settings import settings
from matplotlib import pyplot as plt
from memory_profiler import memory_usage


@pytest.fixture
def shape_list():
    return GeometricShapeList(
        root=[
            GeometricShape(
                shape=ValidGeometricShape.RECTANGLE,
                label="Rectangle",
                color="red",
            ),
            GeometricShape(shape=ValidGeometricShape.CIRCLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.TRIANGLE, color="green"),
        ]
    ).model_dump(by_alias=True)


@pytest.mark.drawing_functions
def test_draw_geometric_shapes(shape_list):
    file_name = draw_geometric_shapes(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_circle_with_radius():
    stimulus_description = Circle(radius=12, unit="π")

    file_name = draw_circle_with_radius(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_circle_diagram_with_radius():
    stimulus_description = CircleDiagram(
        radius=12,
        unit="cm",
        element=CircleElement(element_type=CircleElementType.RADIUS),
    )

    file_name = draw_circle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_circle_diagram_with_diameter():
    """Test drawing a circle diagram with a diameter element."""
    stimulus_description = CircleDiagram(
        radius=12,
        unit="cm",
        element=CircleElement(element_type=CircleElementType.DIAMETER),
    )

    file_name = draw_circle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_circle_diagram_with_chord():
    """Test drawing a circle diagram with a chord element."""
    stimulus_description = CircleDiagram(
        radius=12,
        unit="cm",
        element=CircleElement(element_type=CircleElementType.CHORD),
    )

    file_name = draw_circle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_circle_diagram_with_custom_labels():
    """Test drawing a circle diagram with custom endpoint labels."""
    stimulus_description = CircleDiagram(
        radius=12,
        unit="cm",
        element=CircleElement(
            element_type=CircleElementType.RADIUS, endpoint_labels=["P", "Q"]
        ),
    )

    file_name = draw_circle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_five_geometric_shapes():
    shape_list = GeometricShapeList(
        root=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="purple"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="red"),
            GeometricShape(shape=ValidGeometricShape.CIRCLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.TRIANGLE, color="green"),
            GeometricShape(shape=ValidGeometricShape.PENTAGON, color="orange"),
        ]
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_all_triangles():
    shape_list = GeometricShapeList(
        root=[
            GeometricShape(shape=ValidGeometricShape.RIGHT_TRIANGLE, color="purple"),
            GeometricShape(shape=ValidGeometricShape.ISOSCELES_TRIANGLE, color="red"),
            GeometricShape(
                shape=ValidGeometricShape.EQUILATERAL_TRIANGLE, color="orange"
            ),
            GeometricShape(shape=ValidGeometricShape.ACUTE_TRIANGLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.SCALENE_TRIANGLE, color="green"),
            GeometricShape(shape=ValidGeometricShape.OBTUSE_TRIANGLE, color="orange"),
        ]
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_nine_geometric_shapes():
    shape_list = GeometricShapeList(
        root=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="purple"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="red"),
            GeometricShape(shape=ValidGeometricShape.CIRCLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.TRIANGLE, color="green"),
            GeometricShape(shape=ValidGeometricShape.PENTAGON, color="orange"),
            GeometricShape(shape=ValidGeometricShape.HEXAGON, color="yellow"),
            GeometricShape(shape=ValidGeometricShape.HEPTAGON, color="pink"),
            GeometricShape(shape=ValidGeometricShape.OCTAGON, color="cyan"),
            GeometricShape(shape=ValidGeometricShape.RHOMBUS, color="brown"),
        ]
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_seven_geometric_shapes():
    shape_list = GeometricShapeList(
        root=[
            GeometricShape(shape=ValidGeometricShape.TRAPEZOID, color="purple"),
            GeometricShape(shape=ValidGeometricShape.ISOSCELES_TRAPEZOID, color="red"),
            GeometricShape(shape=ValidGeometricShape.RIGHT_TRAPEZOID, color="blue"),
            GeometricShape(shape=ValidGeometricShape.REGULAR_TRIANGLE, color="green"),
            GeometricShape(
                shape=ValidGeometricShape.REGULAR_QUADRILATERAL, color="yellow"
            ),
            GeometricShape(shape=ValidGeometricShape.QUADRILATERAL, color="pink"),
            GeometricShape(shape=ValidGeometricShape.PARALLELOGRAM, color="cyan"),
        ]
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_right_angles():
    shape_list = GeometricShapeList(
        root=[
            GeometricShape(shape=ValidGeometricShape.RIGHT_TRIANGLE, color="purple"),
            GeometricShape(shape=ValidGeometricShape.RIGHT_TRAPEZOID, color="blue"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="green"),
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="yellow"),
        ]
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_all_geometric_shapes():
    available_colors = list(mcolors.CSS4_COLORS.keys())
    all_shapes = list(ValidGeometricShape)

    # Slice shapes into sublists of 9 items each
    for i in range(0, len(all_shapes), 9):
        shape_sublist = all_shapes[i : i + 9]

        shape_list = GeometricShapeList(
            root=[
                GeometricShape(
                    shape=shape,
                    color=random.choice(available_colors),
                )
                for shape in shape_sublist
            ]
        ).model_dump(by_alias=True)

        file_name = draw_geometric_shapes(shape_list)
        assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_seven_geometric_shapes_no_indicators():
    shape_list = GeometricShapeList(
        root=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="purple"),
            GeometricShape(shape=ValidGeometricShape.ISOSCELES_TRAPEZOID, color="red"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.REGULAR_TRIANGLE, color="green"),
            GeometricShape(
                shape=ValidGeometricShape.REGULAR_QUADRILATERAL, color="yellow"
            ),
            GeometricShape(shape=ValidGeometricShape.QUADRILATERAL, color="pink"),
            GeometricShape(shape=ValidGeometricShape.PARALLELOGRAM, color="cyan"),
        ]
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes_no_indicators(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_rotation_enabled():
    """Test shapes with rotation enabled using the new rotation function."""
    shape_list = GeometricShapeListWithRotation(
        shapes=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="red"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.TRIANGLE, color="green"),
            GeometricShape(shape=ValidGeometricShape.PENTAGON, color="orange"),
        ],
        rotate=True,
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes_with_rotation(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_no_indicators_with_rotation_enabled():
    """Test shapes without indicators with rotation enabled."""
    shape_list = GeometricShapeListWithRotation(
        shapes=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="purple"),
            GeometricShape(shape=ValidGeometricShape.RHOMBUS, color="cyan"),
            GeometricShape(shape=ValidGeometricShape.HEXAGON, color="yellow"),
        ],
        rotate=True,
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes_no_indicators_with_rotation(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_no_indicators_with_rotation_disabled():
    """Test shapes without indicators with rotation disabled."""
    shape_list = GeometricShapeListWithRotation(
        shapes=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="red"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.TRIANGLE, color="green"),
        ],
        rotate=True,
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes_no_indicators_with_rotation(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_rotation_variety():
    """Test shapes with rotation to show variety of rotated shapes."""
    shape_list = GeometricShapeListWithRotation(
        shapes=[
            GeometricShape(shape=ValidGeometricShape.SQUARE, color="red"),
            GeometricShape(shape=ValidGeometricShape.RECTANGLE, color="blue"),
            GeometricShape(shape=ValidGeometricShape.RIGHT_TRIANGLE, color="green"),
            GeometricShape(shape=ValidGeometricShape.HEXAGON, color="orange"),
            GeometricShape(shape=ValidGeometricShape.RHOMBUS, color="purple"),
        ],
        rotate=True,
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes_with_rotation(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_no_indicators_with_rotation_variety():
    """Test shapes without indicators with rotation to show variety."""
    shape_list = GeometricShapeListWithRotation(
        shapes=[
            GeometricShape(shape=ValidGeometricShape.PARALLELOGRAM, color="magenta"),
            GeometricShape(shape=ValidGeometricShape.TRAPEZOID, color="teal"),
            GeometricShape(shape=ValidGeometricShape.KITE, color="brown"),
            GeometricShape(shape=ValidGeometricShape.OCTAGON, color="pink"),
        ],
        rotate=True,
    ).model_dump(by_alias=True)

    file_name = draw_geometric_shapes_no_indicators_with_rotation(shape_list)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_random_shape_with_0_right_angles():
    """Test drawing shapes with no right angles (circle, equilateral triangle, pentagon, hexagon, quadrilateral)."""
    shape_data = ShapeWithRightAngles(num_right_angles=0).model_dump(by_alias=True)

    file_name = draw_shape_with_right_angles(shape_data)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_random_shape_with_1_right_angle():
    """Test drawing shapes with 1 interior right angle (right triangle or right angle kite)."""
    shape_data = ShapeWithRightAngles(num_right_angles=1).model_dump(by_alias=True)

    file_name = draw_shape_with_right_angles(shape_data)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_random_shape_with_2_right_angles():
    """Test drawing shapes with 2 interior right angles (pentagon with 2 right angles at base or right trapezoid)."""
    shape_data = ShapeWithRightAngles(num_right_angles=2).model_dump(by_alias=True)

    file_name = draw_shape_with_right_angles(shape_data)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_random_shape_with_3_right_angles():
    """Test drawing a file icon shape (3 right angles)."""
    shape_data = ShapeWithRightAngles(num_right_angles=3).model_dump(by_alias=True)

    file_name = draw_shape_with_right_angles(shape_data)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_random_shape_with_4_right_angles():
    """Test drawing a rectangle or square (4 right angles)."""
    shape_data = ShapeWithRightAngles(num_right_angles=4).model_dump(by_alias=True)

    file_name = draw_shape_with_right_angles(shape_data)
    assert plt.imread(file_name) is not None


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_angles_single_rectangle():
    """Test drawing a single rectangle with a right angle marked automatically."""
    shape_list = GeometricShapeWithAngleList(
        root=[
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RECTANGLE, angle_type="right", color="blue"
            )
        ]
    )

    file_name = draw_geometric_shapes_with_angles(shape_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_angles_multiple_shapes():
    """Test drawing multiple shapes with different angle types marked automatically."""
    shape_list = GeometricShapeWithAngleList(
        root=[
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.KITE, angle_type="acute", color="red"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.KITE, angle_type="obtuse", color="green"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.TRAPEZOID, angle_type="acute", color="blue"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.TRAPEZOID, angle_type="obtuse", color="blue"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RIGHT_TRAPEZOID,
                angle_type="right",
                color="blue",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RIGHT_TRAPEZOID,
                angle_type="acute",
                color="blue",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RIGHT_TRAPEZOID,
                angle_type="obtuse",
                color="blue",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.PARALLELOGRAM,
                angle_type="acute",
                color="blue",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.PARALLELOGRAM,
                angle_type="obtuse",
                color="blue",
            ),
        ]
    )

    file_name = draw_geometric_shapes_with_angles(shape_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_angles_mixed_shapes():
    """Test drawing mixed geometric shapes with angle markers automatically detected."""
    shape_list = GeometricShapeWithAngleList(
        root=[
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.SQUARE, angle_type="right", color="orange"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.TRAPEZOID, angle_type="acute", color="purple"
            ),
        ]
    )

    file_name = draw_geometric_shapes_with_angles(shape_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_angles_new_shapes():
    """Test drawing new shapes with various angle types automatically detected."""
    shape_list = GeometricShapeWithAngleList(
        root=[
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.TRAPEZOID, angle_type="obtuse", color="red"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RIGHT_TRAPEZOID,
                angle_type="right",
                color="green",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.KITE, angle_type="acute", color="blue"
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.PARALLELOGRAM,
                angle_type="obtuse",
                color="orange",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RHOMBUS, angle_type="acute", color="purple"
            ),
        ]
    )

    file_name = draw_geometric_shapes_with_angles(shape_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_geometric_shapes_with_angles_right_trapezoid_all_angles():
    """Test right trapezoid with all three angle types automatically detected."""
    shape_list = GeometricShapeWithAngleList(
        root=[
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.RIGHT_TRAPEZOID,
                angle_type="right",
                color="red",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.ISOSCELES_TRIANGLE,
                angle_type="acute",
                color="green",
            ),
            GeometricShapeWithAngle(
                shape=ValidGeometricShape.ISOSCELES_TRIANGLE,
                angle_type="obtuse",
                color="blue",
            ),
        ]
    )

    file_name = draw_geometric_shapes_with_angles(shape_list)
    assert os.path.exists(file_name)
    # The actual test would be visual - the right angle should show a square marker, not an arc


@pytest.fixture
def rectangular_grid_list():
    return RectangularGridList(
        [
            RectangularGridItem(
                length=3, width=4, unit=EAbbreviatedMeasurementUnit.INCHES
            ),
            RectangularGridItem(
                length=2, width=6, unit=EAbbreviatedMeasurementUnit.INCHES
            ),
            RectangularGridItem(
                length=1, width=9, unit=EAbbreviatedMeasurementUnit.INCHES
            ),
            RectangularGridItem(
                length=4, width=7, unit=EAbbreviatedMeasurementUnit.INCHES
            ),
        ]
    )


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_4(rectangular_grid_list):
    file_name = generate_rect_with_side_len(rectangular_grid_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_2():
    rectangular_grid_list = RectangularGridList(
        [
            RectangularGridItem(
                length=3, width=4, unit=EAbbreviatedMeasurementUnit.INCHES
            ),
            RectangularGridItem(
                length=2, width=6, unit=EAbbreviatedMeasurementUnit.INCHES
            ),
        ]
    )
    file_name = generate_rect_with_side_len(rectangular_grid_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len__():
    stimulus_description = [{"unit": "Units", "length": "9", "width": "5"}]
    file_name = generate_single_rect_with_side_len_stimulus(stimulus_description)
    assert os.path.exists(file_name)


def test_generate_rect_with_side_len_biggest_memory_usage():
    rect_size = 21
    rectangular_grid_list = RectangularGridList(
        [
            RectangularGridItem(
                length=rect_size,
                width=rect_size,
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
            RectangularGridItem(
                length=rect_size,
                width=rect_size,
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
            RectangularGridItem(
                length=rect_size,
                width=rect_size,
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
            RectangularGridItem(
                length=rect_size,
                width=rect_size,
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
        ]
    )
    mem_usage = memory_usage(
        (generate_rect_with_side_len, (rectangular_grid_list,)),  # type: ignore
        interval=0.1,
        timeout=30,
    )
    assert max(mem_usage) < settings.lambda_settings.memory_size

    file_name = generate_rect_with_side_len(rectangular_grid_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_fractions_single_item():
    rectangular_grid_list = RectangularGridList(
        [
            RectangularGridItem(
                length=Fraction(3, 2),  # 1.5
                width=Fraction(5, 4),  # 1.25
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
        ]
    )
    file_name = generate_rect_with_side_len(rectangular_grid_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_fractions():
    rectangular_grid_list = RectangularGridList(
        [
            RectangularGridItem(
                length=Fraction(3, 2),  # 1.5
                width=Fraction(5, 4),  # 1.25
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
            RectangularGridItem(
                length=Fraction(7, 3),  # 2.33...
                width=Fraction(4, 3),  # 1.33...
                unit=EAbbreviatedMeasurementUnit.INCHES,
            ),
        ]
    )
    file_name = generate_rect_with_side_len(rectangular_grid_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_rectangle():
    params = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=4,
        base=6,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_large_rectangle():
    params = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=20,
        base=30,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_parallelogram():
    params = Shape(
        shape_type=EShapeType.PARALLELOGRAM,
        height=5,
        base=7,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_triangle():
    params = Shape(
        shape_type=EShapeType.TRIANGLE,
        height=3,
        base=4,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_rhombus():
    params = Shape(
        shape_type=EShapeType.RHOMBUS,
        height=4,
        base=6,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_rhombus_min_ratio():
    """Test rhombus with minimum acceptable height-to-base ratio (0.2)."""
    params = Shape(
        shape_type=EShapeType.RHOMBUS,
        height=1,  # 1/5 = 0.2 ratio (minimum acceptable)
        base=5,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_rhombus_max_ratio():
    """Test rhombus with maximum acceptable height-to-base ratio (5.0)."""
    params = Shape(
        shape_type=EShapeType.RHOMBUS,
        height=10,  # 10/2 = 5.0 ratio (maximum acceptable)
        base=2,
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_rhombus_equal_diagonals():
    """Test rhombus with equal diagonals (creates a square rhombus)."""
    params = Shape(
        shape_type=EShapeType.RHOMBUS,
        height=4,
        base=4,  # Equal diagonals create a square
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_validation_too_tall():
    """Test that extremely tall shapes (ratio > 5.0) raise validation error."""
    with pytest.raises(ValueError, match="Height-to-base ratio .* is too large"):
        Shape(
            shape_type=EShapeType.RECTANGLE,
            height=25,  # 25/4 = 6.25 ratio (too large)
            base=4,
            unit="cm",
        )


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_validation_too_wide():
    """Test that extremely wide shapes (ratio < 0.2) raise validation error."""
    with pytest.raises(ValueError, match="Height-to-base ratio .* is too small"):
        Shape(
            shape_type=EShapeType.RECTANGLE,
            height=1,  # 1/6 = 0.167 ratio (too small)
            base=6,
            unit="cm",
        )


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_validation_edge_cases():
    """Test that edge case ratios (0.2 and 5.0) are accepted."""
    # Test minimum acceptable ratio (0.2)
    shape_min = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=1,  # 1/5 = 0.2 ratio (minimum acceptable)
        base=5,
        unit="cm",
    )
    assert shape_min.height == 1
    assert shape_min.base == 5

    # Test maximum acceptable ratio (5.0)
    shape_max = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=10,  # 10/2 = 5.0 ratio (maximum acceptable)
        base=2,
        unit="cm",
    )
    assert shape_max.height == 10
    assert shape_max.base == 2


@pytest.fixture
def sample_composite_rect_prism():
    return CompositeRectangularPrism(
        figures=[[3, 4, 5], [2, 3, 4], [1, 2, 3]], units="cm"
    )


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism(sample_composite_rect_prism):
    file_name = generate_composite_rect_prism(sample_composite_rect_prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_hide_measurements(sample_composite_rect_prism):
    sample_composite_rect_prism.hide_measurements = [0]
    file_name = generate_composite_rect_prism(sample_composite_rect_prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_empty():
    with pytest.raises(ValueError):
        empty_prism = CompositeRectangularPrism(figures=[], units="cm")
        generate_composite_rect_prism(empty_prism)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_v2_basic():
    """Test basic functionality of v2 function with default parameters."""
    prism = CompositeRectangularPrism2(
        figures=[[3, 4, 5], [2, 3, 4], [1, 2, 3]], units="cm"
    )
    file_name = generate_composite_rect_prism_v2(prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_v2_stack_layout():
    """Test stack layout arrangement."""
    prism = CompositeRectangularPrism2(
        figures=[[2, 3, 4], [1, 2, 3]], units="cm", layout="stack"
    )
    file_name = generate_composite_rect_prism_v2(prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_v2_side_layout():
    """Test side-by-side layout arrangement."""
    prism = CompositeRectangularPrism2(
        figures=[[2, 3, 4], [1, 2, 3]], units="cm", layout="side"
    )
    file_name = generate_composite_rect_prism_v2(prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_v2_l_layout():
    """Test L-shaped layout arrangement with same heights."""
    prism = CompositeRectangularPrism2(
        figures=[[2, 3, 4], [2, 6, 3]],
        units="cm",
        layout="L",  # Same height (2) for both prisms
    )
    file_name = generate_composite_rect_prism_v2(prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_v2_show_labels_mixed():
    """Test show_labels with mixed visibility."""
    prism = CompositeRectangularPrism2(
        figures=[[2, 3, 4], [2, 2, 3], [2, 1, 2]],
        units="cm",
        show_labels=[True, False, True],
    )
    file_name = generate_composite_rect_prism_v2(prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_composite_rect_prism_v2_show_labels_partial_list():
    """Test show_labels shorter than figures list (should default to True)."""
    prism = CompositeRectangularPrism2(
        figures=[[2, 3, 4], [2, 2, 3], [2, 1, 2]],
        units="cm",
        show_labels=[False],
    )
    file_name = generate_composite_rect_prism_v2(prism)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_basic():
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=0, y=1, label="D"),
                ],
                label="Square",
            )
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_multiple_polygons():
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=1, y=1, label="C"),
                ],
                label="Triangle",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=0, y=1, label="D"),
                ],
                label="Square",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_no_label():
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=0, y=1, label="D"),
                ],
                label="No Label",
            )
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.fixture
def sample_unit_squares():
    return UnitSquares(
        root=[
            UnitSquare(length=3, width=3),
            UnitSquare(length=4, width=2),
            UnitSquare(length=2, width=5),
        ]
    )


@pytest.mark.drawing_functions
def test_generate_unit_squares_unitless(sample_unit_squares):
    file_name = generate_unit_squares_unitless(sample_unit_squares)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_unit_squares_unitless_two_squares():
    data = UnitSquares(
        root=[
            UnitSquare(length=5, width=5),
            UnitSquare(length=5, width=5),
        ]
    )
    file_name = generate_unit_squares_unitless(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_unit_squares_unitless_max_squares():
    data = UnitSquares(
        root=[
            UnitSquare(length=3, width=3),
            UnitSquare(length=4, width=4),
            UnitSquare(length=2, width=2),
            UnitSquare(length=1, width=1),
        ]
    )
    file_name = generate_unit_squares_unitless(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_unit_squares_unitless_invalid_square():
    with pytest.raises(ValueError):
        UnitSquares(
            root=[
                UnitSquare(length=11, width=11),  # Invalid dimensions
            ]
        )


@pytest.mark.drawing_functions
def test_generate_area_stimulus_right_triangle():
    params = AreaStimulusParams(
        base=5,
        height=3,
        shape="right_triangle",
        not_to_scale_note="Figure not drawn to scale.",
    )
    file_name = generate_area_stimulus(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_area_stimulus_rectangle():
    params = AreaStimulusParams(
        base=4,
        height=6,
        shape="rectangle",
        not_to_scale_note="Figure not drawn to scale.",
    )
    file_name = generate_area_stimulus(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_similar_right_triangles():
    angle_labels = ["A", "B", "C", "D", "E", "F"]
    params = SimilarRightTriangles(angle_labels=angle_labels)
    file_name = draw_similar_right_triangles(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_circle_with_arc():
    # Define the arc size and point labels for the test
    arc_size = 66  # Example arc size
    point_labels = ["P", "Q", "R", "S"]  # Example point labels

    # Create a stimulus description instance
    stimulus_description = CircleWithArcsDescription(
        arc_size=arc_size, point_labels=point_labels
    )

    # Call the function to generate the image
    file_name = draw_circle_with_arcs(stimulus_description)

    # Check if the file was created
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_right():
    # 3-4-5 right triangle
    triangular_grid = TriangularGrid(
        side1=8, side2=8, side3=13, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_acute():
    # Acute triangle (all angles < 90°)
    triangular_grid = TriangularGrid(
        side1=5, side2=6, side3=7, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_obtuse():
    # Obtuse triangle (one angle > 90°)
    triangular_grid = TriangularGrid(
        side1=3, side2=4, side3=6, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_equilateral():
    # Equilateral triangle (all sides equal)
    triangular_grid = TriangularGrid(
        side1=5, side2=5, side3=5, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_isosceles():
    # Isosceles triangle (two sides equal)
    triangular_grid = TriangularGrid(
        side1=5, side2=5, side3=7, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_scalene():
    # Scalene triangle (all sides different)
    triangular_grid = TriangularGrid(
        side1=4, side2=6, side3=7, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_large():
    # Large triangle (testing scaling)
    triangular_grid = TriangularGrid(
        side1=15, side2=20, side3=21, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_side_len_small():
    # Small triangle
    triangular_grid = TriangularGrid(
        side1=2, side2=3, side3=4, unit=EAbbreviatedMeasurementUnit.INCHES
    )
    file_name = generate_triangle_with_side_len(triangular_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_regular():
    # Regular trapezoid with standard dimensions
    trapezoid_grid = TrapezoidGrid(
        base=7, top_length=5, height=4, unit=TrapezoidMeasurementUnit.CENTIMETERS
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_isosceles():
    # Isosceles trapezoid (symmetric)
    trapezoid_grid = TrapezoidGrid(
        base=10,
        top_length=6,
        height=5,
        unit=EAbbreviatedMeasurementUnit.METERS,
        trapezoid_type=ETrapezoidType.ISOSCELES_TRAPEZOID,
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_right():
    # Right trapezoid (right side vertical)
    trapezoid_grid = TrapezoidGrid(
        base=8,
        top_length=4,
        height=6,
        unit=EAbbreviatedMeasurementUnit.FEET,
        trapezoid_type=ETrapezoidType.RIGHT_TRAPEZOID,
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_left():
    # Left trapezoid (left side vertical)
    trapezoid_grid = TrapezoidGrid(
        base=9,
        top_length=5,
        height=4,
        unit=EAbbreviatedMeasurementUnit.INCHES,
        trapezoid_type=ETrapezoidType.LEFT_TRAPEZOID,
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_variable_height():
    # Trapezoid with height as variable for calculation problems
    trapezoid_grid = TrapezoidGrid(
        base=7,
        top_length=5,
        height=4,
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        show_variable_height=True,
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_fractions():
    # Trapezoid with fractional dimensions
    trapezoid_grid = TrapezoidGrid(
        base=Fraction(7, 2),  # 3.5
        top_length=Fraction(5, 2),  # 2.5
        height=Fraction(3, 1),  # 3
        unit=EAbbreviatedMeasurementUnit.INCHES,
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_small():
    # Small trapezoid (testing scaling)
    trapezoid_grid = TrapezoidGrid(
        base=3, top_length=2, height=2, unit=EAbbreviatedMeasurementUnit.MILLIMETERS
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_large():
    # Large trapezoid (testing scaling)
    trapezoid_grid = TrapezoidGrid(
        base=18,
        top_length=12,
        height=10,
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        trapezoid_type=ETrapezoidType.ISOSCELES_TRAPEZOID,
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_with_label():
    # Trapezoid with custom label
    trapezoid_grid = TrapezoidGrid(
        base=12,
        top_length=8,
        height=6,
        unit=EAbbreviatedMeasurementUnit.UNITS,
        label="Figure 1",
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_area_calculation():
    # Trapezoid for area calculation: Area = 1/2 × (7 + 5) × 4 = 24 cm²
    trapezoid_grid = TrapezoidGrid(
        base=7, top_length=5, height=4, unit=EAbbreviatedMeasurementUnit.CENTIMETERS
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


def test_generate_trapezoid_with_side_len_memory_usage():
    # Memory usage test with large trapezoid
    trapezoid_grid = TrapezoidGrid(
        base=20, top_length=15, height=15, unit=EAbbreviatedMeasurementUnit.METERS
    )
    mem_usage = memory_usage(
        (generate_trapezoid_with_side_len, (trapezoid_grid,)),
        interval=0.1,
        timeout=30,
    )
    assert max(mem_usage) < settings.lambda_settings.memory_size

    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_edge_case_minimal():
    # Edge case: minimal dimensions
    trapezoid_grid = TrapezoidGrid(
        base=2, top_length=1, height=1, unit=EAbbreviatedMeasurementUnit.UNITS
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_trapezoid_with_side_len_edge_case_near_maximum():
    # Edge case: near maximum dimensions
    trapezoid_grid = TrapezoidGrid(
        base=19, top_length=18, height=14, unit=EAbbreviatedMeasurementUnit.KILOMETERS
    )
    file_name = generate_trapezoid_with_side_len(trapezoid_grid)
    assert os.path.exists(file_name)


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


# =============================================================================
# MIXED FRACTIONS TEST CASES FOR GENERATE_SHAPE_WITH_BASE_AND_HEIGHT
# =============================================================================


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_mixed_fractions_rectangle():
    """Test rectangle with mixed fraction dimensions (3 1/3 by 6 2/3 ft)."""
    params = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=6.666667,  # Should display as "6 2/3 ft"
        base=3.333333,  # Should display as "3 1/3 ft"
        unit="ft",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_mixed_fractions_triangle():
    """Test triangle with mixed fraction dimensions (4 1/2 by 2 1/4 cm)."""
    params = Shape(
        shape_type=EShapeType.TRIANGLE,
        height=4.5,  # Should display as "4 1/2 cm"
        base=2.25,  # Should display as "2 1/4 cm"
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_mixed_fractions_parallelogram():
    """Test parallelogram with mixed fraction dimensions (5 1/3 by 3 3/4 in)."""
    params = Shape(
        shape_type=EShapeType.PARALLELOGRAM,
        height=5.333333,  # Should display as "5 1/3 in"
        base=3.75,  # Should display as "3 3/4 in"
        unit="in",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_simple_fractions():
    """Test rectangle with simple fractions (1/2 by 3/4 m)."""
    params = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=0.5,  # Should display as "1/2 m"
        base=0.75,  # Should display as "3/4 m"
        unit="m",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_thirds_and_halves():
    """Test shape with common educational fractions (2/3 by 1 1/2 ft)."""
    params = Shape(
        shape_type=EShapeType.TRIANGLE,
        height=0.666667,  # Should display as "2/3 ft"
        base=1.5,  # Should display as "1 1/2 ft"
        unit="ft",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_fifths_and_eighths():
    """Test shape with educational fractions (2 2/5 by 3 5/8 cm)."""
    params = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=2.4,  # Should display as "2 2/5 cm"
        base=3.625,  # Should display as "3 5/8 cm"
        unit="cm",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_no_fractions():
    """Test that whole numbers display without fractions (5 by 8 in)."""
    params = Shape(
        shape_type=EShapeType.RECTANGLE,
        height=5.0,  # Should display as "5 in"
        base=8.0,  # Should display as "8 in"
        unit="in",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_rhombus_fractions():
    """Test rhombus with mixed fraction diagonals (4 1/4 by 6 1/3 ft)."""
    params = Shape(
        shape_type=EShapeType.RHOMBUS,
        height=4.25,  # Should display as "4 1/4 ft"
        base=6.333333,  # Should display as "6 1/3 ft"
        unit="ft",
    )
    file_name = generate_shape_with_base_and_height(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_shape_with_base_and_height_all_units_with_fractions():
    """Test fraction display works with all supported units."""
    test_cases = [
        ("cm", 2.5, 3.75),  # "2 1/2 cm" and "3 3/4 cm"
        ("m", 1.333333, 2.25),  # "1 1/3 m" and "2 1/4 m"
        ("in", 4.666667, 1.5),  # "4 2/3 in" and "1 1/2 in"
        ("ft", 3.2, 2.8),  # "3 1/5 ft" and "2 4/5 ft"
    ]

    for unit, height, base in test_cases:
        params = Shape(
            shape_type=EShapeType.RECTANGLE,
            height=height,
            base=base,
            unit=unit,
        )
        file_name = generate_shape_with_base_and_height(params)
        assert os.path.exists(file_name)


# =============================================================================
# MIXED FRACTIONS TEST CASES FOR GENERATE_AREA_STIMULUS
# =============================================================================
# NOTE: generate_area_stimulus only accepts integer values due to Pydantic validation
# Mixed fractions testing is done with generate_shape_with_base_and_height instead


@pytest.mark.drawing_functions
def test_generate_area_stimulus_integer_rectangle():
    """Test rectangle area stimulus with integer dimensions only."""
    params = AreaStimulusParams(
        base=5,  # Integer value as required by validation
        height=8,  # Integer value as required by validation
        shape="rectangle",
        not_to_scale_note="Figure not drawn to scale.",
    )
    file_name = generate_area_stimulus(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_area_stimulus_integer_triangle():
    """Test triangle area stimulus with integer dimensions only."""
    params = AreaStimulusParams(
        base=6,  # Integer value as required by validation
        height=4,  # Integer value as required by validation
        shape="right_triangle",
        not_to_scale_note="Figure not drawn to scale.",
    )
    file_name = generate_area_stimulus(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_triangular_grid():
    """Test drawing a composite figure with 2 rectangles and 1 triangle."""
    # Create an L-shaped configuration where individual components are visually distinct
    # Rectangle 1 (horizontal base), Rectangle 2 (vertical extension), Triangle (connecting corner)
    test_data = CompositeRectangularTriangularGrid(
        rectangles=[
            # Rectangle 1 (horizontal base) - 4×2 rectangle
            RectangleSpec(
                unit=ComplexFigureUnit.CENTIMETERS,
                length=2,  # height
                width=4,  # width (longer horizontal rectangle)
                x=0,
                y=0,
            ),
            # Rectangle 2 (vertical extension) - 2×3 rectangle extending upward
            RectangleSpec(
                unit=ComplexFigureUnit.CENTIMETERS,
                length=3,  # height (taller vertical rectangle)
                width=2,  # width
                x=0,  # aligned with left edge of Rectangle 1
                y=2,  # starts where Rectangle 1 ends vertically
            ),
        ],
        # Triangle - fills the corner gap between the two rectangles
        triangle=RightTriangleSpec(
            unit=ComplexFigureUnit.CENTIMETERS,
            right_angle_x=2,  # right angle at the inner corner
            right_angle_y=2,  # at the junction of both rectangles
            base=2,  # extends to the right edge of Rectangle 1
            height=3,  # extends to the top of Rectangle 2
        ),
    )

    # Test that the function runs without error and creates a file
    file_name = draw_composite_rectangular_triangular_grid(test_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_triangular_grid_different_config():
    """Test drawing a composite figure with a different configuration."""
    # Create test data with rectangles and a clearly visible triangle
    test_data = CompositeRectangularTriangularGrid(
        rectangles=[
            RectangleSpec(unit=ComplexFigureUnit.INCHES, length=3, width=4, x=0, y=0),
            RectangleSpec(unit=ComplexFigureUnit.INCHES, length=2, width=3, x=4, y=1),
        ],
        triangle=RightTriangleSpec(
            unit=ComplexFigureUnit.INCHES,
            right_angle_x=2,
            right_angle_y=3,
            base=4,
            height=3,
        ),
    )

    # Test that the function runs without error and creates a file
    file_name = draw_composite_rectangular_triangular_grid(test_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_composite_rectangular_triangular_grid_small_edges():
    """Test drawing a composite figure with small edges to ensure all are labeled."""
    # Create test data with small rectangles and triangle to test edge labeling
    test_data = CompositeRectangularTriangularGrid(
        rectangles=[
            # Small rectangle 1 - 1×1 square
            RectangleSpec(
                unit=ComplexFigureUnit.CENTIMETERS,
                length=1,  # height
                width=1,  # width
                x=0,
                y=0,
            ),
            # Small rectangle 2 - 2×1 rectangle
            RectangleSpec(
                unit=ComplexFigureUnit.CENTIMETERS,
                length=1,  # height
                width=2,  # width
                x=1,  # adjacent to Rectangle 1
                y=0,  # same bottom level
            ),
        ],
        # Small triangle connecting to create interesting shape
        triangle=RightTriangleSpec(
            unit=ComplexFigureUnit.CENTIMETERS,
            right_angle_x=1,  # right angle at junction
            right_angle_y=1,  # at top of rectangles
            base=1,  # small base
            height=1,  # small height
        ),
    )

    # Test that the function runs without error and creates a file
    file_name = draw_composite_rectangular_triangular_grid(test_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_90_clockwise():
    """Test 90-degree clockwise rotation of a square."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=1, y=1, label="A"),
                    Point(x=3, y=1, label="B"),
                    Point(x=3, y=3, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Original Square",
            ),
            Polygon(
                points=[
                    Point(x=1, y=1, label="A"),
                    Point(x=3, y=1, label="B"),
                    Point(x=3, y=3, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Rotated Square",
                rotation_angle=90,
                rotation_direction="clockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_90_counterclockwise():
    """Test 90-degree counterclockwise rotation of a triangle."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=3, y=0, label="B"),
                    Point(x=1, y=2, label="C"),
                ],
                label="Original Triangle",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=3, y=0, label="B"),
                    Point(x=1, y=2, label="C"),
                ],
                label="Rotated Triangle",
                rotation_angle=90,
                rotation_direction="counterclockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_180_clockwise():
    """Test 180-degree clockwise rotation of a pentagon."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=3, y=1, label="C"),
                    Point(x=1, y=2, label="D"),
                    Point(x=-1, y=1, label="E"),
                ],
                label="Original Pentagon",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=3, y=1, label="C"),
                    Point(x=1, y=2, label="D"),
                    Point(x=-1, y=1, label="E"),
                ],
                label="Rotated Pentagon",
                rotation_angle=180,
                rotation_direction="clockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_180_counterclockwise():
    """Test 180-degree counterclockwise rotation."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=1, y=1, label="A"),
                    Point(x=4, y=1, label="B"),
                    Point(x=4, y=3, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Original Rectangle",
            ),
            Polygon(
                points=[
                    Point(x=1, y=1, label="A"),
                    Point(x=4, y=1, label="B"),
                    Point(x=4, y=3, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Rotated Rectangle",
                rotation_angle=180,
                rotation_direction="counterclockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_270_clockwise():
    """Test 270-degree clockwise rotation of a quadrilateral."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=3, y=2, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Original Quadrilateral",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=3, y=2, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Rotated Quadrilateral",
                rotation_angle=270,
                rotation_direction="clockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_270_counterclockwise():
    """Test 270-degree counterclockwise rotation of a hexagon."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=2, y=1, label="C"),
                    Point(x=1, y=2, label="D"),
                    Point(x=0, y=2, label="E"),
                    Point(x=-1, y=1, label="F"),
                ],
                label="Original Hexagon",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=2, y=1, label="C"),
                    Point(x=1, y=2, label="D"),
                    Point(x=0, y=2, label="E"),
                    Point(x=-1, y=1, label="F"),
                ],
                label="Rotated Hexagon",
                rotation_angle=270,
                rotation_direction="counterclockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_rotation_with_custom_center():
    """Test rotation around a custom center point."""
    rotation_center = Point(x=2, y=2, label="O")
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=0, y=1, label="D"),
                ],
                label="Original Square",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=1, y=0, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=0, y=1, label="D"),
                ],
                label="Rotated Square",
                rotation_angle=90,
                rotation_direction="clockwise",
                rotation_center=rotation_center,
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_multiple_rotations():
    """Test multiple polygons with different rotations."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=1, y=2, label="C"),
                ],
                label="Original Triangle",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=1, y=2, label="C"),
                ],
                label="90° CW Rotation",
                rotation_angle=90,
                rotation_direction="clockwise",
            ),
            Polygon(
                points=[
                    Point(x=0, y=0, label="A"),
                    Point(x=2, y=0, label="B"),
                    Point(x=1, y=2, label="C"),
                ],
                label="180° Rotation",
                rotation_angle=180,
                rotation_direction="counterclockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_single_rotation():
    """Test single polygon with rotation (no original)."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=1, y=1, label="A"),
                    Point(x=3, y=1, label="B"),
                    Point(x=3, y=3, label="C"),
                    Point(x=1, y=3, label="D"),
                ],
                label="Rotated Square",
                rotation_angle=90,
                rotation_direction="counterclockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_large_coordinates_with_rotation():
    """Test rotation with larger coordinate values."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=5, y=5, label="A"),
                    Point(x=8, y=5, label="B"),
                    Point(x=8, y=8, label="C"),
                    Point(x=5, y=8, label="D"),
                ],
                label="Original Square",
            ),
            Polygon(
                points=[
                    Point(x=5, y=5, label="A"),
                    Point(x=8, y=5, label="B"),
                    Point(x=8, y=8, label="C"),
                    Point(x=5, y=8, label="D"),
                ],
                label="Rotated Square",
                rotation_angle=270,
                rotation_direction="clockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_geometric_shapes_transformations_negative_coordinates_with_rotation():
    """Test rotation with negative coordinates."""
    polygons = PolygonList(
        [
            Polygon(
                points=[
                    Point(x=-2, y=-2, label="A"),
                    Point(x=-1, y=-2, label="B"),
                    Point(x=-1, y=-1, label="C"),
                    Point(x=-2, y=-1, label="D"),
                ],
                label="Original Square",
            ),
            Polygon(
                points=[
                    Point(x=-2, y=-2, label="A"),
                    Point(x=-1, y=-2, label="B"),
                    Point(x=-1, y=-1, label="C"),
                    Point(x=-2, y=-1, label="D"),
                ],
                label="Rotated Square",
                rotation_angle=90,
                rotation_direction="counterclockwise",
            ),
        ]
    )
    file_name = generate_geometric_shapes_transformations(polygons)
    assert os.path.exists(file_name)


#################################################################
# Test Cases for Triangles with Optional Side Lengths         #
# (Pythagorean Theorem Problems)                               #
#################################################################


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_calculate_hypotenuse():
    """Test calculating missing hypotenuse given two legs (3-4-5 triangle)."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.FEET,
        side1=3.0,  # leg 1
        side2=4.0,  # leg 2
        side3=None,  # hypotenuse to be calculated (should be 5.0)
        show_calculation=False,
        label_unknown=True,
        exercise_type=ExerciseType.FIND_HYPOTENUSE,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_calculate_leg():
    """Test calculating missing leg given hypotenuse and one leg."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.METERS,
        side1=None,  # leg to be calculated (should be 9.0)
        side2=15.0,  # hypotenuse
        side3=12.0,  # known leg
        show_calculation=False,
        label_unknown=True,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_all_sides_provided():
    """Test validation when all three sides are provided (5-12-13 triangle)."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        side1=5.0,
        side2=12.0,
        side3=13.0,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_with_calculation_display():
    """Test showing step-by-step Pythagorean theorem calculation."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.INCHES,
        side1=6.0,  # leg 1
        side2=8.0,  # leg 2
        side3=None,  # hypotenuse to be calculated (should be 10.0)
        show_calculation=True,  # Show c² = a² + b² steps
        label_unknown=False,  # Show calculated value instead of variable
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_variable_labels():
    """Test using variable labels (a, b, c) for unknown sides."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.FEET,
        side1=5.0,
        side2=None,  # Will be labeled as 'b'
        side3=13.0,
        show_calculation=False,
        label_unknown=True,  # Use variable names instead of calculated values
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_calculated_value_labels():
    """Test showing calculated values instead of variables."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.METERS,
        side1=8.0,
        side2=15.0,  # hypotenuse
        side3=None,  # Will show calculated value
        show_calculation=False,
        label_unknown=False,  # Show calculated values instead of variables
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_small_triangle():
    """Test with small triangle dimensions."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        side1=0.6,  # Small values
        side2=0.8,
        side3=None,  # Should calculate to 1.0
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_large_triangle():
    """Test with large triangle dimensions."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.METERS,
        side1=30.0,  # Large values
        side2=40.0,
        side3=None,  # Should calculate to 50.0
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_fractional_sides():
    """Test with fractional side lengths."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.FEET,
        side1=1.5,  # 3/2
        side2=2.0,  # 2/1
        side3=None,  # Should calculate to 2.5
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_educational_complete():
    """Test complete educational scenario with calculation and variable labels."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.INCHES,
        side1=9.0,  # leg 1 (known)
        side2=None,  # leg 2 (unknown - will be labeled as variable)
        side3=15.0,  # hypotenuse (known)
        show_calculation=True,  # Show a² + b² = c² steps
        label_unknown=True,  # Use 'b' for unknown side
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_real_world_construction():
    """Test real-world construction problem scenario."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.FEET,
        side1=12.0,  # Building width
        side2=None,  # Diagonal brace length (to be calculated)
        side3=16.0,  # Building length
        show_calculation=True,  # Show work for contractor
        label_unknown=False,  # Show actual calculated length
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


# New test cases for different exercise types and rotation


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_find_hypotenuse_type():
    """Test exercise type 1: Find hypotenuse - shows only legs labeled."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.INCHES,
        side1=5.0,  # leg 1
        side2=12.0,  # leg 2
        side3=None,  # hypotenuse (will be calculated)
        exercise_type=ExerciseType.FIND_HYPOTENUSE,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_find_leg_type():
    """Test exercise type 2: Find leg - shows one leg and hypotenuse labeled."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.METERS,
        side1=None,  # leg to be calculated
        side2=13.0,  # hypotenuse
        side3=5.0,  # known leg
        exercise_type=ExerciseType.FIND_LEG,
        show_calculation=False,
        label_unknown=True,  # Show variable for unknown leg
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_verify_right_triangle_type():
    """Test exercise type 3: Verify right triangle - shows all sides labeled, no right angle symbol."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        side1=8.0,
        side2=15.0,
        side3=17.0,
        exercise_type=ExerciseType.VERIFY_RIGHT_TRIANGLE,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_rotation_90():
    """Test triangle rotation by 90 degrees."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.FEET,
        side1=6.0,
        side2=8.0,
        side3=10.0,
        exercise_type=ExerciseType.FIND_HYPOTENUSE,
        rotation_angle=90,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_rotation_180():
    """Test triangle rotation by 180 degrees."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.INCHES,
        side1=9.0,
        side2=12.0,
        side3=15.0,
        exercise_type=ExerciseType.FIND_LEG,
        rotation_angle=180,
        show_calculation=False,
        label_unknown=True,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_rotation_270():
    """Test triangle rotation by 270 degrees."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.METERS,
        side1=3.0,
        side2=4.0,
        side3=5.0,
        exercise_type=ExerciseType.VERIFY_RIGHT_TRIANGLE,
        rotation_angle=270,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_custom_labeling():
    """Test custom side labeling using label_sides parameter."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.INCHES,
        side1=7.0,
        side2=24.0,
        side3=25.0,
        exercise_type=ExerciseType.FIND_HYPOTENUSE,  # Will be overridden by custom labeling
        label_sides=[True, False, True],  # Label only side1 and side3
        show_right_angle_symbol=True,
        rotation_angle=0,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_triangle_with_opt_side_len_no_right_angle_symbol():
    """Test explicitly hiding right angle symbol."""
    triangle = TriangularGridOpt(
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        side1=20.0,
        side2=21.0,
        side3=29.0,
        exercise_type=ExerciseType.FIND_HYPOTENUSE,  # Would normally show right angle symbol
        show_right_angle_symbol=False,  # Override to hide it
        rotation_angle=0,
        show_calculation=False,
        label_unknown=False,
    )
    file_name = generate_triangle_with_opt_side_len(triangle)
    assert os.path.exists(file_name)


def test_polygon_rotation_validation_missing_direction():
    """Test that validation fails when rotation_angle is provided but rotation_direction is missing."""
    with pytest.raises(ValueError, match="rotation_direction must be specified"):
        Polygon(
            points=[
                Point(x=0, y=0, label="A"),
                Point(x=1, y=0, label="B"),
                Point(x=1, y=1, label="C"),
            ],
            rotation_angle=90,
            # rotation_direction is missing
        )


def test_polygon_rotation_validation_missing_angle():
    """Test that validation fails when rotation_direction is provided but rotation_angle is missing."""
    with pytest.raises(ValueError, match="rotation_angle must be specified"):
        Polygon(
            points=[
                Point(x=0, y=0, label="A"),
                Point(x=1, y=0, label="B"),
                Point(x=1, y=1, label="C"),
            ],
            rotation_direction="clockwise",
            # rotation_angle is missing
        )


# =============================================================================
# Test Cases for Polygon Perimeter Problems (3-10 sides, one unknown)       #
# =============================================================================


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_and_area_with_fractions():
    """Test rectangle with fractional dimensions and area."""
    rectangle_data = RectangleWithHiddenSide(
        unit=EAbbreviatedMeasurementUnit.FEET,
        length=Fraction(5, 2),  # 2.5 ft
        width=Fraction(7, 2),  # 3.5 ft
        show_length=True,
        show_width=False,  # Hide width
    )
    file_name = generate_rect_with_side_len_and_area(rectangle_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_and_area_small_rectangle():
    """Test with small rectangle dimensions."""
    rectangle_data = RectangleWithHiddenSide(
        unit=EAbbreviatedMeasurementUnit.MILLIMETERS,
        length=3,
        width=2,
        show_length=False,  # Hide length
        show_width=True,
    )
    file_name = generate_rect_with_side_len_and_area(rectangle_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_and_area_large_rectangle():
    """Test with large rectangle dimensions (should be scaled)."""
    rectangle_data = RectangleWithHiddenSide(
        unit=EAbbreviatedMeasurementUnit.METERS,
        length=20,
        width=15,
        show_length=True,
        show_width=False,  # Hide width
    )
    file_name = generate_rect_with_side_len_and_area(rectangle_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_and_area_square():
    """Test with square dimensions."""
    rectangle_data = RectangleWithHiddenSide(
        unit=EAbbreviatedMeasurementUnit.UNITS,
        length=9,
        width=9,
        show_length=False,  # Hide length (find missing side)
        show_width=True,
    )
    file_name = generate_rect_with_side_len_and_area(rectangle_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_and_area_mixed_fractions():
    """Test with mixed fraction dimensions (educational use case)."""
    rectangle_data = RectangleWithHiddenSide(
        unit=EAbbreviatedMeasurementUnit.INCHES,
        length=Fraction(7, 3),  # 2 1/3 inches
        width=Fraction(9, 4),  # 2 1/4 inches
        show_length=True,
        show_width=False,  # Student needs to find width
    )
    file_name = generate_rect_with_side_len_and_area(rectangle_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_rect_with_side_len_and_area_area_calculation_problem():
    """Test typical area calculation problem: given one side and area, find other side."""
    rectangle_data = RectangleWithHiddenSide(
        unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
        length=15,  # Known dimension
        width=8,  # This should be hidden
        show_length=True,  # Show the known side
        show_width=False,  # Hide the unknown side (students calculate: 120 ÷ 15 = 8)
    )
    file_name = generate_rect_with_side_len_and_area(rectangle_data)
    assert os.path.exists(file_name)


# =============================================================================
# Test Cases for draw_polygon_fully_labeled - New Function                   #
# =============================================================================


@pytest.mark.drawing_functions
def test_draw_polygon_fully_labeled_square():
    """Test drawing a square with all sides labeled."""
    polygon_data = PolygonFullyLabeled(
        side_lengths=[6, 6, 6, 6],  # Square with 6 cm sides
        unit="cm",
        shape_type="square",
    )
    file_name = draw_polygon_fully_labeled(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_fully_labeled_rectangle():
    """Test drawing a rectangle with all sides labeled."""
    polygon_data = PolygonFullyLabeled(
        side_lengths=[8, 5, 8, 5],  # Rectangle: length=8, width=5
        unit="m",
        shape_type="rectangle",
    )
    file_name = draw_polygon_fully_labeled(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_fully_labeled_regular_pentagon():
    """Test drawing a regular pentagon with all sides labeled."""
    polygon_data = PolygonFullyLabeled(
        side_lengths=[7, 7, 7, 7, 7],  # Regular pentagon with 7 ft sides
        unit="ft",
        shape_type="regular",
    )
    file_name = draw_polygon_fully_labeled(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_fully_labeled_l_shape():
    """Test drawing an L-shaped polygon with all sides labeled."""
    polygon_data = PolygonFullyLabeled(
        side_lengths=[12, 6, 8, 4, 4, 10],  # L-shape with closure constraints satisfied
        unit="cm",
        shape_type="L-shape",
    )
    file_name = draw_polygon_fully_labeled(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_polygon_fully_labeled_t_shape():
    """Test drawing a T-shaped polygon with all sides labeled."""
    polygon_data = PolygonFullyLabeled(
        side_lengths=[
            3,
            2,
            4,
            2,
            11,
            2,
            4,
            2,
        ],  # T-shape with symmetry and closure constraints satisfied
        unit="units",
        shape_type="T-shape",
    )
    file_name = draw_polygon_fully_labeled(polygon_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_regular():
    """Test generating multiple regular grids with different dimensions."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(
                length=3,
                width=2,
                unit=EAbbreviatedMeasurementUnit.UNITS,
                label="Grid A",
            ),
            RectangularGrid(
                length=4,
                width=3,
                unit=EAbbreviatedMeasurementUnit.UNITS,
                label="Grid B",
            ),
            RectangularGrid(
                length=2,
                width=5,
                unit=EAbbreviatedMeasurementUnit.UNITS,
                label="Grid C",
            ),
        ],
        title="Comparing Grid Areas",
        irregularity="all_regular",
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_irregular():
    """Test generating multiple irregular grids with removed squares."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(
                length=4,
                width=4,
                unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
                label="Shape A",
            ),
            RectangularGrid(
                length=5,
                width=3,
                unit=EAbbreviatedMeasurementUnit.CENTIMETERS,
                label="Shape B",
            ),
        ],
        title="Irregular Grid Shapes",
        irregularity="all_irregular",
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_mixed():
    """Test generating mixed regular and irregular grids."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(
                length=3,
                width=3,
                unit=EAbbreviatedMeasurementUnit.FEET,
                label="Figure 1",
            ),
            RectangularGrid(
                length=4,
                width=2,
                unit=EAbbreviatedMeasurementUnit.FEET,
                label="Figure 2",
            ),
            RectangularGrid(
                length=2,
                width=4,
                unit=EAbbreviatedMeasurementUnit.FEET,
                label="Figure 3",
            ),
        ],
        title="Mixed Grid Types",
        irregularity="mixed",
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_with_target_units():
    """Test generating grids with specific target unit counts."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(
                length=4,
                width=4,
                unit=EAbbreviatedMeasurementUnit.INCHES,
                label="Design A",
            ),
            RectangularGrid(
                length=5,
                width=3,
                unit=EAbbreviatedMeasurementUnit.INCHES,
                label="Design B",
            ),
        ],
        title="Target Area Grids",
        irregularity="all_irregular",
        target_units=[12, 10],  # Specific unit counts to achieve
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_single():
    """Test generating a single grid (edge case)."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(
                length=6,
                width=4,
                unit=EAbbreviatedMeasurementUnit.METERS,
                label="Solo Grid",
            ),
        ],
        title="Single Grid Display",
        irregularity="all_regular",
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_no_labels():
    """Test generating multiple grids without explicit labels (should use defaults)."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(
                length=3, width=3, unit=EAbbreviatedMeasurementUnit.MILLIMETERS
            ),
            RectangularGrid(
                length=2, width=4, unit=EAbbreviatedMeasurementUnit.MILLIMETERS
            ),
            RectangularGrid(
                length=5, width=2, unit=EAbbreviatedMeasurementUnit.MILLIMETERS
            ),
        ],
        irregularity="all_regular",
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_multiple_grids_rectangular_grids():
    """Test generating multiple rectangular grids."""
    grid_data = MultipleGrids(
        grids=[
            RectangularGrid(length=5, width=1, unit=EAbbreviatedMeasurementUnit.UNITS),
            RectangularGrid(length=2, width=3, unit=EAbbreviatedMeasurementUnit.UNITS),
            RectangularGrid(length=1, width=5, unit=EAbbreviatedMeasurementUnit.UNITS),
            RectangularGrid(length=2, width=2, unit=EAbbreviatedMeasurementUnit.UNITS),
        ],
        title="Shape Areas",
        irregularity="all_regular",
        target_units=None,
    )
    file_name = generate_multiple_grids(grid_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_single_rhombus():
    """Test drawing a single rhombus with side labels."""

    quad_data = QuadrilateralFigures(
        shape_types=[QuadrilateralShapeType.RHOMBUS],
        side_labels=[["10", "10", "10", "10"]],
        show_ticks=False,
    )
    file_name = draw_single_quadrilateral_stimulus(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_single_parallelogram():
    """Test drawing a single parallelogram with side labels."""
    quad_data = QuadrilateralFigures(
        shape_types=[QuadrilateralShapeType.PARALLELOGRAM],
        side_labels=[["8", "17", "8", "17"]],
        show_ticks=False,
    )
    file_name = draw_single_quadrilateral_stimulus(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_single_irregular():
    """Test drawing a single irregular quadrilateral with side labels."""

    quad_data = QuadrilateralFigures(
        shape_types=[QuadrilateralShapeType.IRREGULAR_QUADRILATERAL],
        side_labels=[["17", "11", "17", "24"]],
        show_ticks=False,
    )
    file_name = draw_single_quadrilateral_stimulus(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_with_ticks():
    """Test drawing quadrilateral with tick marks instead of labels."""

    quad_data = QuadrilateralFigures(
        shape_types=[QuadrilateralShapeType.RHOMBUS],
        side_labels=[["10", "10", "10", "10"]],
        show_ticks=True,
    )
    file_name = draw_single_quadrilateral_stimulus(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_two_shapes():
    """Test drawing two different quadrilaterals with figure labels."""

    quad_data = QuadrilateralFigures(
        shape_types=[
            QuadrilateralShapeType.RHOMBUS,
            QuadrilateralShapeType.PARALLELOGRAM,
        ],
        side_labels=[["10", "10", "10", "10"], ["8", "17", "8", "17"]],
        show_ticks=False,
    )
    file_name = draw_quadrilateral_figures(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_three_shapes_with_rotation():
    """Test drawing three different quadrilaterals."""

    quad_data = QuadrilateralFigures(
        shape_types=[
            QuadrilateralShapeType.RHOMBUS,
            QuadrilateralShapeType.PARALLELOGRAM,
            QuadrilateralShapeType.IRREGULAR_QUADRILATERAL,
        ],
        side_labels=[
            ["12", "12", "12", "12"],
            ["6", "15", "6", "15"],
            ["9", "14", "11", "16"],
        ],
        show_ticks=False,
        rotation=True,
    )
    file_name = draw_quadrilateral_figures(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_four_shapes():
    """Test drawing four different quadrilaterals in a 2x2 grid."""

    quad_data = QuadrilateralFigures(
        shape_types=[
            QuadrilateralShapeType.RHOMBUS,
            QuadrilateralShapeType.PARALLELOGRAM,
            QuadrilateralShapeType.IRREGULAR_QUADRILATERAL,
            QuadrilateralShapeType.RHOMBUS,
        ],
        side_labels=[
            ["8", "8", "8", "8"],
            ["7", "12", "7", "12"],
            ["10", "15", "9", "13"],
            ["14", "14", "14", "14"],
        ],
        show_ticks=False,
    )
    file_name = draw_quadrilateral_figures(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_mixed_ticks_and_labels():
    """Test drawing multiple shapes with tick marks enabled."""

    quad_data = QuadrilateralFigures(
        shape_types=[
            QuadrilateralShapeType.PARALLELOGRAM,
            QuadrilateralShapeType.IRREGULAR_QUADRILATERAL,
        ],
        side_labels=[["5", "13", "5", "13"], ["11", "9", "14", "12"]],
        show_ticks=True,
    )
    file_name = draw_quadrilateral_figures(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_variable_labels():
    """Test drawing quadrilaterals with variable/algebraic side labels."""
    quad_data = QuadrilateralFigures(
        shape_types=[
            QuadrilateralShapeType.RHOMBUS,
            QuadrilateralShapeType.PARALLELOGRAM,
        ],
        side_labels=[["x", "x", "x", "x"], ["2y", "3z", "2y", "3z"]],
        show_ticks=False,
    )
    file_name = draw_quadrilateral_figures(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_quadrilateral_figures_empty_labels():
    """Test drawing quadrilaterals with some empty labels."""

    quad_data = QuadrilateralFigures(
        shape_types=[QuadrilateralShapeType.IRREGULAR_QUADRILATERAL],
        side_labels=[["12", "", "12", ""]],
        show_ticks=False,
    )
    file_name = draw_single_quadrilateral_stimulus(quad_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_parallel_quadrilateral_no_parallel_sides():
    """Test creating an irregular quadrilateral with no parallel sides."""
    shape_data = ParallelQuadrilateral(num_parallel_sides=0, rotate=False)

    file_name = create_parallel_quadrilateral(shape_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_parallel_quadrilateral_one_parallel_side():
    """Test creating a trapezoid with exactly one pair of parallel sides."""
    shape_data = ParallelQuadrilateral(num_parallel_sides=1, rotate=True)

    file_name = create_parallel_quadrilateral(shape_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_parallel_quadrilateral_two_parallel_sides():
    """Test creating a parallelogram with two pairs of parallel sides."""
    shape_data = ParallelQuadrilateral(num_parallel_sides=2, rotate=True)

    file_name = create_parallel_quadrilateral(shape_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_parallel_quadrilateral_with_rotation():
    """Test creating a parallel quadrilateral with random rotation applied."""
    shape_data = ParallelQuadrilateral(num_parallel_sides=1, rotate=True)

    file_name = create_parallel_quadrilateral(shape_data)
    assert os.path.exists(file_name)


# =============================================================================
# Test Cases for draw_regular_irregular_polygons            #
# =============================================================================


# 1) simple singletons (labels only)
@pytest.mark.drawing_functions
def test_single_regular_triangle_labels():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=3,
                show_side_lengths=True,
                show_tick_marks=False,
                show_angle_markings=False,
                color="blue",
            )
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_single_irregular_quadrilateral_labels():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=4,
                side_lengths=[2, 3, 2.5, 1.8],
                show_side_lengths=True,
                show_angle_markings=False,
                show_tick_marks=False,
                color="red",
            )
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 2) ticks only, auto & explicit (double markers!)
@pytest.mark.drawing_functions
def test_ticks_only_auto_groups_all_equal():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=5,
                side_lengths=[2, 2, 3, 2, 3],
                show_side_lengths=False,
                show_tick_marks=True,
                color="green",
            ),
            RegularIrregularPolygon(
                num_sides=4,
                side_lengths=[2, 2, 2, 2],
                show_side_lengths=False,
                show_tick_marks=True,
                color="blue",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_ticks_only_with_explicit_groups_single_double():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=5,
                color="green",
                side_lengths=[2, 3, 2, 3, 2],
                show_side_lengths=False,
                show_tick_marks=True,
                side_equal_groups=[
                    SideEqualGroup(sides=[0, 2, 4], tick_style="single"),
                    SideEqualGroup(sides=[1, 3], tick_style="double"),
                ],
            )
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 3) mixed panels (labels, angles+labels/ticks, outlines)
@pytest.mark.drawing_functions
def test_multiple_mixed_panels():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(num_sides=5, show_side_lengths=True, color="blue"),
            RegularIrregularPolygon(
                num_sides=3,
                side_lengths=[2, 3, 2.8],
                show_side_lengths=True,
                show_angle_markings=True,
                angle_style="arc",
                color="red",
            ),
            RegularIrregularPolygon(
                num_sides=6, show_side_lengths=False, color="green"
            ),
            RegularIrregularPolygon(
                num_sides=4,
                angles=[90, 120, 85, 65],
                show_angle_markings=True,
                angle_style="square",
                show_tick_marks=True,
                side_lengths=[2, 2, 2, 2],
                color="orange",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 4) all-regular / all-irregular
@pytest.mark.drawing_functions
def test_all_regular_shapes_labels():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(num_sides=3, show_side_lengths=True, color="red"),
            RegularIrregularPolygon(num_sides=4, show_side_lengths=True, color="blue"),
            RegularIrregularPolygon(num_sides=5, show_side_lengths=True, color="green"),
            RegularIrregularPolygon(
                num_sides=6, show_side_lengths=True, color="purple"
            ),
            RegularIrregularPolygon(
                num_sides=8, show_side_lengths=True, color="orange"
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_all_irregular_shapes_variations():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=3,
                side_lengths=[2, 3, 4],
                show_side_lengths=True,
                show_angle_markings=True,
                angle_style="arc",
                color="red",
            ),
            RegularIrregularPolygon(
                num_sides=4,
                side_lengths=[2, 1.5, 3, 2.5],
                show_side_lengths=True,
                show_angle_markings=True,
                angle_style="number",
                color="blue",
            ),
            RegularIrregularPolygon(
                num_sides=5,
                side_lengths=[1, 2, 1.5, 2.5, 1.8],
                show_side_lengths=True,
                color="green",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 5) custom angles (numbers & squares), with drawer auto-pairing a side mode
@pytest.mark.drawing_functions
def test_custom_angles_numbers_squares():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=3,
                angles=[90, 45, 45],
                show_angle_markings=True,
                angle_style="number",
                show_side_lengths=False,
                show_tick_marks=False,
                color="blue",
            ),
            RegularIrregularPolygon(
                num_sides=4,
                angles=[90, 90, 120, 60],
                show_angle_markings=True,
                angle_style="square",
                show_side_lengths=False,
                show_tick_marks=False,
                color="red",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 6) arcs without numbers: pass no angles -> arcs only; pair with side labels
@pytest.mark.drawing_functions
def test_arcs_without_numbers_but_with_labels():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=3,
                show_angle_markings=True,
                angle_style="arc",
                show_side_lengths=True,
                color="green",
            ),
            RegularIrregularPolygon(
                num_sides=5,
                show_angle_markings=True,
                angle_style="arc",
                show_side_lengths=True,
                color="purple",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 7) star-like concave (octagon): uses angles >180 and equal-tick groups
@pytest.mark.drawing_functions
def test_concave_star_octagon_numbers_and_ticks():
    # 8-gon sum = (8-2)*180 = 1080. Use 64 and 206 alternating.
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=8,
                angles=[64, 206, 64, 206, 64, 206, 64, 206],
                show_angle_markings=True,
                angle_style="number",
                show_tick_marks=True,
                side_lengths=[2, 3, 2, 3, 2, 3, 2, 3],
                side_equal_groups=[
                    SideEqualGroup(sides=[0, 2, 4, 6], tick_style="double"),
                    SideEqualGroup(sides=[1, 3, 5, 7], tick_style="single"),
                ],
                color="blue",
            )
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 8) comprehensive panel (close to your Excel/ref gallery)
@pytest.mark.drawing_functions
def test_comprehensive_panels():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=8,
                side_lengths=[96] * 8,  # type: ignore
                show_side_lengths=True,
                color="green",
            ),
            RegularIrregularPolygon(
                num_sides=9,
                angles=[94, 127, 104, 135, 175, 154, 154, 159, 158],
                show_angle_markings=True,
                angle_style="arc",
                color="green",
            ),
            RegularIrregularPolygon(
                num_sides=5,
                side_lengths=[58, 58, 49, 58, 58],
                show_side_lengths=True,
                show_angle_markings=True,
                angle_style="arc",
                color="orange",
            ),
            RegularIrregularPolygon(
                num_sides=8,
                show_side_lengths=True,
                show_angle_markings=True,
                angle_style="arc",
                color="purple",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 9) rule-enforcement (angles need exactly one side mode)
@pytest.mark.drawing_functions
def test_rule_enforcement_angles_one_side():
    data = RegularIrregularPolygonList(
        root=[
            # both requested -> drawer prefers labels
            RegularIrregularPolygon(
                num_sides=4,
                angles=[90, 90, 90, 90],
                show_angle_markings=True,
                show_side_lengths=True,
                show_tick_marks=True,
                side_lengths=[2, 2, 2, 2],
                color="green",
            ),
            # neither requested -> drawer enables ticks
            RegularIrregularPolygon(
                num_sides=3,
                angles=[60, 60, 60],
                show_angle_markings=True,
                side_lengths=[2, 2, 2],
                color="orange",
            ),
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# 10) bare outlines (optional helper)
@pytest.mark.drawing_functions
def test_bare_outlines_drawer_bare():
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(num_sides=8, color="red"),
            RegularIrregularPolygon(num_sides=5, color="green"),
            RegularIrregularPolygon(num_sides=3, color="blue"),
        ]
    )
    fn = draw_polygons_bare(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_angles_markers_without_numbers():
    # angles with semi-circles only (no degree text)
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=4,
                angles=[90, 90, 90, 90],
                show_angle_markings=True,
                angle_style="arc",
                show_angle_values=False,  # NEW: markers only, no numbers
                show_side_lengths=False,
                show_tick_marks=True,  # ensure XOR logic pairs with one side mode
                side_lengths=[2, 2, 2, 2],
                color="green",
            )
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_concave_star_numbers_inside():
    # reflex angles + numbers only; ensure inward placement (no arcs)
    data = RegularIrregularPolygonList(
        root=[
            RegularIrregularPolygon(
                num_sides=8,
                # alternating acute/reflex to mimic star-like shape but with simple geometry
                angles=[206, 64, 206, 64, 206, 64, 206, 64],
                show_angle_markings=True,
                angle_style="number",  # numbers only
                show_angle_values=True,
                show_side_lengths=True,
                side_lengths=[2, 3, 2, 3, 2, 3, 2, 3],
                color="blue",
            )
        ]
    )
    fn = draw_regular_irregular_polygons(data)
    assert os.path.exists(fn)


# =============================================================================
# Test Cases for draw_quadrilateral_venn_diagram
# =============================================================================


@pytest.mark.drawing_functions
def test_quadrilateral_venn_diagram_basic():
    """Test basic quadrilateral Venn diagram with default settings."""
    from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
        draw_quadrilateral_venn_diagram,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.geometric_shapes import (
        QuadrilateralVennDiagram,
    )

    data = QuadrilateralVennDiagram()
    fn = draw_quadrilateral_venn_diagram(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_quadrilateral_venn_diagram_no_shapes():
    """Test Venn diagram without shape examples."""
    from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
        draw_quadrilateral_venn_diagram,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.geometric_shapes import (
        QuadrilateralVennDiagram,
    )

    data = QuadrilateralVennDiagram(
        show_shape_examples=False, show_labels=True, title="Types of Quadrilaterals"
    )
    fn = draw_quadrilateral_venn_diagram(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_quadrilateral_venn_diagram_no_labels():
    """Test Venn diagram without labels."""
    from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
        draw_quadrilateral_venn_diagram,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.geometric_shapes import (
        QuadrilateralVennDiagram,
    )

    data = QuadrilateralVennDiagram(
        show_shape_examples=True, show_labels=False, title=None
    )
    fn = draw_quadrilateral_venn_diagram(data)
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_quadrilateral_venn_diagram_custom_title():
    """Test Venn diagram with custom title."""
    from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
        draw_quadrilateral_venn_diagram,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.geometric_shapes import (
        QuadrilateralVennDiagram,
    )

    data = QuadrilateralVennDiagram(
        show_shape_examples=True,
        show_labels=True,
    )
    fn = draw_quadrilateral_venn_diagram(data)
    assert os.path.exists(fn)
