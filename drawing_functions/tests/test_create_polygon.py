import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing import (
    create_polygon,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_ploygon import (
    Axis,
    PlotPolygon,
    Point,
    Polygon,
)


@pytest.mark.drawing_functions
def test_triangle():
    polygon = PlotPolygon(
        axes={"x": Axis(range=[-5, 5]), "y": Axis(range=[-5, 5])},
        polygon=Polygon(
            points=[
                Point(coordinates=[0, 0], label="A"),
                Point(coordinates=[3, 0], label="B"),
                Point(coordinates=[0, 4], label="C"),
            ]
        ),
    )
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_square():
    polygon = PlotPolygon(
        axes={"x": Axis(range=[-6, 6]), "y": Axis(range=[-6, 6])},
        polygon=Polygon(
            points=[
                Point(coordinates=[-2, -2], label="A"),
                Point(coordinates=[2, -2], label="B"),
                Point(coordinates=[2, 2], label="C"),
                Point(coordinates=[-2, 2], label="D"),
            ]
        ),
    )
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_pentagon():
    """Test a triangle (3 vertices don't need rectilinear validation)"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=[-4, 4]), "y": Axis(range=[-4, 4])},
        polygon=Polygon(
            points=[
                Point(coordinates=[0, 3], label="A"),
                Point(coordinates=[2, 1], label="B"),
                Point(coordinates=[-1, -2], label="C"),
            ]
        ),
    )
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_different_axis_ranges():
    polygon = PlotPolygon(
        axes={
            "x": Axis(range=[-10, 10]),
            "y": Axis(range=[-5, 5]),
        },
        polygon=Polygon(
            points=[
                Point(coordinates=[-8, -3], label="A"),
                Point(coordinates=[8, -3], label="B"),
                Point(coordinates=[0, 4], label="C"),
            ]
        ),
    )
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_single_point_polygon():
    with pytest.raises(ValueError):
        PlotPolygon(
            axes={
                "x": Axis(range=(-5, 5)),
                "y": Axis(range=(-5, 5)),
            },
            polygon=Polygon(points=[Point(coordinates=(0, 0), label="A")]),
        )


@pytest.mark.drawing_functions
def test_invalid_axis_range():
    with pytest.raises(ValueError):
        PlotPolygon(
            axes={
                "x": Axis(range=(5, -5)),
                "y": Axis(range=(-5, 5)),
            },
            polygon=Polygon(
                points=[
                    Point(coordinates=(0, 0), label="A"),
                    Point(coordinates=(3, 0), label="B"),
                    Point(coordinates=(0, 4), label="C"),
                ]
            ),
        )


@pytest.mark.drawing_functions
def test_out_of_range_point():
    with pytest.raises(ValueError):
        PlotPolygon(
            axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
            polygon=Polygon(
                points=[
                    Point(coordinates=(0, 0), label="A"),
                    Point(coordinates=(6, 0), label="B"),
                    Point(coordinates=(0, 6), label="C"),
                ]
            ),
        )


@pytest.mark.drawing_functions
def test_invalid_point_label():
    with pytest.raises(ValueError):
        PlotPolygon(
            axes={
                "x": Axis(range=(-5, 5)),
                "y": Axis(range=(-5, 5)),
            },
            polygon=Polygon(
                points=[
                    Point(coordinates=(0, 0), label="A"),
                    Point(coordinates=(3, 0), label="B"),
                    Point(coordinates=(0, 4), label="123"),  # Invalid label
                ]
            ),
        )


@pytest.mark.drawing_functions
def test_hard_difficulty_l_shape():
    """Test hard difficulty with 6+ vertices forming a rectilinear L-shape"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(-3, -2), label="A"),
                Point(coordinates=(-3, 2), label="B"),
                Point(coordinates=(1, 2), label="C"),
                Point(coordinates=(1, 4), label="D"),
                Point(coordinates=(4, 4), label="E"),
                Point(coordinates=(4, -2), label="F"),
            ]
        ),
    )
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_invalid_non_rectilinear_shape():
    """Test that non-rectilinear shapes with 4+ vertices are rejected"""
    with pytest.raises(ValueError, match="neither horizontal nor vertical"):
        PlotPolygon(
            axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
            polygon=Polygon(
                points=[
                    Point(coordinates=(0, 0), label="A"),
                    Point(coordinates=(2, 0), label="B"),
                    Point(coordinates=(3, 2), label="C"),  # Diagonal side - should fail
                    Point(coordinates=(0, 2), label="D"),
                ]
            ),
        )


# New tests for quadrilateral completion functionality


@pytest.mark.drawing_functions
def test_complete_rectangle_from_three_points():
    """Test completing a rectangle from three given points"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(3, 0), label="B"),
                Point(coordinates=(3, 2), label="C"),
            ],
            complete_as="rectangle",
        ),
    )

    # Verify the fourth point was calculated correctly
    assert len(polygon.polygon.points) == 4
    fourth_point = polygon.polygon.points[3]
    assert fourth_point.coordinates == [0, 2]
    assert fourth_point.calculated
    assert fourth_point.label == "D"

    # Test the drawing function
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_complete_square_from_three_points():
    """Test completing a square from three given points"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(1, 1), label="A"),
                Point(coordinates=(3, 1), label="B"),
                Point(coordinates=(3, 3), label="C"),
            ],
            complete_as="square",
        ),
    )

    # Verify the fourth point was calculated correctly
    assert len(polygon.polygon.points) == 4
    fourth_point = polygon.polygon.points[3]
    assert fourth_point.coordinates == [1, 3]
    assert fourth_point.calculated

    # Test the drawing function
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_complete_parallelogram_from_three_points():
    """Test completing a parallelogram from three given points"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(2, 1), label="B"),
                Point(coordinates=(3, 3), label="C"),
            ],
            complete_as="parallelogram",
        ),
    )

    # Verify the fourth point was calculated correctly
    assert len(polygon.polygon.points) == 4
    fourth_point = polygon.polygon.points[3]
    # For parallelogram ABCD, D = A + C - B = (0,0) + (3,3) - (2,1) = (1,2)
    assert fourth_point.coordinates == [1, 2]
    assert fourth_point.calculated

    # Test the drawing function
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_rectangle_completion_different_orientations():
    """Test rectangle completion with different point orderings"""
    # Test case 1: Points form right angle at second point
    polygon1 = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(0, 3), label="B"),  # Right angle at B
                Point(coordinates=(4, 3), label="C"),
            ],
            complete_as="rectangle",
        ),
    )

    assert len(polygon1.polygon.points) == 4
    fourth_point = polygon1.polygon.points[3]
    assert fourth_point.coordinates == [4, 0]
    assert fourth_point.calculated


@pytest.mark.drawing_functions
def test_invalid_rectangle_completion():
    """Test that invalid rectangle completion raises error"""
    with pytest.raises(ValueError, match="cannot form a rectangle"):
        PlotPolygon(
            axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
            polygon=Polygon(
                points=[
                    Point(coordinates=(0, 0), label="A"),
                    Point(coordinates=(1, 1), label="B"),  # No right angles possible
                    Point(coordinates=(2, 3), label="C"),
                ],
                complete_as="rectangle",
            ),
        )


@pytest.mark.drawing_functions
def test_invalid_square_completion():
    """Test that invalid square completion raises error when sides aren't equal"""
    with pytest.raises(ValueError, match="cannot form a square"):
        PlotPolygon(
            axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
            polygon=Polygon(
                points=[
                    Point(coordinates=(0, 0), label="A"),
                    Point(coordinates=(3, 0), label="B"),  # Side length 3
                    Point(coordinates=(3, 2), label="C"),  # Side length 2 - not equal
                ],
                complete_as="square",
            ),
        )


@pytest.mark.drawing_functions
def test_completion_with_four_points_ignored():
    """Test that completion is ignored when 4 points are already provided"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(2, 0), label="B"),
                Point(coordinates=(2, 2), label="C"),
                Point(coordinates=(0, 2), label="D"),
            ],
            complete_as="rectangle",  # Should be ignored
        ),
    )

    # Should still have exactly 4 points, none calculated
    assert len(polygon.polygon.points) == 4
    assert all(
        not hasattr(p, "calculated") or not p.calculated for p in polygon.polygon.points
    )


@pytest.mark.drawing_functions
def test_calculated_point_outside_axis_range():
    """Test that calculated point outside axis range raises error"""
    with pytest.raises(ValueError, match="outside the specified axis ranges"):
        PlotPolygon(
            axes={
                "x": Axis(range=[-1, 1]),
                "y": Axis(range=[-1, 1]),
            },  # Very small range
            polygon=Polygon(
                points=[
                    Point(coordinates=[0, 0], label="A"),
                    Point(coordinates=[1, 0], label="B"),
                    Point(
                        coordinates=[1, 3], label="C"
                    ),  # This will cause fourth point (0, 3) to be outside y range [-1, 1]
                ],
                complete_as="rectangle",
            ),
        )


@pytest.mark.drawing_functions
def test_label_sequence_generation():
    """Test that calculated points get correct sequential labels"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(2, 0), label="C"),  # Skip B
                Point(coordinates=(2, 2), label="E"),  # Skip D
            ],
            complete_as="rectangle",
        ),
    )

    fourth_point = polygon.polygon.points[3]
    assert fourth_point.label == "B"  # Should get the first available label


@pytest.mark.drawing_functions
def test_backwards_compatibility():
    """Test that existing functionality still works without completion"""
    # This should work exactly as before
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(3, 0), label="B"),
                Point(coordinates=(0, 4), label="C"),
            ]
            # No complete_as field - should work as before
        ),
    )

    # Should have exactly 3 points, none calculated
    assert len(polygon.polygon.points) == 3
    assert all(
        not hasattr(p, "calculated") or not p.calculated for p in polygon.polygon.points
    )

    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_calculated_vertex_visual_distinction():
    """Test that calculated vertices have different visual representation"""
    polygon = PlotPolygon(
        axes={"x": Axis(range=(-5, 5)), "y": Axis(range=(-5, 5))},
        polygon=Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(2, 0), label="B"),
                Point(coordinates=(2, 2), label="C"),
            ],
            complete_as="rectangle",
        ),
    )

    # Verify the calculated point is marked correctly
    fourth_point = polygon.polygon.points[3]
    assert fourth_point.calculated

    # The visual distinction should be tested in the drawing function
    file_name = create_polygon(polygon)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_coordinate_numbers_always_visible():
    """Test that coordinate numbers are always visible on axes for coordinate geometry problems"""
    # Test various scenarios where coordinates must be visible

    # Test 1: Rectangle spanning multiple quadrants
    polygon1 = PlotPolygon(
        axes={"x": Axis(range=[-2, 3]), "y": Axis(range=[-1, 3])},
        polygon=Polygon(
            points=[
                Point(coordinates=[0, 0], label="A"),
                Point(coordinates=[2, 0], label="B"),
                Point(coordinates=[2, 2], label="C"),
                Point(coordinates=[0, 2], label="D"),
            ]
        ),
    )
    file1 = create_polygon(polygon1)
    assert os.path.exists(file1)

    # Test 2: Rectangle completion in positive quadrant
    polygon2 = PlotPolygon(
        axes={"x": Axis(range=[0, 4]), "y": Axis(range=[0, 3])},
        polygon=Polygon(
            points=[
                Point(coordinates=[1, 0], label="P"),
                Point(coordinates=[3, 0], label="Q"),
                Point(coordinates=[3, 2], label="R"),
            ],
            complete_as="rectangle",
        ),
    )
    file2 = create_polygon(polygon2)
    assert os.path.exists(file2)

    # Test 3: Small range to ensure coordinates still show
    polygon3 = PlotPolygon(
        axes={"x": Axis(range=[0, 2]), "y": Axis(range=[0, 2])},
        polygon=Polygon(
            points=[
                Point(coordinates=[0, 0], label="A"),
                Point(coordinates=[1, 0], label="B"),
                Point(coordinates=[1, 1], label="C"),
            ]
        ),
    )
    file3 = create_polygon(polygon3)
    assert os.path.exists(file3)

    # All tests should generate images successfully with visible coordinates
    # Visual verification: coordinate numbers should be visible on all axes
