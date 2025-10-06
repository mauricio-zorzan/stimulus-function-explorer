import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.linear_diagram import (
    IntersectionPoint,
    Line,
    LinearDiagram,
)
from pydantic import ValidationError


def test_valid_linear_diagram():
    diagram = LinearDiagram(
        lines=[Line(slope=1, y_intercept=0), Line(slope=-1, y_intercept=4)],
        intersection_point=IntersectionPoint(x=2, y=2),
    )
    assert isinstance(diagram, LinearDiagram)
    assert len(diagram.lines) == 2
    assert diagram.intersection_point.x == 2
    assert diagram.intersection_point.y == 2


def test_invalid_intersection_point():
    with pytest.raises(
        ValueError,
        match="The given intersection point .* does not match the calculated intersection point",
    ):
        LinearDiagram(
            lines=[Line(slope=1, y_intercept=0), Line(slope=-1, y_intercept=4)],
            intersection_point=IntersectionPoint(x=3, y=3),
        )


def test_parallel_lines():
    with pytest.raises(
        ValueError, match="The lines are parallel and have no intersection point"
    ):
        LinearDiagram(
            lines=[Line(slope=1, y_intercept=0), Line(slope=1, y_intercept=2)],
            intersection_point=IntersectionPoint(x=0, y=0),
        )


def test_identical_lines():
    with pytest.raises(
        ValueError,
        match="The lines are identical and have infinite intersection points",
    ):
        LinearDiagram(
            lines=[Line(slope=1, y_intercept=2), Line(slope=1, y_intercept=2)],
            intersection_point=IntersectionPoint(x=0, y=2),
        )


def test_intersection_at_origin():
    diagram = LinearDiagram(
        lines=[Line(slope=1, y_intercept=0), Line(slope=-1, y_intercept=0)],
        intersection_point=IntersectionPoint(x=0, y=0),
    )
    assert isinstance(diagram, LinearDiagram)


def test_invalid_number_of_lines():
    with pytest.raises(ValidationError, match="List should have at least 2 items"):
        LinearDiagram(
            lines=[Line(slope=1, y_intercept=0)],
            intersection_point=IntersectionPoint(x=0, y=0),
        )

    with pytest.raises(ValidationError, match="List should have at most 2 items"):
        LinearDiagram(
            lines=[
                Line(slope=1, y_intercept=0),
                Line(slope=2, y_intercept=1),
                Line(slope=3, y_intercept=2),
            ],
            intersection_point=IntersectionPoint(x=0, y=0),
        )


def test_number_greater_than_8():
    with pytest.raises(
        ValueError,
        match="No y-intercept in the stimulus description should be greater than 8",
    ):
        LinearDiagram(
            lines=[Line(slope=9, y_intercept=9), Line(slope=1, y_intercept=4)],
            intersection_point=IntersectionPoint(x=2, y=4),
        )

    with pytest.raises(
        ValueError,
        match="No coordinate in the intersection point should be greater than 8",
    ):
        LinearDiagram(
            lines=[Line(slope=1, y_intercept=2), Line(slope=2, y_intercept=4)],
            intersection_point=IntersectionPoint(x=9, y=4),
        )


def test_valid_fractional_slope():
    diagram = LinearDiagram(
        lines=[Line(slope=0, y_intercept=2), Line(slope=-0.25, y_intercept=4)],
        intersection_point=IntersectionPoint(x=8, y=2),
    )
    assert isinstance(diagram, LinearDiagram)
    assert diagram.intersection_point.x == 8
    assert diagram.intersection_point.y == 2


def test_almost_integer_intersection():
    # This should fail because the calculated intersection point is not close to integers
    with pytest.raises(
        ValueError,
        match="The calculated intersection point .* is not sufficiently close to integer coordinates",
    ):
        LinearDiagram(
            lines=[Line(slope=1 / 3, y_intercept=1), Line(slope=-1 / 2, y_intercept=3)],
            intersection_point=IntersectionPoint(x=2, y=2),
        )

    # This should fail because the given point doesn't match the calculated point
    with pytest.raises(
        ValueError,
        match="The given intersection point .* does not match the calculated intersection point",
    ):
        LinearDiagram(
            lines=[Line(slope=1, y_intercept=0), Line(slope=-1, y_intercept=4)],
            intersection_point=IntersectionPoint(x=3, y=3),
        )

    # This should pass as it's the correct integer intersection point
    diagram = LinearDiagram(
        lines=[Line(slope=1, y_intercept=0), Line(slope=-1, y_intercept=4)],
        intersection_point=IntersectionPoint(x=2, y=2),
    )
    assert isinstance(diagram, LinearDiagram)


def test_incorrect_intersection_point():
    with pytest.raises(
        ValueError,
        match="The given intersection point \\(-1, 3\\) does not match the calculated intersection point \\(2, 0\\)",
    ):
        LinearDiagram(
            lines=[Line(slope=-2, y_intercept=4), Line(slope=1, y_intercept=-2)],
            intersection_point=IntersectionPoint(x=-1, y=3),
        )


def test_floating_point_precision():
    diagram = LinearDiagram(
        lines=[Line(slope=1 / 3, y_intercept=0), Line(slope=-1 / 3, y_intercept=2)],
        intersection_point=IntersectionPoint(x=3, y=1),
    )
    assert isinstance(diagram, LinearDiagram)
    assert diagram.intersection_point.x == 3
    assert diagram.intersection_point.y == 1
