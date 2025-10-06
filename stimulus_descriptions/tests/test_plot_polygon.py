import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_ploygon import (
    Point,
    Polygon,
)
from pydantic import ValidationError


def test_valid_polygon():
    valid_points = [
        Point(coordinates=(0, 0), label="A"),
        Point(coordinates=(2, 0), label="B"),
        Point(coordinates=(2, 2), label="C"),
        Point(coordinates=(0, 2), label="D"),
    ]
    polygon = Polygon(points=valid_points)
    assert len(polygon.points) == 4


def test_polygon_too_few_points():
    with pytest.raises(ValidationError):
        Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(1, 1), label="B"),
            ]
        )


def test_polygon_too_many_points():
    with pytest.raises(ValidationError):
        Polygon(points=[Point(coordinates=(i, i), label=f"A{i}") for i in range(6)])


def test_polygon_self_intersection():
    with pytest.raises(ValidationError, match="The polygon has overlapping lines"):
        Polygon(
            points=[
                Point(coordinates=(0, 0), label="A"),
                Point(coordinates=(2, 0), label="B"),
                Point(coordinates=(0, 2), label="C"),
                Point(coordinates=(2, 2), label="D"),
                Point(coordinates=(1, 1), label="E"),  # Central hole point
            ]
        )
