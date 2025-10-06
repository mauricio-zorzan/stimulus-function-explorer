import logging
import math
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class Line(BaseModel):
    slope: Optional[float] = Field(
        ...,
        description="The slope of the line. None represents a vertical line (infinite slope).",
    )
    intercept: float = Field(..., description="The intercept of the line.")
    label: Literal["A", "B", "C", "D"] = Field(
        ..., description="A single character label for the line."
    )


class LinesOfSymmetry(StimulusDescription):
    shape_coordinates: List[List[float]] = Field(
        ...,
        description="A list of coordinate pairs defining the vertices of the shape.",
    )
    lines: List[Line] = Field(
        ...,
        description="A list of lines defined by their slope, intercept, and label.",
    )

    @model_validator(mode="after")
    def validate_shape_and_lines(cls, values):
        if len(values.shape_coordinates) < 3:
            raise ValueError("Shape must have at least 3 vertices.")
        if len(values.lines) > 4:
            raise ValueError("There can be no more than 4 lines.")
        return values

    @model_validator(mode="after")
    def validate_single_line_of_symmetry(cls, values):
        if not values.lines:
            return values  # Empty list of lines is allowed

        lines_of_symmetry = [
            line
            for line in values.lines
            if is_line_of_symmetry(values.shape_coordinates, line)
        ]

        if len(lines_of_symmetry) != 1:
            raise ValueError(
                f"There must be exactly one true line of symmetry when lines are present. "
                f"Found {len(lines_of_symmetry)}."
            )

        return values

    @model_validator(mode="after")
    def validate_lines_inside_shape(cls, values):
        for line in values.lines:
            if not line_has_point_inside_shape(line, values.shape_coordinates):
                raise ValueError(
                    f"Line {line.label} does not have a point strictly inside the shape."
                )
        return values

    @model_validator(mode="after")
    def validate_shape_coordinates_order(cls, values):
        coords = values.shape_coordinates
        if len(coords) < 3:
            raise ValueError("Shape must have at least 3 vertices.")

        # Check for self-intersection
        if is_self_intersecting(coords):
            raise ValueError("Shape coordinates form a self-intersecting polygon.")

        return values

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext") -> None:
        """
        Additional validation for specific substandard configurations
        """
        logging.info(self)
        super().pipeline_validate(pipeline_context)

        if pipeline_context.standard_id == "CCSS.MATH.CONTENT.4.G.A.3+2":
            if self.lines:
                raise ValueError(
                    f"For substandard {pipeline_context.standard_id}, "
                    f"there should be no lines in the lines list."
                )


def is_line_of_symmetry(
    shape_coordinates: List[Tuple[float, float]], line: Line
) -> bool:
    # Split the shape into two parts
    part1, part2 = split_shape(shape_coordinates, line)

    # Calculate areas of both parts
    area1 = polygon_area(part1)
    area2 = polygon_area(part2)

    # Compare areas with a small tolerance for floating-point errors
    return math.isclose(area1, area2, rel_tol=1e-9)


def split_shape(
    shape_coordinates: List[Tuple[float, float]], line: Line
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    part1, part2 = [], []

    for i in range(len(shape_coordinates)):
        p1 = shape_coordinates[i]
        p2 = shape_coordinates[(i + 1) % len(shape_coordinates)]

        side1 = point_side_of_line(p1, line)
        side2 = point_side_of_line(p2, line)

        if side1 >= 0:
            part1.append(p1)
        if side1 <= 0:
            part2.append(p1)

        if side1 * side2 < 0:
            # Line intersects the edge, calculate intersection point
            intersection = line_intersection(p1, p2, line)
            if intersection:
                part1.append(intersection)
                part2.append(intersection)

    return part1, part2


def point_side_of_line(point: Tuple[float, float], line: Line) -> float:
    x, y = point
    if line.slope is None:  # Vertical line
        return x - line.intercept
    return y - (line.slope * x + line.intercept)


def line_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float], line: Line
) -> Optional[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2

    if line.slope is None:  # Vertical line
        if x1 == x2:  # The edge is also vertical
            return None  # Parallel lines, no intersection
        x = line.intercept
        y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    else:
        if x1 == x2:  # Vertical edge
            x = x1
            y = line.slope * x + line.intercept
        else:
            m1 = (y2 - y1) / (x2 - x1)
            if math.isclose(m1, line.slope, rel_tol=1e-9):
                return None  # Parallel lines, no intersection
            x = (line.intercept - y1 + m1 * x1) / (m1 - line.slope)
            y = m1 * (x - x1) + y1

    return (x, y)


def polygon_area(vertices: List[Tuple[float, float]]) -> float:
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def line_intersects_shape(
    line: Line, shape_coordinates: List[Tuple[float, float]]
) -> bool:
    # Check if the line intersects any edge of the shape
    for i in range(len(shape_coordinates)):
        p1 = shape_coordinates[i]
        p2 = shape_coordinates[(i + 1) % len(shape_coordinates)]
        if line_segments_intersect(line, p1, p2):
            return True

    # If no intersections, check if the midpoint of the line is inside the shape
    midpoint = get_line_midpoint(line, shape_coordinates)
    return is_point_inside_polygon(midpoint, shape_coordinates)


def line_segments_intersect(
    line: Line, p1: Tuple[float, float], p2: Tuple[float, float]
) -> bool:
    # Check if the line segment (p1, p2) intersects with the given line
    x1, y1 = p1
    x2, y2 = p2

    if line.slope is None:  # Vertical line
        if x1 <= line.intercept <= x2 or x2 <= line.intercept <= x1:
            return True
    else:
        y_intercept = line.slope * x1 + line.intercept
        if (y1 <= y_intercept <= y2) or (y2 <= y_intercept <= y1):
            return True

    return False


def get_line_midpoint(
    line: Line, shape_coordinates: List[Tuple[float, float]]
) -> Tuple[float, float]:
    # Get the bounding box of the shape
    x_coords, y_coords = zip(*shape_coordinates)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    if line.slope is None:  # Vertical line
        x = line.intercept
        y = (min_y + max_y) / 2
    else:
        x = (min_x + max_x) / 2
        y = line.slope * x + line.intercept

    return (x, y)


def is_point_inside_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_has_point_inside_shape(
    line: Line, shape_coordinates: List[Tuple[float, float]]
) -> bool:
    # Get the bounding box of the shape
    x_coords, y_coords = zip(*shape_coordinates)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Check multiple points along the line
    num_points = 100  # Increase this for more precision
    for i in range(num_points):
        t = i / (num_points - 1)
        if line.slope is None:  # Vertical line
            x = line.intercept
            y = min_y + t * (max_y - min_y)
        else:
            x = min_x + t * (max_x - min_x)
            y = line.slope * x + line.intercept

        point = (x, y)
        if is_point_strictly_inside_polygon(point, shape_coordinates):
            return True

    return False


def is_point_strictly_inside_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    if is_point_on_polygon_edge(point, polygon):
        return False
    return is_point_inside_polygon(point, polygon)


def is_point_on_polygon_edge(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if the point is on the edge
        if (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1):
            # Calculate the distance from the point to the line segment
            distance = abs(
                (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1
            ) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if math.isclose(distance, 0, abs_tol=1e-9):
                return True

    return False


def is_self_intersecting(coords: List[Tuple[float, float]]) -> bool:
    n = len(coords)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # Skip adjacent edges
            if segments_intersect(
                coords[i], coords[(i + 1) % n], coords[j], coords[(j + 1) % n]
            ):
                return True
    return False


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
