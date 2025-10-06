import math
from typing import Annotated, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from shapely.geometry import Polygon as ShapelyPolygon


class Axis(BaseModel):
    range: list[int] = Field(
        ...,
        max_length=2,
        min_length=2,
        description="The range of the axis, used for identification",
    )

    @field_validator("range")
    def validate_range(cls, v):
        start, end = v
        if start >= end:
            raise ValueError("Range start must be less than range end")
        if abs(start) > 10 or abs(end) > 10:
            raise ValueError("Range values must have an absolute value of 10 or less")
        return v


class Point(BaseModel):
    coordinates: list[float] = Field(
        ...,
        max_length=2,
        min_length=2,
        description="The x and y coordinates of the point",
    )
    label: Annotated[
        str, StringConstraints(min_length=1, max_length=1, pattern=r"^[A-Za-z]$")
    ] = Field(..., description="The label of the point, used for identification")

    # New field to mark calculated vertices
    calculated: bool = Field(
        default=False,
        description="Whether this point was calculated (for missing vertices) or provided",
    )

    @field_validator("coordinates")
    def validate_integer_coordinates(cls, v):
        """Ensure coordinates are integers as required by assessment boundary"""
        for coord in v:
            if not isinstance(coord, int) and not coord.is_integer():
                raise ValueError("Coordinates must be integers")
        return [int(coord) for coord in v]


class Polygon(BaseModel):
    points: list[Point] = Field(min_length=3, max_length=8)

    # New optional field for quadrilateral completion
    complete_as: Optional[Literal["rectangle", "square", "parallelogram"]] = Field(
        default=None,
        description="When 3 points are provided, complete as this type of quadrilateral",
    )

    @model_validator(mode="after")
    def calculate_missing_vertex(self):
        """Calculate the fourth vertex if requested and only 3 points provided"""
        if self.complete_as is not None and len(self.points) == 3:
            # Extract the three given points
            p1, p2, p3 = self.points

            # Calculate the fourth vertex based on the type
            fourth_point = self._calculate_fourth_vertex(p1, p2, p3, self.complete_as)

            # Add the calculated point to the polygon
            self.points.append(fourth_point)

        return self

    def _calculate_fourth_vertex(
        self, p1: Point, p2: Point, p3: Point, shape_type: str
    ) -> Point:
        """Calculate the fourth vertex to complete the specified quadrilateral type"""
        x1, y1 = p1.coordinates
        x2, y2 = p2.coordinates
        x3, y3 = p3.coordinates

        if shape_type == "parallelogram":
            # For parallelogram ABCD, if we have A, B, C then D = A + C - B (vector addition)
            # This ensures opposite sides are parallel and equal
            x4 = x1 + x3 - x2
            y4 = y1 + y3 - y2

        elif shape_type in ["rectangle", "square"]:
            # For rectangle/square, we need to find which two points form a right angle
            # Try all three possible configurations and pick the valid one
            fourth_point = self._find_rectangle_fourth_vertex(
                (x1, y1), (x2, y2), (x3, y3), shape_type
            )
            if fourth_point is None:
                raise ValueError(f"The three given points cannot form a {shape_type}")
            x4, y4 = fourth_point

        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

        # Ensure coordinates are integers
        x4, y4 = int(round(x4)), int(round(y4))

        # Determine the next label in sequence
        labels = [p.label for p in [p1, p2, p3]]
        next_label = self._get_next_label(labels)

        return Point(coordinates=[x4, y4], label=next_label, calculated=True)

    def _find_rectangle_fourth_vertex(self, p1, p2, p3, shape_type):
        """Find the fourth vertex that makes a rectangle or square"""
        points = [p1, p2, p3]

        # Try each configuration where two consecutive points form a side
        for i in range(3):
            a = points[i]
            b = points[(i + 1) % 3]
            c = points[(i + 2) % 3]

            # Check if angle ABC is 90 degrees (or close to it)
            if self._is_right_angle(a, b, c):
                # Calculate fourth point D such that ABCD is a rectangle
                # Vector from B to A
                ba_x, ba_y = a[0] - b[0], a[1] - b[1]
                # Vector from B to C
                bc_x, bc_y = c[0] - b[0], c[1] - b[1]

                # Fourth point D = C + (B - A) = C + BA vector
                d_x = c[0] + ba_x
                d_y = c[1] + ba_y

                # For square, verify all sides are equal length
                if shape_type == "square":
                    side1_len = math.sqrt(ba_x**2 + ba_y**2)
                    side2_len = math.sqrt(bc_x**2 + bc_y**2)
                    if not math.isclose(side1_len, side2_len, rel_tol=1e-9):
                        continue  # Not a square, try next configuration

                return (d_x, d_y)

        return None

    def _is_right_angle(self, a, b, c):
        """Check if angle ABC is approximately 90 degrees"""
        # Vectors BA and BC
        ba_x, ba_y = a[0] - b[0], a[1] - b[1]
        bc_x, bc_y = c[0] - b[0], c[1] - b[1]

        # Dot product should be 0 for perpendicular vectors
        dot_product = ba_x * bc_x + ba_y * bc_y
        return abs(dot_product) < 1e-9

    def _get_next_label(self, existing_labels):
        """Get the next alphabetical label"""
        existing_set = set(existing_labels)

        # Try uppercase letters first
        for i in range(26):
            label = chr(ord("A") + i)
            if label not in existing_set:
                return label

        # If all uppercase are used, try lowercase
        for i in range(26):
            label = chr(ord("a") + i)
            if label not in existing_set:
                return label

        raise ValueError("No available labels remaining")

    @model_validator(mode="after")
    def validate_polygon_shape(self):
        # Convert points to a list of tuples
        point_coords = [p.coordinates for p in self.points]
        point_coords.append(point_coords[0])  # Close the polygon

        # Create a Shapely polygon
        poly = ShapelyPolygon(point_coords)

        # Check if the polygon is simple (no self-intersections)
        if not poly.is_simple:
            raise ValueError("The polygon has overlapping lines")

        return self

    @model_validator(mode="after")
    def validate_rectilinear_shape(self):
        """Validate that all sides are horizontal or vertical for rectilinear shapes"""
        if len(self.points) >= 4:  # Only apply to polygons with 4+ vertices
            # Skip rectilinear validation for parallelograms since they don't need to be rectilinear
            if self.complete_as == "parallelogram":
                return self

            coords = [p.coordinates for p in self.points]

            for i in range(len(coords)):
                p1 = coords[i]
                p2 = coords[(i + 1) % len(coords)]

                # Check if side is neither horizontal nor vertical
                if p1[0] != p2[0] and p1[1] != p2[1]:
                    raise ValueError(
                        f"Side from {p1} to {p2} is neither horizontal nor vertical. "
                        "For 4+ vertex polygons, all sides must be horizontal or vertical to form rectilinear shapes."
                    )

        return self

    @model_validator(mode="after")
    def validate_suitable_for_area_calculation(self):
        """Ensure polygon is suitable for area calculation through rectangular decomposition"""
        if (
            len(self.points) >= 6
        ):  # Apply stricter validation for hard difficulty (6+ vertices)
            coords = [p.coordinates for p in self.points]

            # Check that coordinates align properly for rectangular decomposition
            x_coords = sorted(set(coord[0] for coord in coords))
            y_coords = sorted(set(coord[1] for coord in coords))

            # For proper rectangular decomposition, we need at least 3 distinct x and y values
            if len(x_coords) < 3 or len(y_coords) < 3:
                raise ValueError(
                    "Complex polygons (6+ vertices) must have at least 3 distinct x-coordinates "
                    "and 3 distinct y-coordinates for proper area calculation"
                )

            # Ensure the shape can be decomposed into rectangles (basic check)
            # All vertices should lie on the grid formed by x_coords and y_coords
            for coord in coords:
                if coord[0] not in x_coords or coord[1] not in y_coords:
                    raise ValueError(
                        "All vertices must align with the coordinate grid for rectangular decomposition"
                    )

        return self


class PlotPolygon(StimulusDescription):
    axes: dict[Literal["x", "y"], Axis] = Field(
        ..., description="Dictionary containing 'x' and 'y' axes information"
    )
    polygon: Polygon = Field(
        ...,
        description="The polygon represented by points to be plotted on the graph",
    )

    @model_validator(mode="after")
    def validate_polygon(self):
        x_range = self.axes["x"].range
        y_range = self.axes["y"].range

        for point in self.polygon.points:
            x, y = point.coordinates
            if x < x_range[0] or x > x_range[1] or y < y_range[0] or y > y_range[1]:
                raise ValueError(
                    f"Point {point.label} ({x}, {y}) is outside the specified axis ranges."
                )

        return self


if __name__ == "__main__":
    PlotPolygon.generate_assistant_function_schema("mcq4")
