from enum import Enum
from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box


class EAbbreviatedMeasurementUnit(str, Enum):
    INCHES = "in"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"
    FEET = "ft"
    METERS = "m"
    UNITS = "Units"
    KILOMETERS = "km"


class RectangleSpec(BaseModel):
    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )
    length: int = Field(..., ge=1, le=10, description="The length of the rectangle.")
    width: int = Field(..., ge=1, le=10, description="The width of the rectangle.")
    x: int = Field(..., description="The x-coordinate of the bottom-left corner.")
    y: int = Field(..., description="The y-coordinate of the bottom-left corner.")


class RightTriangleSpec(BaseModel):
    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )
    right_angle_x: int = Field(..., description="The x-coordinate of the right-angle vertex.")
    right_angle_y: int = Field(..., description="The y-coordinate of the right-angle vertex.")
    base: int = Field(..., ge=1, le=10, description="The length of the base (horizontal side from right angle).")
    height: int = Field(..., ge=1, le=10, description="The length of the height (vertical side from right angle).")


class CompositeRectangularGrid(StimulusDescription):
    rectangles: List[RectangleSpec] = Field(
        ...,
        description="List of two rectangles with position and size.",
    )

    @model_validator(mode="after")
    def check_two_rectangles(cls, values):
        if len(values.rectangles) != 2:
            raise ValueError("Exactly two rectangles must be provided.")
        r1, r2 = values.rectangles
        if (
            r1.x == r2.x
            and r1.y == r2.y
            and r1.width == r2.width
            and r1.length == r2.length
            and r1.unit == r2.unit
        ):
            raise ValueError(
                "The two rectangles must not be identical (completely overlapping)."
            )
        # Use shapely to check for intersection or touching
        b1 = box(r1.x, r1.y, r1.x + r1.width, r1.y + r1.length)
        b2 = box(r2.x, r2.y, r2.x + r2.width, r2.y + r2.length)
        if not b1.intersects(b2) and not b1.touches(b2):
            raise ValueError(
                "The two rectangles must touch or overlap (not be disjoint)."
            )
        return values


class CompositeRectangularTriangularGrid(StimulusDescription):
    """
    Stimulus description for a composite figure made of 2 rectangles and 1 triangle.
    """
    rectangles: List[RectangleSpec] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="List of two rectangles with position and size.",
    )
    triangle: RightTriangleSpec = Field(
        ...,
        description="A right-angled triangle defined by right-angle vertex, base, and height.",
    )

    @model_validator(mode="after")
    def validate_composite_figure(cls, values):
        rectangles = values.rectangles
        triangle = values.triangle
        
        # Check that all shapes have the same unit
        units = [r.unit for r in rectangles] + [triangle.unit]
        if len(set(units)) > 1:
            raise ValueError("All shapes must have the same unit of measurement.")
        
        # Check that rectangles are not identical
        r1, r2 = rectangles
        if (
            r1.x == r2.x
            and r1.y == r2.y
            and r1.width == r2.width
            and r1.length == r2.length
        ):
            raise ValueError(
                "The two rectangles must not be identical (completely overlapping)."
            )
        
        # Create shapely geometries for validation
        rect1_poly = box(r1.x, r1.y, r1.x + r1.width, r1.y + r1.length)
        rect2_poly = box(r2.x, r2.y, r2.x + r2.width, r2.y + r2.length)
        
        # Create triangle polygon from right-angle vertex, base, and height
        # The triangle vertices are: right-angle vertex, base endpoint, height endpoint
        triangle_vertices = [
            (triangle.right_angle_x, triangle.right_angle_y),  # Right-angle vertex
            (triangle.right_angle_x + triangle.base, triangle.right_angle_y),  # Base endpoint
            (triangle.right_angle_x, triangle.right_angle_y + triangle.height)  # Height endpoint
        ]
        triangle_poly = ShapelyPolygon(triangle_vertices)
        
        # Check that triangle is valid (has non-zero area)
        if triangle_poly.area <= 0:
            raise ValueError("Triangle must have a positive area (base and height must be greater than 0).")
        
        # Check that all shapes touch or overlap (not completely disjoint)
        shapes = [rect1_poly, rect2_poly, triangle_poly]
        connected = False
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if shapes[i].intersects(shapes[j]) or shapes[i].touches(shapes[j]):
                    connected = True
                    break
            if connected:
                break
        
        if not connected:
            raise ValueError("At least some of the shapes must touch or overlap to form a composite figure.")
        
        return values


if __name__ == "__main__":
    CompositeRectangularGrid.generate_assistant_function_schema("mcq4")
    CompositeRectangularTriangularGrid.generate_assistant_function_schema("mcq4")
