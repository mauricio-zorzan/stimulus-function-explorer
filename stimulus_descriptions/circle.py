from enum import Enum
from typing import Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field


class Circle(StimulusDescription):
    radius: float = Field(..., description="The radius of the circle.")
    unit: str = Field(..., description="The unit of measurement for the radius.")


class CircleElementType(str, Enum):
    RADIUS = "radius"
    DIAMETER = "diameter"
    CHORD = "chord"

class CircleElement(BaseModel):
    """Represents an element to be drawn on the circle (radius, diameter, or chord)"""
    element_type: CircleElementType = Field(..., description="Type of element to draw")
    endpoint_labels: list[str] = Field(default=["A", "B"], description="Labels for the endpoints of the element")


class CircleDiagram(Circle):
    """
    Circle diagram that can display a single radius, diameter, or chord element
    Inherits radius and unit from the base Circle class
    """
    element: Optional[CircleElement] = Field(
        default=None,
        description="Single element to draw (radius, diameter, or chord)"
    )


if __name__ == "__main__":
    Circle.generate_assistant_function_schema("mcq4")
