from enum import Enum
from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_validator


class ShapeEnum(str, Enum):
    ellipse = "ellipse"
    rectangle = "rectangle"
    diamond = "diamond"
    plaintext = "plaintext"


class FlowchartNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node")
    label: Optional[str] = Field(None, description="Label for the node", max_length=50)
    shape: ShapeEnum = Field(..., description="Shape of the node")

    # No additional validators needed


class FlowchartEdge(BaseModel):
    from_: str = Field(..., description="ID of the starting node")
    to: str = Field(..., description="ID of the ending node")
    label: Optional[str] = Field(None, description="Label for the edge", max_length=40)


class FlowchartData(BaseModel):
    nodes: List[FlowchartNode] = Field(
        ..., description="List of nodes in the flowchart", min_length=1, max_length=10
    )
    edges: List[FlowchartEdge] = Field(
        ...,
        description="List of edges connecting the nodes",
        min_length=1,
        max_length=12,
    )
    orientation: Optional[str] = Field(
        "horizontal",
        description="Orientation of the flowchart",
    )

    @field_validator("orientation")
    @classmethod
    def validate_orientation(cls, v):
        if v not in ("horizontal", "vertical"):
            raise ValueError("Orientation must be 'horizontal' or 'vertical'")
        return v


class Flowchart(StimulusDescription):
    flowchart: FlowchartData = Field(..., description="Flowchart data")
