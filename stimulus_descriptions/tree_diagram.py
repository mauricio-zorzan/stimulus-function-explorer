from typing import Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field


class TreeDiagramNode(BaseModel):
    label: str = Field(
        ..., description="Label for this node (e.g., 'L', 'M', 'H', 'T')"
    )
    left: Optional["TreeDiagramNode"] = Field(
        None, description="Left child node (or None if leaf)"
    )
    right: Optional["TreeDiagramNode"] = Field(
        None, description="Right child node (or None if leaf)"
    )

    class Config:
        arbitrary_types_allowed = True


TreeDiagramNode.update_forward_refs()


class TreeDiagram(StimulusDescription):
    root: TreeDiagramNode = Field(..., description="Root node of the tree diagram")
    title: Optional[str] = Field(None, description="Optional title for the diagram")
