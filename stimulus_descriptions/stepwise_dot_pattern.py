from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from .stimulus_description import StimulusDescription


class StepwiseShapePatternStep(BaseModel):
    rows: int = Field(..., description="Number of rows of shapes in this step.")
    columns: int = Field(..., description="Number of columns of shapes in this step.")
    color: str = Field(default="#e78be7", description="Color of the shapes (hex or named color).")
    shape: Literal['circle', 'square', 'triangle'] = Field('circle', description="Shape type: 'circle', 'square', or 'triangle'.")
    rotation: Optional[float] = Field(default=0.0, description="Rotation in degrees (only used for triangles; ignored for other shapes).")
    label: Optional[str] = Field(default=None, description="Optional label for this step (e.g., 'step 1').")

class StepwiseShapePattern(StimulusDescription):
    steps: List[StepwiseShapePatternStep] = Field(..., min_length=1, description="List of step definitions for the shape pattern.")
    shape_size: float = Field(default=1.0, description="Relative size of the shapes.")
    spacing: float = Field(default=0.5, description="Horizontal spacing between steps.") 