from enum import Enum
from typing import List, Optional

from pydantic import Field, model_validator

from .stimulus_description import StimulusDescription

# Configurable parameters - can be changed as per requirements
MAX_ROWS = 2
MAX_TOTAL_SHAPES = 10  # Maximum total shapes in the diagram
DEFAULT_SHAPE_SIZE = 0.8  # Default shape size for multi-row layouts
SINGLE_ROW_SHAPE_SIZE = 0.5  # Smaller shape size for single row layouts


class RatioObjectShape(str, Enum):
    CIRCLE = "circle"
    SQUARE = "square"
    TRIANGLE = "triangle"
    STAR = "star"
    HEXAGON = "hexagon"


class RatioObjectCell(StimulusDescription):
    shape: RatioObjectShape = Field(
        ..., description="Shape: circle, square, triangle, star, or hexagon."
    )
    color: str = Field(..., description="Matplotlib color string or hex code.")


class RatioObjectArray(StimulusDescription):
    rows: int = Field(
        ..., ge=1, le=MAX_ROWS, description=f"Rows in the grid (max {MAX_ROWS})."
    )
    columns: int = Field(..., ge=1, le=12, description="Columns in the grid.")
    objects: List[List[RatioObjectCell]] = Field(
        ...,
        description=(
            "2D list of cells (row-major). The rendered image must include 2–4 distinct shapes."
        ),
    )
    shape_size: Optional[float] = Field(
        default=None,
        ge=0.3,
        le=1.0,
        description=f"Size of shapes (0.3-1.0). If not specified, uses {DEFAULT_SHAPE_SIZE} for multi-row or {SINGLE_ROW_SHAPE_SIZE} for single row.",
    )

    @model_validator(mode="after")
    def _validate_total_shapes(self):
        """Validate that total number of shapes doesn't exceed maximum."""
        total_shapes = self.rows * self.columns
        if total_shapes > MAX_TOTAL_SHAPES:
            raise ValueError(
                f"Total shapes ({total_shapes}) exceeds maximum allowed ({MAX_TOTAL_SHAPES}). "
                f"Reduce rows or columns to stay within limit."
            )
        return self

    def get_effective_shape_size(self) -> float:
        """Get the effective shape size based on rows and user input."""
        if self.shape_size is not None:
            return self.shape_size

        # Auto-adjust based on number of rows
        if self.rows == 1:
            return SINGLE_ROW_SHAPE_SIZE
        else:
            return DEFAULT_SHAPE_SIZE


if __name__ == "__main__":
    RatioObjectArray.generate_assistant_function_schema(type="mcq4")
