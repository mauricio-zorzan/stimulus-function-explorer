from enum import Enum
from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator, model_validator


class QuadrilateralShapeType(str, Enum):
    """Types of quadrilateral shapes that can be drawn."""

    RHOMBUS = "rhombus"
    IRREGULAR_QUADRILATERAL = "irregular_quadrilateral"
    PARALLELOGRAM = "parallelogram"
    PARALLELOGRAM_WITH_HEIGHT = "parallelogram_with_height"


class QuadrilateralFigures(StimulusDescription):
    """
    Defines quadrilateral figures with configurable properties.

    Can draw 1-4 quadrilateral shapes with the following features:
    - Shape types: rhombus, parallelogram, or irregular quadrilateral
    - Side labels: custom labels for each side of each shape
    - Tick marks: option to show red tick marks instead of side labels
    - Optional rotation: random rotation when enabled, standard orientation when disabled
    - Random colors for visual variety
    - Figure labels: automatically assigned as "Figure 1", "Figure 2", etc. for multiple shapes
    """

    shape_types: List[QuadrilateralShapeType] = Field(
        min_length=1,
        max_length=4,
        description="List of shape types to draw (1-4 shapes). Each can be rhombus, parallelogram, or irregular_quadrilateral.",
    )

    @field_validator("shape_types", mode="before")
    @classmethod
    def normalize_shape_types_case(cls, v):
        """Convert uppercase shape type names to lowercase to match enum values."""
        if isinstance(v, list):
            return [item.lower() if isinstance(item, str) else item for item in v]
        return v

    side_labels: List[List[str]] = Field(
        description="List of side labels for each shape. Each inner list should have 4 labels for the 4 sides of each quadrilateral."
    )

    show_ticks: bool = Field(
        default=False,
        description="If True, shows red tick marks on sides instead of side labels. If False, shows the side labels.",
    )

    rotation: bool = Field(
        default=False,
        description="If True, applies random rotation to each quadrilateral shape. If False, shapes appear in standard orientation.",
    )

    @model_validator(mode="after")
    def validate_side_labels_match_shapes(self):
        """Validate that side_labels has the correct structure for the number of shapes."""
        if len(self.side_labels) != len(self.shape_types):
            raise ValueError(
                f"Number of side_labels lists ({len(self.side_labels)}) must match number of shape_types ({len(self.shape_types)})"
            )

        for i, labels in enumerate(self.side_labels):
            if len(labels) != 4:
                raise ValueError(
                    f"Each quadrilateral must have exactly 4 side labels. Shape {i+1} has {len(labels)} labels."
                )

        return self


class ParallelogramWithHeight(StimulusDescription):
    """
    Defines a parallelogram with height line.
    Features:
    - Support for base, height, and slant side labels
    - Dashed perpendicular height line
    - Right angle marker where height meets base
    - Variable slant angle determined by the height and slant side measurements
      (angle varies between ~45° and ~75° to ensure clear visibility)
    """

    base_label: str = Field(
        description="Label for the base of the parallelogram (e.g., '8 cm' or 'b')"
    )
    height_label: str = Field(
        description="Label for the height of the parallelogram (e.g., '6 cm' or 'h')"
    )
    slant_side_label: str = Field(
        description="Label for the slant side of the parallelogram (e.g., '8 cm')"
    )

    @model_validator(mode="after")
    def validate_measurements(self) -> "ParallelogramWithHeight":
        """Validate that parallelogram measurements are geometrically reasonable."""
        base = (
            float(self.base_label.replace("cm", ""))
            if self.base_label not in ["b", "h"]
            else None
        )
        height = (
            float(self.height_label.replace("cm", ""))
            if self.height_label not in ["b", "h"]
            else None
        )
        slant = float(self.slant_side_label.replace("cm", ""))

        # Basic geometric constraints
        if height and slant:
            if height >= slant:
                raise ValueError(
                    "Height must be less than slant side (geometric constraint)"
                )

            # Limit angle to avoid extremely skewed parallelograms
            if height < slant * 0.3:  # Roughly 70 degree angle
                raise ValueError(
                    "Height should be at least 30% of slant side to avoid extreme angles"
                )

        if base and slant:
            # Keep base and slant within reasonable proportions
            if base >= slant * 2:
                raise ValueError(
                    "Base cannot be more than or equal to twice the slant side"
                )
            if slant >= base * 2:
                raise ValueError(
                    "Slant side cannot be more than or equal to twice the base"
                )

        return self
