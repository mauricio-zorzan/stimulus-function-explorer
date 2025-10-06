from typing import List, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator, model_validator


class PolygonPerimeter(StimulusDescription):
    """
    Defines a polygon for perimeter calculation problems.

    The polygon has known side lengths with visible labels and unknown sides with no labels.
    Used for problems where students need to find the missing side lengths
    given the perimeter or other side lengths.

    Supports:
    - Regular polygons with 3-10 sides (triangle, quadrilateral, pentagon, hexagon, heptagon, octagon, enneagon, decagon)
    - Irregular polygons with 3-10 sides (scales according to actual side length values)
    - L-shaped polygons (6 sides in an L configuration)
    - T-shaped polygons (8 sides in a T configuration)
    """

    side_lengths: List[int] = Field(
        min_length=3,
        max_length=10,
        description="List of side lengths as positive integers (3-10 sides supported)",
    )
    unknown_side_indices: List[int] = Field(
        default_factory=list,
        description="List of indices (0-based) of sides that should be marked with '?' instead of showing their lengths. If empty, all sides show their measurements.",
    )
    unit: str = Field(
        description="Unit string for side lengths (e.g., 'cm', 'm', 'units')",
    )
    shape_type: Optional[Literal["regular", "irregular", "L-shape", "T-shape"]] = Field(
        default="regular",
        description="Type of polygon shape: 'regular' for standard polygons with equal visual sides, 'irregular' for polygons that scale to actual side lengths, 'L-shape' for L-shaped polygons, 'T-shape' for T-shaped polygons",
    )

    @field_validator("side_lengths")
    @classmethod
    def validate_side_lengths_for_shape_type(cls, v, info):
        if "shape_type" not in info.data:
            return v

        shape_type = info.data.get("shape_type", "regular")

        if shape_type == "L-shape" and len(v) != 6:
            raise ValueError("L-shaped polygons must have exactly 6 sides")
        elif shape_type == "T-shape" and len(v) != 8:
            raise ValueError("T-shaped polygons must have exactly 8 sides")
        elif shape_type in ["regular", "irregular"] and not (3 <= len(v) <= 10):
            raise ValueError("Regular and irregular polygons must have 3-10 sides")

        return v

    @model_validator(mode="after")
    def validate_shape_closure(self):
        """Validate that side lengths form valid closed shapes for L-shape and T-shape polygons."""
        shape_type = self.shape_type
        side_lengths = self.side_lengths
        violations = []

        if shape_type == "L-shape" and len(side_lengths) == 6:
            # L-shape closure constraints
            if not (side_lengths[1] + side_lengths[3] == side_lengths[5]):
                violations.append(
                    f"Vertical closure: {side_lengths[1]} + {side_lengths[3]} = {side_lengths[1] + side_lengths[3]} ≠ {side_lengths[5]}"
                )
            if not (side_lengths[2] + side_lengths[4] == side_lengths[0]):
                violations.append(
                    f"Horizontal closure: {side_lengths[2]} + {side_lengths[4]} = {side_lengths[2] + side_lengths[4]} ≠ {side_lengths[0]}"
                )

        elif shape_type == "T-shape" and len(side_lengths) == 8:
            # T-shape closure constraints
            if not (side_lengths[1] == side_lengths[7]):
                violations.append(
                    f"Symmetry constraint: side_lengths[1] = {side_lengths[1]} ≠ {side_lengths[7]} = side_lengths[7]"
                )
            if not (side_lengths[2] == side_lengths[6]):
                violations.append(
                    f"Symmetry constraint: side_lengths[2] = {side_lengths[2]} ≠ {side_lengths[6]} = side_lengths[6]"
                )
            if not (side_lengths[3] == side_lengths[5]):
                violations.append(
                    f"Symmetry constraint: side_lengths[3] = {side_lengths[3]} ≠ {side_lengths[5]} = side_lengths[5]"
                )
            if not (
                side_lengths[1] + side_lengths[3] == side_lengths[5] + side_lengths[7]
            ):
                violations.append(
                    f"Vertical closure: {side_lengths[1]} + {side_lengths[3]} = {side_lengths[1] + side_lengths[3]} ≠ {side_lengths[5] + side_lengths[7]} = {side_lengths[5]} + {side_lengths[7]}"
                )
            if not (
                side_lengths[0] + side_lengths[2] + side_lengths[6] == side_lengths[4]
            ):
                violations.append(
                    f"Horizontal closure: {side_lengths[0]} + {side_lengths[2]} + {side_lengths[6]} = {side_lengths[0] + side_lengths[2] + side_lengths[6]} ≠ {side_lengths[4]}"
                )
        elif shape_type == "regular":
            if not (sum(side_lengths) == side_lengths[0] * len(side_lengths)):
                violations.append(
                    f"Regular polygon closure: {sum(side_lengths)} ≠ {side_lengths[0]} * {len(side_lengths)}"
                )
        elif shape_type == "irregular":
            if len(side_lengths) > 8:
                violations.append(
                    f"Irregular polygon must have at most 8 sides, but got {len(side_lengths)}"
                )

        if violations:
            violation_list = "\n  - ".join([""] + violations)
            raise ValueError(
                f"{shape_type} validation failed with {len(violations)} constraint violation(s):{violation_list}"
            )

        return self

    @field_validator("unknown_side_indices")
    @classmethod
    def validate_unknown_side_indices(cls, v, info):
        if "side_lengths" not in info.data:
            return v

        side_lengths = info.data.get("side_lengths", [])
        if not side_lengths:
            return v

        # Check that all indices are valid for the number of sides
        for index in v:
            if index < 0 or index >= len(side_lengths):
                raise ValueError(
                    f"Unknown side index {index} is out of range for {len(side_lengths)} sides"
                )

        # Check for duplicate indices
        if len(v) != len(set(v)):
            raise ValueError("Unknown side indices must be unique")

        return v

    class Config:
        json_encoders = {int: lambda i: int(i)}


if __name__ == "__main__":
    PolygonPerimeter.generate_assistant_function_schema("mcq4")
