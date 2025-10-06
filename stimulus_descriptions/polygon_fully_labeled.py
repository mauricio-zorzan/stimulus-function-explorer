from typing import List, Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator, model_validator


class PolygonFullyLabeled(StimulusDescription):
    """
    Defines a polygon where all sides are labeled with their measurements.
    
    Used for educational problems where students need to see all side lengths,
    such as calculating perimeter when all measurements are given.
    
    Supports:
    - Square (4 equal sides)
    - Rectangle (4 sides, opposite sides equal)
    - Regular polygons with 3-10 sides (triangle, pentagon, hexagon, heptagon, octagon, enneagon, decagon)
    - L-shaped polygons (6 sides in an L configuration)
    - T-shaped polygons (8 sides in a T configuration)
    """
    
    side_lengths: List[int] = Field(
        min_length=3,
        max_length=10,
        description="List of side lengths as positive integers (3-10 sides supported)",
    )
    unit: str = Field(
        description="Unit string for side lengths (e.g., 'cm', 'm', 'units')",
    )
    shape_type: Literal["square", "rectangle", "regular", "L-shape", "T-shape"] = Field(
        default="regular",
        description="Type of polygon shape: 'square' for 4 equal sides, 'rectangle' for 4 sides with opposite sides equal, 'regular' for standard polygons with equal visual sides, 'L-shape' for L-shaped polygons, 'T-shape' for T-shaped polygons"
    )

    @field_validator("side_lengths")
    @classmethod
    def validate_side_lengths_for_shape_type(cls, v, info):
        if "shape_type" not in info.data:
            return v
            
        shape_type = info.data.get("shape_type", "regular")
        
        if shape_type == "square" and len(v) != 4:
            raise ValueError("Square must have exactly 4 sides")
        elif shape_type == "rectangle" and len(v) != 4:
            raise ValueError("Rectangle must have exactly 4 sides")
        elif shape_type == "L-shape" and len(v) != 6:
            raise ValueError("L-shaped polygons must have exactly 6 sides")
        elif shape_type == "T-shape" and len(v) != 8:
            raise ValueError("T-shaped polygons must have exactly 8 sides")
        elif shape_type == "regular" and not (3 <= len(v) <= 10):
            raise ValueError("Regular polygons must have 3-10 sides")
            
        return v

    @model_validator(mode='after')
    def validate_shape_constraints(self):
        """Validate that side lengths match the shape type constraints."""
        shape_type = self.shape_type
        side_lengths = self.side_lengths
        violations = []
        
        if shape_type == "square" and len(side_lengths) == 4:
            # All sides must be equal for a square
            if not all(side == side_lengths[0] for side in side_lengths):
                violations.append(f"Square must have all equal sides, but got: {side_lengths}")
                
        elif shape_type == "rectangle" and len(side_lengths) == 4:
            # Opposite sides must be equal for a rectangle
            if not (side_lengths[0] == side_lengths[2] and side_lengths[1] == side_lengths[3]):
                violations.append(f"Rectangle must have opposite sides equal (sides 0,2 and sides 1,3), but got: {side_lengths}")
                
        elif shape_type == "L-shape" and len(side_lengths) == 6:
            # L-shape closure constraints
            if not (side_lengths[1] + side_lengths[3] == side_lengths[5]):
                violations.append(f"L-shape vertical closure: {side_lengths[1]} + {side_lengths[3]} = {side_lengths[1] + side_lengths[3]} ≠ {side_lengths[5]}")
            if not (side_lengths[2] + side_lengths[4] == side_lengths[0]):
                violations.append(f"L-shape horizontal closure: {side_lengths[2]} + {side_lengths[4]} = {side_lengths[2] + side_lengths[4]} ≠ {side_lengths[0]}")
                
        elif shape_type == "T-shape" and len(side_lengths) == 8:
            # T-shape closure constraints
            if not (side_lengths[1] == side_lengths[7]):
                violations.append(f"T-shape symmetry constraint: side_lengths[1] = {side_lengths[1]} ≠ {side_lengths[7]} = side_lengths[7]")
            if not (side_lengths[2] == side_lengths[6]):
                violations.append(f"T-shape symmetry constraint: side_lengths[2] = {side_lengths[2]} ≠ {side_lengths[6]} = side_lengths[6]")
            if not (side_lengths[3] == side_lengths[5]):
                violations.append(f"T-shape symmetry constraint: side_lengths[3] = {side_lengths[3]} ≠ {side_lengths[5]} = side_lengths[5]")
            if not (side_lengths[1] + side_lengths[3] == side_lengths[5] + side_lengths[7]):
                violations.append(f"T-shape vertical closure: {side_lengths[1]} + {side_lengths[3]} = {side_lengths[1] + side_lengths[3]} ≠ {side_lengths[5] + side_lengths[7]} = {side_lengths[5]} + {side_lengths[7]}")
            if not (side_lengths[0] + side_lengths[2] + side_lengths[6] == side_lengths[4]):
                violations.append(f"T-shape horizontal closure: {side_lengths[0]} + {side_lengths[2]} + {side_lengths[6]} = {side_lengths[0] + side_lengths[2] + side_lengths[6]} ≠ {side_lengths[4]}")
        
        if violations:
            violation_list = '\n  - '.join([''] + violations)
            raise ValueError(f"{shape_type} validation failed with {len(violations)} constraint violation(s):{violation_list}")
                
        return self

    class Config:
        json_encoders = {int: lambda i: int(i)}


if __name__ == "__main__":
    PolygonFullyLabeled.generate_assistant_function_schema("mcq4") 