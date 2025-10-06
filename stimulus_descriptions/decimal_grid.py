from enum import Enum
from typing import List, Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_validator


class ComparisonLevel(str, Enum):
    """Generic comparison complexity levels."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class DecimalGrid(StimulusDescription):
    """
    Decimal grids showing squares divided into either 10 or 100 equal parts.
    Used for decimal representation and fraction-to-decimal conversion exercises.

    - division=10: Creates 1x10 rectangular grids
    - division=100: Creates 10x10 square grids

    When shaded_squares exceeds the division value, multiple grids are created in rows.
    The shaded squares fill column by column (left to right, top to bottom within each column).
    Maximum of 10 fully shaded grids are allowed, arranged in up to 2 rows (5 grids per row).

    Examples:
    - 235 shaded with division=100: 2 fully shaded grids + 1 grid with 35 shaded squares
    - 25 shaded with division=10: 2 fully shaded grids + 1 grid with 5 shaded squares
    - 850 shaded with division=100: 8 fully shaded grids + 1 grid with 50 shaded squares
    """

    division: Literal[10, 100] = Field(
        ...,
        description="The number of equal parts the square is divided into. Must be either 10 or 100.",
    )

    shaded_squares: int = Field(
        ...,
        description="The number of squares to shade in the grid. Can exceed division value to create multiple grids (max 10 full grids).",
    )

    @field_validator("shaded_squares")
    @classmethod
    def validate_shaded_squares(cls, v, info):
        """Validate that shaded_squares is non-negative and doesn't exceed maximum allowed."""
        if hasattr(info, "data") and "division" in info.data:
            division = info.data["division"]
            max_allowed = division * 10  # Maximum 10 fully shaded grids
            if v < 0:
                raise ValueError("shaded_squares must be non-negative")
            if v > max_allowed:
                raise ValueError(
                    f"shaded_squares ({v}) cannot exceed maximum of {max_allowed} (10 full grids of {division})"
                )
        return v


class DecimalComparison(BaseModel):
    """A pair of decimal values for comparison exercises."""

    decimal_1: float = Field(..., description="First decimal value to compare")
    decimal_2: float = Field(..., description="Second decimal value to compare")
    complexity_level: ComparisonLevel = Field(
        ..., description="Complexity level of the comparison"
    )
    color_1: str = Field(
        default="lightblue", description="Color for first decimal grid"
    )
    color_2: str = Field(
        default="lightblue", description="Color for second decimal grid"
    )

    @field_validator("decimal_1", "decimal_2")
    @classmethod
    def validate_decimal_range(cls, v):
        """Ensure decimal values are within reasonable range for visualization."""
        v = round(v, 2)
        if v < 0 or v >= 10:
            raise ValueError(
                "Decimal values must be between 0 and 9.99 for proper grid visualization"
            )
        return v

    @field_validator("complexity_level")
    @classmethod
    def validate_complexity_constraints(cls, v, info):
        """Apply complexity-based validation rules."""
        if (
            hasattr(info, "data")
            and "decimal_1" in info.data
            and "decimal_2" in info.data
        ):
            d1, d2 = info.data["decimal_1"], info.data["decimal_2"]
            whole_part_1, whole_part_2 = int(d1), int(d2)

            # Basic level: values without whole number parts
            if v == ComparisonLevel.BASIC and (whole_part_1 != 0 or whole_part_2 != 0):
                raise ValueError("Basic level requires values less than 1")
            # Intermediate level: values with matching whole number parts
            elif v == ComparisonLevel.INTERMEDIATE and whole_part_1 != whole_part_2:
                raise ValueError(
                    "Intermediate level requires matching whole number parts"
                )
        return v


class DecimalComparisonList(StimulusDescription):
    """Collection of decimal comparison pairs for visual comparison exercises."""

    comparisons: List[DecimalComparison] = Field(
        ..., description="List of decimal comparison pairs"
    )
    show_solutions: bool = Field(
        default=False, description="Whether to display solution indicators"
    )

    @field_validator("comparisons")
    @classmethod
    def validate_comparison_list(cls, v):
        """Ensure comparison list is not empty."""
        if not v:
            raise ValueError("At least one comparison pair must be provided")
        return v


class DecimalMultiplication(StimulusDescription):
    """
    Decimal multiplication visualization for products of the form 0.a × 0.b or 0.a × 1.b.

    Uses 10×10 grids (100 squares each) to show multiplication visually:
    
    For 0.a × 0.b (single grid):
    1. First shades b×10 squares with a base color (representing 0.b)
    2. Then shades a rows of that shaded section with a pattern/darker color (representing 0.a × 0.b)
    
    For 0.a × 1.b (two grids side by side):
    1. First shades entire first grid + b×10 squares of second grid with base color (representing 1.b)
    2. Then shades a rows of the entire shaded section with pattern/darker color (representing 0.a × 1.b)

    Examples:
    - 0.1 × 0.8: Shade 80 squares, then add pattern to 1 row (8 squares) = 0.08
    - 0.3 × 0.6: Shade 60 squares, then add pattern to 3 rows (18 squares) = 0.18
    - 0.2 × 1.3: Shade 130 squares (full grid + 30), then add pattern to 2 rows (26 squares) = 0.26
    """

    decimal_factors: List[float] = Field(
        ...,
        description="Two decimal factors: first of form 0.a, second of form 0.b or 1.b where a,b are single digits",
        min_length=2,
        max_length=2,
    )

    @field_validator("decimal_factors", mode="before")
    @classmethod
    def validate_decimal_factors(cls, v):
        """Validate that we have exactly 2 decimals: first of form 0.a, second of form 0.b or 1.b."""
        # Check if v is a list or tuple
        if not isinstance(v, (list, tuple)):
            raise ValueError("decimal_factors must be a list or tuple")

        if len(v) != 2:
            raise ValueError("Must provide exactly 2 decimal factors")

        for i, factor in enumerate(v):
            if not isinstance(factor, (int, float)):
                raise ValueError(f"Factor {i+1} must be a number")

            # First factor must be between 0 and 1 (exclusive)
            if i == 0:
                if factor <= 0 or factor >= 1:
                    raise ValueError(
                        f"First factor ({factor}) must be between 0 and 1 (exclusive)"
                    )
            # Second factor can be between 0 and 2 (exclusive)
            else:
                if factor <= 0 or factor >= 2:
                    raise ValueError(
                        f"Second factor ({factor}) must be between 0 and 2 (exclusive)"
                    )

            # Check if it's a single decimal place by checking the original value
            # Convert to string to check decimal places more reliably
            factor_str = f"{factor:.10f}".rstrip("0").rstrip(".")
            if "." in factor_str:
                decimal_places = len(factor_str.split(".")[1])
                if decimal_places > 1:
                    raise ValueError(
                        f"Factor {i+1} ({factor}) must be a single decimal place (e.g., 0.1, 0.2, 1.3, 1.9)"
                    )

            # Ensure it's exactly one decimal place
            rounded_factor = round(factor, 1)
            if abs(factor - rounded_factor) > 1e-10:
                raise ValueError(
                    f"Factor {i+1} ({factor}) must be a single decimal place (e.g., 0.1, 0.2, 1.3, 1.9)"
                )

        return [round(f, 1) for f in v]  # Return rounded values


if __name__ == "__main__":
    DecimalGrid.generate_assistant_function_schema("mcq4")
    DecimalComparisonList.generate_assistant_function_schema("mcq4")
    DecimalMultiplication.generate_assistant_function_schema("mcq4")
