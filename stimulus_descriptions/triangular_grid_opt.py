import math
from enum import Enum
from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    EAbbreviatedMeasurementUnit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class ExerciseType(Enum):
    """Types of Pythagorean theorem exercises."""

    FIND_HYPOTENUSE = "find_hypotenuse"  # Given two legs, find hypotenuse
    FIND_LEG = "find_leg"  # Given leg and hypotenuse, find missing leg
    VERIFY_RIGHT_TRIANGLE = "verify_right_triangle"  # Determine if triangle is right


class TriangularGridOpt(StimulusDescription):
    """
    Triangular grid with optional side lengths for Pythagorean theorem problems.

    This model allows 2-3 sides to be specified, and can calculate missing sides
    using the Pythagorean theorem when applicable. Designed for real-world
    problem solving using the Pythagorean theorem in two dimensions.

    Supports three types of exercises:
    1. Find hypotenuse: Given two legs, find the hypotenuse
    2. Find leg: Given a leg and hypotenuse, find the missing leg
    3. Verify right triangle: Determine if triangle with given sides is right
    """

    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )
    side1: Optional[float] = Field(
        None,
        ge=0.1,
        le=50.0,
        description="The length of the first side (can be left as None to calculate using Pythagorean theorem).",
    )
    side2: Optional[float] = Field(
        None,
        ge=0.1,
        le=50.0,
        description="The length of the second side (can be left as None to calculate using Pythagorean theorem).",
    )
    side3: Optional[float] = Field(
        None,
        ge=0.1,
        le=50.0,
        description="The length of the third side (can be left as None to calculate using Pythagorean theorem).",
    )
    show_calculation: bool = Field(
        default=False,
        description="Whether to show the Pythagorean theorem calculation for missing sides.",
    )
    label_unknown: bool = Field(
        default=True,
        description="Whether to label unknown sides with variable names (a, b, c) instead of calculated values.",
    )
    exercise_type: ExerciseType = Field(
        default=ExerciseType.FIND_HYPOTENUSE,
        description="The type of exercise: find_hypotenuse, find_leg, or verify_right_triangle.",
    )
    rotation_angle: int = Field(
        default=0,
        description="Rotation angle in degrees (0, 90, 180, 270).",
    )
    show_right_angle_symbol: Optional[bool] = Field(
        default=None,
        description="Whether to show the right angle symbol. If None, determined by exercise_type.",
    )
    label_sides: Optional[List[bool]] = Field(
        default=None,
        description="List of 3 booleans indicating which sides to label [side1, side2, side3]. If None, determined by exercise_type.",
    )

    @model_validator(mode="after")
    def validate_triangle_and_calculate_missing_side(self):
        """
        Validates the triangle and calculates missing sides using Pythagorean theorem.
        """
        # Validate rotation angle
        if self.rotation_angle not in [0, 90, 180, 270]:
            raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees.")

        sides = [self.side1, self.side2, self.side3]
        known_sides = [s for s in sides if s is not None]

        # Must have at least 2 sides to work with Pythagorean theorem
        if len(known_sides) < 2:
            raise ValueError(
                "At least 2 sides must be provided to use the Pythagorean theorem."
            )

        # If all 3 sides are provided, validate they form a valid triangle
        if len(known_sides) == 3:
            # Check triangle inequality
            sorted_sides = sorted(known_sides)
            if sorted_sides[0] + sorted_sides[1] <= sorted_sides[2]:
                raise ValueError(
                    "The provided sides do not form a valid triangle (triangle inequality violated)."
                )

            # For verify_right_triangle exercise, we don't enforce it to be a right triangle
            if self.exercise_type != ExerciseType.VERIFY_RIGHT_TRIANGLE:
                # Check if it's a right triangle using Pythagorean theorem converse
                # Allow for small floating point errors
                a, b, c = sorted_sides
                if not math.isclose(a**2 + b**2, c**2, rel_tol=1e-9):
                    raise ValueError(
                        "The provided sides do not form a right triangle. "
                        "This function is designed for Pythagorean theorem problems with right triangles."
                    )

        # If exactly 2 sides are provided, calculate the third using Pythagorean theorem
        elif len(known_sides) == 2:
            # Determine which side is missing and calculate it based on exercise type
            provided_values = [s for s in sides if s is not None]
            a, b = sorted(provided_values)

            if self.exercise_type == ExerciseType.FIND_HYPOTENUSE:
                # Calculate hypotenuse: c² = a² + b²
                calculated_side = math.sqrt(a**2 + b**2)
            elif self.exercise_type == ExerciseType.FIND_LEG:
                # Calculate missing leg: a² = c² - b² (where c is hypotenuse)
                # Assume the larger provided side is the hypotenuse
                hypotenuse = max(provided_values)
                leg = min(provided_values)
                if hypotenuse <= leg:
                    raise ValueError(
                        "For find_leg exercise, one side must be longer (hypotenuse)."
                    )
                calculated_side = math.sqrt(hypotenuse**2 - leg**2)
            else:  # VERIFY_RIGHT_TRIANGLE
                # Default to calculating hypotenuse
                calculated_side = math.sqrt(a**2 + b**2)

            # Assign the calculated side to the missing position
            if self.side1 is None:
                self.side1 = calculated_side
            elif self.side2 is None:
                self.side2 = calculated_side
            elif self.side3 is None:
                self.side3 = calculated_side

        # Set default values for display options based on exercise type
        if self.show_right_angle_symbol is None:
            self.show_right_angle_symbol = (
                self.exercise_type != ExerciseType.VERIFY_RIGHT_TRIANGLE
            )

        if self.label_sides is None:
            if self.exercise_type == ExerciseType.FIND_HYPOTENUSE:
                # Label only the two legs (find which sides are legs vs hypotenuse)
                sides_list = [self.side1, self.side2, self.side3]
                # All sides should be non-None at this point
                non_none_sides = [s for s in sides_list if s is not None]
                if len(non_none_sides) == 3:
                    max_side_value = max(non_none_sides)
                    max_side_index = sides_list.index(max_side_value)
                    self.label_sides = [i != max_side_index for i in range(3)]
                else:
                    self.label_sides = [True, True, True]
            elif self.exercise_type == ExerciseType.FIND_LEG:
                # Label one leg and the hypotenuse
                sides_list = [self.side1, self.side2, self.side3]
                # All sides should be non-None at this point
                non_none_sides = [s for s in sides_list if s is not None]
                if len(non_none_sides) == 3:
                    max_side_value = max(non_none_sides)
                    min_side_value = min(non_none_sides)
                    max_side_index = sides_list.index(max_side_value)
                    min_side_index = sides_list.index(min_side_value)
                    self.label_sides = [
                        i == max_side_index or i == min_side_index for i in range(3)
                    ]
                else:
                    self.label_sides = [True, True, True]
            else:  # VERIFY_RIGHT_TRIANGLE
                # Label all sides
                self.label_sides = [True, True, True]

        return self

    def get_triangle_type(self) -> str:
        """Returns the type of triangle based on the Pythagorean theorem."""
        sides = [self.side1, self.side2, self.side3]
        known_sides = [s for s in sides if s is not None]

        if len(known_sides) == 3:
            sorted_sides = sorted(known_sides)
            a, b, c = sorted_sides

            if math.isclose(a**2 + b**2, c**2, rel_tol=1e-9):
                return "right"
            elif a**2 + b**2 > c**2:
                return "acute"
            else:
                return "obtuse"

        return "right"  # Default for Pythagorean theorem problems

    def get_side_values(self) -> tuple[float, float, float]:
        """Returns all three side values, with calculated values filled in."""
        return (self.side1 or 0.0, self.side2 or 0.0, self.side3 or 0.0)
