from enum import Enum

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator


class AngleRange(str, Enum):
    """Generic angle range categories for organizational purposes."""

    BASIC = "basic"  # 0 - 180 degrees
    INTERMEDIATE = "intermediate"  # 180 - 360 degrees
    ADVANCED = "advanced"  # > 360 degrees


class CircleAngle(StimulusDescription):
    """
    A circle with degree markings showing an angle measurement exercise.

    The circle has dashes marking every 15 degrees with major degree labels
    at cardinal directions (0°, 90°, 180°, 270°).
    """

    angle_measure: int = Field(
        ..., description="The angle measure in degrees, must be a multiple of 15"
    )

    start_position: int = Field(
        default=0,
        description="Starting position of the angle in degrees (default 0° - rightmost position)",
    )

    range_category: AngleRange = Field(
        ..., description="Category of angle range for organizational purposes"
    )

    show_question: bool = Field(
        default=False, description="Whether to show the measurement question"
    )

    sector_color: str = Field(
        default="lightgreen", description="Color of the shaded angle sector"
    )

    @field_validator("angle_measure")
    @classmethod
    def validate_angle_multiple_of_15(cls, v):
        """Ensure angle is a multiple of 15 degrees."""
        if v % 15 != 0:
            raise ValueError("Angle measure must be a multiple of 15 degrees")
        if v <= 0 or v >= 360:
            raise ValueError("Angle measure must be between 1 and 359 degrees")
        return v

    @field_validator("start_position")
    @classmethod
    def validate_start_position(cls, v):
        """Ensure start position is valid."""
        if v < 0 or v >= 360:
            raise ValueError("Start position must be between 0 and 359 degrees")
        if v % 15 != 0:
            raise ValueError("Start position must be a multiple of 15 degrees")
        return v


if __name__ == "__main__":
    CircleAngle.generate_assistant_function_schema("mcq4")
