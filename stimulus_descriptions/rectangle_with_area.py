from fractions import Fraction

from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    EAbbreviatedMeasurementUnit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator


class RectangleWithHiddenSide(StimulusDescription):
    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )
    length: Fraction | int = Field(
        ...,
        description="The length of the rectangle, can be a fraction or integer.",
    )
    width: Fraction | int = Field(
        ...,
        description="The width of the rectangle, can be a fraction or integer.",
    )
    show_length: bool = Field(
        ...,
        description="Whether to show the length dimension. If False, length will be labeled with '?'.",
    )
    show_width: bool = Field(
        ...,
        description="Whether to show the width dimension. If False, width will be labeled with '?'.",
    )

    @field_validator("length", "width")
    @classmethod
    def validate_length_width(cls, v: Fraction | int) -> Fraction | int:
        if isinstance(v, int):
            if v < 1 or v > 100:
                raise ValueError("Integer dimensions must be between 1 and 100")
        elif isinstance(v, Fraction):
            if v <= 0 or v > 100:
                raise ValueError(
                    "Fraction dimensions must be positive and less than 100"
                )
        return v


if __name__ == "__main__":
    RectangleWithHiddenSide.generate_assistant_function_schema("mcq4") 