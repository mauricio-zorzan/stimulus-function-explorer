from enum import Enum
from fractions import Fraction

from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    EAbbreviatedMeasurementUnit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field, field_validator


class LabelingStyle(str, Enum):
    """Style for unit labeling - abbreviated or full names"""
    ABBREVIATED = "abbreviated"  # Default: cm, in, ft, etc.
    FULL_NAMES = "full_names"    # centimeters, inches, feet, etc.


class RectangularGridItem(BaseModel):
    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )
    length: Fraction | int = Field(
        ...,
        description="The length of the grid, can be a fraction or integer.",
    )
    width: Fraction | int = Field(
        ...,
        description="The width of the grid, can be a fraction or integer.",
    )
    labeling_style: LabelingStyle = Field(
        default=LabelingStyle.ABBREVIATED,
        description="Style for unit labeling: 'abbreviated' (cm, in) or 'full_names' (centimeters, inches)"
    )

    @field_validator("length", "width")
    @classmethod
    def validate_dimensions(cls, v: Fraction | int) -> Fraction | int:
        if isinstance(v, int):
            if v < 1 or v > 21:
                raise ValueError("Integer dimensions must be between 1 and 21")
        elif isinstance(v, Fraction):
            if v <= 0 or v > 21:
                raise ValueError(
                    "Fraction dimensions must be positive and less than 21"
                )
        return v


class RectangularGridList(StimulusDescriptionList[RectangularGridItem]):
    root: list[RectangularGridItem] = Field(max_length=4)


if __name__ == "__main__":
    RectangularGridList.generate_assistant_function_schema("mcq4")
