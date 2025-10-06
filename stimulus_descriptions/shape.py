from enum import Enum

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class EShapeType(str, Enum):
    RECTANGLE = "rectangle"
    PARALLELOGRAM = "parallelogram"
    TRIANGLE = "triangle"
    RHOMBUS = "rhombus"


class Shape(StimulusDescription):
    shape_type: EShapeType
    height: float = Field(
        ...,
        description="The height of the shape.",
    )
    base: float = Field(
        ...,
        description="The base width of the shape.",
    )
    unit: str = Field(
        ...,
        description="The unit of measurement for the height and base of the shape.",
    )

    @model_validator(mode="after")
    def validate_dimension_ratio(self):
        """Validate that the height-to-base ratio is not extreme to prevent bad images."""
        if self.base <= 0:
            raise ValueError("Base must be positive")
        if self.height <= 0:
            raise ValueError("Height must be positive")

        ratio = self.height / self.base

        # Define reasonable bounds for the ratio (1/5 to 5/1)
        min_ratio = 0.2  # 1/5
        max_ratio = 5.0  # 5/1

        if ratio < min_ratio:
            raise ValueError(
                f"Height-to-base ratio {ratio:.2f} is too small. "
                f"Ratio must be at least {min_ratio} (height should be at least {min_ratio * 100}% of base)."
            )
        if ratio > max_ratio:
            raise ValueError(
                f"Height-to-base ratio {ratio:.2f} is too large. "
                f"Ratio must be at most {max_ratio} (height should be at most {max_ratio * 100}% of base)."
            )

        return self


if __name__ == "__main__":
    Shape.generate_assistant_function_schema("mcq4")
