from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    EAbbreviatedMeasurementUnit,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class TriangularGrid(StimulusDescription):
    unit: EAbbreviatedMeasurementUnit = Field(
        ..., description="The unit of measurement, e.g., 'in' for inches."
    )
    side1: int = Field(
        ...,
        ge=1,
        le=21,
        description="The length of the first side, must be a natural number less than 21.",
    )
    side2: int = Field(
        ...,
        ge=1,
        le=21,
        description="The length of the second side, must be a natural number less than 21.",
    )
    side3: int = Field(
        ...,
        ge=1,
        le=21,
        description="The length of the third side, must be a natural number less than 21.",
    )

    @model_validator(mode="after")
    def validate_triangle_inequality(self):
        sides = sorted([self.side1, self.side2, self.side3])
        if sides[0] + sides[1] <= sides[2]:
            raise ValueError(
                "The sum of any two sides must be greater than the third side"
            )
        return self
