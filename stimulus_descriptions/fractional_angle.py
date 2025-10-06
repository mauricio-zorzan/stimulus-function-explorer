from fractions import Fraction

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator, model_validator


class FractionalAngle(StimulusDescription):
    """
    A circle divided into equal parts with a shaded sector representing a fraction.

    Shows the relationship between fractions and angle measures in a complete circle.
    The circle is divided into equal parts based on the denominator, with the
    numerator determining how many parts are shaded.
    """

    numerator: int = Field(
        ..., description="Numerator of the fraction (number of shaded parts)"
    )

    denominator: int = Field(
        ...,
        description="Denominator of the fraction (total number of equal parts)",
    )

    sector_color: str = Field(
        default="lightblue", description="Color of the shaded sector"
    )

    show_fraction_label: bool = Field(
        default=True, description="Whether to show the fraction label on the diagram"
    )

    show_angle_measure: bool = Field(
        default=False, description="Whether to show the calculated angle measure"
    )

    @field_validator("numerator", "denominator")
    @classmethod
    def validate_single_digit(cls, v):
        """Ensure numerator and denominator are single-digit numbers."""
        if v < 1 or v > 9:
            raise ValueError("Values must be single-digit numbers (1-9)")
        return v

    @field_validator("denominator")
    @classmethod
    def validate_denominator_factor_of_360(cls, v):
        """Ensure denominator is a factor of 360."""
        if 360 % v != 0:
            raise ValueError("Denominator must be a factor of 360")
        return v

    @model_validator(mode="after")
    def validate_proper_fraction(self):
        """Ensure the fraction is proper (numerator < denominator)."""
        if self.numerator >= self.denominator:
            raise ValueError("Numerator must be less than denominator")
        return self

    @property
    def fraction(self) -> Fraction:
        """Get the fraction representation."""
        return Fraction(self.numerator, self.denominator)

    @property
    def angle_measure(self) -> float:
        """Calculate the angle measure in degrees."""
        return float(self.fraction * 360)

    @property
    def is_unit_fraction(self) -> bool:
        """Check if this is a unit fraction (numerator = 1)."""
        return self.numerator == 1


if __name__ == "__main__":
    FractionalAngle.generate_assistant_function_schema("mcq4")
