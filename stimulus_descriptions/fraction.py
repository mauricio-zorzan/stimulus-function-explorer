import re
from enum import Enum
from typing import TYPE_CHECKING, Annotated, List, Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class FractionShape(str, Enum):
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"


class Fraction(BaseModel):
    """
    Fraction model stimulus description.
    Every rectangle is the same orientation with fractions shown in vertical bars with blue shading starting on the left.
    Every circle is the same orientation with blue shaded slices shown anti-clockwise starting at 12 o'clock.
    """

    shape: Annotated[FractionShape, BeforeValidator(lambda x: x.lower())] = Field(
        description="The shape used to represent the fraction."
    )

    fraction: str = Field(
        pattern=r"^\d+/\d+$",
        description="The label for the x-axis of the line plot with correct context for question. Maximum denominator is 25.",
    )

    @model_validator(mode="before")
    def validate_fraction(cls, fraction):
        validate_fraction(fraction.get("fraction"))

        # Validate triangle fractions
        shape = fraction.get("shape", "").lower()
        fraction_str = fraction.get("fraction", "")

        if shape == "triangle" and fraction_str:
            try:
                numerator, denominator = map(int, fraction_str.split("/"))
                # Triangles can only be divided into halves (2), thirds (3), or sixths (6)
                if denominator not in [2, 3, 6]:
                    raise ValueError(
                        f"Triangles can only be divided into halves (1/2), thirds (1/3), or sixths (1/6). Got denominator: {denominator}"
                    )
            except (ValueError, AttributeError) as e:
                if "can only be divided" not in str(e):
                    raise ValueError(
                        f"Invalid fraction format for triangle: {fraction_str}"
                    )
                raise e

        return fraction


class FractionPair(BaseModel):
    """
    Fraction Pair model stimulus description with the first fraction shaded in blue and the second fraction shaded in green.
    """

    shape: Annotated[FractionShape, BeforeValidator(lambda x: x.lower())] = Field(
        description="The shape used to represent the fraction."
    )

    fractions: list[str] = Field(
        min_length=1,
        description=(
            "The fractions that make up the group of fractions on the figure which all share the same denominator. "
            "Each fraction must match the pattern '^\\d+/\\d+$'. The denominator can be a maximum of 25."
        ),
    )

    @field_validator("fractions", mode="before")
    @classmethod
    def validate_fractions(cls, fractions):
        pattern = r"^\d+/\d+$"
        for fraction in fractions:
            if not re.match(pattern, fraction):
                raise ValueError(f"{fraction} is not a valid fraction")
        return fractions

    @model_validator(mode="before")
    def validate_triangle_fractions(cls, values):
        # Validate triangle fractions
        shape = values.get("shape", "").lower()
        fractions = values.get("fractions", [])

        if shape == "triangle" and fractions:
            for fraction_str in fractions:
                try:
                    numerator, denominator = map(int, fraction_str.split("/"))
                    # Triangles can only be divided into halves (2), thirds (3), or sixths (6)
                    if denominator not in [2, 3, 6]:
                        raise ValueError(
                            f"Triangles can only be divided into halves (1/2), thirds (1/3), or sixths (1/6). Got denominator: {denominator} in fraction: {fraction_str}"
                        )
                except (ValueError, AttributeError) as e:
                    if "can only be divided" not in str(e):
                        raise ValueError(
                            f"Invalid fraction format for triangle: {fraction_str}"
                        )
                    raise e

        return values


class FractionPairSet(FractionPair):
    """
    Fraction Pair model stimulus description with the fractions shaded in the color specified in the color field.
    """

    color: Literal["blue", "green", "red", "orange", "purple", "yellow"] | None = Field(
        default="blue", description="The color of shaded parts of the current shape"
    )


def validate_fraction(fraction_str):
    numerator, denominator = map(int, fraction_str.split("/"))

    if denominator < numerator or denominator >= 25:
        raise ValueError("denominator is larger and not greater than 25")

    return True


def validate_fractions_same_denominator(fractions: list[str]):
    if not fractions:
        return True

    denominators = set()
    total_numerator = 0

    for fraction in fractions:
        numerator, denominator = map(int, fraction.split("/"))
        denominators.add(denominator)
        total_numerator += numerator

    if len(denominators) != 1:
        raise ValueError("All fractions must share the same denominator.")

    common_denominator = denominators.pop()

    if total_numerator > common_denominator:
        raise ValueError(
            f"The sum of numerators ({total_numerator}) is greater than the common denominator ({common_denominator})."
        )

    return True


class FractionSet(StimulusDescription):
    fractions: List[str] = Field(
        ...,
        description="The label for the x-axis of the line plot with correct context for question.",
        min_length=2,
        max_length=2,
    )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        assert all(
            validate_fraction(fraction) for fraction in self.fractions
        ), f"{pipeline_context.standard_id}: All fractions pair must be valid fractions."


class FractionList(StimulusDescriptionList[Fraction]):
    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        match pipeline_context.standard_id:
            case "CCSS.MATH.CONTENT.4.NF.B.3.B+2":
                current_shape = self.root[0].shape
                assert all(
                    fraction.shape == current_shape for fraction in self.root
                ), f"{pipeline_context.standard_id}: All fractions must have the same shape."


class FractionPairSetList(StimulusDescriptionList[FractionPairSet]):
    """Each fractionally shaded shape is labelled as figure 1, 2, 3, etc."""

    pass


class FractionPairList(StimulusDescriptionList[FractionPair]):
    """Each fractionally shaded shape is labelled as figure 1, 2, 3, etc."""

    def validate_same_denominator(self, pipeline_context: "QuestionGeneratorContext"):
        assert all(
            validate_fractions_same_denominator(fraction.fractions)
            for fraction in self.root
        ), f"{pipeline_context.standard_id}: All fractions pair must have same denominator."

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        match pipeline_context.standard_id:
            case "CCSS.MATH.CONTENT.4.NF.B.3.D+1":
                assert all(
                    len(fraction.fractions) == 2 for fraction in self.root
                ), f"{pipeline_context.standard_id}: All fractions pair must in sets of twos."

                assert all(
                    validate_fraction(fraction.fractions[0])
                    and validate_fraction(fraction.fractions[1])
                    for fraction in self.root
                ), f"{pipeline_context.standard_id}: All fractions pair must be valid fractions."
                self.validate_same_denominator(pipeline_context)
            case "CCSS.MATH.CONTENT.5.NF.A.2+1":
                self.validate_same_denominator(pipeline_context)


class DividedShape(BaseModel):
    shape: FractionShape = Field(
        description="The shape used to represent the fraction."
    )
    denominator: int = Field(description="The denominator of the fraction.", ge=1)


class DividedShapeList(StimulusDescriptionList[DividedShape]):
    pass


class UnequalFraction(BaseModel):
    """
    Unequal Fraction model stimulus description.
    Represents a shape that can be either equally or unequally divided.
    All shapes are fully shaded.
    """

    shape: Annotated[FractionShape, BeforeValidator(lambda x: x.lower())] = Field(
        description="The shape used to represent the fraction."
    )
    divided_parts: int = Field(
        ge=2,
        le=25,
        description="The number of parts the shape is divided into. Maximum is 25.",
    )
    equally_divided: bool = Field(
        description="Whether the shape is divided into equal parts (True) or unequal parts (False)."
    )

    @model_validator(mode="after")
    def validate_triangle_constraints(self):
        """
        Validate triangle division constraints:
        - equally_divided=True: can only have 2, 3, or 6 parts (radial divisions)
        - equally_divided=False: can have 2-6 parts (vertical line divisions)
        """
        if self.shape == FractionShape.TRIANGLE:
            if self.equally_divided:
                # For equally divided triangles, only 2, 3, or 6 parts are allowed (radial divisions)
                if self.divided_parts not in [2, 3, 6]:
                    raise ValueError(
                        f"Equally divided triangles can only have 2, 3, or 6 parts. Got: {self.divided_parts}"
                    )
            else:
                # For unequally divided triangles, 2-6 parts are allowed (vertical line divisions)
                if self.divided_parts < 2 or self.divided_parts > 6:
                    raise ValueError(
                        f"Unequally divided triangles can only have 2-6 parts. Got: {self.divided_parts}"
                    )

        return self


class UnequalFractionList(StimulusDescriptionList[UnequalFraction]):
    pass


class MixedFraction(BaseModel):
    """
    Mixed Fraction model stimulus description for fractions greater than 1.
    Can represent improper fractions and mixed numbers.
    """

    shape: Annotated[FractionShape, BeforeValidator(lambda x: x.lower())] = Field(
        description="The shape used to represent the fraction."
    )
    fraction: str = Field(
        pattern=r"^\d+/\d+$",
        description="The fraction to represent. Can be greater than 1 (improper fraction).",
    )


class MixedFractionList(StimulusDescriptionList[MixedFraction]):
    """List of mixed fractions that can be greater than 1. Displayed in a 2x2 grid format."""

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        # Ensure between 1 and 4 models for 2x2 grid
        assert (
            1 <= len(self.root) <= 4
        ), f"{pipeline_context.standard_id}: Must have between 1 and 4 models for 2x2 grid display."
        # Ensure no fraction results in more than 4 shapes
        for fraction in self.root:
            numerator, denominator = map(int, fraction.fraction.split("/"))
            whole_shapes = numerator // denominator
            shapes_to_draw = whole_shapes + (1 if numerator % denominator > 0 else 0)
            assert shapes_to_draw <= 4, (
                f"{pipeline_context.standard_id}: Fraction {fraction.fraction} would require {shapes_to_draw} shapes, "
                f"but the maximum allowed is 4."
            )


class WholeFractionalShapes(StimulusDescription):
    """
    Schema for drawing 1-5 fully shaded shapes representing whole fractions.
    Each shape is divided into the same number of parts (common denominator) and fully shaded.
    Represents fractions like 3/3, 6/3, 9/3, etc. arranged in a single row without labels.
    """

    count: int = Field(
        ge=1,
        le=5,
        description="Number of whole units to represent (1-5 fully shaded shapes)",
    )
    shape: FractionShape = Field(description="The shape type: circle or rectangle")
    divisions: int = Field(
        ge=1,
        le=14,
        description="Common denominator - number of equal parts each shape is divided into",
    )


class FractionStrips(StimulusDescription):
    """
    Schema for drawing 2-3 stacked rectangles showing fraction decomposition.
    First rectangle is always whole with "1" in the middle.
    Second rectangle shows unit fractions (1/n) split from the whole.
    Third rectangle (optional) shows further decomposition of a unit fraction.
    """

    splits: Literal[2, 3] = Field(
        description="Number of rectangle strips to draw (2 or 3)"
    )

    first_division: int = Field(
        ge=2,
        le=10,
        description="Number of equal parts to split the whole into for the second rectangle (max 20 to ensure smallest fraction is 1/20)",
    )

    second_division: int | None = Field(
        default=None,
        ge=2,
        le=10,
        description="Number of equal parts to split one unit fraction into for the third rectangle (only used when splits=3, limited to ensure smallest fraction is 1/20)",
    )

    @model_validator(mode="after")
    def validate_splits_and_divisions(self):
        if self.splits == 3 and self.second_division is None:
            raise ValueError("second_division is required when splits=3")
        if self.splits == 2 and self.second_division is not None:
            raise ValueError("second_division should not be provided when splits=2")

        # Ensure smallest fraction is not smaller than 1/20
        if self.splits == 3 and self.second_division is not None:
            if self.first_division * self.second_division > 21:
                raise ValueError(
                    f"Product of first_division ({self.first_division}) and second_division ({self.second_division}) cannot exceed 21 to ensure smallest fraction is at least 1/21"
                )

        return self


class FractionNumber(BaseModel):
    """
    Represents a fraction with numerator and denominator.
    """

    numerator: int = Field(ge=1, le=100, description="The numerator of the fraction")
    denominator: int = Field(ge=1, le=25, description="The denominator of the fraction")

    @model_validator(mode="after")
    def validate_fraction_constraints(self):
        if self.denominator > 25:
            raise ValueError("Denominator cannot exceed 25")
        return self


class DivisionModel(StimulusDescription):
    """
    Schema for drawing fraction bar models for division exercises.
    Supports division problems like 6/7 ÷ 3 or 3 ÷ 3/4.
    Creates visual bar models showing the division process.
    """

    dividend: int | FractionNumber = Field(
        description="The dividend in the division problem. Can be a whole number or a fraction object with numerator/denominator."
    )

    divisor: int | FractionNumber = Field(
        description="The divisor in the division problem. Can be a whole number or a fraction object with numerator/denominator."
    )

    @model_validator(mode="after")
    def validate_division_model(self):
        """Validate that at least one operand is a fraction and divisor is not zero."""
        # Check for zero divisor
        if isinstance(self.divisor, int) and self.divisor == 0:
            raise ValueError("Divisor cannot be zero")
        elif isinstance(self.divisor, FractionNumber) and self.divisor.numerator == 0:
            raise ValueError("Divisor cannot be zero")

        # Ensure at least one operand is a fraction (otherwise it's just integer division)
        dividend_is_fraction = isinstance(self.dividend, FractionNumber)
        divisor_is_fraction = isinstance(self.divisor, FractionNumber)

        if not (dividend_is_fraction or divisor_is_fraction):
            raise ValueError(
                "At least one operand must be a fraction for fraction division models"
            )

        # Validate reasonable ranges for visualization
        if isinstance(self.dividend, int):
            if self.dividend <= 0 or self.dividend > 20:
                raise ValueError(
                    "Whole number dividend must be between 1 and 20 for visualization"
                )

        if isinstance(self.divisor, int):
            if self.divisor <= 0 or self.divisor > 20:
                raise ValueError(
                    "Whole number divisor must be between 1 and 20 for visualization"
                )

        return self


if __name__ == "__main__":
    FractionList.generate_assistant_function_schema("mcq4")
    FractionPairList.generate_assistant_function_schema("mcq4")
    FractionSet.generate_assistant_function_schema("mcq4")
