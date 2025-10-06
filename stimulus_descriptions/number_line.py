from typing import TYPE_CHECKING

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class Range(BaseModel):
    min: int = Field(..., description="The minimum value on the number line")
    max: int = Field(..., description="The maximum value on the number line")


class ExtendedRange(BaseModel):
    min: float = Field(..., description="The minimum value on the number line")
    max: float = Field(..., description="The maximum value on the number line")


class Point(BaseModel):
    label: str = Field(
        ...,
        pattern=r"^-?[a-zA-Z]$|^-?\d+$|^-?\d+/\d+$|^-?\d+(\.\d+)?\s?(°C|°F|mm|cm|m|km|mg|g|kg|lb|oz|in|ft|ml|l|gal|fl oz)$",
        description="The label for this point on the number line, either a letter or a number or a decimal or a fraction.",
    )
    value: float = Field(..., description="The value of this point on the number line")


class NumberLine(StimulusDescription):
    """Number line with a range of values.
    Minor ticks are automatically added to the number line.
    If the range is greater than 120, the minor ticks are set to 10.
    If the range is greater than 15, the minor ticks are set to 1.
    If the range is less than 15, the major ticks are set to 1.
    """

    range: Range = Field(
        ...,
        description="The range of values on the number line.",
    )
    points: list[Point] = Field(
        ...,
        min_length=0,
        json_schema_extra={"uniqueItems": True},
        description="List of optional points to be displayed on the number line, these are shown as additionally labelled points on the number line, so use this to add a label or mark a specific point.",
    )
    minor_divisions: int | None = Field(
        default=None,
        exclude=True,
        description="Not used at the moment. Forwarded from unit fraction number line.",
    )

    @model_validator(mode="after")
    def validate_points_within_range(self):
        for point in self.points:
            if not (self.range.min <= point.value <= self.range.max):
                raise ValueError(
                    f"Point {point.label} with value {point.value} is outside the specified range."
                )
        return self


class FixedStepNumberLine(StimulusDescription):
    """Number line with a fixed step size for major ticks.
    This allows precise control over the spacing of major tick marks
    instead of using automatic step size calculation.
    """

    range: Range = Field(
        ...,
        description="The range of values on the number line.",
    )
    points: list[Point] = Field(
        ...,
        min_length=0,
        json_schema_extra={"uniqueItems": True},
        description="List of optional points to be displayed on the number line, these are shown as additionally labelled points on the number line, so use this to add a label or mark a specific point.",
    )
    step_size: float = Field(
        ...,
        gt=0,
        description="The step size for major ticks on the number line. Must be greater than 0.",
    )
    minor_divisions: int = Field(
        default=2,
        ge=1,
        description="Number of minor divisions between each major tick. Default is 2 (one minor tick in the middle).",
    )

    @model_validator(mode="after")
    def validate_points_within_range(self):
        for point in self.points:
            if not (self.range.min <= point.value <= self.range.max):
                raise ValueError(
                    f"Point {point.label} with value {point.value} is outside the specified range."
                )
        return self


class UnitFractionPoint(Point):
    value: float = Field(
        ...,
        le=1,
        ge=0,
        description="The value of this point on the number line",
    )
    label: str = Field(
        ...,
        pattern=r"^[a-zA-Z]$|^0$|^1$|^\d+\.\d+$|^\d+\/\d+$",
        description="The label for this point on the number line, either a letter or a decimal or a fraction.",
    )


class UnitFractionNumberLine(StimulusDescription):
    """Unit fraction number line with range 0 to 1 with major ticks only at 0 and 1."""

    range: Range = Field(
        default=Range(min=0, max=1),
        exclude=True,
        description="The range of values on the number line.",
    )
    points: list[UnitFractionPoint] = Field(
        ...,
        min_length=1,
        description="List of optional points to be displayed on the number line, these are shown as additionally labelled points on the number line, so use this to add a label or mark a specific point. Must be 0 <= point.value <= 1.",
    )
    minor_divisions: int | None = Field(
        default=None,
        description="The number of divisions to make minor ticks on the number line, between 0 and 1. None means there will be no minor ticks. The minor ticks don't have lables so use the label to show points of interest.",
    )

    @model_validator(mode="after")
    def validate_points_within_range(self):
        for point in self.points:
            if not (self.range.min <= point.value <= self.range.max):
                raise ValueError(
                    f"Point {point.label} with value {point.value} is outside the specified range."
                )
        if not self.points:
            raise ValueError(
                "No points provided for unit fraction number line, please provide at least one point."
            )

        return self


class ExtendedUnitFractionPoint(Point):
    value: float = Field(
        ...,
        ge=0,
        description="The value of this point on the number line",
    )
    label: str = Field(
        ...,
        pattern=r"^[a-zA-Z]$|^0$|^\d+/\d+$",
        description="The label for this point on the number line, either a letter, 0, or a fraction.",
    )


class ExtendedUnitFractionDotPoint(BaseModel):
    label: str = Field(..., description="Label for the dot point (e.g., 'A')")
    value: float = Field(
        ...,
        description="Length/amount that the blue bar represents (e.g., 2/4 = 0.5). The bar will extend this much from the start tick.",
    )
    dot_start_tick: int = Field(
        default=0,
        ge=0,
        description="Which tick the blue bar should start from (0-based). Default is 0 (start of number line). The bar extends 'value' units from this starting position.",
    )
    red: bool = Field(
        default=False,
        description="Whether the dot point should be red instead of blue. Default is False (blue).",
    )


class ExtendedUnitFractionNumberLine(StimulusDescription):
    """Extended unit fraction number line with endpoints always labeled (0 and a fraction), and a single dot point on a minor tick."""

    range: ExtendedRange = Field(...)
    minor_divisions: int = Field(...)
    endpoint_fraction: str = Field(
        ..., description="Fraction label for the endpoint (e.g., '8/4')"
    )
    dot_point: ExtendedUnitFractionDotPoint = Field(...)
    show_all_tick_labels: bool = Field(
        default=False,
        description="Whether to show fraction labels on all minor divisions (e.g., 1/3, 2/3). Default is False.",
    )
    labeled_fraction: str | None = Field(
        default=None,
        description=(
            "Optional fraction string (e.g., '3/4') to label on the number line. Only 0 and this fraction will be labelled; the endpoint is not labelled."
        ),
    )

    @model_validator(mode="after")
    def validate_endpoint_and_dot(self):
        # Validate endpoint_fraction matches range.max
        try:
            endpoint_value = eval(self.endpoint_fraction)
        except Exception:
            raise ValueError(
                "endpoint_fraction must be a valid fraction string, e.g., '8/4'"
            )
        if abs(endpoint_value - self.range.max) > 1e-6:
            raise ValueError(
                f"endpoint_fraction value {endpoint_value} must match range.max {self.range.max}"
            )
        # Calculate step size for validation
        step = (self.range.max - self.range.min) / self.minor_divisions

        # Validate dot_start_tick and bar length are within bounds
        if self.dot_point.dot_start_tick > self.minor_divisions:
            raise ValueError(
                f"dot_start_tick ({self.dot_point.dot_start_tick}) cannot be greater than minor_divisions ({self.minor_divisions})."
            )

        # Calculate where the bar would end
        bar_end_position = (
            self.range.min
            + (self.dot_point.dot_start_tick * step)
            + self.dot_point.value
        )
        if bar_end_position > self.range.max:
            raise ValueError(
                f"Bar extends beyond the number line range. Bar end position ({bar_end_position}) exceeds range.max ({self.range.max}). "
                f"Reduce the value ({self.dot_point.value}) or start tick ({self.dot_point.dot_start_tick})."
            )

        # Validate labeled_fraction if provided
        if self.labeled_fraction is not None:
            try:
                # Interpret labeled_fraction as a fraction of the endpoint
                labeled_ratio = eval(self.labeled_fraction)
            except Exception:
                raise ValueError("labeled_fraction must be a valid fraction string like '3/4'")
            labeled_value = self.range.min + labeled_ratio * (self.range.max - self.range.min)
            if not (self.range.min - 1e-9 <= labeled_value <= self.range.max + 1e-9):
                raise ValueError("labeled_fraction must correspond to a value within the number line range")
            # Must be on a minor tick
            rel = labeled_value - self.range.min
            if abs((rel / step) - round(rel / step)) > 1e-6:
                raise ValueError("labeled_fraction must fall exactly on a minor tick")
            # Must be different from dot point value
            if abs(labeled_value - self.dot_point.value) < 1e-9:
                raise ValueError("labeled_fraction must be different from dot_point.value")

        return self


class MultiExtendedUnitFractionNumberLine(StimulusDescription):
    """Multiple extended unit fraction number lines with bars displayed vertically."""

    number_lines: list[ExtendedUnitFractionNumberLine] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="List of extended unit fraction number lines to display vertically, each with a bar from 0 to dot_point.value",
    )
    show_minor_division_labels: bool = Field(
        default=False,
        description="Whether to show fraction labels on all minor divisions (e.g., 1/3, 2/3). Default is False.",
    )


class LabeledUnitFractionNumberLine(StimulusDescription):
    """Unit fraction number line with all minor divisions labeled as fractions."""

    range: ExtendedRange = Field(...)
    minor_divisions: int = Field(...)
    endpoint_fraction: str = Field(
        ..., description="Fraction label for the endpoint (e.g., '8/4')"
    )

    @model_validator(mode="after")
    def validate_endpoint(self):
        # Validate endpoint_fraction matches range.max
        try:
            endpoint_value = eval(self.endpoint_fraction)
        except Exception:
            raise ValueError(
                "endpoint_fraction must be a valid fraction string, e.g., '8/4'"
            )
        if abs(endpoint_value - self.range.max) > 1e-6:
            raise ValueError(
                f"endpoint_fraction value {endpoint_value} must match range.max {self.range.max}"
            )
        return self


class MultiLabeledUnitFractionNumberLine(StimulusDescription):
    """Multiple labeled unit fraction number lines displayed vertically."""

    number_lines: list[LabeledUnitFractionNumberLine] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="List of labeled unit fraction number lines to display vertically with all minor divisions labeled as fractions",
    )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)

        match pipeline_context.standard_id:
            case "CCSS.MATH.CONTENT.3.NF.A.3.A+2":
                if len(self.number_lines) != 2:
                    raise ValueError(
                        f"Need exactly 2 number lines for CCSS.MATH.CONTENT.3.NF.A.3.A+2, but got {len(self.number_lines)}"
                    )


class DecimalComparisonNumberLine(StimulusDescription):
    """Decimal comparison number line with exactly 10 divisions.
    Increments are either 0.1 or 0.01 based on the range.
    The range should span exactly 1.0 (for 0.1 increments) or 0.1 (for 0.01 increments).
    """

    range: ExtendedRange = Field(
        ...,
        description="The range of values on the number line. Should span exactly 1.0 or 0.1 to create 10 equal divisions.",
    )
    points: list[Point] = Field(
        default=[],
        min_length=0,
        json_schema_extra={"uniqueItems": True},
        description="Optional list of points to be displayed on the number line. Can be empty for just the number line.",
    )
    label_interval: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Interval for labeling ticks. 1 = label all ticks, 2 = label every 2nd tick, 5 = label every 5th tick, etc. Must be between 1 and 10.",
    )

    @model_validator(mode="after")
    def validate_decimal_comparison_settings(self):
        # Validate that points are within range
        for point in self.points:
            if not (self.range.min <= point.value <= self.range.max):
                raise ValueError(
                    f"Point {point.label} with value {point.value} is outside the specified range."
                )

        # Validate that the range creates exactly 10 divisions
        range_span = self.range.max - self.range.min

        # Check for 0.1 increments (span should be 1.0)
        if abs(range_span - 1.0) < 1e-10:
            increment = 0.1
        # Check for 0.01 increments (span should be 0.1)
        elif abs(range_span - 0.1) < 1e-10:
            increment = 0.01
        else:
            raise ValueError(
                f"Range span must be exactly 1.0 (for 0.1 increments) or 0.1 (for 0.01 increments). Got {range_span}"
            )

        # Validate that start value is aligned with increment
        if increment == 0.1:
            # For 0.1 increments, start should be a multiple of 0.1
            if abs((self.range.min * 10) - round(self.range.min * 10)) > 1e-10:
                raise ValueError(
                    f"For 0.1 increments, start value must be a multiple of 0.1. Got {self.range.min}"
                )
        else:  # increment == 0.01
            # For 0.01 increments, start should be a multiple of 0.01
            if abs((self.range.min * 100) - round(self.range.min * 100)) > 1e-10:
                raise ValueError(
                    f"For 0.01 increments, start value must be a multiple of 0.01. Got {self.range.min}"
                )

        return self


if __name__ == "__main__":
    NumberLine.generate_assistant_function_schema("mcq4")
    print(UnitFractionNumberLine.model_json_schema())
