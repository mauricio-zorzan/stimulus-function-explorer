import math
from enum import Enum
from typing import TYPE_CHECKING

from content_generators.additional_content.stimulus_image.drawing_functions.constants import (
    CM_TO_INCH_RATIO,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class MeasurementUnit(str, Enum):
    INCHES = "inches"
    CENTIMETERS = "centimeters"
    CM = "cm"
    IN = "in"


class MeasuredItemName(str, Enum):
    PENCIL = "pencil"
    ARROW = "arrow"
    STRAW = "straw"


class Ruler(BaseModel):
    unit: MeasurementUnit = Field(
        ..., description="The unit of measurement for the ruler"
    )
    length: float | None = Field(
        default=0, exclude=True, description="The length of the ruler", le=30, ge=1
    )
    length_in_cm: float | None = Field(
        default=0,
        exclude=True,
        description="The length of the ruler in centimeters",
        le=30,
        ge=1,
    )

    @model_validator(mode="before")
    def validate_unit(cls, values):
        if values["unit"] == "cm":
            values["unit"] = MeasurementUnit.CENTIMETERS.value
        elif values["unit"] == "in":
            values["unit"] = MeasurementUnit.INCHES.value
        return values

    @classmethod
    def convert_to_unit(
        cls, value: float, unit: MeasurementUnit, target_unit: MeasurementUnit
    ) -> float:
        if unit == target_unit:
            return value
        elif (
            unit == MeasurementUnit.INCHES
            and target_unit == MeasurementUnit.CENTIMETERS
        ):
            return value * CM_TO_INCH_RATIO
        elif (
            unit == MeasurementUnit.CENTIMETERS
            and target_unit == MeasurementUnit.INCHES
        ):
            return value / CM_TO_INCH_RATIO
        else:
            raise ValueError(f"Invalid unit conversion from {unit} to {target_unit}")


class MeasuredItem(BaseModel):
    name: MeasuredItemName = Field(
        ..., description="The name of the item being measured."
    )
    label: str | None = Field(
        description="The label of the item being measured.",
        min_length=1,
        default=None,
    )
    length: float = Field(..., description="The length of the item.", ge=1, le=30)
    start_position: float = Field(
        default=0.0,
        description="The starting position of the item on the ruler (0.0 means starts at zero mark).",
        ge=0.0,
        le=25.0,
    )
    length_in_cm: float | None = Field(
        default=1,
        exclude=True,
        description="The length of the item in centimeters",
        le=30,
        ge=1,
    )
    ruler: Ruler = Field(..., description="The ruler used to measure this item.")


class RulerStimulus(StimulusDescription):
    items: list[MeasuredItem] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="List of items to be measured (maximum 4 items)",
    )

    @model_validator(mode="after")
    def validate_items_and_rulers(self):
        max_length_cm = 0
        for item in self.items:
            if not item.ruler:
                raise ValueError("Each item must have a ruler specified")
            # Calculate total space needed (start_position + length) in centimeters
            start_position_cm = Ruler.convert_to_unit(
                item.start_position,
                item.ruler.unit,
                MeasurementUnit.CENTIMETERS,
            )
            item_length_cm = Ruler.convert_to_unit(
                item.length,
                item.ruler.unit,
                MeasurementUnit.CENTIMETERS,
            )
            total_space_needed_cm = start_position_cm + item_length_cm
            max_length_cm = max(max_length_cm, total_space_needed_cm)

        for item in self.items:
            self.adjust_individual_ruler(item, max_length_cm)

        return self

    @classmethod
    def adjust_individual_ruler(cls, item: MeasuredItem, max_length_cm: float):
        # For inch rulers, calculate the required ruler length
        if item.ruler.unit == MeasurementUnit.INCHES:
            # Calculate total space needed
            total_space_needed = item.start_position + item.length

            # Only adjust ruler length if current length is insufficient
            required_length = math.ceil(total_space_needed)
            if item.ruler.length is None or item.ruler.length < required_length:
                item.ruler.length = required_length

            # Only apply visual constraints if the ruler would be too long for display
            # This preserves existing behavior for tests while preventing visual overflow
            if (
                item.ruler.length > 8.0
            ):  # More generous limit for backward compatibility
                # Reduce object length to fit within reasonable bounds
                max_ruler_inches = 8.0
                if total_space_needed > max_ruler_inches:
                    new_length = max_ruler_inches - item.start_position
                    if new_length < 1.0:  # Minimum object length
                        # If start position is too high, adjust both
                        item.start_position = max_ruler_inches - 1.0
                        item.length = 1.0
                    else:
                        item.length = new_length

                    total_space_needed = item.start_position + item.length
                    item.ruler.length = math.ceil(total_space_needed)

            # Convert to centimeters for coordinate system
            item.ruler.length_in_cm = Ruler.convert_to_unit(
                item.ruler.length, item.ruler.unit, MeasurementUnit.CENTIMETERS
            )
        else:
            # For centimeter rulers, use the original logic
            total_space_needed = item.start_position + item.length
            required_length = math.ceil(total_space_needed)
            if item.ruler.length is None or item.ruler.length < required_length:
                item.ruler.length = required_length
            item.ruler.length_in_cm = item.ruler.length

        # Set item length in cm for drawing
        item.length_in_cm = Ruler.convert_to_unit(
            item.length, item.ruler.unit, MeasurementUnit.CENTIMETERS
        )

        # Validate centimeter ruler length
        if item.ruler.unit == MeasurementUnit.CENTIMETERS:
            # Use a small epsilon to handle floating point imprecision
            if (
                item.ruler.length_in_cm > 12.001
            ):  # Allow for small floating point errors
                raise ValueError(
                    f"Centimeter rulers cannot be longer than 12 centimeters (got {item.ruler.length_in_cm:.3f}cm)"
                )

        if item.length_in_cm > 30:
            raise ValueError(
                "The length of the item in centimeters cannot be greater than 30"
            )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        match pipeline_context.standard_id:
            case "CCSS.MATH.CONTENT.2.MD.A.4":
                assert all(
                    item.ruler.unit == self.items[0].ruler.unit for item in self.items
                ), "All items must be measured in the same unit"
            case "CCSS.MATH.CONTENT.2.MD.D.9+1":
                assert all(
                    item.ruler.unit == self.items[0].ruler.unit for item in self.items
                ), "All items must be measured in the same unit"
            case "CCSS.MATH.CONTENT.2.MD.D.9+2":
                assert len(self.items) == 2, "There must be exactly 2 items"
                assert any(
                    item.ruler.unit == MeasurementUnit.INCHES for item in self.items
                ), "At least one item must be measured in inches"
                assert any(
                    item.ruler.unit == MeasurementUnit.CENTIMETERS
                    for item in self.items
                ), "At least one item must be measured in centimeters"
                assert (
                    self.items[0].length_in_cm == self.items[1].length_in_cm
                ), "The lengths of the items must be the same"
                assert (
                    self.items[0].name == self.items[1].name
                ), "The names of the items must be the same"
            case "CCSS.MATH.CONTENT.3.MD.B.4+1":
                assert all(
                    item.ruler.unit == MeasurementUnit.INCHES for item in self.items
                ), "All items must be measured in inches"


if __name__ == "__main__":
    RulerStimulus.generate_assistant_function_schema("mcq4")
