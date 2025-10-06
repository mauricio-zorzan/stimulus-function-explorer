import re
from enum import Enum
from typing import TYPE_CHECKING

from content_generators.additional_content.stimulus_image.stimulus_descriptions.ruler import (
    MeasuredItemName,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescriptionList,
)
from content_generators.types import SortableEnum
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class MeasurementUnitImage(str, Enum):
    BUTTON = "button"
    UNIT_SQUARES = "unit squares"


class UnitDisplayError(str, SortableEnum):
    GAP = "gap"
    OVERLAP = "overlap"


class MeasuredObject(BaseModel):
    object_name: MeasuredItemName = Field(
        ...,
        description="The name of the object being measured displaying the corresponding image.",
    )
    length: int = Field(
        ...,
        ge=1,
        le=12,
        description="The length of the object in units of the specified unit of measurement.",
    )
    unit: MeasurementUnitImage | None = Field(
        default=None,
        description="The unit of measurement displayed as images measuring the item. If omitted, then no unit images are displayed.",
    )
    label: str = Field(
        ...,
        description="The label for the object, which is displayed above the image, usually the name of the object but if an object is re-used (e.g. 2 pencils), there must be a letter suffixed to the name (e.g. 'Pencil A').",
    )
    unit_display_error: UnitDisplayError | None = Field(
        default=None,
        description="The error in the unit display, redacted to have no errors.",
    )


class MeasurementComparison(StimulusDescriptionList[MeasuredObject]):
    root: list[MeasuredObject] = Field(
        ...,
        description="List of objects to be measured and compared",
        min_length=2,
        max_length=3,
    )

    @property
    def lengths(self) -> list[int]:
        return [obj.length for obj in self.root]

    @property
    def units(self) -> list[MeasurementUnitImage | None]:
        return [obj.unit for obj in self.root]

    @property
    def unit_display_errors(self) -> list[UnitDisplayError | None]:
        return [obj.unit_display_error for obj in self.root]

    @model_validator(mode="after")
    def validate_objects(self):
        # Check if units are consistent when present
        if self.units and len(set(self.units)) != 1:
            raise ValueError(
                "All objects must use the same unit of measurement when present"
            )
        return self

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)

        if pipeline_context.standard_id == "CCSS.MATH.CONTENT.1.MD.A.1+1":
            if len(self) != 3:
                raise ValueError(
                    "Exactly three objects are required for ordering by length"
                )
            if any(self.units):
                raise ValueError("Units should not be specified for this sub-standard")

            # Ensure lengths are sufficiently different
            if not (
                max(self.lengths) - min(self.lengths) > 1
                and len(set(self.lengths)) == 3
            ):
                raise ValueError(
                    "For ordering, object lengths must be sufficiently different"
                )

            # Ensure that the lengths are not mentioned in the explanations for this sub-standard
            if pipeline_context.question is not None:
                pipeline_context.question.assert_all_explanations_dont_contain(
                    re.compile(r"\d+")
                )

        elif pipeline_context.standard_id == "CCSS.MATH.CONTENT.1.MD.A.1+2":
            # Compare the lengths of two objects indirectly by using a third object
            if len(self) != 2:
                raise ValueError(
                    "Exactly two objects are required for indirect comparison with units"
                )
            if not any(self.units):
                raise ValueError("Units should be specified for this sub-standard")

        elif pipeline_context.standard_id == "CCSS.MATH.CONTENT.1.MD.A.2+1":
            # Express the length of an object as a whole number of length units by laying multiple copies of a shorter object end to end
            if len(self) != 2:
                raise ValueError("Exactly two objects should be measured")
            if not any(self.units):
                raise ValueError("Units should be specified for this sub-standard")

            # Check if one length is a multiple of the other
            if max(self.lengths) % min(self.lengths) != 0:
                raise ValueError("One length should be a multiple of the other")

        elif pipeline_context.standard_id == "CCSS.MATH.CONTENT.1.MD.A.2+2":
            # Understand that the length measurement of an object is the number of same-size length units that span it with no gaps or overlaps
            if len(self) != 3:
                raise ValueError(
                    "Only three objects should be measured for this standard"
                )
            # Check if all objects have the same length and it's at least 3
            if not (len(set(self.lengths)) == 1 and min(self.lengths) >= 3):
                raise ValueError(
                    "All objects should have the same length of at least 3"
                )

            # Check for correct unit display errors
            if set(self.unit_display_errors) != {
                None,
                UnitDisplayError.GAP,
                UnitDisplayError.OVERLAP,
            }:
                raise ValueError(
                    "There should be one object with no error, one with gaps, and one with overlaps"
                )


if __name__ == "__main__":
    MeasurementComparison.generate_assistant_function_schema("mcq3")
