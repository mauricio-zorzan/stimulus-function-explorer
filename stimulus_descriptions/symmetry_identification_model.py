import logging
from typing import TYPE_CHECKING, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class IdentificationLine(BaseModel):
    slope: Optional[float] = Field(
        ...,
        description="The slope of the line. None represents a vertical line (infinite slope).",
    )
    intercept: float = Field(..., description="The intercept of the line.")
    label: Literal["vertical", "horizontal", "diagonal", "not_symmetry"] = Field(
        ..., description="A descriptive label for the line."
    )


class SymmetryIdentification(StimulusDescription):
    shape_type: Literal[
        "flower", "sun", "diamond", "heart", "house", "wheel", "football", "polygon"
    ] = Field(..., description="Type of shape to generate")
    shape_coordinates: List[List[float]] = Field(
        default_factory=lambda: [[0, 0]],
        description="Dummy coordinates (not used for special shapes)",
    )
    lines: List[IdentificationLine] = Field(
        ...,
        description="A list of lines for symmetry identification",
    )

    @model_validator(mode="after")
    def validate_identification_task(cls, values):
        # Allow dummy coordinates for special shapes
        if values.shape_type in [
            "flower",
            "sun",
            "diamond",
            "heart",
            "house",
            "wheel",
            "football",
        ]:
            # No validation needed for special shapes
            pass
        else:
            # For polygons, require real coordinates
            if len(values.shape_coordinates) < 3:
                raise ValueError("Polygon must have at least 3 vertices.")

        if len(values.lines) > 4:
            raise ValueError("There can be no more than 4 lines.")

        return values

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext") -> None:
        """
        Additional validation for specific substandard configurations
        """
        logging.info(self)
        super().pipeline_validate(pipeline_context)

        if pipeline_context.standard_id == "CCSS.MATH.CONTENT.4.G.A.3+2":
            # For symmetry identification, we need exactly one line to test
            if len(self.lines) != 1:
                raise ValueError(
                    f"For substandard {pipeline_context.standard_id} (Identify whether a dotted line is a line of symmetry), "
                    f"exactly one line is required, got {len(self.lines)}."
                )
