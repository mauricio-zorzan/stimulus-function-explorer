from typing import Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class AreaStimulusParams(StimulusDescription):
    base: int = Field(..., ge=1, description="The base of the shape.")
    height: int = Field(..., ge=1, description="The height of the shape.")
    shape: Literal["right_triangle", "rectangle"] = Field(
        ..., description="The type of shape, either 'right_triangle' or 'rectangle'."
    )
    not_to_scale_note: str = Field(
        default="Figure not drawn to scale.",
        description="Note indicating the figure is not to scale.",
    )


if __name__ == "__main__":
    AreaStimulusParams.generate_assistant_function_schema("sat-math")
