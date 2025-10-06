from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field


class Bar(BaseModel):
    label: str = Field(
        ...,
        description="The label for the bar (shown, so add the length here if the student needs it)",
    )
    length: float = Field(..., description="The length of the bar (not shown)")


class BarModel(StimulusDescriptionList[Bar]):
    root: list[Bar] = Field(
        ...,
        description="List of bars with labels and lengths",
        min_length=2,
        max_length=2,
    )


if __name__ == "__main__":
    BarModel.generate_assistant_function_schema("mcq4")
