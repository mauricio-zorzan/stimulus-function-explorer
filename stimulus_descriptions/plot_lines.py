from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescriptionList,
)
from pydantic import AliasChoices, BaseModel, Field


class Line(BaseModel):
    slope: float = Field(..., description="The slope of the line")
    y_intercept: float = Field(
        ...,
        description="The y-intercept of the line",
        validation_alias=AliasChoices("y-intercept", "yIntercept", "y_intercept"),
    )
    label: str = Field(..., description="The label for the line")


class PlotLines(StimulusDescriptionList[Line]):
    """A graph with multiple lines plotted with a visible range of -10 to 10 on the x-axis and the y-axis"""

    root: list[Line] = Field(..., min_length=1, description="List of lines to plot")


if __name__ == "__main__":
    PlotLines.generate_assistant_function_schema("mcq4")
