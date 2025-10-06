from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field


class Axis(BaseModel):
    label: str = Field(..., description="The label for the axis.")
    range: List[float] = Field(..., description="The range of the axis.")


class Line(BaseModel):
    intercept: float = Field(..., description="The y-intercept of the line.")
    slope: float = Field(..., description="The slope of the line.")


class PlotLine(StimulusDescription):
    x_axis: Axis = Field(..., description="The x-axis of the plot")
    y_axis: Axis = Field(..., description="The y-axis of the plot")
    line: Line = Field(..., description="The line to be plotted.")
    point: list[float] | None = Field(
        default=None,
        description="Optional point to display the coordinates of the point along the line.",
    )


if __name__ == "__main__":
    print(PlotLine.generate_assistant_function_schema("mcq4"))
