# stimulus_descriptions/line_graph_description.py

from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field


class LineDataPoint(BaseModel):
    """Represents a single (x, y) coordinate on the line graph."""

    x: float = Field(..., description="The value on the x-axis.")
    y: float = Field(..., description="The value on the y-axis.")


class LineGraphSeries(BaseModel):
    """Represents a single line (or series) to be plotted on the graph."""

    data_points: List[LineDataPoint] = Field(
        ..., description="A list of (x, y) coordinates for the line."
    )
    label: Optional[str] = Field(
        default=None, description="The name of the line, to be shown in the legend."
    )
    color: Optional[str] = Field(
        default=None,
        description="The color of the line (e.g., 'blue', 'red', '#FF5733'). If not provided, a default color will be used.",
    )


class LineGraph(BaseModel):
    """Defines the complete structure for a single line graph."""

    title: str = Field(..., description="The main title of the line graph.")
    x_axis_label: str = Field(..., description="The label for the x-axis.")
    y_axis_label: str = Field(..., description="The label for the y-axis.")
    data_series: List[LineGraphSeries] = Field(
        ...,
        description="A list of one or more lines to be plotted on the graph.",
        min_length=1,
    )


class LineGraphList(StimulusDescriptionList[LineGraph]):
    """A container for a single line graph, mirroring the project structure."""

    root: list[LineGraph] = Field(
        ...,
        description="A list containing exactly one line graph definition.",
        min_length=1,
        max_length=1,
    )


if __name__ == "__main__":
    LineGraphList.generate_assistant_function_schema("mcq4")
