from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field


class Point(BaseModel):
    label: str = Field(..., description="The label for the point")
    x: float = Field(..., description="The x-coordinate of the point")
    y: float = Field(..., description="The y-coordinate of the point")


class PointOneQuadrant(BaseModel):
    label: str = Field(..., description="The label for the point")
    x: float = Field(..., ge=0, description="The x-coordinate of the point")
    y: float = Field(..., ge=0, description="The y-coordinate of the point")


class PointPlotWithContext(StimulusDescription):
    x_title: str = Field(default="", description="The label for the x-axis")
    y_title: str = Field(default="", description="The label for the y-axis")
    points: list[PointOneQuadrant] = Field(..., description="The points to be plotted")


class PointPlot(StimulusDescription):
    points: list[Point] = Field(..., description="The points to be plotted")


class PointList(StimulusDescriptionList[Point]):
    root: list[Point] = Field(..., description="The points to be plotted")
