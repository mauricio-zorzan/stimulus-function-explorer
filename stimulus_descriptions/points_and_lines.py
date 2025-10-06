from typing import Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import AliasChoices, BaseModel, Field


class Point(BaseModel):
    label: str = Field(..., description="The label for the point.")
    coordinates: list[float] = Field(
        ..., description="The coordinates of the point as x and y respectively."
    )


class Line(BaseModel):
    start_label: str = Field(
        ...,
        description="The label of the start point of the line.",
        validation_alias=AliasChoices("start", "start_label"),
    )
    end_label: str = Field(
        ...,
        description="The label of the end point of the line.",
        validation_alias=AliasChoices("end", "end_label"),
    )
    type: Literal["line", "ray", "segment"] = Field(
        ..., description="The type of the line."
    )


class PointsAndLines(StimulusDescription):
    points: list[Point] = Field(
        ..., description="A list of points with labels and coordinates."
    )
    lines: list[Line] = Field(
        ..., description="A list of lines defined by start and end points."
    )
