from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_validator


class DataSeries(BaseModel):
    label: str = Field(..., description="Label for the data series")
    x_values: List[float] = Field(..., description="X values for the series")
    y_values: List[float] = Field(..., description="Y values for the series")
    marker: Optional[str] = Field(None, description="Marker style for the data series")

    @field_validator("x_values")
    @classmethod
    def check_x_y_length(cls, v, info):
        y_values = info.data.get("y_values")
        if y_values and len(v) != len(y_values):
            raise ValueError("Length of x_values and y_values must be the same")
        return v

    @field_validator("y_values")
    @classmethod
    def check_y_x_length(cls, v, info):
        x_values = info.data.get("x_values")
        if x_values and len(v) != len(x_values):
            raise ValueError("Length of x_values and y_values must be the same")
        return v


class AxisOptions(BaseModel):
    label: str = Field(..., description="Label for the axis")
    range: Optional[List[float]] = Field(
        None, description="Range for the axis as [min, max]"
    )

    @field_validator("range")
    @classmethod
    def check_range(cls, v):
        if v and len(v) != 2:
            raise ValueError("Range must be a list of two numbers [min, max]")
        if v and v[0] >= v[1]:
            raise ValueError("First value in range must be less than second value")
        return v


class MultiLineGraph(StimulusDescription):
    data_series: List[DataSeries] = Field(
        ..., description="List of data series to plot", min_length=1, max_length=5
    )
    x_axis: AxisOptions = Field(..., description="Options for the x-axis")
    y_axis: AxisOptions = Field(..., description="Options for the y-axis")
    title: Optional[str] = Field(None, description="Title of the graph", max_length=80)

    @field_validator("data_series")
    @classmethod
    def validate_data_series(cls, v):
        if not (1 <= len(v) <= 5):
            raise ValueError("Number of data series must be between 1 and 5")
        return v

    @field_validator("title")
    @classmethod
    def check_title_length(cls, v):
        if v is None:
            return v
        if len(v.split()) > 15:
            raise ValueError("Title must not exceed 15 words")
        return v
