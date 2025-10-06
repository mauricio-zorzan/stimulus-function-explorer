from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_validator


class DataItem(BaseModel):
    group: str = Field(..., description="Name of the group")
    condition: str = Field(..., description="Name of the condition")
    value: float = Field(..., description="Value for the bar")
    error: Optional[float] = Field(None, description="Error bar value (optional)")


class GroupedBarChart(StimulusDescription):
    data: List[DataItem]
    y_label: str = Field(..., description="Label for the y-axis")
    x_label: str = Field(..., description="Label for the x-axis")
    title: str = Field(..., description="Title of the chart", max_length=80)

    @field_validator("data")
    def check_group_and_condition_constraints(cls, v):
        groups = {item.group for item in v}
        conditions = {item.condition for item in v}
        if not (1 <= len(groups) <= 5):
            raise ValueError("Number of unique groups must be between 1 and 5")
        if not (1 <= len(conditions) <= 5):
            raise ValueError("Number of unique conditions must be between 1 and 5")
        return v

    @field_validator("title")
    def check_title_length(cls, v):
        if len(v.split()) > 10:
            raise ValueError("Title must not exceed 15 words")
        return v
