from typing import Annotated, List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, BeforeValidator, Field


def convert_to_int(value):
    """
    Convert the value to an integer if it is a string. Required for some legacy generators.
    """
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Cannot convert {value} to an integer")
    return value


class Axis(BaseModel):
    label: str = Field(..., description="The label for the axis.")
    min_value: Annotated[int, BeforeValidator(convert_to_int)] = Field(
        ..., description="The minimum value for the axis."
    )
    max_value: Annotated[int, BeforeValidator(convert_to_int)] = Field(
        ..., description="The maximum value for the axis."
    )


class Point(BaseModel):
    x: Annotated[int, BeforeValidator(convert_to_int)] = Field(
        ..., description="The x-coordinate of the point."
    )
    y: Annotated[int, BeforeValidator(convert_to_int)] = Field(
        ..., description="The y-coordinate of the point."
    )


class ScatterPlot(StimulusDescription):
    title: str = Field(..., description="The title of the scatter plot.")
    x_axis: Axis = Field(..., description="The x-axis of the scatter plot.")
    y_axis: Axis = Field(..., description="The y-axis of the scatter plot.")
    points: List[Point] = Field(
        ..., description="The list of points in the scatter plot."
    )


if __name__ == "__main__":
    ScatterPlot.generate_assistant_function_schema("mcq4")
