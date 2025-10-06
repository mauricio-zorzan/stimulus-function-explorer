from typing import List, Union

from pydantic import Field, field_validator, model_validator

from .stimulus_description import StimulusDescription


class StatsBarDiagram(StimulusDescription):
    x_axis_label: str = Field(
        description="The label for the x-axis of the bar diagram."
    )
    y_axis_label: str = Field(
        description="The label for the y-axis of the bar diagram."
    )
    x_axis_data: List[Union[int, str]] = Field(
        description="The data points for the x-axis.", min_length=5, max_length=10
    )
    y_axis_data: List[Union[int, float]] = Field(
        description="The data points for the y-axis.", min_length=5, max_length=10
    )

    @field_validator("x_axis_data", "y_axis_data")
    def validate_integer_values(cls, v):
        try:
            integers = [int(x) for x in v]
        except ValueError:
            raise ValueError("All values must be integers")
        return integers

    @model_validator(mode="after")
    def validate_x_axis_sequential(cls, values):
        x_data = values.x_axis_data
        if not all(x_data[i] < x_data[i + 1] for i in range(len(x_data) - 1)):
            raise ValueError("x_axis_data must be in sequential order")
        return values

    @model_validator(mode="after")
    def validate_data_length_match(cls, values):
        if len(values.x_axis_data) != len(values.y_axis_data):
            raise ValueError("x_axis_data and y_axis_data must have the same length")
        return values

    model_config = {"populate_by_name": True}


if __name__ == "__main__":
    StatsBarDiagram.generate_assistant_function_schema("sat-math")
