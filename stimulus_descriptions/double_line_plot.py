from typing import List

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription


class DataPoint(BaseModel):
    """
    A data point for a double line plot.
    """

    class Config:
        description = "A data point for a double line plot."

    value: int = Field(
        description="The x-axis value for the data point as a number.",
    )
    frequency: int = Field(
        description="The frequency or count of occurrences at this value.", le=8, ge=0
    )


class Dataset(BaseModel):
    """
    A dataset for a double line plot.
    """

    class Config:
        description = "A dataset for a double line plot."

    title: str = Field(description="The title of the dataset for the double line plot.")
    data_points: List[DataPoint] = Field(
        description="List of data points and their frequencies for the dataset."
    )

    @model_validator(mode="after")
    def validate_data_points(cls, dataset):
        dataset.data_points.sort(
            key=lambda x: x.value
        )  # Sort data points by value_parsed
        return dataset


class Range(BaseModel):
    """
    The range of the x-axis of the double line plot with appropriate size for the question.
    """

    class Config:
        description = "The range of the x-axis of the double line plot with appropriate size for the question."

    min: int = Field(
        description="The minimum range value of the x-axis of the double line plot."
    )
    max: int = Field(
        description="The maximum range value of the x-axis of the double line plot."
    )


class DoubleLinePlot(StimulusDescription):
    """
    Double Line plot stimulus description
    """

    class Config:
        description = "Double Line plot stimulus description"

    x_axis_label: str = Field(
        description="The label for the x-axis of the double line plot with correct context for the question."
    )
    range: Range = Field(
        description="The range of the x-axis of the double line plot with appropriate size for the question.",
        exclude=True,
    )

    datasets: List[Dataset] = Field(
        description="The list of the two datasets for the double line plot.",
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="before")
    def set_range_and_fill_zeros(cls, double_line_plot):
        all_x_values = [
            point["value"]
            for dataset in double_line_plot["datasets"]
            for point in dataset["data_points"]
        ]
        min_x = min(all_x_values)
        max_x = max(all_x_values)
        double_line_plot["range"] = Range(min=min_x, max=max_x)
        max_length_x_value = max(len(str(x)) for x in all_x_values)
        if max_length_x_value > 3:
            raise ValueError(
                f"The maximum length of the x-axis str repr for value is {max_length_x_value} which is greater than the maximum length of 3."
            )
        elif max_length_x_value > 1:
            assert (
                max_x - min_x <= 12
            ), "The range of the x-axis is greater than 12 with double or triple digit values."
        else:
            assert (
                max_x - min_x <= 20
            ), "The range of the x-axis is greater than 20 with double/single digit values."

        # Fill all the omitted values with 0's
        for dataset in double_line_plot["datasets"]:
            min_x = double_line_plot["range"].min
            max_x = double_line_plot["range"].max
            filled_data_points = []
            for x in range(min_x, max_x + 1):
                matching_points = [
                    point for point in dataset["data_points"] if point["value"] == x
                ]
                if matching_points:
                    filled_data_points.append(matching_points[0])
                else:
                    filled_data_points.append({"value": x, "frequency": 0})
            dataset["data_points"] = filled_data_points
        return double_line_plot


if __name__ == "__main__":
    DoubleLinePlot.generate_assistant_function_schema(type="mcq4")
