from typing import List, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """A single data point with x and y coordinates."""

    x: float = Field(..., description="The x-coordinate of the data point.")
    y: float = Field(..., description="The y-coordinate of the data point.")


class ScatterplotData(BaseModel):
    """Configuration for a single scatterplot option."""

    title: str = Field(..., description="The title of the scatterplot option.")
    x_label: str = Field(..., description="The label for the x-axis.")
    y_label: str = Field(..., description="The label for the y-axis.")
    x_min: float = Field(..., description="The minimum value for the x-axis.")
    x_max: float = Field(..., description="The maximum value for the x-axis.")
    y_min: float = Field(..., description="The minimum value for the y-axis.")
    y_max: float = Field(..., description="The maximum value for the y-axis.")
    data_points: List[DataPoint] = Field(
        ...,
        description="The list of data points to plot in the scatterplot.",
        min_length=1,
        max_length=20,
    )
    is_correct: bool = Field(
        default=True,
        description="Whether this scatterplot correctly represents the table data.",
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Type of error if incorrect: 'swapped_coordinates', 'missing_points', 'extra_points', 'shifted_points'",
    )


class TableData(BaseModel):
    """Configuration for the data table."""

    headers: List[str] = Field(
        ..., description="The column headers for the table.", min_length=2, max_length=5
    )
    rows: List[List[str]] = Field(
        ...,
        description="The rows of data in the table. Each row should have the same number of elements as headers.",
        min_length=1,
        max_length=20,
    )


class TableAndMultiScatterplots(StimulusDescription):
    """
    Stimulus description for creating a table alongside multiple scatterplot options.

    This addresses scatter plot interpretation standards by presenting tabular data
    and asking students to identify which scatterplot correctly represents the data.
    Supports multiple choice questions with correct and incorrect options.
    """

    table: TableData = Field(
        ..., description="The table data to display alongside the scatterplots."
    )

    scatterplots: List[ScatterplotData] = Field(
        ...,
        description="The list of scatterplot options to display. At least one should be correct.",
        min_length=1,
        max_length=4,
    )

    layout: str = Field(
        default="vertical",
        description="Layout arrangement: 'vertical' (table above plots) or 'horizontal' (table beside plots).",
    )


if __name__ == "__main__":
    TableAndMultiScatterplots.generate_assistant_function_schema("mcq4")
