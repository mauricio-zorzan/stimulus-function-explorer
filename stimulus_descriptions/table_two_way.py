from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class TableTwoWay(StimulusDescription):
    table_title: str = Field(..., description="The title of the table.")
    rows_title: list[dict[str, str]] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of row titles with labels 'label_x' where x is the row number.",
    )
    columns_title: list[dict[str, str]] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of column titles with labels 'label_x' where x is the column number.",
    )
    data: list[dict[str, int]] = Field(
        ...,
        description="List of data entries with keys 'x' where x is the column number.",
    )
