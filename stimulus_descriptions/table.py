from typing import List, Optional  # noqa: I001

from pydantic import AliasChoices, BaseModel, Field, field_validator

from content_generators.additional_content.stimulus_image.drawing_functions.common import (
    TextHelper,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)


# Define the ColumnEntry model
class TableColumn(BaseModel):
    label: str


# Define the RowEntry model with aliases for fields "1" and "2"
class TableRow(BaseModel):
    field_1: str = Field(
        ...,
        alias="1",
        validation_alias=AliasChoices("field_1", "1"),
        coerce_numbers_to_str=True,
    )
    field_2: str = Field(
        ...,
        alias="2",
        validation_alias=AliasChoices("field_2", "2"),
        coerce_numbers_to_str=True,
    )


class Table(StimulusDescription):
    columns: list[TableColumn] = Field(
        ..., description="List of columns with their labels."
    )
    rows: list[TableRow] = Field(
        ..., description="List of rows with their data.", max_length=15
    )

    def extract_and_wrap_data(self, text_helper: TextHelper, wrap_length: int):
        # Extract columns
        column_labels = [col.label for col in self.columns]
        # Extract data rows
        data_rows = [[row.field_1, row.field_2] for row in self.rows]

        # Wrap text for columns
        wrapped_columns = [
            text_helper.wrap_text(label, wrap_length) for label in column_labels
        ]

        # Wrap text for data rows
        wrapped_data_rows = [
            [text_helper.wrap_text(str(item), wrap_length) for item in row]
            for row in data_rows
        ]

        # Create data rows as a list of dictionaries with wrapped column labels as keys
        data_rows = [dict(zip(wrapped_columns, row)) for row in wrapped_data_rows]

        return wrapped_columns, data_rows

    def calculate_max_cell_dimensions(
        self, text_helper: TextHelper, data_rows, columns
    ):
        max_cell_width = 0
        max_cell_height = 0
        for row in data_rows:
            for value in row:
                cell_width = text_helper.get_text_width(str(value), pad=2)
                cell_height = text_helper.get_text_height(str(value), pad=0.1)
                max_cell_width = max(max_cell_width, cell_width)
                max_cell_height = max(max_cell_height, cell_height)

        for col in columns:
            cell_width = text_helper.get_text_width(col, pad=2)
            cell_height = text_helper.get_text_height(col, pad=0.5)
            max_cell_width = max(max_cell_width, cell_width)
            max_cell_height = max(max_cell_height, cell_height)
        return max_cell_width, max_cell_height


class DataTable(StimulusDescription):
    """Draw a data table with a title and optional metadata."""

    headers: List[str] = Field(
        ..., description="List of column headers", max_length=8, min_length=1
    )
    data: List[List[str]] = Field(
        ...,
        description="List of rows, each row is a list of cell values. Possibly use variables to help with table construction questions.",
        min_length=2,
        max_length=15,
    )
    title: Optional[str] = Field(
        None,
        max_length=40,
        description="Title of the data table, must not exceed 40 characters.",
    )
    metadata: Optional[str] = Field(
        None,
        max_length=40,
        description="Footer or additional metadata for the data table, must not exceed 40 characters.",
    )

    @field_validator("data")
    @classmethod
    def check_data(cls, v, info):
        headers = info.data.get("headers")
        num_columns = len(headers)
        for row in v:
            if len(row) != num_columns:
                raise ValueError(
                    "All rows must have the same number of columns as headers"
                )
        v = [[str(item) for item in row] for row in v]  # Handle intermittent numbers
        return v


class DataTableGroup(StimulusDescription):
    """Draw a group of data tables arranged in a grid layout."""

    tables: List[DataTable] = Field(
        ...,
        description="List of data tables to be grouped together",
        min_length=1,
        max_length=6,
    )
    group_title: Optional[str] = Field(
        None,
        max_length=60,
        description="Overall title for the group of tables, must not exceed 60 characters.",
    )
    layout: Optional[str] = Field(
        "auto",
        description="Layout arrangement for tables. Options: 'auto' (automatic grid), 'horizontal' (single row), 'vertical' (single column)",
    )

    @field_validator("layout")
    @classmethod
    def validate_layout(cls, v):
        valid_layouts = {"auto", "horizontal", "vertical"}
        if v not in valid_layouts:
            raise ValueError(f"Layout must be one of: {', '.join(valid_layouts)}")
        return v


if __name__ == "__main__":
    Table.generate_assistant_function_schema("mcq4")
