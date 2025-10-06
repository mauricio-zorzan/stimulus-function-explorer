from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription


class DataItem(BaseModel):
    col1: int = Field(..., alias="1")
    col2: int = Field(..., alias="2")
    col3: int = Field(..., alias="3")
    row_total: int = Field(..., alias="4")


class ProbabilityDiagram(StimulusDescription):
    rows_title: List[Dict[str, str]] = Field(..., min_length=3, max_length=3)
    columns_title: List[Dict[str, str]] = Field(..., min_length=4, max_length=4)
    data: List[DataItem] = Field(..., min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_structure_and_totals(self):
        # Validate row and column labels
        for i, row in enumerate(self.rows_title, 1):
            if f"label_{i}" not in row:
                raise ValueError(f"Row {i} label is incorrect. Expected 'label_{i}'")

        for i, col in enumerate(self.columns_title, 1):
            if f"label_{i}" not in col:
                raise ValueError(f"Column {i} label is incorrect. Expected 'label_{i}'")

        data = self.data

        # Validate column totals
        for col in range(1, 4):
            column_total = sum(getattr(row, f"col{col}") for row in data[:2])
            if getattr(data[2], f"col{col}") != column_total:
                raise ValueError(
                    f"Column {col} total is incorrect. Expected {column_total}, got {getattr(data[2], f'col{col}')}"
                )

        # Validate row totals
        for i, row in enumerate(data):
            row_total = sum(getattr(row, f"col{col}") for col in range(1, 4))
            if row.row_total != row_total:
                raise ValueError(
                    f"Row {i+1} total is incorrect. Expected {row_total}, got {row.row_total}"
                )

        return self

    model_config = {"populate_by_name": True}


if __name__ == "__main__":
    ProbabilityDiagram.generate_assistant_function_schema("sat-math")
