import re
from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_validator


class Dimensions(BaseModel):
    columns: int = Field(
        ..., description="the number of columns in the area model data"
    )
    rows: int = Field(..., description="the number of rows in the area model data")


class Headers(BaseModel):
    """Defines the headers for the area model, which represent the decomposition of products or dividends into their constituent parts.
    The headers are either the decomposition pieces of the product or dividend or a single letter representing a variable to assist with partial and explanation questions.
    """

    columns: List[str | int | float] = Field(
        ...,
        description=(
            "The decomposed pieces of the product or dividend to be displayed on the top of the area model, left to right. "
            "Can be single letters (A, B), numbers (4, 2), fractions (2/5, 3/4), decimals (1.5, 2.75), or question marks (?) for unknown values."
        ),
    )
    rows: list[str | int | float] = Field(
        ...,
        description=(
            "The decomposed pieces of the product or dividend to be displayed on the left side of the area model, top to bottom. "
            "Can be single letters (A, B), numbers (4, 2), fractions (2/5, 3/4), decimals (1.5, 2.75), or question marks (?) for unknown values."
        ),
    )

    @field_validator("columns", "rows", mode="before")
    @classmethod
    def validate_headers(cls, headers):
        # Updated pattern to allow:
        # - Single letters: A, B, C
        # - Numbers: 4, 2, 3
        # - Fractions: 2/5, 3/4, 1/3
        # - Decimals: 1.5, 2.75
        # - Question marks: ? (for unknown values)
        pattern = r"^([a-zA-Z]|\d+(\.\d+)?|\d+/\d+|\?)$"
        for item in headers:
            if isinstance(item, str) and not re.match(pattern, item):
                raise ValueError(f"{item} does not match the required pattern")
        return headers


class AreaModel(StimulusDescription):
    """An area model with headers and data. Every string in the data is either a number or a letter representation."""

    dimensions: Dimensions = Field(..., description="The dimensions of the area model")
    headers: Headers = Field(
        ...,
        description="The headers of the area model as either the number or a letter representation.",
    )
    data: List[List[str | int | float]] = Field(
        ...,
        description=(
            "The partials of the product or dividend to be displayed inside of the area model, left to right, top to bottom. "
            "Can be single/double letters (A, AC, BD), numbers (12, 6), fractions (6/5, 9/4), decimals (1.5, 2.75), or question marks (?) for unknown values."
        ),
    )

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, data):
        # Updated pattern to allow:
        # - Single/double letters: A, AC, BD
        # - Numbers: 12, 6, 10
        # - Fractions: 6/5, 9/4, 5/3
        # - Decimals: 1.5, 2.75
        # - Question marks: ? (for unknown values)
        pattern = r"^([a-zA-Z]{1,2}|\d+(\.\d+)?|\d+/\d+|\?)$"
        for row in data:
            for item in row:
                if isinstance(item, str) and not re.match(pattern, item):
                    raise ValueError(f"{item} does not match the required pattern")
        return data

    class Config:
        json_schema_extra = {
            "Example Multiplication": [
                {
                    "dimensions": {"columns": 2, "rows": 2},
                    "headers": {"columns": ["A", "B"], "rows": ["C", "D"]},
                    "data": [["AC", "AD"], ["BC", "BD"]],
                }
            ]
        }


if __name__ == "__main__":
    print(AreaModel.model_json_schema())
