from pydantic import Field, field_validator

from .stimulus_description import StimulusDescription


class DivideItemsIntoArray(StimulusDescription):
    """
    Stimulus description for creating a visual representation of items arranged in a rectangular array.
    Shows small circles arranged in rows and columns to form a grid pattern.
    """
    
    num_rows: int = Field(
        ..., 
        description="The number of rows in the array",
        ge=1,
        le=10
    )
    
    num_columns: int = Field(
        ...,
        description="The number of columns in the array",
        ge=1,
        le=10
    )
    
    @field_validator("num_rows", "num_columns")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Must be a positive integer")
        return v 