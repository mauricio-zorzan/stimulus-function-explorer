from pydantic import Field

from .common import DrawableItem
from .stimulus_description import StimulusDescription


class ObjectArray(StimulusDescription):
    object_name: DrawableItem = Field(
        ..., description="The type of object to be arranged in the array"
    )
    rows: int = Field(..., description="The number of rows in the array", ge=2, le=7)
    columns: int = Field(
        ..., description="The number of columns in the array", ge=2, le=7
    )


if __name__ == "__main__":
    ObjectArray.generate_assistant_function_schema(type="mcq4")
