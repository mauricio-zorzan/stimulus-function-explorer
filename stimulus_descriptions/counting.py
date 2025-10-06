from pydantic import Field

from .common import DrawableItem
from .stimulus_description import StimulusDescription


class Counting(StimulusDescription):
    object_name: DrawableItem = Field(
        ..., description="The type of object to be counted"
    )
    count: int = Field(
        ..., description="The number of objects to be drawn", ge=5, le=20
    )


if __name__ == "__main__":
    Counting.generate_assistant_function_schema(type="mcq3")
