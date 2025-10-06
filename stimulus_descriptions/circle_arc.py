from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class CircleWithArcsDescription(StimulusDescription):
    """
    This schema represents the description of a circle with intersecting lines.
    """

    arc_size: float = Field(
        ...,
        ge=10,
        le=80,
        description="The size of the arc in degrees, ranging from 10 to 80.",
    )
    point_labels: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="A list of four labels for the points at the arc endpoints.",
    )

    def validate(self):
        if len(self.point_labels) != 4:
            raise ValueError("point_labels must contain exactly 4 labels.")


if __name__ == "__main__":
    CircleWithArcsDescription.generate_assistant_function_schema("sat-math")
