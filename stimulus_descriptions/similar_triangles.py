from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class SimilarRightTriangles(StimulusDescription):
    angle_labels: List[str] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Labels for the angles of the two similar right triangles.",
    )

    @model_validator(mode="after")
    def validate_angle_labels(self):
        if len(set(self.angle_labels)) != 6:
            raise ValueError("Each angle label must be unique.")
        return self


if __name__ == "__main__":
    SimilarRightTriangles.generate_assistant_function_schema("mcq4")
