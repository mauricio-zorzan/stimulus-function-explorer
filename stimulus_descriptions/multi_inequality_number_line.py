from content_generators.additional_content.stimulus_image.stimulus_descriptions.inequality_number_line import (
    InequalityNumberLine,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, model_validator


class MultiInequalityNumberLine(StimulusDescription):
    """A collection of 4 inequality number lines arranged in a 2x2 grid format."""

    number_lines: list[InequalityNumberLine] = Field(
        min_length=4,
        max_length=4,
        description="Exactly 4 inequality number lines to be displayed in a 2x2 grid.",
    )

    @model_validator(mode="after")
    def validate_number_lines(self):
        if len(self.number_lines) != 4:
            raise ValueError(
                "MultiInequalityNumberLine must contain exactly 4 inequality number lines."
            )
        return self


if __name__ == "__main__":
    MultiInequalityNumberLine.generate_assistant_function_schema("mcq4")
