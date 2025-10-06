from typing import Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class Clock(StimulusDescription):
    """
    Stimulus description for a clock, which can be either analog or digital.
    """

    type: Literal["analog", "digital"] = Field(
        ...,
        description="The clock to be created. This is either analog or digital.",
    )
    hour: int = Field(
        ...,
        description="The hour to be displayed on the clock.",
    )
    minute: int = Field(
        ...,
        description="The minute to be displayed on the clock.",
    )


if __name__ == "__main__":
    Clock.generate_assistant_function_schema("mcq4")
