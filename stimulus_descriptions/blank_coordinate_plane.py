from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field


class BlankCoordinatePlane(StimulusDescription):
    """
    Blank coordinate plane stimulus with configurable axis ranges.
    Default ranges: x=[-10, 10], y=[-10, 10]
    """

    x_axis_title: str | None = Field(default=None)
    y_axis_title: str | None = Field(default=None)
    x_min: int = Field(default=-10, description="Minimum x-axis value")
    x_max: int = Field(default=10, description="Maximum x-axis value")
    y_min: int = Field(default=-10, description="Minimum y-axis value")
    y_max: int = Field(default=10, description="Maximum y-axis value")
