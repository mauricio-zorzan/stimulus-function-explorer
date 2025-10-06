import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class Segment(BaseModel):
    start_coordinate: Tuple[int, int]
    end_coordinate: Tuple[int, int]
    linear: bool = Field(
        ..., description="True if the segment is linear, False if nonlinear"
    )


class GraphingPiecewise(StimulusDescription):
    x_axis_label: Optional[str] = Field(None, description="Label for the x-axis")
    y_axis_label: Optional[str] = Field(None, description="Label for the y-axis")
    segments: List[Segment] = Field(
        ..., description="List of segments in the piecewise function"
    )

    @model_validator(mode="after")
    def check_segments(self) -> "GraphingPiecewise":
        segments = self.segments
        if len(segments) < 3 or len(segments) > 5:
            raise ValueError(
                f"Number of segments must be between 3 and 5. Got {len(segments)} segments."
            )

        for i in range(1, len(segments)):
            if segments[i].start_coordinate != segments[i - 1].end_coordinate:
                raise ValueError(
                    f"Segment {i} does not connect with the previous segment. "
                    f"Expected start_coordinate to be {segments[i-1].end_coordinate}, but got {segments[i].start_coordinate}"
                )

        return self

    @model_validator(mode="after")
    def check_coordinates_within_range(self) -> "GraphingPiecewise":
        max_distance = 10
        for i, segment in enumerate(self.segments):
            start_x, start_y = segment.start_coordinate
            end_x, end_y = segment.end_coordinate

            if (
                abs(start_x) > max_distance
                or abs(start_y) > max_distance
                or abs(end_x) > max_distance
                or abs(end_y) > max_distance
            ):
                raise ValueError(
                    f"All coordinates must be within {max_distance} units of the origin. "
                    f"Found out-of-range coordinate in segment {i}: "
                    f"start ({start_x}, {start_y}), end ({end_x}, {end_y})"
                )
        return self

    @model_validator(mode="after")
    def check_low_y_coordinate(self) -> "GraphingPiecewise":
        has_low_y = any(
            segment.start_coordinate[1] <= 1 or segment.end_coordinate[1] <= 1
            for segment in self.segments
        )
        if not has_low_y:
            raise ValueError(
                "At least one segment must have a coordinate with a y-value less than or equal to 1."
            )
        return self

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext") -> None:
        """
        Additional validation here for substandard configurations
        """
        logging.info(self)
        super().pipeline_validate(pipeline_context)
        if pipeline_context.standard_id in {
            "CCSS.MATH.CONTENT.8.F.B.5+2",
        }:
            self.ensure_positive_coordinates()

    def ensure_positive_coordinates(self) -> None:
        """
        Ensures that all coordinates in the segments are positive.
        Raises a ValueError if any coordinate is negative.
        """
        for i, segment in enumerate(self.segments):
            start_x, start_y = segment.start_coordinate
            end_x, end_y = segment.end_coordinate

            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                raise ValueError(
                    f"All coordinates must be positive for substandard CCSS.MATH.CONTENT.8.F.B.5+2. "
                    f"Found negative coordinate in segment {i}: "
                    f"start ({start_x}, {start_y}), end ({end_x}, {end_y})"
                )
