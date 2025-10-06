from typing import List

from pydantic import BaseModel, Field, model_validator

from .stimulus_description import StimulusDescription


class Range(BaseModel):
    """Range between min and max must be less than 5 hours."""

    min: int = Field(
        ..., description="The minimum hour value on the number line", ge=1, le=12
    )
    max: int = Field(
        ..., description="The maximum hour value on the number line", ge=1, le=12
    )


class TimePoint(BaseModel):
    """A time point on the number line clock model."""

    label: str = Field(
        ...,
        description="The label for this time point, either the time (e.g 3:45 PM) or (e.g Start: 3:45 PM)",
    )
    hour: int = Field(..., ge=1, le=12, description="The hour value")
    minute: int = Field(..., ge=0, le=59, multiple_of=5, description="The minute value")


class NumberLineClockStimulus(StimulusDescription):
    """A number line clock model with a range of hours and a list of time points.
    If multiple points, they must be at least 30 minutes apart and within the range.
    If using labels such as "Start: 3:45 PM" with 2 words, ensure that the points are at least an hour apart.
    """

    range: Range = Field(
        ..., description="The range of hours displayed on the number line"
    )
    points: List[TimePoint] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="List of time points to be displayed on the number line",
    )

    @model_validator(mode="after")
    def validate_range_and_points(cls, values):
        # 1. Range must be less than 5
        min_hour, max_hour = values.range.min, values.range.max
        range_length = (max_hour - min_hour) % 12 or 12
        if range_length >= 5:
            raise ValueError("Range must be less than 5 hours")

        # 4. If multiple points, they must be at least 30 minutes apart
        if len(values.points) == 2:
            time1 = (values.points[0].hour % 12) * 60 + values.points[0].minute
            time2 = (values.points[1].hour % 12) * 60 + values.points[1].minute
            time_diff = min((time2 - time1) % 720, (time1 - time2) % 720)
            if time_diff < 30:
                raise ValueError("Multiple points must be at least 30 minutes apart")

        # 5. All points must be within the specified range
        min_time = (min_hour % 12) * 60
        max_time = ((max_hour % 12) * 60 - 1) % 720  # -1 to exclude the max hour itself
        for point in values.points:
            point_time = (point.hour % 12) * 60 + point.minute
            if range_length < 12:
                if min_time <= max_time:
                    if not (min_time <= point_time <= max_time):
                        raise ValueError(
                            f"Point {point.hour}:{point.minute:02d} is outside the specified range"
                        )
                else:  # Range crosses 12 o'clock
                    if not (point_time >= min_time or point_time <= max_time):
                        raise ValueError(
                            f"Point {point.hour}:{point.minute:02d} is outside the specified range"
                        )
            # If range_length == 12, all points are valid, so no check needed

        return values

    class Config:
        json_schema_extra = {
            "example": {
                "range": {"min": 7, "max": 10},
                "points": [{"label": "Start", "hour": 8, "minute": 45}],
            }
        }


if __name__ == "__main__":
    NumberLineClockStimulus.generate_assistant_function_schema(type="mcq4")
