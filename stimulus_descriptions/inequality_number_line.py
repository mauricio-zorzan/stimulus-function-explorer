from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator


class Range(BaseModel):
    """A maximum and minimum value on the number line. Total range difference must be less than 20."""

    min: int = Field(ge=-99, le=99)
    max: int = Field(ge=-99, le=99)

    @model_validator(mode="after")
    def check_range(self):
        if self.max - self.min > 20:
            raise ValueError(f"A range of {self.max - self.min} is too large.")
        return self


class Point(BaseModel):
    fill: bool = Field(
        default=False,
        description="Whether the point is filled in. If false the point is an non-filled in circle.",
    )
    value: int = Field(ge=-99, le=99)


class Line(BaseModel):
    """A line on the number line, depicting a sub-set of the range to show the inequality.
    At least one of the min or max values must be provided.
    If a value is None, the line will be the full length of the number line on that side."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def check_min_or_max_not_none(self):
        if self.min is None and self.max is None:
            raise ValueError(
                "At least one of 'min' or 'max' must not be None for a line to be valid."
            )
        return self


class InequalityNumberLine(StimulusDescription):
    """A number line with a range and two points on the line and a line depicting a sub-set of the range."""

    range: Range
    points: list[Point] = Field(min_length=1, max_length=2)
    line: Line

    @model_validator(mode="after")
    def check_points_and_lines(self):
        points = self.points
        line = self.line

        valid_points = {
            point.value for point in points
        }  # Create a set of valid points for faster lookup

        min_value = line.min
        max_value = line.max

        # Check if min and max are either None, or exist in the set of valid points
        if (min_value is not None and min_value not in valid_points) or (
            max_value is not None and max_value not in valid_points
        ):
            raise ValueError(
                f"Line with min {min_value} and max {max_value} has invalid point references."
            )

        for point in points:
            point_value = point.value
            is_valid_point = False

            min_value = line.min
            max_value = line.max

            if point_value == min_value or point_value == max_value:
                is_valid_point = True
                break

            if not is_valid_point:
                raise ValueError(
                    f"Point {point_value} is not the min or max of any line."
                )

        range_min = self.range.min
        range_max = self.range.max
        min_value = line.min or range_min
        max_value = line.max or range_max

        if min_value == range_min and max_value == range_max:
            raise ValueError(
                "Invalid Line: Line is the full length of the number line."
            )

        if max_value is not None and min_value is not None:
            if max_value - min_value < 1:
                raise ValueError("Invalid line: max - min must be greater than 0.")
        return self


if __name__ == "__main__":
    InequalityNumberLine.generate_assistant_function_schema("mcq4")
