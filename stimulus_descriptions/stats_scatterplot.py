from typing import List

import numpy as np
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator


class Point(BaseModel):
    x: float
    y: float
    model_config = {"populate_by_name": True}


class LineOfBestFit(BaseModel):
    slope: float
    intercept: float
    model_config = {"populate_by_name": True}


class StatsScatterplot(StimulusDescription):
    points: List[Point] = Field(..., min_length=5, max_length=15)
    line_of_best_fit: LineOfBestFit

    @model_validator(mode="after")
    def validate_point_range(self):
        x_values = [point.x for point in self.points]
        y_values = [point.y for point in self.points]

        if (
            max(x_values) > 15
            or min(x_values) < -15
            or max(y_values) > 15
            or min(y_values) < -15
        ):
            raise ValueError(
                "All point coordinates must be within the range of -15 to 15."
            )

        return self

    @model_validator(mode="after")
    def validate_unique_points(self):
        point_set = set((point.x, point.y) for point in self.points)
        if len(point_set) != len(self.points):
            raise ValueError("All points must be unique.")

        return self

    @model_validator(mode="after")
    def validate_line_of_best_fit(self):
        x = np.array([point.x for point in self.points])
        y = np.array([point.y for point in self.points])

        # Calculate the actual line of best fit
        slope, intercept = np.polyfit(x, y, 1)

        # Check if the provided line of best fit is close to the actual one
        if (
            abs(slope - self.line_of_best_fit.slope) > 0.5
            or abs(intercept - self.line_of_best_fit.intercept) > 0.5
        ):
            raise ValueError(
                "The provided line of best fit does not match the actual line of best fit within the allowed margin of error."
            )

        return self

    @model_validator(mode="after")
    def validate_points_on_line(self):
        """Validate that at least two points exist on the provided line of best fit"""
        slope = self.line_of_best_fit.slope
        intercept = self.line_of_best_fit.intercept

        # Tolerance for considering a point to be on the line
        tolerance = 0.1
        epsilon = 1e-9  # small buffer to account for floating-point rounding errors

        points_on_line = 0
        for point in self.points:
            # Calculate expected y value for this x on the line
            expected_y = slope * point.x + intercept
            # Check if the actual y value is close to the expected y value
            if abs(point.y - expected_y) <= tolerance + epsilon:
                points_on_line += 1

        if points_on_line < 2:
            raise ValueError(
                f"At least two points must exist on the provided line of best fit. "
                f"Only {points_on_line} point(s) found on the line (tolerance: {tolerance})."
            )

        return self


if __name__ == "__main__":
    StatsScatterplot.generate_assistant_function_schema("sat-math")
