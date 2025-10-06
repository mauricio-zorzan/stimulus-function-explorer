import math
from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Line(BaseModel):
    slope: float
    y_intercept: int


class IntersectionPoint(BaseModel):
    x: int
    y: int


class LinearDiagram(StimulusDescription):
    lines: List[Line] = Field(..., min_length=2, max_length=2)
    intersection_point: IntersectionPoint

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_diagram(self):
        self.validate_max_number()
        self.validate_intersection_point()
        return self

    def validate_max_number(self):
        max_allowed = 8
        for line in self.lines:
            if abs(line.y_intercept) > max_allowed:
                raise ValueError(
                    f"No y-intercept in the stimulus description should be greater than {max_allowed}"
                )
        if (
            abs(self.intersection_point.x) > max_allowed
            or abs(self.intersection_point.y) > max_allowed
        ):
            raise ValueError(
                f"No coordinate in the intersection point should be greater than {max_allowed}"
            )

    def validate_intersection_point(self):
        line1, line2 = self.lines

        if line1.slope == line2.slope:
            if line1.y_intercept == line2.y_intercept:
                raise ValueError(
                    "The lines are identical and have infinite intersection points"
                )
            else:
                raise ValueError(
                    "The lines are parallel and have no intersection point"
                )

        # Calculate the intersection point
        x = (line2.y_intercept - line1.y_intercept) / (line1.slope - line2.slope)
        y = line1.slope * x + line1.y_intercept

        # Round to nearest integer for comparison
        x_rounded = round(x)
        y_rounded = round(y)

        # Check if the given intersection point matches the calculated (rounded) point
        if (
            x_rounded != self.intersection_point.x
            or y_rounded != self.intersection_point.y
        ):
            raise ValueError(
                f"The given intersection point ({self.intersection_point.x}, {self.intersection_point.y}) "
                f"does not match the calculated intersection point ({x_rounded}, {y_rounded})"
            )

        # Check if the calculated point is sufficiently close to an integer
        tolerance = 1e-9
        if not (
            math.isclose(x, x_rounded, abs_tol=tolerance)
            and math.isclose(y, y_rounded, abs_tol=tolerance)
        ):
            raise ValueError(
                f"The calculated intersection point ({x}, {y}) is not sufficiently close to integer coordinates"
            )

        return self


if __name__ == "__main__":
    LinearDiagram.generate_assistant_function_schema("sat-math")
