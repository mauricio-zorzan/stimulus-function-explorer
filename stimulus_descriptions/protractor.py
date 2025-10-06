from typing import Literal

from pydantic import BaseModel, Field

from .stimulus_description import StimulusDescriptionList


class ProtractorPoint(BaseModel):
    label: Literal["R", "S", "T", "V"] = Field(
        ..., description="The label of the point on the protractor"
    )
    degree: int = Field(
        ..., description="The degree measurement of the point", ge=5, le=180
    )


class Protractor(StimulusDescriptionList):
    """A protractor stimulus with a list of points. The origin is Point P and a line is drawn to Point Q at 0 degrees."""

    root: list[ProtractorPoint] = Field(
        ..., min_length=4, max_length=4, description="List of points on the protractor"
    )

    @classmethod
    def model_validator(cls, values):
        """
        Check that each point is at least 5 degrees apart from each other
        """
        points = values.get("root", [])

        # Sort points by degree
        sorted_points = sorted(points, key=lambda x: x.degree)

        # Check the difference between adjacent points
        for i in range(len(sorted_points)):
            current = sorted_points[i].degree
            next_point = sorted_points[(i + 1) % len(sorted_points)].degree

            # Handle the case where we're comparing the last point with the first
            if i == len(sorted_points) - 1:
                diff = (next_point + 360 - current) % 360
            else:
                diff = next_point - current

            if diff < 5:
                raise ValueError(
                    f"Points must be at least 5 degrees apart. Found {diff} degrees between {sorted_points[i].label} and {sorted_points[(i + 1) % len(sorted_points)].label}"
                )

        return values


if __name__ == "__main__":
    Protractor.generate_assistant_function_schema("mcq4")
