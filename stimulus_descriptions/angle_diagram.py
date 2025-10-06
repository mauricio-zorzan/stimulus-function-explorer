from functools import cached_property
from typing import TYPE_CHECKING, Optional, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class Angle(BaseModel):
    measure: Union[float, str] = Field(
        ...,
        description="The measure of the angle in degrees, or a variable string (e.g., '?', 'x', '2y+5') for unknown angles",
    )
    positioning_measure: Optional[float] = Field(
        None,
        description="Actual measure for positioning when measure is a variable string",
    )
    points: list[str] = Field(
        ..., description="The labels of the points forming the angle"
    )

    @model_validator(mode="after")
    def validate_measure(self):
        if isinstance(self.measure, str):
            # Allow any string for variables (e.g., "?", "x", "2y+5", etc.)
            if self.positioning_measure is None:
                raise ValueError(
                    "positioning_measure is required when measure is a variable string"
                )
            if not (1 <= self.positioning_measure <= 179):
                raise ValueError(
                    "positioning_measure must be between 1 and 179 degrees"
                )
        else:
            if not (1 <= self.measure <= 179):
                raise ValueError(
                    "Numeric angle measure must be between 1 and 179 degrees"
                )
            if self.positioning_measure is not None:
                raise ValueError(
                    "positioning_measure should only be provided when measure is a variable string"
                )
        return self

    def get_numeric_measure(self) -> float:
        """Get numeric measure for positioning calculations."""
        if isinstance(self.measure, str):
            assert self.positioning_measure is not None  # Guaranteed by validation
            return float(self.positioning_measure)
        return float(self.measure)

    def get_display_text(self) -> str:
        """Get text to display on the diagram."""
        if isinstance(self.measure, str):
            return self.measure  # Returns the variable string (e.g., "?", "x", "2y+5")
        return f"{int(self.measure)}°"


class Angles(BaseModel):
    angles: list[Angle] = Field(
        ...,
        description="A list of angles forming the diagram. The sum of all angle measures must be exactly 90, 180, or 360 degrees. The angles must be sequential and share a common center point and adjacent angles share connecting points.",
    )

    @model_validator(mode="after")
    def validate_angle_sum(self):
        """Validate that angles sum to exactly 90, 180, or 360 degrees and no total angle is included."""
        # Calculate total of all angle measures
        total = sum(angle.get_numeric_measure() for angle in self.angles)

        # Check if total is one of the valid sums
        valid_totals = [90, 180, 360]
        if total not in valid_totals:
            raise ValueError(
                f"The sum of all angle measures must be exactly 90, 180, or 360 degrees. "
                f"Current sum is {total}°. Valid totals are: {valid_totals}"
            )

        # Check for potential "total" angles that shouldn't be included
        # Flag angles that are suspiciously close to common totals
        for angle in self.angles:
            measure = angle.get_numeric_measure()
            if measure in [180, 360] and len(self.angles) > 1:
                # If we have multiple angles and one equals a common total, it might be incorrectly included
                other_angles_sum = sum(
                    a.get_numeric_measure() for a in self.angles if a != angle
                )
                if other_angles_sum > 0:  # If there are other meaningful angles
                    raise ValueError(
                        f"Angle with measure {measure}° appears to be a total angle and should not be "
                        f"included in the angles array. Include only the individual component angles, "
                        f"not their sum."
                    )

        return self

    @model_validator(mode="after")
    def validate_sequential_point_sharing(self):
        """Validate that all angles share a common center point and adjacent angles share connecting points."""
        if len(self.angles) < 2:
            return self  # Single angle doesn't need sequential validation

        # Extract center points - should all be the same (middle point in each angle's points list)
        center_points = [angle.points[1] for angle in self.angles]
        if len(set(center_points)) != 1:
            raise ValueError(
                f"All angles must share the same center point. Found center points: {center_points}"
            )

        total = sum(angle.get_numeric_measure() for angle in self.angles)

        # Validate that adjacent angles share their connecting points
        angles = self.angles.copy()
        if total == 360:
            angles.append(angles[0])

        for i in range(len(angles) - 1):
            current_angle = angles[i]
            next_angle = angles[i + 1]

            # The ending point of current angle should be the starting point of next angle
            current_end_point = current_angle.points[2]  # Third point is the ending ray
            next_start_point = next_angle.points[0]  # First point is the starting ray

            if current_end_point != next_start_point:
                raise ValueError(
                    f"Adjacent angles must share connecting points. "
                    f"Angle {i + 1} (points: {current_angle.points}) ends at '{current_end_point}', "
                    f"but angle {i + 2} (points: {next_angle.points}) starts at '{next_start_point}'. "
                    f"These should be the same point for proper sequential placement."
                )

        return self


class AngleDiagram(StimulusDescription):
    """An angle diagram is a diagram with multiple angles with a common shared point. The central point for all angles is the first angle's second point. For example, ABC, CBD, and DBE are all angles with a common shared point B."""

    diagram: Angles = Field(
        ..., description="A dictionary containing the angles information"
    )

    @model_validator(mode="after")
    def validate_angles(self):
        angles_info = self.diagram.angles
        reference_point = angles_info[0].points[1]
        adjusted_points = [
            self.adjust_points(angle.points, reference_point)[1]
            for angle in angles_info
        ]
        if len(set(adjusted_points)) != 1:
            raise ValueError("Angles must share a common central point")
        return self

    def adjust_points(self, points, reference_point=None):
        if reference_point is None:
            reference_point = self.reference_point
        if points[1] != reference_point:
            if points[0] == reference_point:
                return [points[1], points[0], points[2]]
            else:
                return [points[2], points[1], points[0]]
        return points

    @cached_property
    def reference_point(self):
        reference_point = self.diagram.angles[0].points[1]
        points = [
            self.adjust_points(angle.points, reference_point)[1]
            for angle in self.diagram.angles
        ]
        if len(set(points)) != 1:
            raise ValueError("Angles must share a common central point")
        return reference_point

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        # Get all unique letters in the points
        unique_letters = set()
        for angle in self.diagram.angles:
            unique_letters.update(angle.points)
        unique_letters.discard(self.reference_point)

        unique_letters = list(unique_letters)
        point_combinations = []
        for i in range(len(unique_letters)):
            for j in range(i + 1, len(unique_letters)):
                point_combinations.append(
                    [
                        min(unique_letters[i], unique_letters[j]),
                        self.reference_point,
                        max(unique_letters[i], unique_letters[j]),
                    ]
                )

        total = sum(angle.get_numeric_measure() for angle in self.diagram.angles)
        expected_total_points = (
            len(self.diagram.angles) if total == 360 else len(self.diagram.angles) + 1
        )

        # Check that the pipeline_context has enough points
        if len(unique_letters) < expected_total_points:
            raise ValueError(
                f"Not enough unique points to form {len(self.diagram.angles)} angles. "
                f"Need at least {len(self.diagram.angles) + 1} points, but only {len(unique_letters)} are available."
            )


if __name__ == "__main__":
    AngleDiagram.generate_assistant_function_schema("mcq4")
