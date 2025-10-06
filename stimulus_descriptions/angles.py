from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field, model_validator


class Angle(BaseModel):
    measure: int = Field(..., gt=0, le=360)
    label: str = Field(..., max_length=10)


class AngleList(StimulusDescriptionList[Angle]):
    root: list[Angle] = Field(..., min_length=3, max_length=4)

    @model_validator(mode="after")
    def validate_angles(self):
        measurements = [angle.measure for angle in self]

        # Check if each angle measurement is valid
        for measure in measurements:
            if not (measure < 80 or measure > 100 or measure == 90):
                raise ValueError(
                    f"Angle measurement {measure} is not valid. Must be < 80, > 100, or exactly 90."
                )

        # Check the differences between any two angle measurements
        for i in range(len(measurements)):
            for j in range(i + 1, len(measurements)):
                if abs(measurements[i] - measurements[j]) < 15:
                    raise ValueError(
                        f"Difference between angle measurements {measurements[i]} and {measurements[j]} is less than 15 degrees."
                    )

        return self


class SingleAngle(StimulusDescription):
    """Stimulus description for drawing a single angle."""

    measure: int = Field(..., gt=0, le=360, description="The angle measure in degrees")


if __name__ == "__main__":
    AngleList.generate_assistant_function_schema("mcq4")
