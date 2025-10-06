import re
from typing import Annotated

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import AliasChoices, BaseModel, BeforeValidator, Field, model_validator


def parse_measure(v):
    if isinstance(v, str):
        v = re.sub(r"[^\d]", "", v)
        try:
            return int(v)
        except ValueError:
            return v
    return v


def validate_measures(v: list[str | int]):
    # First measure should be convertible to int
    try:
        if isinstance(v[0], str):
            int(re.sub(r"[^\d]", "", v[0]))
    except ValueError:
        raise ValueError("First measure must be a valid integer")

    # Second measure can be any string (label)
    return v


class TrianglePoint(BaseModel):
    label: str = Field(..., description="The label of the point")


class Angle(BaseModel):
    vertex: str = Field(..., description="The vertex of the angle")
    measure: Annotated[str | int, BeforeValidator(parse_measure)] = Field(
        ...,
        description="The measure of the angle in degrees or the letter label for the angle.",
    )


class Ray(BaseModel):
    start_label: str = Field(
        ...,
        description="The starting label of the ray.",
        validation_alias=AliasChoices("start", "start_label"),
    )
    measures: Annotated[list[str | int], BeforeValidator(validate_measures)] = Field(
        ...,
        description="The measures associated with the ray",
        min_length=2,
        max_length=2,
    )


class Triangle(BaseModel):
    points: list[TrianglePoint] = Field(
        ..., description="The points of the triangle", min_length=3, max_length=3
    )
    angles: list[Angle] = Field(
        ..., description="The angles of the triangle.", min_length=3, max_length=3
    )


class TriangleStimulusDescription(StimulusDescription):
    triangle: Triangle = Field(
        ..., description="The triangle with point and angle information."
    )


class RightTriangle(BaseModel):
    points: list[TrianglePoint] = Field(
        ..., description="The points of the triangle", min_length=3, max_length=3
    )
    angles: list[Angle] = Field(
        ...,
        description="The angles of the triangle with one being 90 degrees.",
        min_length=3,
        max_length=3,
    )
    rays: list[Ray] = Field(
        ..., description="The rays in the triangle", min_length=1, max_length=1
    )

    @model_validator(mode="after")
    def check_right_angle(self):
        angles = self.angles
        if not any(angle.measure == 90 for angle in angles):
            raise ValueError("One angle must be 90 degrees")
        return self


class RightTriangleWithRay(StimulusDescription):
    triangle: RightTriangle = Field(
        ..., description="The right triangle with ray information."
    )


if __name__ == "__main__":
    RightTriangleWithRay.generate_assistant_function_schema("mcq4")
