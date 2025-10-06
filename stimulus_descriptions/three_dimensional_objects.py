from typing import List, Literal, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import AliasChoices, BaseModel, Field


class Base3DShape(BaseModel):
    shape: Literal[
        "sphere", "pyramid", "cube", "rectangular prism", "cone", "cylinder"
    ] = Field(
        ...,
        validation_alias=AliasChoices("shape"),
    )
    label: str = Field(..., description="The label for the 3D shape")
    faces: str | None = Field(default=None, description="The faces of the 3D shape")

    class Config:
        extra = "allow"


class Sphere(Base3DShape):
    shape: Literal["sphere"] = "sphere"
    radius: float | None = Field(
        default=None, description="The radius of the sphere", ge=3, le=10
    )


class Pyramid(Base3DShape):
    shape: Literal["pyramid"] = "pyramid"
    side: float | None = Field(
        default=None, description="The side length of the pyramid base", ge=3, le=10
    )
    height: float | None = Field(
        default=None, description="The height of the pyramid", ge=3, le=10
    )


class RectangularPrism(Base3DShape):
    shape: Literal["rectangular prism"] = "rectangular prism"
    height: int | None = Field(
        default=None, description="The height of the rectangular prism", ge=3, le=10
    )
    width: int | None = Field(
        default=None, description="The width of the rectangular prism", ge=3, le=10
    )
    length: int | None = Field(
        default=None, description="The length of the rectangular prism", ge=3, le=10
    )


class Cube(Base3DShape):
    shape: Literal["cube"] = "cube"
    height: float | None = Field(
        default=None, description="The height of the cube", ge=3, le=10
    )
    width: float | None = Field(
        default=None, description="The width of the cube", ge=3, le=10
    )
    length: float | None = Field(
        default=None, description="The length of the cube", ge=3, le=10
    )


class Cone(Base3DShape):
    shape: Literal["cone"] = "cone"
    height: float | None = Field(
        default=None, description="The height of the cone", ge=3, le=10
    )
    radius: float | None = Field(
        default=None, description="The radius of the cone", ge=3, le=10
    )


class Cylinder(Base3DShape):
    shape: Literal["cylinder"] = "cylinder"
    height: float | None = Field(
        default=None, description="The height of the cylinder", ge=3, le=10
    )
    radius: float | None = Field(
        default=None, description="The radius of the cylinder", ge=3, le=10
    )


ShapeUnion = Union[Sphere, Pyramid, RectangularPrism, Cone, Cylinder, Cube]


class ThreeDimensionalObjectsList(StimulusDescription):
    """A list of 3D objects to draw all oriented in the same direction - upright"""

    shapes: List[ShapeUnion] = Field(
        default_factory=list,
        min_length=1,
        max_length=9,
        description="The list of 3D objects to draw.",
    )
    units: str = Field(
        default="units",
        description="The unit of measurement for dimensions (e.g., 'cm', 'm', 'in', 'ft', 'units')",
    )


class CrossSectionQuestion(StimulusDescription):
    """A 3D shape with a cross-section question showing a cutting plane and multiple choice options"""

    shape: ShapeUnion = Field(
        ..., description="The 3D shape to create a cross-section question for"
    )
    correct_cross_section: str = Field(
        ...,
        description="The correct 2D cross-section shape (e.g., 'circle', 'triangle', 'rectangle', 'square')",
    )
    correct_letter: str = Field(
        ...,
        description="The letter (a, b, c, d) indicating the position of the correct answer",
    )


if __name__ == "__main__":
    import pyperclip

    pyperclip.copy(ThreeDimensionalObjectsList.model_json_schema())
