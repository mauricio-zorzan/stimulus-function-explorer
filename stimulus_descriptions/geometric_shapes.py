from __future__ import annotations

from enum import Enum
from typing import Literal

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field, field_validator, model_validator


class ValidGeometricShape(str, Enum):
    SQUARE = "square"
    RECTANGLE = "rectangle"
    TRIANGLE = "triangle"
    CIRCLE = "circle"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    HEPTAGON = "heptagon"
    OCTAGON = "octagon"
    RHOMBUS = "rhombus"
    TRAPEZOID = "trapezoid"
    ISOSCELES_TRAPEZOID = "isosceles trapezoid"
    RIGHT_TRAPEZOID = "right trapezoid"
    ISOSCELES_TRIANGLE = "isosceles triangle"
    RIGHT_TRIANGLE = "right triangle"
    SCALENE_TRIANGLE = "scalene triangle"
    REGULAR_TRIANGLE = "regular triangle"
    EQUILATERAL_TRIANGLE = "equilateral triangle"
    REGULAR_QUADRILATERAL = "regular quadrilateral"
    QUADRILATERAL = "quadrilateral"
    REGULAR_PENTAGON = "regular pentagon"
    REGULAR_HEXAGON = "regular hexagon"
    REGULAR_HEPTAGON = "regular heptagon"
    REGULAR_OCTAGON = "regular octagon"
    OBTUSE_TRIANGLE = "obtuse triangle"
    ACUTE_TRIANGLE = "acute triangle"
    PARALLELOGRAM = "parallelogram"
    KITE = "kite"


class GeometricShape(BaseModel):
    shape: ValidGeometricShape = Field(description="The shape that is drawn.")
    color: str = Field(
        default="blue",
        description="The color of the fill and outline of the shape.",
    )
    label: str | None = Field(
        default=None,
        description="The optional label of the geometric shape. Use this to help with selection problems, if selecting, name them as figure 1, figure 2, etc or you can use letters like A, B, C, etc. Defaults to the name of the shape if not provided.",
    )

    @field_validator("shape", mode="before")
    @classmethod
    def normalize_shape_case(cls, v):
        """Convert uppercase shape names to lowercase to match enum values."""
        if isinstance(v, str):
            # Convert uppercase to lowercase to match enum values
            return v.lower()
        return v


class GeometricShapeList(StimulusDescriptionList[GeometricShape]):
    root: list[GeometricShape] = Field(
        default_factory=list,
        min_length=1,
        max_length=9,
        description="The list of geometric shapes to draw.",
    )

    @model_validator(mode="after")
    def set_figure_labels_if_all_default(self):
        """
        If all shapes have no labels (None or empty string),
        set them to 'Figure 1', 'Figure 2', etc.
        If at least one shape has a label, leave all labels unchanged.
        """
        if not self.root:
            return self

        # Check if all labels are default (match the capitalized shape name)
        all_default_labels = True
        for shape in self.root:
            if shape.label is not None and shape.label != "":
                all_default_labels = False
                break

        # If all labels are default, replace with Figure pattern
        if all_default_labels:
            for i, shape in enumerate(self.root, 1):
                shape.label = f"Figure {i}"

        return self


class GeometricShapeListWithRotation(StimulusDescription):
    """
    Stimulus description for geometric shapes with rotation support.
    """

    shapes: list[GeometricShape] = Field(
        default_factory=list,
        min_length=1,
        max_length=9,
        description="The list of geometric shapes to draw.",
    )
    rotate: bool = Field(
        default=False,
        description="Whether to apply random rotation to the shapes. If True, each shape will be rotated by a random angle.",
    )

    @model_validator(mode="after")
    def set_figure_labels_if_all_default(self):
        """
        If all shapes have no labels (None or empty string),
        set them to 'Figure 1', 'Figure 2', etc.
        If at least one shape has a label, leave all labels unchanged.
        """
        if not self.shapes:
            return self

        # Check if all labels are default (match the capitalized shape name)
        all_default_labels = True
        for shape in self.shapes:
            if shape.label is not None and shape.label != "":
                all_default_labels = False
                break

        # If all labels are default, replace with Figure pattern
        if all_default_labels:
            for i, shape in enumerate(self.shapes, 1):
                shape.label = f"Figure {i}"

        return self


class GeometricShapeWithAngle(BaseModel):
    shape: ValidGeometricShape = Field(description="The shape that is drawn.")
    angle_type: Literal["acute", "obtuse", "right"] = Field(
        description="The type of angle to mark: 'acute' (< 90°), 'obtuse' (> 90°), or 'right' (= 90°). The system will automatically find the first vertex with the requested angle type."
    )
    color: str = Field(
        default="blue",
        description="The color of the fill and outline of the shape.",
    )
    label: str | None = Field(
        default=None,
        description="The optional label of the geometric shape. Use this to help with selection problems, if selecting, name them as figure 1, figure 2, etc or you can use letters like A, B, C, etc. Defaults to the name of the shape if not provided.",
    )


class GeometricShapeWithAngleList(StimulusDescriptionList[GeometricShapeWithAngle]):
    root: list[GeometricShapeWithAngle] = Field(
        default_factory=list,
        min_length=1,
        max_length=9,
        description="The list of geometric shapes to draw with angle markings.",
    )

    @model_validator(mode="after")
    def set_figure_labels_if_all_default(self):
        """
        If all shapes have no labels (None or empty string),
        set them to 'Figure 1', 'Figure 2', etc.
        If at least one shape has a label, leave all labels unchanged.
        """
        if not self.root:
            return self

        # Check if all labels are default (match the capitalized shape name)
        all_default_labels = True
        for shape in self.root:
            if shape.label is not None and shape.label != "":
                all_default_labels = False
                break

        # If all labels are default, replace with Figure pattern
        if all_default_labels:
            for i, shape in enumerate(self.root, 1):
                shape.label = f"Figure {i}"

        return self


class ShapeWithRightAngles(StimulusDescription):
    """
    Stimulus description for generating a random shape with a specified number of right angles.
    The shape will have right angle markers to indicate the 90-degree angles.
    Each shape is randomly rotated by 0°, 90°, 180°, or 270° to provide variety.

    Supported right angle counts (interior angles only):
    - 0: Circle, equilateral triangle, pentagon, hexagon, or quadrilateral (no right angles)
    - 1: Right triangle or right angle kite shape
    - 2: Pentagon with 2 right angles at base or right trapezoid
    - 3: File icon shape (triangle + rectangle)
    - 4: Rectangle or square
    """

    num_right_angles: int = Field(
        ge=0,
        le=4,
        description="The number of interior right angles (90-degree angles) the shape should have. Supported values: 0, 1, 2, 3, 4.",
    )


class ParallelQuadrilateral(StimulusDescription):
    """
    Stimulus description for generating a quadrilateral with a specified number of parallel sides.
    
    Supported parallel side counts:
    - 0: Irregular quadrilateral with no parallel sides (random shape each time)
    - 1: Trapezoid with exactly one pair of parallel sides (random dimensions)
    - 2: Parallelogram (square, rectangle, rhombus, or other parallelogram with variation)
    
    The quadrilateral can optionally be rotated by a random angle for additional variety.
    """

    num_parallel_sides: int = Field(
        ge=0,
        le=2,
        description="The number of pairs of parallel sides the quadrilateral should have. Supported values: 0, 1, 2.",
    )
    rotate: bool = Field(
        default=False,
        description="Whether to apply random rotation to the quadrilateral. If True, the shape will be rotated by a random angle between 0-360 degrees.",
    )


if __name__ == "__main__":
    print(GeometricShapeList.model_json_schema(mode="serialization"))
