from enum import Enum
from typing import List, Literal, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, model_validator


class RightPrismType(str, Enum):
    IRREGULAR = "irregular prism"
    OCTAGONAL = "octagonal prism"
    HEXAGONAL = "hexagonal prism"
    PENTAGONAL = "pentagonal prism"
    TRAPEZOIDAL = "trapezoidal prism"
    TRIANGULAR = "triangular prism"
    RECTANGULAR = "rectangular prism"
    CUBE = "cube"


class BaseRightPrism(BaseModel):
    shape: RightPrismType = Field(..., description="The type of right prism")
    label: str = Field(..., description="The label for the prism")
    height: float = Field(
        ..., description="The height of the prism", ge=1, le=40
    )  # Updated to 40

    class Config:
        extra = "allow"


class IrregularPrism(BaseRightPrism):
    shape: Literal[RightPrismType.IRREGULAR] = RightPrismType.IRREGULAR
    base_vertices: List[List[float]] = Field(
        ...,
        description="List of [x, y] coordinates for the irregular base vertices",
        min_length=3,
    )


class OctagonalPrism(BaseRightPrism):
    shape: Literal[RightPrismType.OCTAGONAL] = RightPrismType.OCTAGONAL
    side_length: float = Field(
        ..., description="The side length of the regular octagon base", ge=2, le=8
    )


class HexagonalPrism(BaseRightPrism):
    shape: Literal[RightPrismType.HEXAGONAL] = RightPrismType.HEXAGONAL
    side_length: float = Field(
        ..., description="The side length of the regular hexagon base", ge=2, le=8
    )


class PentagonalPrism(BaseRightPrism):
    shape: Literal[RightPrismType.PENTAGONAL] = RightPrismType.PENTAGONAL
    side_length: float = Field(
        ..., description="The side length of the regular pentagon base", ge=2, le=8
    )


class TrapezoidalPrism(BaseRightPrism):
    shape: Literal[RightPrismType.TRAPEZOIDAL] = RightPrismType.TRAPEZOIDAL
    top_base: float = Field(..., description="The length of the top base", ge=2, le=8)
    bottom_base: float = Field(
        ..., description="The length of the bottom base", ge=2, le=8
    )
    trapezoid_height: float = Field(
        ..., description="The height of the trapezoid base", ge=2, le=8
    )


class TriangularPrism(BaseRightPrism):
    shape: Literal[RightPrismType.TRIANGULAR] = RightPrismType.TRIANGULAR
    side_a: float = Field(
        ..., description="First side length of the triangle base", ge=1, le=40
    )
    side_b: float = Field(
        ..., description="Second side length of the triangle base", ge=1, le=40
    )
    side_c: float = Field(
        ..., description="Third side length of the triangle base", ge=1, le=40
    )


class RectangularPrismRight(BaseRightPrism):
    shape: Literal[RightPrismType.RECTANGULAR] = RightPrismType.RECTANGULAR
    # Either provide width and length, OR base_area
    width: float | None = Field(
        None, description="The width of the rectangular base", ge=2, le=8
    )
    length: float | None = Field(
        None, description="The length of the rectangular base", ge=2, le=8
    )
    base_area: float | None = Field(
        None, description="The area of the rectangular base", ge=4, le=64
    )

    @model_validator(mode="after")
    def validate_rectangular_prism_params(self):
        """Validate that either width+length OR base_area is provided, but not both"""
        has_dimensions = self.width is not None and self.length is not None
        has_base_area = self.base_area is not None

        if has_dimensions and has_base_area:
            raise ValueError(
                "Cannot provide both width/length and base_area. Use either width+length OR base_area."
            )

        if not has_dimensions and not has_base_area:
            raise ValueError("Must provide either width+length OR base_area.")

        return self


class CubeRight(BaseRightPrism):
    shape: Literal[RightPrismType.CUBE] = RightPrismType.CUBE
    # Either provide side_length OR base_area
    side_length: float | None = Field(
        None, description="The side length of the cube", ge=3, le=10
    )
    base_area: float | None = Field(
        None, description="The area of the cube base", ge=9, le=100
    )

    @model_validator(mode="after")
    def validate_cube_params(self):
        """Validate that either side_length OR base_area is provided, but not both"""
        has_side_length = self.side_length is not None
        has_base_area = self.base_area is not None

        if has_side_length and has_base_area:
            raise ValueError(
                "Cannot provide both side_length and base_area. Use either side_length OR base_area."
            )

        if not has_side_length and not has_base_area:
            raise ValueError("Must provide either side_length OR base_area.")

        return self


RightPrismUnion = Union[
    IrregularPrism,
    OctagonalPrism,
    HexagonalPrism,
    PentagonalPrism,
    TrapezoidalPrism,
    TriangularPrism,
    RectangularPrismRight,
    CubeRight,
]


class RightPrismsList(StimulusDescription):
    """A list of right prisms to draw all oriented in the same direction - upright"""

    prisms: List[RightPrismUnion] = Field(
        default_factory=list,
        min_length=1,
        max_length=9,
        description="The list of right prisms to draw.",
    )
    units: str = Field(
        default="units",
        description="The unit of measurement for dimensions (e.g., 'cm', 'm', 'in', 'ft', 'units')",
    )
    show_height: bool = Field(
        default=True, description="Whether to draw the prism height label"
    )
    show_base_area: bool = Field(
        default=True, description="Whether to draw base area labels when appropriate"
    )
