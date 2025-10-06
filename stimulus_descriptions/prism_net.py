from enum import Enum
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class EPrismType(str, Enum):
    RECTANGULAR = "rectangular"
    TRIANGULAR = "triangular"
    PYRAMIDAL = "pyramidal"


class PrismNet(StimulusDescription):
    """A 3D prism net
    Dimensions are configurable based on the specific standard's requirements.
    """

    height: int = Field(
        ...,
        gt=0,
        description="The vertical height of the topmost face of the prism.",
    )
    width: int = Field(
        ...,
        gt=0,
        description="The horizontal width of the topmost face of the prism.",
    )
    length: int = Field(
        ...,
        gt=0,
        description="The vertical length of the central face of the prism.",
    )
    net_type: EPrismType = Field(
        ..., title="Net Type", description="The type of prism net to generate."
    )
    unit_label: str = Field(
        ...,
        title="Unit Label",
        description="The unit label to use for the surface area.",
    )
    label_all_sides: bool = Field(
        default=False,
        title="Label All Sides",
        description="Whether to label all edges of the net with their dimensions. If True, every edge will show its length.",
    )
    blank_net: bool = Field(
        default=False,
        title="Blank Net",
        description="Whether this is a blank net (no dimensions shown). Set based on standard during validation.",
    )

    @property
    def surface_area(self):
        match self.net_type:
            case EPrismType.RECTANGULAR:
                return 2 * (
                    self.height * self.width
                    + self.height * self.length
                    + self.width * self.length
                )
            case EPrismType.TRIANGULAR:
                return (
                    self.height * self.width
                    + self.width * self.length
                    + (self.height**2 + (self.width / 2) ** 2) ** 0.5 * self.length * 2
                )
            case EPrismType.PYRAMIDAL:
                # Surface area for rectangular pyramid: base area + 4 triangular faces
                # For rectangular base: length * width + 2 * triangular faces
                base_area = self.length * self.width
                # Simplified calculation for triangular faces (assuming right pyramid)
                face_area1 = 0.5 * self.length * self.height
                face_area2 = 0.5 * self.width * self.height
                return base_area + 2 * face_area1 + 2 * face_area2

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """Set blank_net based on the standard and validate dimensions for blank net requirements."""
        super().pipeline_validate(pipeline_context)
        self.blank_net = pipeline_context.standard_id == "CCSS.MATH.CONTENT.6.G.A.4+1"

        # For blank net standard, ensure all dimensions differ by at least 3 units
        # Exception: if all dimensions are equal (cube), this is allowed
        if self.blank_net:
            # Check if this is a cube (all dimensions equal)
            is_cube = self.height == self.width == self.length

            if not is_cube:
                min_difference = (
                    3  # Minimum difference required between any two dimensions
                )

                # Check all pairs of dimensions
                dimensions = [
                    (self.height, self.width, "height", "width"),
                    (self.height, self.length, "height", "length"),
                    (self.width, self.length, "width", "length"),
                ]

                for dim1, dim2, name1, name2 in dimensions:
                    difference = abs(dim1 - dim2)
                    if difference < min_difference:
                        raise ValueError(
                            f"For blank net standard, all dimensions must differ by at least {min_difference} units. "
                            f"The {name1} ({dim1}) and {name2} ({dim2}) differ by only {difference} units."
                        )


class RectangularPrismNet(PrismNet):
    net_type: EPrismType = EPrismType.RECTANGULAR


class TriangularPrismNet(PrismNet):
    net_type: EPrismType = EPrismType.TRIANGULAR


class PyramidPrismNet(PrismNet):
    net_type: EPrismType = EPrismType.PYRAMIDAL

    @model_validator(mode="after")
    def validate_pyramid_constraints(self):
        # For pyramidal nets, allow both square and rectangular bases.
        # Support both square and rectangular pyramid bases.
        # Only enforce reasonable height proportions for valid pyramid geometry.
        max_base_dimension = max(self.width, self.length)
        if self.height > max_base_dimension * 3:
            raise ValueError("Pyramid height too large relative to base dimensions")

        return self


class Position(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class CubePrismNet(RectangularPrismNet):
    """A cube net (special case of rectangular prism with equal dimensions)."""

    net_type: EPrismType = EPrismType.RECTANGULAR

    @model_validator(mode="after")
    def validate_cube_dimensions(self):
        """Validate that cube has equal dimensions."""
        if not (self.height == self.width == self.length):
            raise ValueError(
                "Cube must have equal dimensions (height = width = length)"
            )
        return self


class RegularRectangularPrismNet(RectangularPrismNet):
    """A rectangular prism net with potentially different dimensions."""

    net_type: EPrismType = EPrismType.RECTANGULAR


class SquarePyramidPrismNet(PyramidPrismNet):
    """A square pyramid net (equal base dimensions)."""

    net_type: EPrismType = EPrismType.PYRAMIDAL

    @model_validator(mode="after")
    def validate_square_pyramid_dimensions(self):
        """Validate that square pyramid has equal base dimensions."""
        if self.width != self.length:
            raise ValueError(
                "Square pyramid must have equal base dimensions (width = length)"
            )
        return self


class RectangularPyramidPrismNet(PyramidPrismNet):
    """A rectangular pyramid net (potentially different base dimensions)."""

    net_type: EPrismType = EPrismType.PYRAMIDAL


class DualNetsShapeType(str, Enum):
    """Specific shape types for dual nets comparison.

    These are more specific than EPrismType and allow for precise
    shape comparisons in dual nets scenarios.
    """

    CUBE = "cube"
    RECTANGULAR_PRISM = "rectangular_prism"
    TRIANGULAR_PRISM = "triangular_prism"
    SQUARE_PYRAMID = "square_pyramid"
    RECTANGULAR_PYRAMID = "rectangular_pyramid"


class DualPrismNets(StimulusDescription):
    """A pair of nets for 3D shapes - one correct and one incorrect."""

    correct_shape_type: DualNetsShapeType = Field(default=DualNetsShapeType.CUBE)
    incorrect_shape_type: DualNetsShapeType = Field(
        default=DualNetsShapeType.RECTANGULAR_PRISM
    )
    correct_shape_position: Position = Field(default=Position.LEFT)

    @model_validator(mode="after")
    def ensure_distinct_shapes(self):
        if self.incorrect_shape_type == self.correct_shape_type:
            for s in DualNetsShapeType:
                if s != self.correct_shape_type:
                    self.incorrect_shape_type = s
                    break
        return self


class CustomTriangularPrismNet(TriangularPrismNet):
    """Triangular prism net with custom side width."""

    side_w: int = Field(
        ...,
        gt=0,
        description="Custom side width for the triangular prism",
    )
    expected_surface_area: int = Field(
        ...,
        gt=0,
        description="Expected surface area of the triangular prism",
    )
    label_all_sides: bool = Field(
        default=True,  # Override parent's default
        title="Label All Sides",
        description="Whether to label all edges of the net with their dimensions. Defaults to True for custom triangular prism.",
    )

    @model_validator(mode="after")
    def validate_surface_area(self) -> "CustomTriangularPrismNet":
        """Validate that the provided surface area matches calculated value."""
        calculated = (
            self.height * self.width  # Two triangular faces
            + self.width * self.length  # Front rectangular face
            + self.side_w
            * self.length
            * 2  # Two side rectangular faces using provided side_w
        )

        if calculated != self.expected_surface_area:
            raise ValueError(
                f"Expected surface area {self.expected_surface_area} does not match "
                f"calculated area {calculated}. Calculation breakdown:\n"
                f"- Two triangular faces: {self.height * self.width}\n"
                f"- Front rectangular face: {self.width * self.length}\n"
                f"- Two side rectangular faces: {self.side_w * self.length * 2}"
            )
        return self
