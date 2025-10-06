import random
from typing import TYPE_CHECKING, List, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    Point,
    Polygon,
    PolygonDilation,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
    StimulusDescriptionProtocol,
)
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class PolygonDilationStimulus(StimulusDescription, StimulusDescriptionProtocol):
    """
    Stimulus description for generating polygon dilation problems on a coordinate plane.
    Supports configurable complexity levels and polygon types.
    """

    complexity_level: Literal["BASIC", "ADVANCED"] = Field(
        ..., description="Complexity level determining center placement strategy"
    )

    polygon_type: Literal["triangle", "quadrilateral", "pentagon"] = Field(
        "triangle", description="Type of polygon to use for dilation"
    )

    scale_factor: float = Field(
        ...,
        gt=0,
        description="Scale factor for dilation transformation",
    )

    preimage_quadrant: Optional[Literal["I", "II", "III", "IV", "mixed"]] = Field(
        "I",
        description="Quadrant placement preference for preimage polygon",
    )

    show_center: bool = Field(
        True, description="Whether to display the center point on the graph"
    )

    preimage_color: str = Field("blue", description="Color for the preimage polygon")

    image_color: str = Field("red", description="Color for the image polygon")

    center_color: str = Field("black", description="Color for the center point")

    @field_validator("scale_factor")
    def validate_scale_factor(cls, v):
        """Ensure scale factor creates a meaningful transformation."""
        if abs(v - 1.0) < 1e-6:
            raise ValueError("Scale factor must create a visible transformation")
        return v

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """Pipeline validation for generated content."""
        if pipeline_context is None:
            raise ValueError("Validation context is None")
        super().pipeline_validate(pipeline_context)

    def generate_polygon_dilation(self) -> PolygonDilation:
        """
        Generate a PolygonDilation object based on the stimulus parameters.
        Includes bounds checking to ensure all points stay within reasonable limits.

        Returns:
            PolygonDilation: Complete dilation specification ready for drawing
        """
        # Generate a valid dilation configuration with bounds checking
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                # Generate center based on complexity level
                if self.complexity_level == "BASIC":
                    center = Point(x=0, y=0, label="O")
                else:  # ADVANCED
                    # Choose center strategically based on scale factor and polygon size
                    center = self._generate_smart_center()

                # Generate preimage polygon
                preimage = self._generate_preimage_polygon()

                # Calculate image polygon using dilation formula
                image = self._calculate_image_polygon(preimage, center)

                # Validate that all points are within bounds
                all_points = preimage.points + image.points + [center]
                if all(self._is_point_in_bounds(point) for point in all_points):
                    return PolygonDilation(
                        preimage=preimage,
                        image=image,
                        scale_factor=self.scale_factor,
                        center_of_dilation=center,
                        show_center=self.show_center,
                        preimage_color=self.preimage_color,
                        image_color=self.image_color,
                        center_color=self.center_color,
                    )
            except Exception:
                continue

        # If we can't generate a valid dilation, fall back to origin center
        center = Point(x=0, y=0, label="O")
        preimage = self._generate_compact_preimage_polygon()
        image = self._calculate_image_polygon(preimage, center)

        return PolygonDilation(
            preimage=preimage,
            image=image,
            scale_factor=self.scale_factor,
            center_of_dilation=center,
            show_center=self.show_center,
            preimage_color=self.preimage_color,
            image_color=self.image_color,
            center_color=self.center_color,
        )

    def _is_point_in_bounds(self, point: Point) -> bool:
        """Check if a point is within reasonable coordinate bounds."""
        return -6 <= point.x <= 6 and -6 <= point.y <= 6

    def _generate_smart_center(self) -> Point:
        """Generate a center point that works well with the scale factor."""
        # For ADVANCED complexity, always try to place center away from origin
        # but be more strategic about ensuring integer coordinates after dilation

        # For fractional scale factors, we need to be more careful about center placement
        # to ensure integer results
        if self.scale_factor == 0.5:
            # For scale factor 0.5, use centers that work well with even distances
            center_options = [
                (0, 2),  # Even y-distance from most points
                (2, 0),  # Even x-distance from most points
                (0, -2),  # Even y-distance from most points
                (-2, 0),  # Even x-distance from most points
            ]
        elif self.scale_factor >= 2.0:
            # For very large scale factors, use centers very close to origin but not at origin
            center_options = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]
        elif self.scale_factor >= 1.5:
            center_options = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, 1),
            ]
        else:  # Other scale factors < 1.5 (reductions)
            # For reductions, we can use centers further from origin since the result will be smaller
            center_options = [
                (-2, -2),
                (-2, 2),
                (2, -2),
                (2, 2),
                (-1, -1),
                (1, 1),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]

        # Choose a center option
        center_x, center_y = random.choice(center_options)
        return Point(x=center_x, y=center_y, label="P")

    def _generate_compact_preimage_polygon(self) -> Polygon:
        """Generate a compact polygon that works well with any scale factor."""
        if self.polygon_type == "triangle":
            points = [
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=1, y=2, label="C"),
            ]
            return Polygon(points=points, label="ABC")
        elif self.polygon_type == "quadrilateral":
            points = [
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
                Point(x=1, y=2, label="D"),
            ]
            return Polygon(points=points, label="ABCD")
        else:  # pentagon
            points = [
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
                Point(x=1, y=2, label="D"),
                Point(x=0, y=1, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _generate_preimage_polygon(self) -> Polygon:
        """Generate preimage polygon based on type and quadrant with scale factor consideration."""
        # Be much more conservative with polygon sizes
        if self.scale_factor >= 3.0:
            return self._generate_tiny_polygon()
        elif self.scale_factor >= 2.0:
            return self._generate_tiny_polygon()  # Use tiny for scale >= 2.0
        elif self.scale_factor >= 1.5:
            return self._generate_small_polygon()
        elif self.scale_factor == 0.5:
            return self._generate_half_scale_polygon()  # Special case for 0.5 scale
        elif self.scale_factor >= 0.5:
            return self._generate_medium_polygon()
        else:  # scale_factor < 0.5 (very small reductions)
            return self._generate_large_polygon()

    def _generate_half_scale_polygon(self) -> Polygon:
        """Generate polygons specifically designed for scale factor 0.5 to ensure integer coordinates."""
        if self.polygon_type == "triangle":
            if self.preimage_quadrant == "I":
                # Points chosen to work well with centers like (0,2), (2,0), etc.
                points = [
                    Point(x=2, y=2, label="A"),
                    Point(x=4, y=2, label="B"),
                    Point(x=2, y=4, label="C"),
                ]
            elif self.preimage_quadrant == "II":
                points = [
                    Point(x=-4, y=2, label="A"),
                    Point(x=-2, y=2, label="B"),
                    Point(x=-2, y=4, label="C"),
                ]
            elif self.preimage_quadrant == "III":
                points = [
                    Point(x=-4, y=-4, label="A"),
                    Point(x=-2, y=-4, label="B"),
                    Point(x=-2, y=-2, label="C"),
                ]
            elif self.preimage_quadrant == "IV":
                points = [
                    Point(x=2, y=-4, label="A"),
                    Point(x=4, y=-4, label="B"),
                    Point(x=2, y=-2, label="C"),
                ]
            else:  # mixed
                points = [
                    Point(x=-2, y=2, label="A"),
                    Point(x=2, y=2, label="B"),
                    Point(x=0, y=-2, label="C"),
                ]
            return Polygon(points=points, label="ABC")

        elif self.polygon_type == "quadrilateral":
            if self.preimage_quadrant == "I":
                # Even coordinates that work well with 0.5 scale factor
                points = [
                    Point(x=2, y=2, label="A"),
                    Point(x=4, y=2, label="B"),
                    Point(x=4, y=4, label="C"),
                    Point(x=2, y=4, label="D"),
                ]
            else:  # Simplified for other quadrants
                points = [
                    Point(x=-2, y=-2, label="A"),
                    Point(x=2, y=-2, label="B"),
                    Point(x=2, y=2, label="C"),
                    Point(x=-2, y=2, label="D"),
                ]
            return Polygon(points=points, label="ABCD")

        else:  # pentagon
            points = [
                Point(x=2, y=2, label="A"),
                Point(x=4, y=2, label="B"),
                Point(x=4, y=4, label="C"),
                Point(x=2, y=4, label="D"),
                Point(x=0, y=2, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _generate_tiny_polygon(self) -> Polygon:
        """Generate tiny polygons for large scale factors (>= 2.0)."""
        if self.polygon_type == "triangle":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=1, y=1, label="A"),
                    Point(x=2, y=1, label="B"),
                    Point(x=1, y=2, label="C"),
                ]
            elif self.preimage_quadrant == "II":
                points = [
                    Point(x=-2, y=1, label="A"),
                    Point(x=-1, y=1, label="B"),
                    Point(x=-1, y=2, label="C"),
                ]
            elif self.preimage_quadrant == "III":
                points = [
                    Point(x=-2, y=-2, label="A"),
                    Point(x=-1, y=-2, label="B"),
                    Point(x=-1, y=-1, label="C"),
                ]
            elif self.preimage_quadrant == "IV":
                points = [
                    Point(x=1, y=-2, label="A"),
                    Point(x=2, y=-2, label="B"),
                    Point(x=1, y=-1, label="C"),
                ]
            else:  # mixed
                points = [
                    Point(x=-1, y=1, label="A"),
                    Point(x=1, y=1, label="B"),
                    Point(x=0, y=-1, label="C"),
                ]
            return Polygon(points=points, label="ABC")

        elif self.polygon_type == "quadrilateral":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=1, y=1, label="A"),
                    Point(x=2, y=1, label="B"),
                    Point(x=2, y=2, label="C"),
                    Point(x=1, y=2, label="D"),
                ]
            else:  # Simplified for other quadrants
                points = [
                    Point(x=-1, y=-1, label="A"),
                    Point(x=1, y=-1, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=-1, y=1, label="D"),
                ]
            return Polygon(points=points, label="ABCD")

        else:  # pentagon - very small for large scale factors
            points = [
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
                Point(x=1, y=2, label="D"),
                Point(x=0, y=1, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _generate_small_polygon(self) -> Polygon:
        """Generate small polygons for moderate scale factors (1.5-2.0)."""
        if self.polygon_type == "triangle":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=1, y=1, label="A"),
                    Point(x=2, y=1, label="B"),
                    Point(x=1, y=2, label="C"),
                ]
            elif self.preimage_quadrant == "II":
                points = [
                    Point(x=-2, y=1, label="A"),
                    Point(x=-1, y=1, label="B"),
                    Point(x=-1, y=2, label="C"),
                ]
            elif self.preimage_quadrant == "III":
                points = [
                    Point(x=-2, y=-2, label="A"),
                    Point(x=-1, y=-2, label="B"),
                    Point(x=-1, y=-1, label="C"),
                ]
            elif self.preimage_quadrant == "IV":
                points = [
                    Point(x=1, y=-2, label="A"),
                    Point(x=2, y=-2, label="B"),
                    Point(x=1, y=-1, label="C"),
                ]
            else:  # mixed
                points = [
                    Point(x=-1, y=1, label="A"),
                    Point(x=1, y=1, label="B"),
                    Point(x=0, y=-1, label="C"),
                ]
            return Polygon(points=points, label="ABC")

        elif self.polygon_type == "quadrilateral":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=1, y=1, label="A"),
                    Point(x=2, y=1, label="B"),
                    Point(x=2, y=2, label="C"),
                    Point(x=1, y=2, label="D"),
                ]
            else:  # Simplified for other quadrants
                points = [
                    Point(x=-1, y=-1, label="A"),
                    Point(x=1, y=-1, label="B"),
                    Point(x=1, y=1, label="C"),
                    Point(x=-1, y=1, label="D"),
                ]
            return Polygon(points=points, label="ABCD")

        else:  # pentagon - small for moderate scale factors
            points = [
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
                Point(x=1, y=2, label="D"),
                Point(x=0, y=1, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _generate_medium_polygon(self) -> Polygon:
        """Generate medium-sized polygons for moderate scale factors."""
        if self.polygon_type == "triangle":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=1, y=1, label="A"),
                    Point(x=3, y=1, label="B"),
                    Point(x=2, y=3, label="C"),
                ]
            elif self.preimage_quadrant == "II":
                points = [
                    Point(x=-3, y=1, label="A"),
                    Point(x=-1, y=1, label="B"),
                    Point(x=-2, y=3, label="C"),
                ]
            elif self.preimage_quadrant == "III":
                points = [
                    Point(x=-3, y=-3, label="A"),
                    Point(x=-1, y=-3, label="B"),
                    Point(x=-2, y=-1, label="C"),
                ]
            elif self.preimage_quadrant == "IV":
                points = [
                    Point(x=1, y=-3, label="A"),
                    Point(x=3, y=-3, label="B"),
                    Point(x=2, y=-1, label="C"),
                ]
            else:  # mixed
                points = [
                    Point(x=-1, y=1, label="A"),
                    Point(x=2, y=1, label="B"),
                    Point(x=1, y=-2, label="C"),
                ]
            return Polygon(points=points, label="ABC")

        elif self.polygon_type == "quadrilateral":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=1, y=1, label="A"),
                    Point(x=3, y=1, label="B"),
                    Point(x=3, y=3, label="C"),
                    Point(x=1, y=3, label="D"),
                ]
            else:  # Simplified for other quadrants
                points = [
                    Point(x=-2, y=-2, label="A"),
                    Point(x=2, y=-2, label="B"),
                    Point(x=2, y=2, label="C"),
                    Point(x=-2, y=2, label="D"),
                ]
            return Polygon(points=points, label="ABCD")

        else:  # pentagon - make this smaller to prevent out-of-bounds
            points = [
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
                Point(x=1, y=3, label="D"),
                Point(x=0, y=2, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _generate_large_polygon(self) -> Polygon:
        """Generate larger polygons for small scale factors (reductions 0.5-1.5)."""
        if self.polygon_type == "triangle":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=2, y=2, label="A"),
                    Point(x=4, y=2, label="B"),
                    Point(x=3, y=4, label="C"),
                ]
            elif self.preimage_quadrant == "II":
                points = [
                    Point(x=-4, y=2, label="A"),
                    Point(x=-2, y=2, label="B"),
                    Point(x=-3, y=4, label="C"),
                ]
            elif self.preimage_quadrant == "III":
                points = [
                    Point(x=-4, y=-4, label="A"),
                    Point(x=-2, y=-4, label="B"),
                    Point(x=-3, y=-2, label="C"),
                ]
            elif self.preimage_quadrant == "IV":
                points = [
                    Point(x=2, y=-4, label="A"),
                    Point(x=4, y=-4, label="B"),
                    Point(x=3, y=-2, label="C"),
                ]
            else:  # mixed
                points = [
                    Point(x=-2, y=2, label="A"),
                    Point(x=3, y=2, label="B"),
                    Point(x=1, y=-3, label="C"),
                ]
            return Polygon(points=points, label="ABC")

        elif self.polygon_type == "quadrilateral":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=2, y=2, label="A"),
                    Point(x=4, y=2, label="B"),
                    Point(x=4, y=4, label="C"),
                    Point(x=2, y=4, label="D"),
                ]
            else:  # Simplified for other quadrants
                points = [
                    Point(x=-3, y=-3, label="A"),
                    Point(x=3, y=-3, label="B"),
                    Point(x=3, y=3, label="C"),
                    Point(x=-3, y=3, label="D"),
                ]
            return Polygon(points=points, label="ABCD")

        else:  # pentagon - make smaller to prevent out-of-bounds
            points = [
                Point(x=2, y=2, label="A"),
                Point(x=3, y=2, label="B"),
                Point(x=3, y=3, label="C"),
                Point(x=2, y=4, label="D"),
                Point(x=1, y=3, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _generate_very_large_polygon(self) -> Polygon:
        """Generate very large polygons for very small scale factors (< 0.5)."""
        if self.polygon_type == "triangle":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=2, y=2, label="A"),
                    Point(x=6, y=2, label="B"),
                    Point(x=4, y=6, label="C"),
                ]
            elif self.preimage_quadrant == "II":
                points = [
                    Point(x=-6, y=2, label="A"),
                    Point(x=-2, y=2, label="B"),
                    Point(x=-4, y=6, label="C"),
                ]
            elif self.preimage_quadrant == "III":
                points = [
                    Point(x=-6, y=-6, label="A"),
                    Point(x=-2, y=-6, label="B"),
                    Point(x=-4, y=-2, label="C"),
                ]
            elif self.preimage_quadrant == "IV":
                points = [
                    Point(x=2, y=-6, label="A"),
                    Point(x=6, y=-6, label="B"),
                    Point(x=4, y=-2, label="C"),
                ]
            else:  # mixed
                points = [
                    Point(x=-2, y=2, label="A"),
                    Point(x=4, y=2, label="B"),
                    Point(x=2, y=-4, label="C"),
                ]
            return Polygon(points=points, label="ABC")

        elif self.polygon_type == "quadrilateral":
            if self.preimage_quadrant == "I":
                points = [
                    Point(x=2, y=2, label="A"),
                    Point(x=6, y=2, label="B"),
                    Point(x=6, y=6, label="C"),
                    Point(x=2, y=6, label="D"),
                ]
            else:  # Simplified for other quadrants
                points = [
                    Point(x=-4, y=-4, label="A"),
                    Point(x=4, y=-4, label="B"),
                    Point(x=4, y=4, label="C"),
                    Point(x=-4, y=4, label="D"),
                ]
            return Polygon(points=points, label="ABCD")

        else:  # pentagon
            points = [
                Point(x=2, y=2, label="A"),
                Point(x=4, y=2, label="B"),
                Point(x=6, y=4, label="C"),
                Point(x=4, y=6, label="D"),
                Point(x=2, y=4, label="E"),
            ]
            return Polygon(points=points, label="ABCDE")

    def _calculate_image_polygon(self, preimage: Polygon, center: Point) -> Polygon:
        """Calculate image polygon using dilation transformation with bounds checking."""
        image_points = []

        for i, pre_point in enumerate(preimage.points):
            # Apply dilation transformation formula
            new_x = center.x + self.scale_factor * (pre_point.x - center.x)
            new_y = center.y + self.scale_factor * (pre_point.y - center.y)

            # Only round if the result is very close to an integer (within tolerance)
            # This preserves the mathematical relationship while ensuring integer coordinates when possible
            tolerance = 1e-6
            if abs(new_x - round(new_x)) < tolerance:
                new_x = round(new_x)
            if abs(new_y - round(new_y)) < tolerance:
                new_y = round(new_y)

            # Strict bounds checking - if any point goes out of bounds, raise an error
            if not (-6 <= new_x <= 6 and -6 <= new_y <= 6):
                raise ValueError(f"Point ({new_x}, {new_y}) is out of bounds")

            image_points.append(Point(x=new_x, y=new_y, label=pre_point.label))

        return Polygon(points=image_points, label=preimage.label + "'")


class PolygonDilationStimulusList(StimulusDescriptionList[PolygonDilationStimulus]):
    """
    List wrapper for polygon dilation stimuli.
    """

    root: List[PolygonDilationStimulus] = Field(
        ...,
        description="A list of polygon dilation stimuli",
        min_length=1,
        max_length=1,
    )


# Helper functions for common dilation scenarios
def create_basic_dilation_stimulus(
    polygon_type: Literal["triangle", "quadrilateral", "pentagon"] = "triangle",
    scale_factor: float = 2.0,
    quadrant: Literal["I", "II", "III", "IV", "mixed"] = "I",
) -> PolygonDilationStimulus:
    """
    Create a basic complexity dilation stimulus.

    Args:
        polygon_type: Type of polygon to dilate
        scale_factor: Scale factor for dilation
        quadrant: Quadrant placement for preimage

    Returns:
        PolygonDilationStimulus: Ready-to-use stimulus
    """
    return PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type=polygon_type,
        scale_factor=scale_factor,
        preimage_quadrant=quadrant,
        show_center=True,
    )


def create_advanced_dilation_stimulus(
    polygon_type: Literal["triangle", "quadrilateral", "pentagon"] = "triangle",
    scale_factor: float = 2.0,
    quadrant: Literal["I", "II", "III", "IV", "mixed"] = "I",
) -> PolygonDilationStimulus:
    """
    Create an advanced complexity dilation stimulus.

    Args:
        polygon_type: Type of polygon to dilate
        scale_factor: Scale factor for dilation
        quadrant: Quadrant placement for preimage

    Returns:
        PolygonDilationStimulus: Ready-to-use stimulus
    """
    return PolygonDilationStimulus(
        complexity_level="ADVANCED",
        polygon_type=polygon_type,
        scale_factor=scale_factor,
        preimage_quadrant=quadrant,
        show_center=True,
    )
