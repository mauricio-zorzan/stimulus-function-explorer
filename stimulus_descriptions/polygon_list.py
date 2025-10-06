from typing import List, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field, model_validator


class Point(BaseModel):
    label: str = Field(
        ...,
        pattern=r"^[A-Z]('|)$",
        description="A distinct, single-letter label for the point or single letter with an apastrophe showcasing a transformed point, ensuring no duplication across both sets.",
    )
    x: float = Field(
        ...,
        description="The x-coordinate of the point, an integer value ranging from -10 to 10.",
        ge=-10,
        le=10,
    )
    y: float = Field(
        ...,
        description="The y-coordinate of the point, an integer value ranging from -10 to 10.",
        ge=-10,
        le=10,
    )


class Polygon(BaseModel):
    label: str = Field(default="No Label", description="The label for the polygon.")
    points: List[Point] = Field(
        ...,
        description="A list of 3 to 6 points forming a polygon.",
        min_length=3,
        max_length=6,
    )
    rotation_angle: Optional[Literal[90, 180, 270]] = Field(
        default=None,
        description="Optional rotation angle in degrees. Must be 90, 180, or 270 degrees.",
    )
    rotation_direction: Optional[Literal["clockwise", "counterclockwise"]] = Field(
        default=None,
        description="Optional rotation direction. Required if rotation_angle is specified.",
    )
    rotation_center: Optional[Point] = Field(
        default=None,
        description="Optional center point for rotation. Defaults to origin (0, 0) if not specified.",
    )

    @model_validator(mode="after")
    def validate_rotation_params(self):
        """Ensure rotation_direction is provided when rotation_angle is specified."""
        if self.rotation_angle is not None and self.rotation_direction is None:
            raise ValueError(
                "rotation_direction must be specified when rotation_angle is provided"
            )
        if self.rotation_angle is None and self.rotation_direction is not None:
            raise ValueError(
                "rotation_angle must be specified when rotation_direction is provided"
            )
        return self


class PolygonList(StimulusDescriptionList[Polygon]):
    root: list[Polygon] = Field(
        ...,
        description="A list of polygons that can represent transformations such as rotations, reflections, translations, and dilations.",
    )


class PolygonDilation(StimulusDescription):
    """
    Stimulus description for polygon dilations on a coordinate plane.
    Shows a preimage and its dilated image for identifying scale factors and centers of dilation.
    Designed for questions about geometric transformations on coordinate planes.
    """

    preimage: Polygon = Field(
        ..., description="The original polygon (preimage) before dilation."
    )

    image: Polygon = Field(
        ..., description="The dilated polygon (image) after dilation."
    )

    scale_factor: float = Field(
        ...,
        gt=0,
        description="The scale factor used for the dilation. Must be positive and not equal to 1.",
    )

    center_of_dilation: Point = Field(
        ..., description="The center point around which the dilation is performed."
    )

    show_center: bool = Field(
        default=True,
        description="Whether to mark and label the center of dilation on the graph.",
    )

    preimage_color: str = Field(
        default="blue", description="Color for the preimage polygon and its vertices."
    )

    image_color: str = Field(
        default="red", description="Color for the image polygon and its vertices."
    )

    center_color: str = Field(
        default="black", description="Color for the center of dilation point."
    )

    @model_validator(mode="after")
    def validate_no_center_overlap(self):
        """Validate that the center of dilation does not coincide with any polygon points."""
        center_x = self.center_of_dilation.x
        center_y = self.center_of_dilation.y

        tolerance = 1e-6  # Small tolerance for floating point comparisons

        # Check preimage points
        for point in self.preimage.points:
            if (
                abs(point.x - center_x) < tolerance
                and abs(point.y - center_y) < tolerance
            ):
                raise ValueError(
                    f"Center of dilation ({center_x}, {center_y}) cannot coincide with "
                    f"preimage point {point.label}({point.x}, {point.y}). This creates "
                    f"visual ambiguity where the point appears to be missing from the diagram."
                )

        # Check image points
        for point in self.image.points:
            if (
                abs(point.x - center_x) < tolerance
                and abs(point.y - center_y) < tolerance
            ):
                raise ValueError(
                    f"Center of dilation ({center_x}, {center_y}) cannot coincide with "
                    f"image point {point.label}({point.x}, {point.y}). This creates "
                    f"visual ambiguity where the point appears to be missing from the diagram."
                )

        return self

    @model_validator(mode="after")
    def validate_dilation_relationship(self):
        """Validate that the image is correctly related to the preimage by the dilation."""
        # Check that both polygons have the same number of vertices
        if len(self.preimage.points) != len(self.image.points):
            raise ValueError("Preimage and image must have the same number of vertices")

        # Verify the dilation relationship for each corresponding vertex
        center_x = self.center_of_dilation.x
        center_y = self.center_of_dilation.y
        scale = self.scale_factor

        tolerance = 1e-6  # Small tolerance for floating point comparisons

        for pre_point, img_point in zip(self.preimage.points, self.image.points):
            # Calculate expected image coordinates based on dilation formula
            # (x', y') = (cx, cy) + scale * ((x, y) - (cx, cy))
            expected_x = center_x + scale * (pre_point.x - center_x)
            expected_y = center_y + scale * (pre_point.y - center_y)

            # Check if the actual image point matches the expected coordinates
            if (
                abs(img_point.x - expected_x) > tolerance
                or abs(img_point.y - expected_y) > tolerance
            ):
                raise ValueError(
                    f"Image point {img_point.label}({img_point.x}, {img_point.y}) does not match "
                    f"expected dilation of preimage point {pre_point.label}({pre_point.x}, {pre_point.y}) "
                    f"with center ({center_x}, {center_y}) and scale factor {scale}. "
                    f"Expected: ({expected_x:.3f}, {expected_y:.3f})"
                )

        return self
