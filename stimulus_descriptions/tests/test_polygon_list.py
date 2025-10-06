import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    Point,
    Polygon,
    PolygonDilation,
)
from pydantic import ValidationError


class TestPoint:
    """Test cases for the Point model."""

    def test_valid_point(self):
        """Test creating a valid point."""
        point = Point(x=5.5, y=-3.2, label="A")
        assert point.x == 5.5
        assert point.y == -3.2
        assert point.label == "A"

    def test_point_with_apostrophe(self):
        """Test point with apostrophe in label."""
        point = Point(x=1, y=2, label="A'")
        assert point.label == "A'"

    def test_invalid_label_format(self):
        """Test that invalid label formats raise validation errors."""
        with pytest.raises(ValidationError):
            Point(x=1, y=2, label="AB")  # Too many characters

        with pytest.raises(ValidationError):
            Point(x=1, y=2, label="a")  # Lowercase not allowed

        with pytest.raises(ValidationError):
            Point(x=1, y=2, label="1")  # Numbers not allowed


class TestPolygon:
    """Test cases for the Polygon model."""

    def test_valid_polygon(self):
        """Test creating a valid polygon."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
            Point(x=1, y=1, label="C"),
        ]
        polygon = Polygon(points=points, label="Triangle")
        assert len(polygon.points) == 3
        assert polygon.label == "Triangle"

    def test_rotation_with_valid_params(self):
        """Test polygon with valid rotation parameters."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
            Point(x=1, y=1, label="C"),
        ]
        polygon = Polygon(
            points=points,
            label="Triangle",
            rotation_angle=90,
            rotation_direction="clockwise",
            rotation_center=Point(x=0, y=0, label="O"),
        )
        assert polygon.rotation_angle == 90
        assert polygon.rotation_direction == "clockwise"

    def test_rotation_angle_without_direction_raises_error(self):
        """Test that rotation angle without direction raises validation error."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
            Point(x=1, y=1, label="C"),
        ]
        with pytest.raises(
            ValidationError, match="rotation_direction must be specified"
        ):
            Polygon(
                points=points,
                label="Triangle",
                rotation_angle=90,
                # Missing rotation_direction
            )

    def test_rotation_direction_without_angle_raises_error(self):
        """Test that rotation direction without angle raises validation error."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
            Point(x=1, y=1, label="C"),
        ]
        with pytest.raises(ValidationError, match="rotation_angle must be specified"):
            Polygon(
                points=points,
                label="Triangle",
                rotation_direction="clockwise",
                # Missing rotation_angle
            )

    def test_invalid_rotation_angle(self):
        """Test that invalid rotation angles raise validation errors."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
            Point(x=1, y=1, label="C"),
        ]
        with pytest.raises(ValidationError):
            Polygon(
                points=points,
                label="Triangle",
                rotation_angle=45,  # type: ignore  # Invalid: not 90, 180, or 270
                rotation_direction="clockwise",
            )

    def test_too_few_points_raises_error(self):
        """Test that polygons with fewer than 3 points raise validation errors."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
        ]
        with pytest.raises(ValidationError):
            Polygon(points=points, label="Line")  # Only 2 points

    def test_too_many_points_raises_error(self):
        """Test that polygons with more than 6 points raise validation errors."""
        points = [
            Point(x=0, y=0, label="A"),
            Point(x=1, y=0, label="B"),
            Point(x=2, y=0, label="C"),
            Point(x=3, y=0, label="D"),
            Point(x=4, y=0, label="E"),
            Point(x=5, y=0, label="F"),
            Point(x=6, y=0, label="G"),  # 7 points - too many
        ]
        with pytest.raises(ValidationError):
            Polygon(points=points, label="Heptagon")


class TestPolygonDilation:
    """Test cases for the PolygonDilation model."""

    def test_valid_dilation_scale_2(self):
        """Test a valid dilation with scale factor 2."""
        preimage = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="ABC",
        )

        # Correct dilation: each point distance from origin is doubled
        image = Polygon(
            points=[
                Point(x=2, y=2, label="A"),
                Point(x=4, y=2, label="B"),
                Point(x=4, y=4, label="C"),
            ],
            label="A'B'C'",
        )

        center = Point(x=0, y=0, label="O")

        dilation = PolygonDilation(
            preimage=preimage, image=image, scale_factor=2.0, center_of_dilation=center
        )

        assert dilation.scale_factor == 2.0
        assert dilation.show_center is True  # Default value

    def test_valid_dilation_scale_half(self):
        """Test a valid dilation with scale factor 0.5."""
        preimage = Polygon(
            points=[
                Point(x=4, y=2, label="P"),
                Point(x=6, y=2, label="Q"),
                Point(x=6, y=4, label="R"),
            ],
            label="PQR",
        )

        # Scale factor 0.5 from origin
        image = Polygon(
            points=[
                Point(x=2, y=1, label="P"),
                Point(x=3, y=1, label="Q"),
                Point(x=3, y=2, label="R"),
            ],
            label="P'Q'R'",
        )

        center = Point(x=0, y=0, label="O")

        dilation = PolygonDilation(
            preimage=preimage, image=image, scale_factor=0.5, center_of_dilation=center
        )

        assert dilation.scale_factor == 0.5

    def test_valid_dilation_non_origin_center(self):
        """Test a valid dilation with center not at origin."""
        preimage = Polygon(
            points=[
                Point(x=2, y=2, label="A"),
                Point(x=3, y=2, label="B"),
                Point(x=3, y=3, label="C"),
            ],
            label="ABC",
        )

        # Dilation with center at (1, 1) and scale factor 2
        # Point (2,2): (1,1) + 2*((2,2)-(1,1)) = (1,1) + 2*(1,1) = (3,3)
        # Point (3,2): (1,1) + 2*((3,2)-(1,1)) = (1,1) + 2*(2,1) = (5,3)
        # Point (3,3): (1,1) + 2*((3,3)-(1,1)) = (1,1) + 2*(2,2) = (5,5)
        image = Polygon(
            points=[
                Point(x=3, y=3, label="A"),
                Point(x=5, y=3, label="B"),
                Point(x=5, y=5, label="C"),
            ],
            label="A'B'C'",
        )

        center = Point(x=1, y=1, label="C")

        dilation = PolygonDilation(
            preimage=preimage, image=image, scale_factor=2.0, center_of_dilation=center
        )

        assert dilation.center_of_dilation.x == 1
        assert dilation.center_of_dilation.y == 1

    def test_invalid_scale_factor_zero(self):
        """Test that scale factor of 0 raises validation error."""
        preimage = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="ABC",
        )

        image = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="A'B'C'",
        )

        center = Point(x=0, y=0, label="O")

        with pytest.raises(ValidationError):
            PolygonDilation(
                preimage=preimage,
                image=image,
                scale_factor=0.0,  # Invalid: must be positive
                center_of_dilation=center,
            )

    def test_invalid_negative_scale_factor(self):
        """Test that negative scale factor raises validation error."""
        preimage = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="ABC",
        )

        image = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="A'B'C'",
        )

        center = Point(x=0, y=0, label="O")

        with pytest.raises(ValidationError):
            PolygonDilation(
                preimage=preimage,
                image=image,
                scale_factor=-2.0,  # Invalid: must be positive
                center_of_dilation=center,
            )

    def test_mismatched_vertex_count_raises_error(self):
        """Test that preimage and image with different vertex counts raise validation error."""
        preimage = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="Triangle",
        )

        # Image has 4 points instead of 3
        image = Polygon(
            points=[
                Point(x=2, y=2, label="A"),
                Point(x=4, y=2, label="B"),
                Point(x=4, y=4, label="C"),
                Point(x=2, y=4, label="D"),  # Extra point
            ],
            label="Quadrilateral",
        )

        center = Point(x=0, y=0, label="O")

        with pytest.raises(ValidationError, match="same number of vertices"):
            PolygonDilation(
                preimage=preimage,
                image=image,
                scale_factor=2.0,
                center_of_dilation=center,
            )

    def test_incorrect_dilation_relationship_raises_error(self):
        """Test that incorrect dilation relationship raises validation error."""
        preimage = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="ABC",
        )

        # Incorrect image - not a proper dilation
        image = Polygon(
            points=[
                Point(
                    x=3, y=3, label="A"
                ),  # Should be (2, 2) for scale factor 2 from origin
                Point(x=4, y=2, label="B"),  # Correct
                Point(x=4, y=4, label="C"),  # Correct
            ],
            label="A'B'C'",
        )

        center = Point(x=0, y=0, label="O")

        with pytest.raises(ValidationError, match="does not match expected dilation"):
            PolygonDilation(
                preimage=preimage,
                image=image,
                scale_factor=2.0,
                center_of_dilation=center,
            )

    def test_custom_colors(self):
        """Test dilation with custom colors."""
        preimage = Polygon(
            points=[
                Point(x=1, y=1, label="A"),
                Point(x=2, y=1, label="B"),
                Point(x=2, y=2, label="C"),
            ],
            label="ABC",
        )

        image = Polygon(
            points=[
                Point(x=2, y=2, label="A"),
                Point(x=4, y=2, label="B"),
                Point(x=4, y=4, label="C"),
            ],
            label="A'B'C'",
        )

        center = Point(x=0, y=0, label="O")

        dilation = PolygonDilation(
            preimage=preimage,
            image=image,
            scale_factor=2.0,
            center_of_dilation=center,
            preimage_color="green",
            image_color="purple",
            center_color="orange",
            show_center=False,
        )

        assert dilation.preimage_color == "green"
        assert dilation.image_color == "purple"
        assert dilation.center_color == "orange"
        assert dilation.show_center is False
