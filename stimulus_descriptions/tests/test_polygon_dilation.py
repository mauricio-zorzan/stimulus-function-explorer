import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_dilation import (
    PolygonDilationStimulus,
    PolygonDilationStimulusList,
    create_advanced_dilation_stimulus,
    create_basic_dilation_stimulus,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    PolygonDilation,
)
from pydantic import ValidationError


class TestPolygonDilationStimulus:
    """Test cases for the PolygonDilationStimulus class."""

    def test_basic_complexity_triangle_dilation(self):
        """Test BASIC complexity triangle dilation with origin center."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="I",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # Verify center is at origin for BASIC complexity
        assert dilation.center_of_dilation.x == 0
        assert dilation.center_of_dilation.y == 0
        assert dilation.center_of_dilation.label == "O"

        # Verify scale factor
        assert dilation.scale_factor == 2.0

        # Verify triangle has 3 points
        assert len(dilation.preimage.points) == 3
        assert len(dilation.image.points) == 3

        # Verify dilation relationship (origin center makes this simple)
        for pre_point, img_point in zip(
            dilation.preimage.points, dilation.image.points
        ):
            assert img_point.x == pre_point.x * 2.0
            assert img_point.y == pre_point.y * 2.0

    def test_advanced_complexity_quadrilateral_dilation(self):
        """Test ADVANCED complexity quadrilateral dilation with non-origin center."""
        stimulus = PolygonDilationStimulus(
            complexity_level="ADVANCED",
            polygon_type="quadrilateral",
            scale_factor=0.5,
            preimage_quadrant="I",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # Verify center is NOT at origin for ADVANCED complexity
        assert not (
            dilation.center_of_dilation.x == 0 and dilation.center_of_dilation.y == 0
        )
        assert dilation.center_of_dilation.label == "P"

        # Verify scale factor
        assert dilation.scale_factor == 0.5

        # Verify quadrilateral has 4 points
        assert len(dilation.preimage.points) == 4
        assert len(dilation.image.points) == 4

        # Verify all coordinates are integers
        for point in dilation.preimage.points + dilation.image.points:
            assert isinstance(point.x, int) or point.x.is_integer()
            assert isinstance(point.y, int) or point.y.is_integer()

    def test_pentagon_dilation(self):
        """Test pentagon dilation generation."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="pentagon",
            scale_factor=1.5,
            preimage_quadrant="I",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # Verify pentagon has 5 points
        assert len(dilation.preimage.points) == 5
        assert len(dilation.image.points) == 5

        # Verify polygon labels
        assert dilation.preimage.label == "ABCDE"
        assert dilation.image.label == "ABCDE'"

    def test_invalid_scale_factor(self):
        """Test that scale factor of 1 is rejected."""
        with pytest.raises(ValidationError):
            PolygonDilationStimulus(
                complexity_level="BASIC",
                polygon_type="triangle",
                scale_factor=1.0,  # Invalid - no transformation
                preimage_quadrant="I",
                show_center=True,
            )

    def test_negative_scale_factor(self):
        """Test that negative scale factor is rejected."""
        with pytest.raises(ValidationError):
            PolygonDilationStimulus(
                complexity_level="BASIC",
                polygon_type="triangle",
                scale_factor=-2.0,  # Invalid - negative
                preimage_quadrant="I",
                show_center=True,
            )

    def test_quadrant_placement_first_quadrant(self):
        """Test preimage placement in first quadrant."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="I",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # All preimage points should be in first quadrant (positive x, positive y)
        for point in dilation.preimage.points:
            assert point.x > 0
            assert point.y > 0

    def test_quadrant_placement_second_quadrant(self):
        """Test preimage placement in second quadrant."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="II",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # All preimage points should be in second quadrant (negative x, positive y)
        for point in dilation.preimage.points:
            assert point.x < 0
            assert point.y > 0

    def test_quadrant_placement_third_quadrant(self):
        """Test preimage placement in third quadrant."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="III",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # All preimage points should be in third quadrant (negative x, negative y)
        for point in dilation.preimage.points:
            assert point.x < 0
            assert point.y < 0

    def test_quadrant_placement_fourth_quadrant(self):
        """Test preimage placement in fourth quadrant."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="IV",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # All preimage points should be in fourth quadrant (positive x, negative y)
        for point in dilation.preimage.points:
            assert point.x > 0
            assert point.y < 0

    def test_color_customization(self):
        """Test custom color settings."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="I",
            show_center=True,
            preimage_color="green",
            image_color="purple",
            center_color="orange",
        )

        dilation = stimulus.generate_polygon_dilation()

        assert dilation.preimage_color == "green"
        assert dilation.image_color == "purple"
        assert dilation.center_color == "orange"

    def test_center_display_toggle(self):
        """Test center display can be toggled."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="I",
            show_center=False,
        )

        dilation = stimulus.generate_polygon_dilation()

        assert dilation.show_center is False

    def test_coordinate_bounds(self):
        """Test that all coordinates stay within reasonable bounds."""
        stimulus = PolygonDilationStimulus(
            complexity_level="ADVANCED",
            polygon_type="quadrilateral",
            scale_factor=3.0,  # Large scale factor
            preimage_quadrant="I",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # Check all coordinates are within bounds
        all_points = (
            dilation.preimage.points
            + dilation.image.points
            + [dilation.center_of_dilation]
        )

        for point in all_points:
            assert -6 <= point.x <= 6
            assert -6 <= point.y <= 6

    def test_dilation_relationship_validation(self):
        """Test that generated dilation maintains proper mathematical relationship."""
        stimulus = PolygonDilationStimulus(
            complexity_level="ADVANCED",
            polygon_type="triangle",
            scale_factor=1.5,
            preimage_quadrant="I",
            show_center=True,
        )

        dilation = stimulus.generate_polygon_dilation()

        # The PolygonDilation model should validate the relationship
        # If this doesn't raise an exception, the relationship is valid
        assert isinstance(dilation, PolygonDilation)

        # Manual verification of transformation formula
        center = dilation.center_of_dilation
        scale = dilation.scale_factor

        for pre_point, img_point in zip(
            dilation.preimage.points, dilation.image.points
        ):
            expected_x = center.x + scale * (pre_point.x - center.x)
            expected_y = center.y + scale * (pre_point.y - center.y)

            # Check that the image point matches the expected dilation transformation
            # Allow for small floating point differences
            assert abs(img_point.x - expected_x) < 1e-6
            assert abs(img_point.y - expected_y) < 1e-6


class TestPolygonDilationStimulusList:
    """Test cases for the PolygonDilationStimulusList class."""

    def test_stimulus_list_creation(self):
        """Test creating a list of polygon dilation stimuli."""
        stimulus = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="I",
            show_center=True,
        )

        stimulus_list = PolygonDilationStimulusList(root=[stimulus])

        assert len(stimulus_list.root) == 1
        assert isinstance(stimulus_list.root[0], PolygonDilationStimulus)

    def test_empty_list_validation(self):
        """Test that empty list is rejected."""
        with pytest.raises(ValidationError):
            PolygonDilationStimulusList(root=[])

    def test_multiple_stimuli_validation(self):
        """Test that multiple stimuli are rejected (max_length=1)."""
        stimulus1 = PolygonDilationStimulus(
            complexity_level="BASIC",
            polygon_type="triangle",
            scale_factor=2.0,
            preimage_quadrant="I",
            show_center=True,
        )

        stimulus2 = PolygonDilationStimulus(
            complexity_level="ADVANCED",
            polygon_type="quadrilateral",
            scale_factor=0.5,
            preimage_quadrant="I",
            show_center=True,
        )

        with pytest.raises(ValidationError):
            PolygonDilationStimulusList(root=[stimulus1, stimulus2])


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_create_basic_dilation_stimulus(self):
        """Test basic complexity stimulus creation helper."""
        stimulus = create_basic_dilation_stimulus(
            polygon_type="triangle", scale_factor=2.0, quadrant="I"
        )

        assert stimulus.complexity_level == "BASIC"
        assert stimulus.polygon_type == "triangle"
        assert stimulus.scale_factor == 2.0
        assert stimulus.preimage_quadrant == "I"
        assert stimulus.show_center is True

    def test_create_advanced_dilation_stimulus(self):
        """Test advanced complexity stimulus creation helper."""
        stimulus = create_advanced_dilation_stimulus(
            polygon_type="quadrilateral", scale_factor=0.75, quadrant="II"
        )

        assert stimulus.complexity_level == "ADVANCED"
        assert stimulus.polygon_type == "quadrilateral"
        assert stimulus.scale_factor == 0.75
        assert stimulus.preimage_quadrant == "II"
        assert stimulus.show_center is True

    def test_basic_complexity_behavior(self):
        """Test basic complexity behavior."""
        stimulus = create_basic_dilation_stimulus()
        dilation = stimulus.generate_polygon_dilation()

        # Verify integer coordinates
        all_points = (
            dilation.preimage.points
            + dilation.image.points
            + [dilation.center_of_dilation]
        )
        for point in all_points:
            assert isinstance(point.x, (int, float)) and point.x == int(point.x)
            assert isinstance(point.y, (int, float)) and point.y == int(point.y)

        # Verify center is provided
        assert dilation.center_of_dilation is not None

        # Verify meaningful transformation
        assert dilation.scale_factor > 0 and dilation.scale_factor != 1

        # Verify center at origin for BASIC
        assert dilation.center_of_dilation.x == 0
        assert dilation.center_of_dilation.y == 0

    def test_advanced_complexity_behavior(self):
        """Test advanced complexity behavior."""
        stimulus = create_advanced_dilation_stimulus()
        dilation = stimulus.generate_polygon_dilation()

        # Verify integer coordinates
        all_points = (
            dilation.preimage.points
            + dilation.image.points
            + [dilation.center_of_dilation]
        )
        for point in all_points:
            assert isinstance(point.x, (int, float)) and point.x == int(point.x)
            assert isinstance(point.y, (int, float)) and point.y == int(point.y)

        # Verify center is provided
        assert dilation.center_of_dilation is not None

        # Verify meaningful transformation
        assert dilation.scale_factor > 0 and dilation.scale_factor != 1

        # Verify center NOT at origin for ADVANCED
        assert not (
            dilation.center_of_dilation.x == 0 and dilation.center_of_dilation.y == 0
        )

    def test_content_generation_support(self):
        """Test that generated content supports typical use cases."""
        # Use case 1: Identify transformation
        stimulus = create_basic_dilation_stimulus(scale_factor=2.0)
        dilation = stimulus.generate_polygon_dilation()

        # Should have distinct preimage and image for identification
        assert dilation.preimage != dilation.image
        assert len(dilation.preimage.points) == len(dilation.image.points)

        # Use case 2: Determine transformation parameters
        # Should provide center and scale factor information
        assert dilation.center_of_dilation is not None
        assert dilation.scale_factor is not None
        assert dilation.scale_factor != 1.0
