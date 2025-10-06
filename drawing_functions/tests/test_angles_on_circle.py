import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.angles_on_circle import (
    draw_circle_angle_measurement,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.angle_on_circle import (
    AngleRange,
    CircleAngle,
)


# Basic range tests (0-180 degrees)
@pytest.mark.drawing_functions
def test_circle_angle_basic_15_degrees():
    """Test basic range with 15 degree angle."""
    stimulus = CircleAngle(
        angle_measure=15, start_position=0, range_category=AngleRange.BASIC
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_basic_45_degrees():
    """Test basic range with 45 degree angle."""
    stimulus = CircleAngle(
        angle_measure=45, start_position=0, range_category=AngleRange.BASIC
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_basic_90_degrees():
    """Test basic range with 90 degree angle (quarter circle)."""
    stimulus = CircleAngle(
        angle_measure=90,
        start_position=0,
        range_category=AngleRange.BASIC,
        sector_color="lightblue",
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_basic_135_degrees():
    """Test basic range with 135 degree angle."""
    stimulus = CircleAngle(
        angle_measure=135, start_position=0, range_category=AngleRange.BASIC
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_basic_180_degrees():
    """Test basic range with 180 degree angle (semicircle)."""
    stimulus = CircleAngle(
        angle_measure=180,
        start_position=0,
        range_category=AngleRange.BASIC,
        sector_color="yellow",
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


# Intermediate range tests (180-360 degrees)
@pytest.mark.drawing_functions
def test_circle_angle_intermediate_195_degrees():
    """Test intermediate range with 195 degree angle."""
    stimulus = CircleAngle(
        angle_measure=195, start_position=0, range_category=AngleRange.INTERMEDIATE
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_intermediate_270_degrees():
    """Test intermediate range with 270 degree angle (three quarters)."""
    stimulus = CircleAngle(
        angle_measure=270,
        start_position=0,
        range_category=AngleRange.INTERMEDIATE,
        sector_color="orange",
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_intermediate_315_degrees():
    """Test intermediate range with 315 degree angle."""
    stimulus = CircleAngle(
        angle_measure=315, start_position=0, range_category=AngleRange.INTERMEDIATE
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_intermediate_345_degrees():
    """Test intermediate range with 345 degree angle."""
    stimulus = CircleAngle(
        angle_measure=345,
        start_position=0,
        range_category=AngleRange.INTERMEDIATE,
        sector_color="pink",
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


# Different starting positions
@pytest.mark.drawing_functions
def test_circle_angle_start_position_90():
    """Test angle starting from 90 degree position."""
    stimulus = CircleAngle(
        angle_measure=60, start_position=90, range_category=AngleRange.BASIC
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_start_position_180():
    """Test angle starting from 180 degree position."""
    stimulus = CircleAngle(
        angle_measure=90, start_position=180, range_category=AngleRange.BASIC
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_start_position_270():
    """Test angle starting from 270 degree position."""
    stimulus = CircleAngle(
        angle_measure=75, start_position=270, range_category=AngleRange.BASIC
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


# Different colors and visual options
@pytest.mark.drawing_functions
def test_circle_angle_no_question():
    """Test circle angle without question text."""
    stimulus = CircleAngle(
        angle_measure=120,
        start_position=0,
        range_category=AngleRange.BASIC,
        show_question=False,
        sector_color="purple",
    )
    file_path = draw_circle_angle_measurement(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_different_colors():
    """Test various sector colors."""
    colors = ["red", "blue", "green", "yellow", "purple"]
    for i, color in enumerate(colors):
        stimulus = CircleAngle(
            angle_measure=60 + i * 15,
            start_position=0,
            range_category=AngleRange.BASIC,
            sector_color=color,
        )
        file_path = draw_circle_angle_measurement(stimulus)
        assert os.path.exists(file_path)


# Edge cases and multiples of 15
@pytest.mark.drawing_functions
def test_circle_angle_all_15_degree_multiples_basic():
    """Test all valid 15-degree multiples in basic range."""
    for angle in range(15, 181, 15):
        stimulus = CircleAngle(
            angle_measure=angle, start_position=0, range_category=AngleRange.BASIC
        )
        file_path = draw_circle_angle_measurement(stimulus)
        assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_circle_angle_all_15_degree_multiples_intermediate():
    """Test all valid 15-degree multiples in intermediate range."""
    for angle in range(195, 360, 15):
        stimulus = CircleAngle(
            angle_measure=angle,
            start_position=0,
            range_category=AngleRange.INTERMEDIATE,
        )
        file_path = draw_circle_angle_measurement(stimulus)
        assert os.path.exists(file_path)


# Cross-quadrant angles
@pytest.mark.drawing_functions
def test_circle_angle_cross_quadrants():
    """Test angles that cross multiple quadrants."""
    test_cases = [
        (150, 45),  # Crosses from Q1 to Q2
        (75, 315),  # Crosses from Q4 to Q1
        (225, 180),  # Crosses from Q3 to Q2
        (300, 90),  # Crosses from Q4 to Q1
    ]

    for angle, start_pos in test_cases:
        range_cat = AngleRange.BASIC if angle <= 180 else AngleRange.INTERMEDIATE
        stimulus = CircleAngle(
            angle_measure=angle, start_position=start_pos, range_category=range_cat
        )
        file_path = draw_circle_angle_measurement(stimulus)
        assert os.path.exists(file_path)


# Validation tests
def test_circle_angle_validation_not_multiple_of_15():
    """Test validation rejection for angles not multiple of 15."""
    with pytest.raises(
        ValueError, match="Angle measure must be a multiple of 15 degrees"
    ):
        CircleAngle(angle_measure=22, range_category=AngleRange.BASIC)


def test_circle_angle_validation_basic_range_exceeded():
    """Test validation rejection for basic range with angle > 180."""
    # Create the stimulus object (this should succeed now)
    stimulus = CircleAngle(angle_measure=195, range_category=AngleRange.BASIC)

    # The error should be raised when trying to draw it
    with pytest.raises(
        ValueError, match="Basic range requires angles between 1 and 180 degrees"
    ):
        draw_circle_angle_measurement(stimulus)


def test_circle_angle_validation_intermediate_range_too_small():
    """Test validation rejection for intermediate range with angle <= 180."""
    # Create the stimulus object (this should succeed now)
    stimulus = CircleAngle(angle_measure=180, range_category=AngleRange.INTERMEDIATE)

    # The error should be raised when trying to draw it
    with pytest.raises(
        ValueError,
        match="Intermediate range requires angles between 181 and 359 degrees",
    ):
        draw_circle_angle_measurement(stimulus)


def test_circle_angle_validation_angle_out_of_bounds():
    """Test validation rejection for angles outside valid range."""
    with pytest.raises(
        ValueError, match="Angle measure must be between 1 and 359 degrees"
    ):
        CircleAngle(angle_measure=360, range_category=AngleRange.INTERMEDIATE)


def test_circle_angle_validation_invalid_start_position():
    """Test validation rejection for invalid start positions."""
    with pytest.raises(
        ValueError, match="Start position must be between 0 and 359 degrees"
    ):
        CircleAngle(
            angle_measure=90, start_position=400, range_category=AngleRange.BASIC
        )


def test_circle_angle_validation_start_position_not_multiple_15():
    """Test validation rejection for start position not multiple of 15."""
    with pytest.raises(
        ValueError, match="Start position must be a multiple of 15 degrees"
    ):
        CircleAngle(
            angle_measure=90, start_position=22, range_category=AngleRange.BASIC
        )
