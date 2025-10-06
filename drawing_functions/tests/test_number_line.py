import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.number_lines import (
    create_decimal_comparison_number_line,
    create_dot_plot,
    create_dual_dot_plot,
    create_extended_unit_fraction_number_line,
    create_fixed_step_number_line,
    create_multi_extended_unit_fraction_number_line_with_bar,
    create_multi_extended_unit_fraction_number_line_with_bar_v2,
    create_multi_extended_unit_fraction_number_line_with_dots,
    create_multi_labeled_unit_fraction_number_line,
    create_number_line,
    create_unit_fraction_number_line,
    create_vertical_number_line,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.dual_stats_line import (
    DualStatsLinePlot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.number_line import (
    DecimalComparisonNumberLine,
    ExtendedRange,
    ExtendedUnitFractionDotPoint,
    ExtendedUnitFractionNumberLine,
    FixedStepNumberLine,
    LabeledUnitFractionNumberLine,
    MultiExtendedUnitFractionNumberLine,
    MultiLabeledUnitFractionNumberLine,
    NumberLine,
    Point,
    Range,
    UnitFractionNumberLine,
    UnitFractionPoint,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_line import (
    StatsLinePlot,
)
from pydantic import ValidationError


@pytest.fixture
def sample_number_line():
    return NumberLine(
        range=Range(min=0, max=10),
        points=[
            Point(value=2, label="A"),
            Point(value=5, label="B"),
            Point(value=8, label="C"),
        ],
    )


@pytest.fixture
def sample_decimal_comparison_number_line_0_1():
    """Sample decimal comparison number line with 0.1 increments."""
    return DecimalComparisonNumberLine(
        range=ExtendedRange(min=2.0, max=3.0),
        points=[
            Point(value=2.3, label="A"),
            Point(value=2.7, label="B"),
        ],
    )


@pytest.fixture
def sample_decimal_comparison_number_line_0_01():
    """Sample decimal comparison number line with 0.01 increments."""
    return DecimalComparisonNumberLine(
        range=ExtendedRange(min=2.20, max=2.30),
        points=[
            Point(value=2.23, label="X"),
            Point(value=2.27, label="Y"),
        ],
    )


@pytest.fixture
def sample_decimal_comparison_number_line_no_points():
    """Sample decimal comparison number line with no points."""
    return DecimalComparisonNumberLine(
        range=ExtendedRange(min=1.5, max=2.5),
        points=[],
    )


@pytest.mark.drawing_functions
def test_create_number_line_basic(sample_number_line):
    file_name = create_number_line(sample_number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line_with_temperature_points():
    number_line = NumberLine(
        range=Range(min=-25, max=20),
        points=[
            Point(label="-20°C", value=-20.0),
            Point(label="-2°C", value=-2.0),
            Point(label="0°C", value=0.0),
            Point(label="5°C", value=5.0),
            Point(label="16°C", value=16.0),
        ],
    )
    file_name = create_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line_with_large_values():
    number_line = NumberLine(
        range=Range(min=700, max=1000),
        points=[
            Point(value=762, label="762"),
            Point(value=987, label="987"),
        ],
    )
    file_name = create_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line():
    # Define the test input
    stim_desc = NumberLine(
        range=Range(min=0, max=56),
        points=[
            Point(value=7, label="A"),
            Point(value=14, label="B"),
            Point(value=21, label="C"),
            Point(value=28, label="D"),
        ],
    )

    # Call the function and capture the output
    file_name = create_number_line(stim_desc)
    assert os.path.exists(file_name)


def test_create_number_line_overlapping_labels():
    # Define the test input
    stim_desc = NumberLine(
        range=Range(min=0, max=56),
        points=[
            Point(value=7, label="A"),
            Point(value=8, label="B"),
            Point(value=9, label="C"),
            Point(value=10, label="D"),
        ],
    )

    # Call the function and capture the output
    file_name = create_number_line(stim_desc)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line_with_edge_case_2():
    number_line = NumberLine(
        range=Range(min=0, max=3),
        points=[
            Point(value=0.5, label="A"),
            Point(value=1.3, label="B"),
            Point(value=2.1, label="C"),
            Point(value=2.9, label="D"),
        ],
    )
    file_name = create_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line_with_edge_case():
    number_line = NumberLine(
        range=Range(min=0, max=3),
        points=[
            Point(value=1, label="A"),
            Point(value=2, label="B"),
        ],
    )
    file_name = create_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line_with_large_range():
    number_line = NumberLine(
        range=Range(min=0, max=100),
        points=[
            Point(value=10, label="A"),
            Point(value=50, label="B"),
            Point(value=90, label="C"),
        ],
    )
    file_name = create_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_number_line_with_invalid_point():
    with pytest.raises(ValueError):
        NumberLine(
            range=Range(min=0, max=10),
            points=[
                Point(value=11, label="A"),  # Invalid point
            ],
        )


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_basic_0_1(
    sample_decimal_comparison_number_line_0_1,
):
    """Test basic decimal comparison number line with 0.1 increments."""
    file_name = create_decimal_comparison_number_line(
        sample_decimal_comparison_number_line_0_1
    )
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_basic_0_01(
    sample_decimal_comparison_number_line_0_01,
):
    """Test basic decimal comparison number line with 0.01 increments."""
    file_name = create_decimal_comparison_number_line(
        sample_decimal_comparison_number_line_0_01
    )
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_no_points(
    sample_decimal_comparison_number_line_no_points,
):
    """Test decimal comparison number line with no points."""
    file_name = create_decimal_comparison_number_line(
        sample_decimal_comparison_number_line_no_points
    )
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_0_1_increments_large_values():
    """Test decimal comparison number line with 0.1 increments and larger values."""
    number_line = DecimalComparisonNumberLine(
        range=ExtendedRange(min=15.0, max=16.0),
        points=[
            Point(value=15.2, label="X"),
            Point(value=15.7, label="Y"),
            Point(value=15.9, label="Z"),
        ],
    )
    file_name = create_decimal_comparison_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_0_01_increments_precise():
    """Test decimal comparison number line with 0.01 increments and precise values."""
    number_line = DecimalComparisonNumberLine(
        range=ExtendedRange(min=3.45, max=3.55),
        points=[
            Point(value=3.47, label="A"),
            Point(value=3.49, label="B"),
            Point(value=3.53, label="C"),
        ],
    )
    file_name = create_decimal_comparison_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_with_label_interval():
    """Test decimal comparison number line with custom label intervals."""
    # Test with label_interval=2 (label every 2nd tick)
    number_line = DecimalComparisonNumberLine(
        range=ExtendedRange(min=0.65, max=0.75),
        points=[
            Point(value=0.68, label="A"),
            Point(value=0.72, label="B"),
        ],
        label_interval=2,
    )
    file_name = create_decimal_comparison_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_decimal_comparison_number_line_with_large_label_interval():
    """Test decimal comparison number line with large label interval (only endpoints)."""
    # Test with label_interval=5 (label positions 0, 5, 10 only)
    number_line = DecimalComparisonNumberLine(
        range=ExtendedRange(min=7.5, max=8.5),
        points=[],
        label_interval=5,
    )
    file_name = create_decimal_comparison_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_unit_fraction_number_line_with_fraction_points():
    number_line = UnitFractionNumberLine(
        points=[
            UnitFractionPoint(label="8/9", value=0.8888888888888888),
        ],
        minor_divisions=9,
    )
    file_name = create_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_unit_fraction_number_line_valid():
    number_line = UnitFractionNumberLine(
        points=[
            UnitFractionPoint(value=0.25, label="A"),
            UnitFractionPoint(value=0.5, label="B"),
            UnitFractionPoint(value=0.75, label="C"),
        ],
    )
    file_name = create_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_unit_fraction_number_line_no_points():
    with pytest.raises(ValidationError):
        UnitFractionNumberLine(
            range=Range(min=0, max=1),
            points=[],  # No points provided
        )


@pytest.mark.drawing_functions
def test_create_number_line_with_biggest_range():
    number_line = NumberLine(
        range=Range(min=0, max=1000),
        points=[
            Point(value=100, label="A"),
            Point(value=500, label="B"),
            Point(value=900, label="C"),
        ],
    )
    file_name = create_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line():
    number_line = NumberLine(
        range=Range(min=0, max=10),
        points=[
            Point(value=2, label="A"),
            Point(value=5, label="B"),
            Point(value=8, label="C"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_temperature_points():
    number_line = NumberLine(
        range=Range(min=-25, max=20),
        points=[
            Point(label="-20°C", value=-20.0),
            Point(label="-2°C", value=-2.0),
            Point(label="0°C", value=0.0),
            Point(label="5°C", value=5.0),
            Point(label="16°C", value=16.0),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_large_values():
    number_line = NumberLine(
        range=Range(min=700, max=1000),
        points=[
            Point(value=762, label="762"),
            Point(value=987, label="987"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_multiple_points():
    number_line = NumberLine(
        range=Range(min=0, max=56),
        points=[
            Point(value=7, label="A"),
            Point(value=14, label="B"),
            Point(value=21, label="C"),
            Point(value=28, label="D"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_edge_case_2():
    number_line = NumberLine(
        range=Range(min=0, max=3),
        points=[
            Point(value=0.5, label="A"),
            Point(value=1.3, label="B"),
            Point(value=2.1, label="C"),
            Point(value=2.9, label="D"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_edge_case():
    number_line = NumberLine(
        range=Range(min=0, max=3),
        points=[
            Point(value=1, label="A"),
            Point(value=2, label="B"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_large_range():
    number_line = NumberLine(
        range=Range(min=0, max=100),
        points=[
            Point(value=10, label="A"),
            Point(value=50, label="B"),
            Point(value=90, label="C"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_vertical_number_line_with_biggest_range():
    number_line = NumberLine(
        range=Range(min=0, max=1000),
        points=[
            Point(value=100, label="A"),
            Point(value=500, label="B"),
            Point(value=900, label="C"),
        ],
    )
    file_name = create_vertical_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_with_all_labels():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=1.75),
        minor_divisions=7,
        endpoint_fraction="7/4",
        dot_point=ExtendedUnitFractionDotPoint(label="A", value=0.5),
        show_all_tick_labels=True,
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_with_a_fraction_tick_labeled():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=1),
        minor_divisions=8,
        endpoint_fraction="1",
        dot_point=ExtendedUnitFractionDotPoint(label="A", value=1 / 8),
        labeled_fraction="6/8",
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_basic():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=2.0),
        minor_divisions=8,
        endpoint_fraction="8/4",
        dot_point=ExtendedUnitFractionDotPoint(label="A", value=1.0),
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_6_3():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=2.0),
        minor_divisions=6,
        endpoint_fraction="6/3",
        dot_point=ExtendedUnitFractionDotPoint(label="B", value=1.0),
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_8_4():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=2.0),
        minor_divisions=4,
        endpoint_fraction="8/4",
        dot_point=ExtendedUnitFractionDotPoint(label="C", value=0.5),
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_3_2():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=1.5),
        minor_divisions=6,
        endpoint_fraction="3/2",
        dot_point=ExtendedUnitFractionDotPoint(label="D", value=0.75),
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_extended_unit_fraction_number_line_7_4():
    number_line = ExtendedUnitFractionNumberLine(
        range=ExtendedRange(min=0.0, max=1.75),
        minor_divisions=7,
        endpoint_fraction="7/4",
        dot_point=ExtendedUnitFractionDotPoint(label="F", value=1.25),
    )
    file_name = create_extended_unit_fraction_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_bar():
    """Test with multiple number lines"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=3,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1 / 3),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=4,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=0.75),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=4,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=0.25),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=2,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=0.5),
            ),
        ]
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_bar(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_bar_custom_start_tick():
    """Test with blue bar starting at different tick positions (not tick 0)"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            # Number line 0 to 1, divided into 6, bar length 2/6 starting at tick 2
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1.0),
                minor_divisions=6,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="A", value=2 / 6, dot_start_tick=2
                ),
            ),
            # Number line 0 to 1, divided into 4, bar length 2/4 starting at tick 1
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1.0),
                minor_divisions=4,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="B", value=2 / 4, dot_start_tick=1
                ),
            ),
            # Number line 0 to 1, divided into 8, bar length 2/8 starting at tick 3
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1.0),
                minor_divisions=8,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="C", value=2 / 8, dot_start_tick=3
                ),
            ),
            # Number line 0 to 1, divided into 3, traditional bar length 2/3 starting at tick 0
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1.0),
                minor_divisions=3,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="D", value=2 / 3, dot_start_tick=0
                ),
            ),
        ]
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_bar(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_labeled_unit_fraction_number_line_basic():
    """Test with multiple labeled number lines"""
    multi_line = MultiLabeledUnitFractionNumberLine(
        number_lines=[
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=3,
                endpoint_fraction="1",
            ),
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=4,
                endpoint_fraction="1",
            ),
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=2,
                endpoint_fraction="1",
            ),
        ]
    )
    file_name = create_multi_labeled_unit_fraction_number_line(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_labeled_unit_fraction_number_line_single():
    """Test with a single labeled number line"""
    multi_line = MultiLabeledUnitFractionNumberLine(
        number_lines=[
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=5,
                endpoint_fraction="1",
            ),
        ]
    )
    file_name = create_multi_labeled_unit_fraction_number_line(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_labeled_unit_fraction_number_line_extended():
    """Test with extended range (greater than 1)"""
    multi_line = MultiLabeledUnitFractionNumberLine(
        number_lines=[
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2.0),
                minor_divisions=8,
                endpoint_fraction="8/4",
            ),
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1.5),
                minor_divisions=6,
                endpoint_fraction="3/2",
            ),
        ]
    )
    file_name = create_multi_labeled_unit_fraction_number_line(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_labeled_unit_fraction_number_line_unsimplified_fractions():
    """Test that fractions are not simplified when endpoint is a whole number > 1"""
    multi_line = MultiLabeledUnitFractionNumberLine(
        number_lines=[
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2.0),
                minor_divisions=4,
                endpoint_fraction="2",
            ),
            LabeledUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=3.0),
                minor_divisions=6,
                endpoint_fraction="3",
            ),
        ]
    )
    file_name = create_multi_labeled_unit_fraction_number_line(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_dots():
    """Test with multiple number lines with colored dot points"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=6,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="P", value=4 / 6, red=True
                ),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=4,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="A", value=2 / 4),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=10,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="B", value=7 / 10),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1),
                minor_divisions=8,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="C", value=5 / 8),
            ),
        ]
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_fixed_step_number_line_basic():
    """Test basic fixed step number line with integer step size."""
    number_line = FixedStepNumberLine(
        range=Range(min=0, max=20),
        step_size=2.0,
        points=[
            Point(value=4, label="A"),
            Point(value=10, label="B"),
            Point(value=16, label="C"),
        ],
    )
    file_name = create_fixed_step_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_fixed_step_number_line_fractional_step():
    """Test fixed step number line with fractional step size and 4 minor divisions."""
    number_line = FixedStepNumberLine(
        range=Range(min=0, max=5),
        step_size=0.5,
        minor_divisions=4,
        points=[
            Point(value=0.8, label="X"),
            Point(value=2.5, label="Y"),
            Point(value=4.0, label="Z"),
        ],
    )
    file_name = create_fixed_step_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_fixed_step_number_line_large_step():
    """Test fixed step number line with large step size, wide range, and 5 minor divisions."""
    number_line = FixedStepNumberLine(
        range=Range(min=30, max=100),
        step_size=10.0,
        minor_divisions=10,
        points=[
            Point(value=35, label="A"),
            Point(value=57, label="B"),
            Point(value=75, label="C"),
        ],
    )
    file_name = create_fixed_step_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_fixed_step_number_line_large_step_and_range():
    """Test fixed step number line with large step size and wide range."""
    number_line = FixedStepNumberLine(
        range=Range(min=500, max=1000),
        step_size=100.0,
        minor_divisions=10,
        points=[
            Point(value=550, label="A"),
            Point(value=750, label="B"),
            Point(value=875, label="C"),
        ],
    )
    file_name = create_fixed_step_number_line(number_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_dots_show_labels():
    """Test dots function with show_minor_division_labels set to True"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=1.0),
                minor_divisions=4,
                endpoint_fraction="1",
                dot_point=ExtendedUnitFractionDotPoint(label="A", value=0.75),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2.0),
                minor_divisions=8,
                endpoint_fraction="8/4",
                dot_point=ExtendedUnitFractionDotPoint(label="B", value=1.5),
            ),
        ],
        show_minor_division_labels=True,
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_dots_mixed_range_1_1_4():
    """Reproduce a 4-line layout over [-5, 5] with one dot per line.
    Dots positioned at approximately: 0.2, 1.2, -1.2, and 1.8."""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            # Line 1: near 0.2
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=10,  # integer ticks
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=0.2),
            ),
            # Line 2: near 1.2
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=10,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.2),
            ),
            # Line 3: near -1.2
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=10,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-1.2),
            ),
            # Line 4: near 1.8
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=10,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.8),
            ),
        ],
        # Show labels on every tick so the axis reads -5, -4, ..., 5
        show_minor_division_labels=True,
        show_integer_tick_labels=True,
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_bar_v2():
    """Test v2 with multiple number lines - uses range 0 to 2 and labels 0, 1, 2"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=6,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=2 / 3),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=8,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.25),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=4,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.5),
            ),
        ]
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_bar_v2(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_bar_v2_single_line():
    """Test v2 with a single number line"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=5,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=0.4),
            ),
        ]
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_bar_v2(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_multi_extended_unit_fraction_number_line_with_bar_v2_edge_values():
    """Test v2 with dot points at edge values (0, 1, 2)"""
    multi_line = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=4,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=0.0),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=4,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.0),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=0.0, max=2),
                minor_divisions=4,
                endpoint_fraction="2",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=2.0),
            ),
        ]
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_bar_v2(multi_line)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dot_plot_with_colors():
    """Test dot plot creation with colored dots and title."""
    # Create test data that meets all validation requirements:
    # - At least 10 data points
    # - 5-8 unique numbers (we have 6: 8, 9, 10, 11, 12, 13)
    # - Numbers are consecutive
    # - At least 50% of unique numbers present at least twice (4 out of 6 = 66.7%)
    dot_plot_data = StatsLinePlot(
        title="Number of Kilometres Run",
        data=[8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 13, 13],
    )

    # Generate multiple plots to test color randomization
    file_names = []
    for i in range(3):
        file_name = create_dot_plot(dot_plot_data)
        assert os.path.exists(file_name)
        file_names.append(file_name)

    # All files should exist
    for file_name in file_names:
        assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dot_plot_edge_case_minimum_data():
    """Test dot plot with minimum valid data (exactly 10 points, 5 unique numbers)."""
    # Edge case: minimum valid data
    # - Exactly 10 data points
    # - Exactly 5 unique numbers (minimum allowed)
    # - At least 50% of unique numbers present at least twice (3 out of 5 = 60%)
    dot_plot_data = StatsLinePlot(
        title="Test Scores", data=[1, 1, 2, 2, 3, 3, 4, 5, 5, 5]
    )

    file_name = create_dot_plot(dot_plot_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dot_plot_edge_case_maximum_unique_numbers():
    """Test dot plot with maximum unique numbers (8 unique numbers)."""
    # Edge case: maximum unique numbers
    # - 8 unique numbers (maximum allowed): 10, 11, 12, 13, 14, 15, 16, 17
    # - At least 50% of unique numbers present at least twice (4 out of 8 = 50%)
    # - More data points to ensure sufficient frequency
    dot_plot_data = StatsLinePlot(
        title="Daily Temperature (°C)",
        data=[10, 10, 11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17, 17, 17, 17, 17, 17],
    )

    file_name = create_dot_plot(dot_plot_data)
    assert os.path.exists(file_name)


## Summary of Test Case


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_basic():
    """Test basic dual dot plot creation with color pairs and enhanced styling (side by side)."""
    # Create test data that meets all validation requirements for both datasets
    # Both datasets use the same x-axis range (8-19) for better comparison
    # Left data: 6 unique numbers (8, 9, 10, 11, 12, 13), at least 50% present twice
    # Right data: 5 unique numbers (15, 16, 17, 18, 19), at least 50% present twice
    dual_plot_data = DualStatsLinePlot(
        top_title="Morning Temperatures (°C)",
        top_data=[8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 13, 13],
        bottom_title="Evening Temperatures (°C)",
        bottom_data=[8, 8, 9, 9, 10, 10, 10, 11, 11, 11, 12, 13, 13, 13, 14],
    )

    # Generate multiple plots to test color pair randomization
    file_names = []
    for i in range(3):
        file_name = create_dual_dot_plot(dual_plot_data)
        assert os.path.exists(file_name)
        file_names.append(file_name)

    # All files should exist
    for file_name in file_names:
        assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_edge_case_minimum_data():
    """Test dual dot plot with minimum valid data for both datasets (side by side)."""
    # Edge case: minimum valid data for both datasets
    # Both datasets use the same x-axis range (1-10) for better comparison
    # Left data: exactly 10 points, 5 unique numbers, 60% present at least twice
    # Right data: exactly 10 points, 5 unique numbers, 60% present at least twice
    dual_plot_data = DualStatsLinePlot(
        top_title="Test Scores - Class A",
        top_data=[1, 1, 2, 2, 3, 3, 4, 5, 5, 5],
        bottom_title="Test Scores - Class B",
        bottom_data=[1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5],
    )

    file_name = create_dual_dot_plot(dual_plot_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_edge_case_maximum_unique_numbers():
    """Test dual dot plot with maximum unique numbers (8 unique numbers each) - side by side."""
    # Edge case: maximum unique numbers for both datasets
    # Both datasets use the same x-axis range (20-37) for better comparison
    # Left data: 8 unique numbers (20-27), exactly 50% present at least twice
    # Right data: 8 unique numbers (30-37), exactly 50% present at least twice
    dual_plot_data = DualStatsLinePlot(
        top_title="Sales - Store A (units)",
        top_data=[
            20,
            20,
            21,
            21,
            22,
            22,
            23,
            23,
            24,
            25,
            26,
            27,
            27,
            27,
            27,
            27,
            27,
            27,
        ],
        bottom_title="Sales - Store B (units)",
        bottom_data=[
            30,
            30,
            31,
            31,
            32,
            32,
            33,
            33,
            34,
            35,
            36,
            37,
            37,
            37,
            37,
            37,
            37,
            37,
        ],
    )

    file_name = create_dual_dot_plot(dual_plot_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_overlapping_ranges():
    """Test dual dot plot with overlapping data ranges for better comparison (side by side)."""
    # Test with overlapping ranges for better visual comparison
    # Both datasets share some x-axis values (10-15) for direct comparison
    dual_plot_data = DualStatsLinePlot(
        top_title="Week 1 Performance",
        top_data=[8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 13, 13, 14, 15, 15],
        bottom_title="Week 2 Performance",
        bottom_data=[10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 15, 15, 16, 17, 17],
    )

    file_name = create_dual_dot_plot(dual_plot_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_create_dual_dot_plot_max_values_limit():
    """Test dual dot plot with more than 10 values to test the limit."""
    # Test with more than 10 data points but within validation limits
    # Each dataset has 8 unique numbers (maximum allowed) with more than 10 total data points
    dual_plot_data = DualStatsLinePlot(
        top_title="Large Dataset A",
        top_data=[
            9,
            9,
            9,
            10,
            11,
            11,
            11,
            12,
            12,
            13,
            14,
            14,
            15,
            16,
            16,
            16,
            16,
            16,
            16,
            16,
        ],  # 8 unique numbers, 20 total points
        bottom_title="Large Dataset B",
        bottom_data=[
            9,
            9,
            10,
            10,
            11,
            11,
            12,
            12,
            13,
            13,
            14,
            14,
            15,
            15,
            16,
            16,
            16,
            16,
            16,
            16,
        ],  # 8 unique numbers, 20 total points
    )

    file_name = create_dual_dot_plot(dual_plot_data)
    assert os.path.exists(file_name)


# --- User-specified multi-line with dots cases to preview outputs ---


@pytest.mark.drawing_functions
def test_multi_with_dots_user_case_1_thirds_negative_one():
    multi = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-1.0),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.0),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-2.0),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=2.0),
            ),
        ],
        show_minor_division_labels=True,
        show_integer_tick_labels=True,
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_multi_with_dots_user_case_2_thirds_decimal_2333():
    multi = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="", value=2.333333333333333
                ),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="", value=-1.3333333333333333
                ),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="", value=1.6666666666666667
                ),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=3,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(
                    label="", value=-2.333333333333333
                ),
            ),
        ],
        show_minor_division_labels=True,
        show_integer_tick_labels=True,
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_multi_with_dots_user_case_3_quarters_mixed_1_25():
    multi = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=1.25),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-1.25),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=2.5),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-2.5),
            ),
        ],
        show_minor_division_labels=True,
        show_integer_tick_labels=True,
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_multi_with_dots_user_case_4_quarters_various():
    multi = MultiExtendedUnitFractionNumberLine(
        number_lines=[
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-2.25),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=2.75),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-2.75),
            ),
            ExtendedUnitFractionNumberLine(
                range=ExtendedRange(min=-5.0, max=5.0),
                minor_divisions=4,
                endpoint_fraction="5",
                dot_point=ExtendedUnitFractionDotPoint(label="", value=-2.0),
            ),
        ],
        show_minor_division_labels=True,
        show_integer_tick_labels=True,
    )
    file_name = create_multi_extended_unit_fraction_number_line_with_dots(multi)
    assert os.path.exists(file_name)
