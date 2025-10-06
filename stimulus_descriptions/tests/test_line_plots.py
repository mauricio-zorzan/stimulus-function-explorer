import os
from unittest.mock import MagicMock

import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.line_plots import (
    DataPoint,
    LinePlot,
    LinePlotList,
    SingleLinePlot,
)
from content_generators.question_generator.schemas import MCQQuestion


def test_assert_correct_volume_is_unique_for_select():
    # Create sample data points
    data_points1 = [
        DataPoint(value="1", frequency=5),
        DataPoint(value="1 1/4", frequency=1),
        DataPoint(value="1 1/2", frequency=7),
        DataPoint(value="1 3/4", frequency=2),
        DataPoint(value="2", frequency=3),
    ]
    data_points2 = [
        DataPoint(value="1", frequency=5),
        DataPoint(value="1 1/4", frequency=2),
        DataPoint(value="1 1/2", frequency=7),
        DataPoint(value="1 3/4", frequency=2),
        DataPoint(value="2", frequency=4),
    ]
    data_points3 = [
        DataPoint(value="1", frequency=5),
        DataPoint(value="1 1/4", frequency=1),
        DataPoint(value="1 1/2", frequency=7),
        DataPoint(value="1 3/4", frequency=2),
        DataPoint(value="2", frequency=3),
    ]
    data_points4 = [
        DataPoint(value="1", frequency=4),
        DataPoint(value="1 1/4", frequency=3),
        DataPoint(value="1 1/2", frequency=6),
        DataPoint(value="1 3/4", frequency=3),
        DataPoint(value="2", frequency=5),
    ]

    # Create sample line plots
    line_plot1 = LinePlot(
        title="Plot 1",
        x_axis_label="Length of Ribbon (in yards)",
        data_points=data_points1,
    )
    line_plot2 = LinePlot(
        title="Plot 2",
        x_axis_label="Length of Ribbon (in yards)",
        data_points=data_points2,
    )
    line_plot3 = LinePlot(
        title="Plot 3",
        x_axis_label="Length of Ribbon (in yards)",
        data_points=data_points3,
    )
    line_plot4 = LinePlot(
        title="Plot 4",
        x_axis_label="Length of Ribbon (in yards)",
        data_points=data_points4,
    )

    # Create a LinePlotList
    line_plot_list = LinePlotList([line_plot1, line_plot2, line_plot3, line_plot4])

    pipeline_context = MagicMock(
        payload=MagicMock(
            placeholders=MagicMock(
                grade="4",
                standard_id="CCSS.MATH.CONTENT.4.MD.B.4+1",
                standard_description="Test description",
            ),
        ),
        question=MCQQuestion.legacy_load(
            {
                "question_text_with_inline_latex": "A class is making a project using ribbon of different lengths, measured in yards. The lengths are: $1$, $1$, $1$, $1$, $1$, $1 \\frac{1}{4}$, $1 \\frac{1}{2}$, $1 \\frac{1}{2}$, $1 \\frac{1}{2}$, $1 \\frac{1}{2}$, $1 \\frac{1}{2}$, $1 \\frac{1}{2}$, $1 \\frac{1}{2}$, $1 \\frac{3}{4}$, $1 \\frac{3}{4}$, $2$, $2$, $2$. Considering all four plots, which one correctly shows the frequency of each measurement?",
                "A_text_with_inline_latex": "Plot 1",
                "A_explanation_text_with_inline_latex": "Answer A is incorrect. Plot 1 shows 7 instances of $1 \\frac{1}{2}$, however, there are 6 instances in the data set.",
                "A_correct": False,
                "B_text_with_inline_latex": "Plot 2",
                "B_explanation_text_with_inline_latex": "Answer B is incorrect. Plot 2 shows 2 instances of $1 \\frac{1}{4}$ and 4 instances of $2$, which are more than in the data set.",
                "B_correct": False,
                "C_text_with_inline_latex": "Plot 3",
                "C_explanation_text_with_inline_latex": "Answer C is correct. Plot 3 correctly represents the measurements' frequency with 5 instances of $1$, 1 instance of $1 \\frac{1}{4}$, 7 instances of $1 \\frac{1}{2}$, 2 instances of $1 \\frac{3}{4}$ and 3 instances of $2$.",
                "C_correct": True,
                "D_text_with_inline_latex": "Plot 4",
                "D_explanation_text_with_inline_latex": "Answer D is incorrect. Plot 4 incorrectly shows fewer instances of $1$ and more instances of $1 \\frac{1}{4}$ than there are in the data set.",
                "D_correct": False,
                "stimulus_description": [
                    {
                        "title": "Plot 1",
                        "x_axis_label": "Length of Ribbon (in yards)",
                        "data_points": [
                            {"value": "1", "frequency": 5},
                            {"value": "1 1/4", "frequency": 1},
                            {"value": "1 1/2", "frequency": 7},
                            {"value": "1 3/4", "frequency": 2},
                            {"value": "2", "frequency": 3},
                        ],
                    },
                    {
                        "title": "Plot 2",
                        "x_axis_label": "Length of Ribbon (in yards)",
                        "data_points": [
                            {"value": "1", "frequency": 5},
                            {"value": "1 1/4", "frequency": 2},
                            {"value": "1 1/2", "frequency": 7},
                            {"value": "1 3/4", "frequency": 2},
                            {"value": "2", "frequency": 4},
                        ],
                    },
                    {
                        "title": "Plot 3",
                        "x_axis_label": "Length of Ribbon (in yards)",
                        "data_points": [
                            {"value": "1", "frequency": 5},
                            {"value": "1 1/4", "frequency": 1},
                            {"value": "1 1/2", "frequency": 7},
                            {"value": "1 3/4", "frequency": 2},
                            {"value": "2", "frequency": 3},
                        ],
                    },
                    {
                        "title": "Plot 4",
                        "x_axis_label": "Length of Ribbon (in yards)",
                        "data_points": [
                            {"value": "1", "frequency": 4},
                            {"value": "1 1/4", "frequency": 3},
                            {"value": "1 1/2", "frequency": 6},
                            {"value": "1 3/4", "frequency": 3},
                            {"value": "2", "frequency": 5},
                        ],
                    },
                ],
            },
        ),
    )

    # Test for uniqueness
    with pytest.raises(
        ValueError,
        match="The set of frequencies .* is not unique among the line plots.",
    ):
        line_plot_list.assert_correct_frequencies_is_unique_for_select(pipeline_context)

    # Modify data points to make frequencies unique
    data_points1[0].frequency = 1

    # This should not raise an error
    line_plot_list.assert_correct_frequencies_is_unique_for_select(pipeline_context)

@pytest.mark.drawing_functions
def test_single_line_plot_validates_and_draws():
    data_points = [
        DataPoint(value="0", frequency=2),
        DataPoint(value="1", frequency=5),
        DataPoint(value="2", frequency=8),
        DataPoint(value="3", frequency=3),
        DataPoint(value="4", frequency=2),
    ]
    single_plot = SingleLinePlot(
        title="Number of Pets in Households",
        x_axis_label="Number of Pets",
        data_points=data_points,
    )
    # Validation should not raise
    single_plot.validate_data_points()
    # Optionally test drawing (skip if matplotlib not available)
    from content_generators.additional_content.stimulus_image.drawing_functions.line_plots import (
        generate_single_line_plot,
    )
    file_name = generate_single_line_plot(single_plot)
    assert os.path.exists(file_name)
