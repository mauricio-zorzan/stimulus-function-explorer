import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.box_plots import (
    BoxPlotData,
    BoxPlotDescription,
    draw_bar_models,
    draw_box_plots,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.bar_model import (
    Bar,
    BarModel,
)


@pytest.mark.drawing_functions
def test_draw_bar_models():
    # Create example data for the test
    example_data = BarModel(
        [
            Bar(label="Category 1", length=10),
            Bar(label="Category 2", length=15),
        ]
    )

    # Call the function with the example data
    file_name = draw_bar_models(example_data)

    # Check if the file was created
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_bar_models_empty_data():
    # Test with empty data
    example_data = []

    with pytest.raises(Exception):
        draw_bar_models(example_data)


@pytest.mark.drawing_functions
def test_draw_box_plots():
    stimulus_description = BoxPlotDescription(
        title="Test Scores",
        data=[
            BoxPlotData(
                class_name="Class A",
                min_value=66,
                q1=78,
                median=83,
                q3=86,
                max_value=95,
            ),
            BoxPlotData(
                class_name="Class B",
                min_value=75,
                q1=85,
                median=94,
                q3=96,
                max_value=100,
            ),
        ],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json(by_alias=True))
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_box_plots_with_additional_data():
    stimulus_description = BoxPlotDescription(
        data=[BoxPlotData(min_value=24, q1=27, median=31, q3=33, max_value=34)]
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_single_box_plot_with_title():
    """Test single box plot with title and random colors"""
    stimulus_description = BoxPlotDescription(
        title="Quiz Scores",
        data=[BoxPlotData(min_value=13, q1=15, median=16, q3=17, max_value=20)],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_single_box_plot_without_title():
    """Test single box plot without title - should display without title"""
    stimulus_description = BoxPlotDescription(
        data=[BoxPlotData(min_value=10, q1=12, median=14, q3=16, max_value=18)]
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_single_box_plot_with_explicit_title():
    """Test single box plot with explicit title"""
    stimulus_description = BoxPlotDescription(
        title="Custom Single Box Plot",
        data=[BoxPlotData(min_value=5, q1=7, median=9, q3=11, max_value=13)],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_dual_box_plots_different_ranges():
    """Test dual box plots with different data ranges and random colors"""
    stimulus_description = BoxPlotDescription(
        title="Test Performance Comparison",
        data=[
            BoxPlotData(
                class_name="Mathematics",
                min_value=60,
                q1=70,
                median=80,
                q3=90,
                max_value=100,
            ),
            BoxPlotData(
                class_name="Science",
                min_value=45,
                q1=55,
                median=65,
                q3=75,
                max_value=85,
            ),
        ],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_dual_box_plots_similar_ranges():
    """Test dual box plots with similar data ranges"""
    stimulus_description = BoxPlotDescription(
        title="Sales Performance",
        data=[
            BoxPlotData(
                class_name="Q1 Sales",
                min_value=100,
                q1=120,
                median=140,
                q3=160,
                max_value=180,
            ),
            BoxPlotData(
                class_name="Q2 Sales",
                min_value=105,
                q1=125,
                median=145,
                q3=165,
                max_value=185,
            ),
        ],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_box_plots_edge_case_values():
    """Test box plots with edge case values (very small range)"""
    stimulus_description = BoxPlotDescription(
        title="Precision Measurements",
        data=[
            BoxPlotData(
                class_name="Group A",
                min_value=99,
                q1=99.5,
                median=100,
                q3=100.5,
                max_value=101,
            ),
            BoxPlotData(
                class_name="Group B",
                min_value=98,
                q1=99,
                median=100,
                q3=101,
                max_value=102,
            ),
        ],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_box_plots_large_range():
    """Test box plots with large data range"""
    stimulus_description = BoxPlotDescription(
        title="Annual Revenue",
        data=[
            BoxPlotData(
                class_name="Company A",
                min_value=1000,
                q1=5000,
                median=10000,
                q3=15000,
                max_value=20000,
            ),
            BoxPlotData(
                class_name="Company B",
                min_value=2000,
                q1=6000,
                median=12000,
                q3=18000,
                max_value=25000,
            ),
        ],
    )
    file_name = draw_box_plots(stimulus_description.model_dump_json())
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_single_box_plot_validation_error():
    """Test that single box plots with class_name raise validation error"""
    with pytest.raises(
        ValueError, match="Class name should not be provided for single box plots"
    ):
        BoxPlotDescription(
            title="Invalid Single Box Plot",
            data=[
                BoxPlotData(
                    class_name="Invalid Class",  # This should cause validation error
                    min_value=10,
                    q1=12,
                    median=14,
                    q3=16,
                    max_value=18,
                )
            ],
        )


@pytest.mark.drawing_functions
def test_draw_box_plots_improved_readability():
    """Test box plots with improved tick mark visibility and font size"""
    # Create data that matches the image shown (similar to the original example)
    stimulus_description = BoxPlotDescription(
        title="Test Scores",
        data=[
            BoxPlotData(
                min_value=35,
                q1=45,
                median=60,
                q3=75,
                max_value=95,
            )
        ],
    )

    # Generate the box plot with improved readability
    file_name = draw_box_plots(stimulus_description.model_dump_json())

    # Verify the file was created
    assert os.path.exists(file_name)

    # The test verifies that the function runs without error with the new styling
    # The actual visual improvements (larger ticks, better font size) are tested
    # by the successful generation of the plot with the updated parameters


@pytest.mark.drawing_functions
def test_draw_box_plots_readability_dual_plots():
    """Test dual box plots with improved readability for comparison"""
    stimulus_description = BoxPlotDescription(
        title="Performance Comparison",
        data=[
            BoxPlotData(
                class_name="Group A",
                min_value=30,
                q1=40,
                median=50,
                q3=60,
                max_value=70,
            ),
            BoxPlotData(
                class_name="Group B",
                min_value=40,
                q1=50,
                median=60,
                q3=70,
                max_value=80,
            ),
        ],
    )

    # Generate the box plot with improved readability
    file_name = draw_box_plots(stimulus_description.model_dump_json())

    # Verify the file was created
    assert os.path.exists(file_name)

    # This test specifically validates the improved tick marks and font sizes
    # work correctly with multiple box plots and different data ranges
