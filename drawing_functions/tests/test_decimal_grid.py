import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.decimal_grid import (
    draw_decimal_comparison,
    draw_decimal_grid,
    draw_decimal_multiplication,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.decimal_grid import (
    ComparisonLevel,
    DecimalComparison,
    DecimalComparisonList,
    DecimalGrid,
    DecimalMultiplication,
)


@pytest.mark.drawing_functions
def test_decimal_grid_division_10_no_shading():
    """Test 10-division grid with no squares shaded."""
    stimulus = DecimalGrid(division=10, shaded_squares=0)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_10_partial_shading():
    """Test 10-division grid with some squares shaded."""
    stimulus = DecimalGrid(division=10, shaded_squares=43)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_10_full_shading():
    """Test 10-division grid with all squares shaded."""
    stimulus = DecimalGrid(division=100, shaded_squares=455)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_100_no_shading():
    """Test 100-division grid with no squares shaded."""
    stimulus = DecimalGrid(division=100, shaded_squares=0)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_100_small_shading():
    """Test 100-division grid with few squares shaded."""
    stimulus = DecimalGrid(division=100, shaded_squares=7)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_100_quarter_shading():
    """Test 100-division grid with quarter squares shaded."""
    stimulus = DecimalGrid(division=100, shaded_squares=25)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_100_majority_shading():
    """Test 100-division grid with most squares shaded."""
    stimulus = DecimalGrid(division=100, shaded_squares=89)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_100_full_shading():
    """Test 100-division grid with all squares shaded."""
    stimulus = DecimalGrid(division=100, shaded_squares=100)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


# Decimal comparison tests
@pytest.mark.drawing_functions
def test_decimal_comparison_basic_single_precision():
    """Test basic comparison with single decimal place values."""
    comparison = DecimalComparison(
        decimal_1=0.5,
        decimal_2=0.8,
        complexity_level=ComparisonLevel.BASIC,
        color_1="purple",
        color_2="purple",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_basic_double_precision():
    """Test basic comparison with double decimal place values."""
    comparison = DecimalComparison(
        decimal_1=0.57,
        decimal_2=0.31,
        complexity_level=ComparisonLevel.BASIC,
        color_1="green",
        color_2="green",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_basic_mixed_precision():
    """Test basic comparison with mixed precision values."""
    comparison = DecimalComparison(
        decimal_1=0.2,
        decimal_2=0.1,
        complexity_level=ComparisonLevel.BASIC,
        color_1="pink",
        color_2="pink",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_intermediate_matching_whole_parts():
    """Test intermediate comparison with matching whole number components."""
    comparison = DecimalComparison(
        decimal_1=2.5,
        decimal_2=2.8,
        complexity_level=ComparisonLevel.INTERMEDIATE,
        color_1="lightblue",
        color_2="lightblue",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_intermediate_double_precision():
    """Test intermediate comparison with double precision values."""
    comparison = DecimalComparison(
        decimal_1=1.57,
        decimal_2=1.31,
        complexity_level=ComparisonLevel.INTERMEDIATE,
        color_1="orange",
        color_2="orange",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_multiple_pairs():
    """Test multiple comparison pairs in single visualization."""
    comparisons = [
        DecimalComparison(
            decimal_1=0.5,
            decimal_2=0.8,
            complexity_level=ComparisonLevel.BASIC,
            color_1="purple",
            color_2="purple",
        ),
        DecimalComparison(
            decimal_1=0.57,
            decimal_2=0.31,
            complexity_level=ComparisonLevel.BASIC,
            color_1="green",
            color_2="green",
        ),
        DecimalComparison(
            decimal_1=2.4,
            decimal_2=2.6,
            complexity_level=ComparisonLevel.INTERMEDIATE,
            color_1="blue",
            color_2="blue",
        ),
    ]
    stimulus = DecimalComparisonList(comparisons=comparisons)
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_equivalent_values():
    """Test comparison with equivalent decimal representations."""
    comparison = DecimalComparison(
        decimal_1=0.5,
        decimal_2=0.50,
        complexity_level=ComparisonLevel.BASIC,
        color_1="yellow",
        color_2="yellow",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_boundary_values():
    """Test comparison with boundary values."""
    comparison = DecimalComparison(
        decimal_1=0.0,
        decimal_2=0.99,
        complexity_level=ComparisonLevel.BASIC,
        color_1="red",
        color_2="red",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_comparison_different_grid_colors():
    """Test comparison with distinct grid coloring."""
    comparison = DecimalComparison(
        decimal_1=0.3,
        decimal_2=0.7,
        complexity_level=ComparisonLevel.BASIC,
        color_1="red",
        color_2="blue",
    )
    stimulus = DecimalComparisonList(comparisons=[comparison])
    file_path = draw_decimal_comparison(stimulus.model_dump())
    assert os.path.exists(file_path)


# Validation tests
@pytest.mark.drawing_functions
def test_decimal_comparison_validation_basic_with_whole_components():
    """Test validation rejection for basic level with whole number components."""
    with pytest.raises(ValueError, match="Basic level requires values less than 1"):
        DecimalComparison(
            decimal_1=1.5, decimal_2=0.8, complexity_level=ComparisonLevel.BASIC
        )


@pytest.mark.drawing_functions
def test_decimal_comparison_validation_intermediate_mismatched_whole_parts():
    """Test validation rejection for intermediate level with mismatched whole parts."""
    with pytest.raises(
        ValueError, match="Intermediate level requires matching whole number parts"
    ):
        DecimalComparison(
            decimal_1=1.5, decimal_2=2.8, complexity_level=ComparisonLevel.INTERMEDIATE
        )


@pytest.mark.drawing_functions
def test_decimal_comparison_validation_out_of_range():
    """Test validation rejection for values outside visualization range."""
    with pytest.raises(ValueError, match="Decimal values must be between 0 and 9.99"):
        DecimalComparison(
            decimal_1=10.5, decimal_2=0.8, complexity_level=ComparisonLevel.BASIC
        )


@pytest.mark.drawing_functions
def test_decimal_comparison_list_validation_empty():
    """Test validation rejection for empty comparison lists."""
    with pytest.raises(
        ValueError, match="At least one comparison pair must be provided"
    ):
        DecimalComparisonList(comparisons=[])


# Decimal multiplication tests
@pytest.mark.drawing_functions
def test_decimal_multiplication_basic_example():
    """Test basic decimal multiplication 0.1 × 0.8."""
    stimulus = DecimalMultiplication(decimal_factors=[0.5, 0.7])
    file_path = draw_decimal_multiplication(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_multiplication_small_factors():
    """Test multiplication with small factors 0.1 × 0.1."""
    stimulus = DecimalMultiplication(decimal_factors=[0.8, 0.3])
    file_path = draw_decimal_multiplication(stimulus)
    assert os.path.exists(file_path)

@pytest.mark.drawing_functions
def test_decimal_multiplication_with_one_factor_greater_than_one():
    """Test multiplication with small factors 0.1 × 0.1."""
    stimulus = DecimalMultiplication(decimal_factors=[0.4, 1.6])
    file_path = draw_decimal_multiplication(stimulus)
    assert os.path.exists(file_path)


# Multi-grid tests for new functionality
@pytest.mark.drawing_functions
def test_decimal_grid_division_10_max_grids():
    """Test 10-division grid with maximum 10 grids (99 shaded squares)."""
    stimulus = DecimalGrid(division=10, shaded_squares=99)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_grid_division_100_three_grids():
    """Test 100-division grid with 3 grids (300 shaded squares)."""
    stimulus = DecimalGrid(division=100, shaded_squares=999)
    file_path = draw_decimal_grid(stimulus.model_dump())
    assert os.path.exists(file_path)

