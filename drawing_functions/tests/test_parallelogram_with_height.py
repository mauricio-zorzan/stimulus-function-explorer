import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.geometric_shapes import (
    draw_parallelogram_with_height,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.quadrilateral_figures import (
    ParallelogramWithHeight,
)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_easy():
    """Test easy case: whole numbers ≤ 10 for base and height"""
    # For base=8, height=6, with ~15° slant, slant_side ≈ 8
    data = ParallelogramWithHeight(
        base_label="8 cm", height_label="6 cm", slant_side_label="8 cm"
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_medium_decimal():
    """Test medium case: decimals less than 2"""
    # For base=1.5, height=1.2, with ~15° slant, slant_side ≈ 1.6
    data = ParallelogramWithHeight(
        base_label="1.5 cm", height_label="1.2 cm", slant_side_label="1.6 cm"
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_medium_large():
    """Test medium case: whole numbers between 10 and 50"""
    # For base=25, height=18, with ~15° slant, slant_side ≈ 20
    data = ParallelogramWithHeight(
        base_label="25 cm", height_label="18 cm", slant_side_label="20 cm"
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_hard_find_base():
    """Test hard case: find base when area is provided"""
    # For height=12, with ~15° slant, reasonable slant_side=14
    data = ParallelogramWithHeight(
        base_label="b", height_label="12 cm", slant_side_label="14 cm"
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_hard_find_height():
    """Test hard case: find height when area is provided"""
    # For base=20, with ~15° slant, reasonable slant_side=22
    data = ParallelogramWithHeight(
        base_label="20 cm", height_label="h", slant_side_label="22 cm"
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_boundary_max():
    """Test maximum allowed values within assessment boundary"""
    # For base=49, height=45, with ~15° slant, slant_side ≈ 47
    data = ParallelogramWithHeight(
        base_label="49 cm", height_label="45 cm", slant_side_label="47 cm"
    )
    file_path = draw_parallelogram_with_height(data)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_boundary_min():
    """Test minimum decimal values that still allow proper slant"""
    # For base=1.9, height=1.5, with ~15° slant, slant_side ≈ 2.0
    data = ParallelogramWithHeight(
        base_label="1.9 cm", height_label="1.5 cm", slant_side_label="2.0 cm"
    )
    file_path = draw_parallelogram_with_height(data)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_proportional():
    """Test case for parallelogram with 9cm base, unknown height, and 6cm slant"""
    data = ParallelogramWithHeight(
        base_label="9 cm", height_label="h", slant_side_label="6 cm"
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_distinct_proportions_1():
    """Test case with very distinct proportions - base much longer than slant"""
    with pytest.raises(
        ValueError, match="Base cannot be more than or equal to twice the slant side"
    ):
        ParallelogramWithHeight(
            base_label="30 cm",  # Much longer base
            height_label="12 cm",
            slant_side_label="15 cm",  # Half of base - should be rejected
        )


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_distinct_proportions_2():
    """Test case with very distinct proportions - slant much longer than base"""
    with pytest.raises(
        ValueError, match="Slant side cannot be more than or equal to twice the base"
    ):
        ParallelogramWithHeight(
            base_label="15 cm",
            height_label="24 cm",
            slant_side_label="30 cm",  # Should be rejected - too large compared to base
        )


@pytest.mark.drawing_functions
def test_parallelogram_with_height_invalid_height():
    """Test that height >= slant is rejected"""
    with pytest.raises(ValueError, match="Height must be less than slant side"):
        ParallelogramWithHeight(
            base_label="10 cm",
            height_label="12 cm",  # Invalid: height > slant
            slant_side_label="10 cm",
        )


@pytest.mark.drawing_functions
def test_parallelogram_with_height_extreme_angle():
    """Test that extremely small angles are rejected"""
    with pytest.raises(ValueError, match="Height should be at least 30%"):
        ParallelogramWithHeight(
            base_label="10 cm",
            height_label="2 cm",  # Too small compared to slant
            slant_side_label="15 cm",
        )


@pytest.mark.drawing_functions
def test_parallelogram_with_height_extreme_proportions():
    """Test that extreme base:slant ratios are rejected"""
    with pytest.raises(
        ValueError, match="Base cannot be more than or equal to twice the slant side"
    ):
        ParallelogramWithHeight(
            base_label="30 cm",  # Too large compared to slant
            height_label="8 cm",
            slant_side_label="10 cm",
        )


@pytest.mark.drawing_functions
def test_draw_parallelogram_with_height_distinct_but_valid():
    """Test case with distinct but valid proportions"""
    data = ParallelogramWithHeight(
        base_label="15 cm",
        height_label="12 cm",
        slant_side_label="20 cm",  # Less than double the base
    )
    file_name = draw_parallelogram_with_height(data)
    assert os.path.exists(file_name)
