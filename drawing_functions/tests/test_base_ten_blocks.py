import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.base_ten_blocks import (
    draw_base_ten_blocks,
    draw_base_ten_blocks_division,
    draw_base_ten_blocks_division_grid,
    draw_base_ten_blocks_grid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.base_ten_block import (
    BaseTenBlock,
    BaseTenBlockDivisionStimulus,
    BaseTenBlockGridStimulus,
    BaseTenBlockStimulus,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_single_block():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=123)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_single_block_maximum():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=999)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_blocks_below_100():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=45), BaseTenBlock(value=5)], operation="addition"
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_blocks_divide():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=45), BaseTenBlock(value=9)], operation="divide"
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_blocks_only_tens():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=10), BaseTenBlock(value=50)], operation="addition"
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_blocks():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=556), BaseTenBlock(value=400)], operation="addition"
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_blocks_maximum():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=999), BaseTenBlock(value=999)], operation="addition"
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_zero_value():
    with pytest.raises(ValueError):
        stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=0)])
        draw_base_ten_blocks(stimulus.model_dump())


@pytest.mark.drawing_functions
def test_max_value():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=1000)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_single_digit():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=5)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_digit():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=42)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_only_tens():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=50)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_only_hundreds():
    stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=300)])
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_two_blocks_different_magnitudes():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=5), BaseTenBlock(value=500)], operation="addition"
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_invalid_number_of_blocks():
    with pytest.raises(ValidationError):
        stimulus = BaseTenBlockStimulus(
            blocks=[
                BaseTenBlock(value=1),
                BaseTenBlock(value=2),
                BaseTenBlock(value=3),
            ],
            operation="addition",
        )
        draw_base_ten_blocks(stimulus.model_dump())


@pytest.mark.drawing_functions
def test_invalid_value():
    with pytest.raises(ValueError):
        stimulus = BaseTenBlockStimulus(blocks=[BaseTenBlock(value=2000)])
        draw_base_ten_blocks(stimulus.model_dump())


@pytest.mark.drawing_functions
def test_decimal_display_single_block():
    stimulus = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=999, display_as_decimal=True)]  # Should show as 1.23
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_display_two_blocks():
    stimulus = BaseTenBlockStimulus(
        blocks=[
            BaseTenBlock(value=45, display_as_decimal=True),  # Should show as 0.45
            BaseTenBlock(value=7, display_as_decimal=True),  # Should show as 0.07
        ],
        operation="addition",
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_display_with_hundreds():
    stimulus = BaseTenBlockStimulus(
        blocks=[
            BaseTenBlock(value=500, display_as_decimal=True),  # Should show as 5.00
            BaseTenBlock(value=123, display_as_decimal=True),  # Should show as 1.23
        ],
        operation="addition",
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_decimal_display_with_hundreds_without_values():
    stimulus = BaseTenBlockStimulus(
        blocks=[
            BaseTenBlock(value=123, display_as_decimal=True),
        ],
        show_values=False,
    )
    file_path = draw_base_ten_blocks(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_mixed_display_mode_validation():
    with pytest.raises(ValueError, match="All blocks must use the same display mode"):
        BaseTenBlockStimulus(
            blocks=[
                BaseTenBlock(value=123, display_as_decimal=True),
                BaseTenBlock(value=45, display_as_decimal=False),
            ],
            operation="addition",
        )


# Grid functionality tests
@pytest.mark.drawing_functions
def test_grid_two_blocks():
    """Test grid with two blocks - typical for division by 2"""
    stimulus = BaseTenBlockGridStimulus(block_value=220, count=3)
    file_path = draw_base_ten_blocks_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_grid_six_blocks():
    """Test grid with maximum allowed blocks (6)"""
    stimulus = BaseTenBlockGridStimulus(block_value=150, count=6)
    file_path = draw_base_ten_blocks_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_165_by_11():
    """Test division layout with 165 ÷ 11 = 15"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=732, divisor=4)
    file_path = draw_base_ten_blocks_division(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_simple():
    """Test division layout with a simple example: 84 ÷ 4 = 21"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=84, divisor=4)
    file_path = draw_base_ten_blocks_division(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_no_labels():
    """Test division layout without labels"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=102, divisor=6)
    file_path = draw_base_ten_blocks_division(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_grid_canonical_15():
    """Test canonical division grid: 150 ÷ 15 = 10 (15 rows × 10 columns)"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=150, divisor=15)
    file_path = draw_base_ten_blocks_division_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_grid_canonical_12():
    """Test canonical division grid: 132 ÷ 12 = 11 (12 rows × 11 columns)"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=132, divisor=12)
    file_path = draw_base_ten_blocks_division_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_grid_canonical_19():
    """Test canonical division grid: 190 ÷ 19 = 10 (19 rows × 10 columns)"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=190, divisor=19)
    file_path = draw_base_ten_blocks_division_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_grid_canonical_11():
    """Test canonical division grid: 165 ÷ 11 = 15 (11 rows × 15 columns)"""
    stimulus = BaseTenBlockDivisionStimulus(dividend=165, divisor=11)
    file_path = draw_base_ten_blocks_division_grid(stimulus.model_dump())
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_division_grid_incorrect_tiling():
    """Test canonical division grid with incorrect tiling for multiple choice questions"""
    stimulus = BaseTenBlockDivisionStimulus(
        dividend=165, divisor=11, incorrect_tiling=True
    )
    file_path = draw_base_ten_blocks_division_grid(stimulus.model_dump())
    assert os.path.exists(file_path)
