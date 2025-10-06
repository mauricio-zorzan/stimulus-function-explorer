from pathlib import Path

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.bar_models import (
    draw_bar_model_stimulus,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.bar_model_stimulus import (
    BarModel,
    BarModelStimulus,
    BarSection,
    ComparisonBarModel,
    create_equal_parts_model,
)
from content_generators.settings import settings


@pytest.fixture
def output_dir():
    """Create a directory for test outputs."""
    output_dir = Path(settings.additional_content_settings.image_destination_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.mark.drawing_functions
def test_draw_single_bar_model(output_dir):
    """Test drawing a single bar model."""
    # Create a bar model with 9 candies divided into 3 equal parts
    stimulus = create_equal_parts_model(
        total=9, num_parts=3, context="candies", highlight_part=0
    )

    # Draw and get the output path
    output_path = draw_bar_model_stimulus(stimulus)
    output_path = Path(output_path)

    # Verify the file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nImage saved to: {output_path.absolute()}")


@pytest.mark.drawing_functions
def test_draw_part_whole_model(output_dir):
    """Test drawing a part-whole model."""
    # Create a part-whole model with known and unknown parts
    stimulus = BarModelStimulus(
        model_type="single_bar",
        single_bar=BarModel(
            total_value=12,
            total_label="12 candies",
            sections=[
                BarSection(value=3, color="#FFE4E1"),  # Light pink
                BarSection(value=4, color="#E0F6FF"),  # Light sky blue
                BarSection(value="?", color="#F0FFF0")  # Light mint for unknown
            ],
            section_width_mode="proportional",
            problem_context="candies"
        )
    )
    
    # Draw and get the output path
    output_path = draw_bar_model_stimulus(stimulus)
    output_path = Path(output_path)
    
    # Verify the file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nImage saved to: {output_path.absolute()}")


@pytest.mark.drawing_functions
def test_draw_comparison_bars(output_dir):
    """Test drawing comparison bars."""
    # Create two bars for comparison
    bar1 = BarModel(
        total_value=12,
        total_label="12 apples",
        sections=[BarSection(value=4, color="#FFE4E1") for _ in range(3)],
        problem_context="apples",
    )

    bar2 = BarModel(
        total_value=8,
        total_label="8 oranges",
        sections=[BarSection(value=4, color="#FFA07A") for _ in range(2)],
        problem_context="oranges",
    )

    # Create comparison bars model
    comparison_model = ComparisonBarModel(
        title="Comparing Fruit Quantities",
        bars=[bar1, bar2],
        comparison_type="difference",
        alignment="left",
    )

    # Create stimulus
    stimulus = BarModelStimulus(
        model_type="comparison_bars", comparison_bars=comparison_model
    )

    # Draw and get the output path
    output_path = draw_bar_model_stimulus(stimulus)
    output_path = Path(output_path)

    # Verify the file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nImage saved to: {output_path.absolute()}")


@pytest.mark.drawing_functions
def test_draw_with_custom_colors(output_dir):
    """Test drawing with custom colors."""
    # Create a bar model with custom colors
    stimulus = BarModelStimulus(
        model_type="single_bar",
        single_bar=BarModel(
            total_value=10,
            total_label="10 items",
            sections=[
                BarSection(value=3, color="#FF0000", highlighted=True),
                BarSection(value=4, color="#00FF00"),
                BarSection(value=3, color="#0000FF"),
            ],
            problem_context="items",
        ),
    )

    # Draw and get the output path
    output_path = draw_bar_model_stimulus(stimulus)
    output_path = Path(output_path)

    # Verify the file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nImage saved to: {output_path.absolute()}")


@pytest.mark.drawing_functions
def test_draw_with_labels(output_dir):
    """Test drawing with section labels."""
    # Create a bar model with labels
    stimulus = BarModelStimulus(
        model_type="single_bar",
        single_bar=BarModel(
            total_value=15,
            total_label="15 candies",
            sections=[
                BarSection(value=5, label="Eaten", highlighted=True),
                BarSection(value=5, label="Shared"),
                BarSection(value=5, label="Left"),
            ],
            problem_context="candies",
        ),
    )

    # Draw and get the output path
    output_path = draw_bar_model_stimulus(stimulus)
    output_path = Path(output_path)

    # Verify the file was created
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nImage saved to: {output_path.absolute()}")
