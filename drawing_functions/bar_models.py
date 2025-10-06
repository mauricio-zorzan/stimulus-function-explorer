import time
from typing import List

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.bar_model_stimulus import (
    BarModel,
    BarModelStimulus,
    BarSection,
    ComparisonBarModel,
)
from content_generators.settings import settings
from matplotlib.patches import Rectangle


def is_unknown_section(section: BarSection) -> bool:
    """Check if a section represents an unknown value (question mark)."""
    return section.value == "?" or (isinstance(section.value, str) and "?" in str(section.value))


def are_equal_parts(sections: List[BarSection]) -> bool:
    """Check if all sections represent equal parts (same numeric values)."""
    numeric_values = []
    for section in sections:
        if isinstance(section.value, (int, float)):
            numeric_values.append(section.value)
        elif section.value != "?" and isinstance(section.value, str):
            try:
                numeric_values.append(float(section.value))
            except ValueError:
                return False  # Non-numeric, non-question mark values mean different categories
    
    # If we have at least 2 numeric values and they're all equal, it's equal parts
    return len(numeric_values) >= 2 and len(set(numeric_values)) == 1


def get_section_color(section: BarSection, all_sections: List[BarSection], section_index: int) -> str:
    """
    Determine section color based on educational context:
    - Equal parts: Same color unless it's unknown (question mark = green highlight)
    - Different categories: Different colors for each section
    - Custom colors: Use section.color if explicitly set
    """
    # If custom color is set, use it
    if section.color:
        return section.color
    
    # If this section is unknown (question mark), highlight it in green
    if is_unknown_section(section):
        return "#E8F5E8"  # Light green for unknowns
    
    # Check if this is an equal parts scenario
    if are_equal_parts(all_sections):
        # All equal parts should be the same color (light blue)
        return "#F0F8FF"
    
    # Different categories get different colors
    category_colors = [
        "#FFE4E1",  # Light pink
        "#E0F6FF",  # Light sky blue  
        "#F0FFF0",  # Light mint
        "#FFF8DC",  # Light cream
        "#E6E6FA",  # Light lavender
        "#FFEFD5",  # Light peach
    ]
    
    return category_colors[section_index % len(category_colors)]


@stimulus_function
def draw_bar_model_stimulus(stimulus: BarModelStimulus) -> str:
    """
    Draw a bar model stimulus and return the file path.

    Returns:
        str: Full path of the saved image
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))

    if stimulus.model_type == "single_bar":
        draw_single_bar(ax, stimulus.single_bar)
    else:
        draw_comparison_bars(ax, stimulus.comparison_bars)

    if stimulus.model_type == "single_bar" and stimulus.single_bar.title:
        ax.set_title(stimulus.single_bar.title)
    elif stimulus.model_type == "comparison_bars" and stimulus.comparison_bars.title:
        ax.set_title(stimulus.comparison_bars.title)

    ax.text(
        0.02, 0.02, stimulus.not_to_scale_note,
        transform=ax.transAxes, fontsize=8, style="italic"
    )
    ax.set_axis_off()
    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/bar_model_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)
    return file_name


def draw_single_bar(ax: plt.Axes, bar_model: BarModel) -> None:
    """Draw a single bar model with compact tape-diagram proportions."""
    # Compact bar dimensions - narrow like tape diagrams
    bar_width = 2.5
    bar_height = 0.6
    start_x = 1.0  # Center the bar
    start_y = 1.2
    
    # Calculate section widths
    if bar_model.section_width_mode == "proportional" and all(
        isinstance(s.value, (int, float)) for s in bar_model.sections
    ):
        total = sum(
            s.value for s in bar_model.sections if isinstance(s.value, (int, float))
        )
        section_widths = [(s.value / total) * bar_width for s in bar_model.sections]
    else:
        section_widths = [bar_width / len(bar_model.sections)] * len(bar_model.sections)

    # Draw sections
    x = start_x
    for i, (section, width) in enumerate(zip(bar_model.sections, section_widths)):
        # Smart color logic based on educational context
        section_color = get_section_color(section, bar_model.sections, i)
        
        # Draw section rectangle
        rect = Rectangle(
            (x, start_y),
            width,
            bar_height,
            facecolor=section_color,
            edgecolor="black",
            linewidth=2,
            alpha=0.7 if is_unknown_section(section) else 0.5,
        )
        ax.add_patch(rect)

        # Add section value
        if bar_model.show_section_values and section.value is not None:
            ax.text(
                x + width / 2, 
                start_y + bar_height / 2, 
                str(section.value), 
                ha="center", 
                va="center",
                fontsize=10
            )

        # Add section label below the bar
        if section.label:
            ax.text(
                x + width / 2, 
                start_y - 0.15, 
                section.label, 
                ha="center", 
                va="top",
                fontsize=9
            )

        x += width

    # Add bracket above the bar to show total span
    bracket_y = start_y + bar_height + 0.1
    ax.plot([start_x, start_x + bar_width], [bracket_y, bracket_y], 'k-', linewidth=1)
    ax.plot([start_x, start_x], [bracket_y, bracket_y - 0.05], 'k-', linewidth=1)
    ax.plot([start_x + bar_width, start_x + bar_width], [bracket_y, bracket_y - 0.05], 'k-', linewidth=1)

    # Add total value above the bracket
    if bar_model.show_total_value and bar_model.total_value is not None:
        ax.text(
            start_x + bar_width / 2, 
            bracket_y + 0.15, 
            str(bar_model.total_value), 
            ha="center", 
            va="bottom",
            fontsize=10
        )

    # Add total label above the total value
    if bar_model.total_label:
        ax.text(
            start_x + bar_width / 2, 
            bracket_y + 0.35, 
            bar_model.total_label, 
            ha="center", 
            va="bottom",
            fontsize=11
        )

    # Set compact plot limits
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.8, 2.2)


def draw_comparison_bars(ax: plt.Axes, comparison: ComparisonBarModel) -> None:
    """Draw comparison bars with compact tape-diagram proportions."""
    num_bars = len(comparison.bars)
    bar_width = 2.5
    bar_height = 0.5
    vertical_spacing = 0.8
    start_x = 1.0

    for i, bar_model in enumerate(comparison.bars):
        y = i * vertical_spacing

        # Calculate section widths
        if bar_model.section_width_mode == "proportional" and all(
            isinstance(s.value, (int, float)) for s in bar_model.sections
        ):
            total = sum(
                s.value for s in bar_model.sections if isinstance(s.value, (int, float))
            )
            section_widths = [(s.value / total) * bar_width for s in bar_model.sections]
        else:
            section_widths = [bar_width / len(bar_model.sections)] * len(bar_model.sections)

        # Draw sections
        x = start_x
        for section_idx, (section, width) in enumerate(zip(bar_model.sections, section_widths)):
            # Smart color logic based on educational context
            section_color = get_section_color(section, bar_model.sections, section_idx)
            
            rect = Rectangle(
                (x, y),
                width,
                bar_height,
                facecolor=section_color,
                edgecolor="black",
                linewidth=1,
                alpha=0.7 if is_unknown_section(section) else 0.5,
            )
            ax.add_patch(rect)

            # Add section value
            if bar_model.show_section_values and section.value is not None:
                ax.text(
                    x + width / 2,
                    y + bar_height / 2,
                    str(section.value),
                    ha="center",
                    va="center",
                    fontsize=9
                )

            # Add section label
            if section.label:
                ax.text(
                    x + width / 2, 
                    y - 0.1, 
                    section.label, 
                    ha="center", 
                    va="top",
                    fontsize=8
                )

            x += width

        # Add bracket above each bar
        bracket_y = y + bar_height + 0.08
        ax.plot([start_x, start_x + bar_width], [bracket_y, bracket_y], 'k-', linewidth=1)
        ax.plot([start_x, start_x], [bracket_y, bracket_y - 0.04], 'k-', linewidth=1)
        ax.plot([start_x + bar_width, start_x + bar_width], [bracket_y, bracket_y - 0.04], 'k-', linewidth=1)

        # Add total value above the bracket
        if bar_model.show_total_value and bar_model.total_value is not None:
            ax.text(
                start_x + bar_width / 2,
                bracket_y + 0.1,
                str(bar_model.total_value),
                ha="center",
                va="bottom",
                fontsize=9
            )

        # Add total label above the total value
        if bar_model.total_label:
            ax.text(
                start_x + bar_width / 2,
                bracket_y + 0.25,
                bar_model.total_label,
                ha="center",
                va="bottom",
                fontsize=10
            )

    # Set compact plot limits
    total_height = num_bars * vertical_spacing
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.2, total_height + 0.3)