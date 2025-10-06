"""Drawing function for fraction addition visual models."""

import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    FractionAdditionModel,
    FractionBar,
)
from content_generators.settings import settings


def draw_fraction_bar(ax, fraction: FractionBar, x_offset: float = 0, y_offset: float = 0):
    """Helper function to draw a single fraction bar showing only the parts being added.
    
    Args:
        ax: Matplotlib axis to draw on
        fraction: FractionBar object with numerator, denominator, and color
        x_offset: Horizontal offset for positioning
        y_offset: Vertical offset for positioning
    """
    bar_width = 3.0  # Total width of the bar
    bar_height = 0.5  # Height of the bar
    # Only draw cells for the numerator (parts being added)
    cell_width = bar_width / fraction.denominator
    
    # Draw only the numerator number of cells
    for i in range(fraction.numerator):
        # Draw the cell (all shaded since we only show the parts being added)
        ax.add_patch(
            patches.Rectangle(
                (x_offset + i * cell_width, y_offset),
                cell_width,
                bar_height,
                edgecolor="black",
                facecolor=fraction.color,
                linewidth=2,
            )
        )
        
        # Add fraction label in each cell
        center_x = x_offset + (i + 0.5) * cell_width
        center_y = y_offset + bar_height / 2
        
        # Draw numerator
        ax.text(
            center_x,
            center_y + 0.12,
            "1",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        
        # Draw fraction line
        line_half_width = min(0.08, cell_width * 0.3)
        ax.plot(
            [center_x - line_half_width, center_x + line_half_width],
            [center_y, center_y],
            "k-",
            linewidth=2,
        )
        
        # Draw denominator
        ax.text(
            center_x,
            center_y - 0.12,
            str(fraction.denominator),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )


@stimulus_function
def draw_fraction_addition_model(model_data: FractionAdditionModel) -> str:
    """
    Function to draw two fraction bars for visual fraction addition.
    Shows the original fractions that students use to solve addition problems.
    
    Args:
        model_data: FractionAdditionModel containing two fractions and display settings
        
    Returns:
        str: Path to the saved image file
    """
    fraction1 = model_data.fraction1
    fraction2 = model_data.fraction2
    
    if model_data.layout == "vertical":
        # Vertical stacking
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Draw first fraction bar
        draw_fraction_bar(ax, fraction1, x_offset=0, y_offset=1.5)
        
        # Draw plus sign if requested
        if model_data.show_plus_sign:
            ax.text(-0.5, 0.75, "+", fontsize=24, fontweight="bold", ha="center", va="center")
        
        # Draw second fraction bar
        draw_fraction_bar(ax, fraction2, x_offset=0, y_offset=0.5)
        
        # Draw divider line if requested
        if model_data.show_divider_line:
            ax.plot([0, 3], [0.2, 0.2], "k-", linewidth=3)
        
        # Set axis limits
        ax.set_xlim(-0.8, 3.3)
        ax.set_ylim(0, 2.2)
        
    else:  # horizontal
        # Side-by-side layout
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # Draw first fraction bar
        draw_fraction_bar(ax, fraction1, x_offset=0, y_offset=0.5)
        
        # Draw plus sign if requested
        if model_data.show_plus_sign:
            ax.text(3.5, 0.75, "+", fontsize=24, fontweight="bold", ha="center", va="center")
        
        # Draw second fraction bar
        draw_fraction_bar(ax, fraction2, x_offset=4, y_offset=0.5)
        
        # Set axis limits
        ax.set_xlim(-0.3, 7.3)
        ax.set_ylim(0.2, 1.3)
    
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    
    # Save the figure
    timestamp = int(time.time())
    dest = settings.additional_content_settings
    file_name = f"{dest.image_destination_folder}/{timestamp}_fraction_addition_model.{dest.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=650,
        transparent=False,
        bbox_inches="tight",
        format=dest.stimulus_image_format,
    )
    plt.close(fig)
    
    return file_name
