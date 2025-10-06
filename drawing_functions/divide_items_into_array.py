import random
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.divide_items_into_array import (
    DivideItemsIntoArray,
)
from content_generators.settings import settings

# Standard colors for the visualization - matte shades
STANDARD_COLORS = [
    "#C65D4F",  # Matte Red
    "#4A7C8C",  # Matte Blue
    "#7A9A5A",  # Matte Green
    "#D4A574",  # Matte Orange
    "#8E7A9A",  # Matte Purple
]


@stimulus_function
def draw_divide_items_into_array(stimulus_description: DivideItemsIntoArray) -> str:
    """
    Draw a visual representation of items arranged in a rectangular array.
    
    Args:
        stimulus_description: The stimulus description containing num_rows and num_columns
        
    Returns:
        str: The filename of the generated image
    """
    num_rows = stimulus_description.num_rows
    num_columns = stimulus_description.num_columns
    
    # Choose one random color for the entire image
    color = random.choice(STANDARD_COLORS)
    
    # Row spacing factor - increased for more space between rows
    row_spacing = 1.05
    
    # Create figure with appropriate size
    fig_width = max(6, num_columns * 1.5)
    fig_height = max(4, num_rows * 1.5 * row_spacing)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set up the plot with some padding
    padding = 0.5
    ax.set_xlim(-padding, num_columns + padding)
    ax.set_ylim(-padding, num_rows * row_spacing + padding)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Circle radius
    circle_radius = 0.3
    
    # Draw row borders first (behind circles)
    for row in range(num_rows):
        # Calculate border position for this row
        border_y = (num_rows - row - 0.5) * row_spacing
        border_x = 0.5
        border_width = num_columns
        border_height = 1
        
        # Draw row border rectangle with rounded corners - same color as divide_into_equal_groups outer border
        row_border = patches.FancyBboxPatch(
            (border_x - 0.4, border_y - 0.4),
            border_width - 0.2,
            border_height - 0.2,
            boxstyle="round,pad=0.08",
            fill=False,
            edgecolor='#666666',
            linewidth=3
        )
        ax.add_patch(row_border)
    
    # Draw the array of circles
    for row in range(num_rows):
        for col in range(num_columns):
            # Calculate position (start from top-left, so invert y)
            center_x = col + 0.5
            center_y = (num_rows - row - 0.5) * row_spacing
            
            # Draw filled circle
            circle = patches.Circle(
                (center_x, center_y),
                circle_radius,
                fill=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1
            )
            ax.add_patch(circle)
    
    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/divide_items_into_array_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    
    return file_name 