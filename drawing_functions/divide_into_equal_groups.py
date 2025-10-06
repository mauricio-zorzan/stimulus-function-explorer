import math
import random
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.divide_into_equal_groups import (
    DivideIntoEqualGroups,
)
from content_generators.settings import settings

# Standard colors for the visualization
STANDARD_COLORS = [
    "#FF0000",  # Red
    "#0000FF",  # Blue
    "#00FF00",  # Green
    "#FFA500",  # Orange
    "#800080",  # Purple
]


@stimulus_function
def draw_divide_into_equal_groups(stimulus_description: DivideIntoEqualGroups) -> str:
    """
    Draw a visual representation of dots divided into equal groups.
    
    Args:
        stimulus_description: The stimulus description containing number_of_dots_per_group and number_of_groups
        
    Returns:
        str: The filename of the generated image
    """
    dots_per_group = stimulus_description.number_of_dots_per_group
    number_of_groups = stimulus_description.number_of_groups
    
    # Choose one random color for the entire image
    color = random.choice(STANDARD_COLORS)
    
    # Calculate layout for big circles with maximum 3 per row
    cols = min(3, number_of_groups)
    rows = math.ceil(number_of_groups / cols)
    
    # Create figure with appropriate size
    fig_width = max(8, cols * 4)
    fig_height = max(6, rows * 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set up the plot
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Big circle radius
    big_circle_radius = 0.4
    
    # Draw each group
    for group_idx in range(number_of_groups):
        # Calculate position for this group's big circle
        row = group_idx // cols
        col = group_idx % cols
        
        # Center position of the big circle
        center_x = col + 0.5
        center_y = rows - row - 0.5
        
        # Use the same color for all groups
        
        # Draw the big circle (outline only) - always dark gray
        big_circle = patches.Circle(
            (center_x, center_y),
            big_circle_radius,
            fill=False,
            edgecolor='#666666',
            linewidth=3
        )
        ax.add_patch(big_circle)
        
        # Calculate positions for small circles inside the big circle
        small_circle_radius = 0.06
        
        # Arrange small circles in a grid pattern inside the big circle
        if dots_per_group <= 4:
            # For small numbers, use a simple arrangement
            positions = _get_simple_positions(dots_per_group, big_circle_radius)
        else:
            # For larger numbers (5-10), use a grid arrangement
            positions = _get_grid_positions(dots_per_group, big_circle_radius, small_circle_radius)
        
        # Draw small circles (filled)
        for pos_x, pos_y in positions:
            small_circle = patches.Circle(
                (center_x + pos_x, center_y + pos_y),
                small_circle_radius,
                fill=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1
            )
            ax.add_patch(small_circle)
    
    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/divide_into_equal_groups_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
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


def _get_simple_positions(count, radius):
    """Get positions for small numbers of dots in a simple arrangement."""
    positions = []
    if count == 1:
        positions = [(0, 0)]
    elif count == 2:
        positions = [(-0.1, 0), (0.1, 0)]
    elif count == 3:
        positions = [(-0.1, 0.1), (0.1, 0.1), (0, -0.1)]
    elif count == 4:
        positions = [(-0.1, 0.1), (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)]
    
    return positions


def _get_grid_positions(count, big_radius, small_radius):
    """Get positions for dots arranged in a grid inside the big circle."""
    positions = []
    
    # Use consistent spacing (same as simple positions: 0.2 between adjacent dots)
    CONSISTENT_SPACING = 0.2
    
    # Handle specific layouts for better visual appearance
    if count == 5:
        # Top row: 3 dots, bottom row: 2 dots (centered)
        positions.extend([(-0.2, 0.1), (0, 0.1), (0.2, 0.1)])
        positions.extend([(-0.1, -0.1), (0.1, -0.1)])
    elif count == 6:
        # Two rows of 3 dots each
        positions.extend([(-0.2, 0.1), (0, 0.1), (0.2, 0.1)])
        positions.extend([(-0.2, -0.1), (0, -0.1), (0.2, -0.1)])
    elif count == 7:
        # Top row: 3 dots, middle row: 3 dots, bottom row: 1 dot
        positions.extend([(-0.2, 0.15), (0, 0.15), (0.2, 0.15)])
        positions.extend([(-0.2, 0), (0, 0), (0.2, 0)])
        positions.extend([(0, -0.15)])
    elif count == 8:
        # Top row: 3 dots, middle row: 3 dots, bottom row: 2 dots
        positions.extend([(-0.2, 0.15), (0, 0.15), (0.2, 0.15)])
        positions.extend([(-0.2, 0), (0, 0), (0.2, 0)])
        positions.extend([(-0.1, -0.15), (0.1, -0.15)])
    elif count == 9:
        # Three rows of 3 dots each
        positions.extend([(-0.2, 0.15), (0, 0.15), (0.2, 0.15)])
        positions.extend([(-0.2, 0), (0, 0), (0.2, 0)])
        positions.extend([(-0.2, -0.15), (0, -0.15), (0.2, -0.15)])
    elif count == 10:
        # Top row: 3 dots, second row: 3 dots, third row: 3 dots, bottom row: 1 dot
        positions.extend([(-0.2, 0.2), (0, 0.2), (0.2, 0.2)])
        positions.extend([(-0.2, 0.05), (0, 0.05), (0.2, 0.05)])
        positions.extend([(-0.2, -0.1), (0, -0.1), (0.2, -0.1)])
        positions.extend([(0, -0.25)])
    else:
        # For other counts, calculate grid layout with consistent spacing
        dots_per_row = min(3, count)  # Maximum 3 dots per row
        rows_needed = math.ceil(count / dots_per_row)
        
        for i in range(count):
            row = i // dots_per_row
            col = i % dots_per_row
            
            # Calculate how many dots are in this row
            dots_in_current_row = min(dots_per_row, count - row * dots_per_row)
            
            # Calculate horizontal position using consistent spacing
            if dots_in_current_row == 1:
                x = 0
            else:
                # Use consistent spacing, but center the row if it has fewer dots
                total_row_width = (dots_in_current_row - 1) * CONSISTENT_SPACING
                row_start = -total_row_width / 2
                x = row_start + col * CONSISTENT_SPACING
            
            # Calculate vertical position using consistent spacing
            if rows_needed == 1:
                y = 0
            else:
                total_height = (rows_needed - 1) * CONSISTENT_SPACING
                y = total_height / 2 - row * CONSISTENT_SPACING
            
            positions.append((x, y))
    
    return positions
