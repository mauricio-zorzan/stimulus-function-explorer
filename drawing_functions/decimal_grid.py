import random
import time

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.decimal_grid import (
    DecimalComparison,
    DecimalComparisonList,
    DecimalGrid,
    DecimalMultiplication,
)
from content_generators.settings import settings
from matplotlib.patches import Rectangle


def get_random_color_pair():
    """
    Returns a random color pair (base_color, darker_edge_color) for visual elements.
    
    This helper function provides consistent color theming across different drawing functions.
    The darker color can be used for edges/borders to create better visual contrast.
    
    Returns:
        tuple: (base_color, darker_edge_color) where both are matplotlib color names
    """
    color_palette = [
        ("lightblue", "steelblue"),
        ("lightgreen", "forestgreen"), 
        ("lightcoral", "darkred"),
        ("wheat", "goldenrod"),
        ("lightgray", "dimgray")
    ]
    
    return random.choice(color_palette)


@stimulus_function
def draw_decimal_grid(stimulus: DecimalGrid):
    """
    Draw decimal grids with shaded squares.

    For division=10: Creates rectangles divided into 10 parts each
    For division=100: Creates 10x10 grids

    Multiple grids are created when shaded_squares exceeds division value.
    Supports up to 10 grids arranged in rows (max 5 grids per row).
    """
    # Calculate number of grids needed
    full_grids = stimulus.shaded_squares // stimulus.division
    remaining_shaded = stimulus.shaded_squares % stimulus.division
    total_grids = full_grids + (1 if remaining_shaded > 0 else 0)

    # If no shaded squares, still show one empty grid
    if total_grids == 0:
        total_grids = 1

    # Cap at 10 grids maximum
    total_grids = min(total_grids, 10)

    # Get random color pair for this stimulus
    shaded_color, edge_color = get_random_color_pair()

    # Calculate grid layout (max 5 grids per row)
    grids_per_row = min(total_grids, 5)
    grid_rows = (total_grids + grids_per_row - 1) // grids_per_row  # Ceiling division

    # Adjust figure size based on grid layout and division type
    if stimulus.division == 10:
        fig_width = max(8, grids_per_row * 2.5)
        fig_height = max(4, grid_rows * 2.5)
    else:  # division == 100
        fig_width = max(8, grids_per_row * 2.5)
        fig_height = max(6, grid_rows * 3.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.axis("off")

    # Determine grid dimensions based on division
    if stimulus.division == 10:
        square_width = 1.0
        square_height = 1.0
        rect_width = square_width / 10
        rect_height = square_height
        grid_spacing_x = 1.3  # Space between grids horizontally
        grid_spacing_y = 1.5  # Space between grids vertically

        for grid_idx in range(total_grids):
            # Calculate grid position in layout
            grid_row = grid_idx // grids_per_row
            grid_col = grid_idx % grids_per_row
            
            grid_x_offset = grid_col * grid_spacing_x
            grid_y_offset = (grid_rows - 1 - grid_row) * grid_spacing_y  # Start from top

            # Determine how many squares to shade in this grid
            if grid_idx < full_grids:
                squares_to_shade = stimulus.division  # Full grid
            elif grid_idx == full_grids and remaining_shaded > 0:
                squares_to_shade = remaining_shaded  # Partial grid
            else:
                squares_to_shade = 0  # Empty grid

            for i in range(10):
                x = grid_x_offset + i * rect_width
                y = grid_y_offset

                # Determine if this rectangle should be shaded
                is_shaded = i < squares_to_shade

                # Choose colors
                facecolor = shaded_color if is_shaded else "white"
                edgecolor = edge_color  # Use same edge color for all squares

                # Draw rectangle
                rect = Rectangle(
                    (x, y),
                    rect_width,
                    rect_height,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.5,
                )
                ax.add_patch(rect)

        # Set plot limits with some padding
        padding = 0.2
        total_width = (grids_per_row - 1) * grid_spacing_x + square_width
        total_height = (grid_rows - 1) * grid_spacing_y + square_height
        ax.set_xlim(-padding, total_width + padding)
        ax.set_ylim(-padding, total_height + padding)

    elif stimulus.division == 100:
        # Use 10x10 grid
        rows, cols = 10, 10
        square_size = 0.4
        grid_spacing_x = 4.5  # Space between grids horizontally
        grid_spacing_y = 4.5  # Space between grids vertically

        for grid_idx in range(total_grids):
            # Calculate grid position in layout
            grid_row = grid_idx // grids_per_row
            grid_col = grid_idx % grids_per_row
            
            grid_x_offset = grid_col * grid_spacing_x
            grid_y_offset = (grid_rows - 1 - grid_row) * grid_spacing_y  # Start from top

            # Determine how many squares to shade in this grid
            if grid_idx < full_grids:
                squares_to_shade = stimulus.division  # Full grid
            elif grid_idx == full_grids and remaining_shaded > 0:
                squares_to_shade = remaining_shaded  # Partial grid
            else:
                squares_to_shade = 0  # Empty grid

            # Draw all squares for this grid
            for row in range(rows):
                for col in range(cols):
                    x = grid_x_offset + col * square_size
                    y = grid_y_offset + (rows - 1 - row) * square_size  # Start from top

                    # Calculate square index (column by column, left to right)
                    square_index = col * rows + row

                    # Determine if this square should be shaded
                    is_shaded = square_index < squares_to_shade

                    # Choose colors
                    facecolor = shaded_color if is_shaded else "white"
                    edgecolor = edge_color  # Use same edge color for all squares

                    # Draw rectangle
                    rect = Rectangle(
                        (x, y),
                        square_size,
                        square_size,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)

        # Set plot limits with some padding
        padding = 0.1
        total_width = (grids_per_row - 1) * grid_spacing_x + cols * square_size
        total_height = (grid_rows - 1) * grid_spacing_y + rows * square_size
        ax.set_xlim(-padding, total_width + padding)
        ax.set_ylim(-padding, total_height + padding)

    else:
        raise ValueError(
            f"Invalid division value: {stimulus.division}. Must be 10 or 100."
        )

    plt.tight_layout()

    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/decimal_grid_{stimulus.division}_{stimulus.shaded_squares}_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_decimal_comparison(stimulus: DecimalComparisonList):
    """
    Draw decimal comparison problems with side-by-side grids and questions.

    Creates visual models for comparing decimals with proper question format
    matching CCSS.MATH.CONTENT.4.NF.C.7+3 requirements.
    """
    num_comparisons = len(stimulus.comparisons)
    fig_height = 8 * num_comparisons
    fig, axes = plt.subplots(num_comparisons, 1, figsize=(14, fig_height))

    if num_comparisons == 1:
        axes = [axes]

    for idx, comparison in enumerate(stimulus.comparisons):
        ax = axes[idx]
        draw_comparison_pair(ax, comparison)

    plt.tight_layout()

    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/decimal_comparison_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def draw_comparison_pair(ax, comparison: DecimalComparison):
    """Draw a single decimal comparison with two grids only."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Determine grid type based on decimal precision
    decimal_1_str = f"{comparison.decimal_1:.2f}".rstrip("0").rstrip(".")
    decimal_2_str = f"{comparison.decimal_2:.2f}".rstrip("0").rstrip(".")

    decimal_1_places = len(decimal_1_str.split(".")[-1]) if "." in decimal_1_str else 0
    decimal_2_places = len(decimal_2_str.split(".")[-1]) if "." in decimal_2_str else 0
    max_places = max(decimal_1_places, decimal_2_places)

    # Use 10-division for 1 decimal place, 100-division for 2 decimal places
    division = 10 if max_places <= 1 else 100

    # Draw left grid (larger and moved closer to center)
    left_center_x = 0.25
    grid_y = 0.55  # Move grids up slightly
    draw_single_grid_at_position(
        ax, comparison.decimal_1, division, left_center_x, grid_y, comparison.color_1
    )

    # Draw right grid (larger and moved closer to center)
    right_center_x = 0.75
    draw_single_grid_at_position(
        ax, comparison.decimal_2, division, right_center_x, grid_y, comparison.color_2
    )

    # Add decimal labels below grids (positioned closer to grids)
    ax.text(
        left_center_x,
        0.25,  # Moved closer to grids
        f"{comparison.decimal_1:.2f}",
        ha="center",
        va="center",
        fontsize=26,  # Slightly larger font
        fontweight="bold",
    )
    ax.text(
        right_center_x,
        0.25,  # Moved closer to grids
        f"{comparison.decimal_2:.2f}",
        ha="center",
        va="center",
        fontsize=26,  # Slightly larger font
        fontweight="bold",
    )


def draw_single_grid_at_position(
    ax,
    decimal_value: float,
    division: int,
    center_x: float,
    center_y: float,
    color: str,
):
    """Draw a single decimal grid at specified position with larger size."""
    decimal_part = decimal_value - int(decimal_value)  # Remove whole number part
    shaded_squares = int(decimal_part * division)

    if division == 10:
        # 1x10 grid (enlarged further)
        grid_width = 0.45  # Increased from 0.4
        grid_height = 0.25  # Increased from 0.2
        rect_width = grid_width / 10
        rect_height = grid_height

        start_x = center_x - grid_width / 2
        start_y = center_y - grid_height / 2

        for i in range(10):
            x = start_x + i * rect_width
            y = start_y

            is_shaded = i < shaded_squares
            facecolor = color if is_shaded else "white"

            rect = Rectangle(
                (x, y),
                rect_width,
                rect_height,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=2,  # Thicker lines
            )
            ax.add_patch(rect)

    elif division == 100:
        # 10x10 grid (enlarged further)
        grid_size = 0.35  # Increased from 0.3
        cell_size = grid_size / 10

        start_x = center_x - grid_size / 2
        start_y = center_y - grid_size / 2

        for row in range(10):
            for col in range(10):
                x = start_x + col * cell_size
                y = start_y + (9 - row) * cell_size  # Start from top

                # Calculate square index (left to right, top to bottom)
                square_index = row * 10 + col

                is_shaded = square_index < shaded_squares
                facecolor = color if is_shaded else "white"

                rect = Rectangle(
                    (x, y),
                    cell_size,
                    cell_size,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1.5,  # Slightly thicker lines
                )
                ax.add_patch(rect)


@stimulus_function
def draw_decimal_multiplication(stimulus: DecimalMultiplication):
    """
    Draw a decimal multiplication visualization for 0.a × 0.b or 0.a × 1.b.
    
    For 0.a × 0.b (single grid):
    1. Second factor (0.b) is represented by shading b columns
    2. First factor (0.a) is represented by adding diagonal lines to a rows of the shaded area
    
    For 0.a × 1.b (two grids):
    1. Second factor (1.b) is represented by shading entire first grid + b columns of second grid
    2. First factor (0.a) is represented by adding diagonal lines to a rows of all shaded areas
    """
    factor1, factor2 = stimulus.decimal_factors
    
    # Convert to appropriate values
    a = int(round(factor1 * 10))  # First factor (will determine rows with pattern)
    
    # Determine if we need one or two grids
    needs_two_grids = factor2 >= 1.0
    
    if needs_two_grids:
        # For 1.b format: decimal part is b
        decimal_part = factor2 - 1.0
        b = int(round(decimal_part * 10))  # Columns in second grid
    else:
        # For 0.b format
        b = int(round(factor2 * 10))  # Columns in first grid
    
    # Define color palette with base colors and their darker variants
    color_palette = [
        ("lightblue", "darkblue"),
        ("lightgreen", "darkgreen"), 
        ("lightcoral", "darkred"),
        ("lightyellow", "orange"),
        ("lightpink", "purple"),
    ]
    
    # Randomly select a color pair
    base_color, product_color = random.choice(color_palette)
    
    # Adjust figure size based on number of grids
    if needs_two_grids:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Grid parameters
    rows, cols = 10, 10
    cell_size = 0.7
    start_x, start_y = 0.5, 0.5
    
    pattern_rows = a  # Number of rows to get diagonal pattern for first factor (0.a)
    
    if needs_two_grids:
        # Draw two grids side by side with no spacing
        for grid_idx in range(2):
            grid_x_offset = grid_idx * cols * cell_size  # No spacing between grids
            
            for row in range(rows):
                for col in range(cols):
                    x = start_x + grid_x_offset + col * cell_size
                    y = start_y + (rows - 1 - row) * cell_size  # Start from top
                    
                    # Determine shading based on grid position
                    if grid_idx == 0:
                        # First grid: always fully shaded (representing the "1" part of 1.b)
                        is_base_shaded = True
                    else:
                        # Second grid: shade first b columns (representing the "0.b" part of 1.b)
                        is_base_shaded = col < b
                    
                    # Add diagonal pattern to first a rows of shaded areas
                    is_product_shaded = row < pattern_rows and is_base_shaded
                    
                    # Choose base color
                    facecolor = base_color if is_base_shaded else "white"
                    
                    # Draw the base rectangle with standard border
                    rect = Rectangle(
                        (x, y),
                        cell_size,
                        cell_size,
                        facecolor=facecolor,
                        edgecolor=product_color,
                        linewidth=1.5,
                    )
                    ax.add_patch(rect)
                    
                    # Add diagonal pattern for product area
                    if is_product_shaded:
                        _add_diagonal_pattern(ax, x, y, cell_size, product_color)
        
        # Add a vertical separator line between the two grids
        separator_x = start_x + cols * cell_size
        ax.plot([separator_x, separator_x], 
                [start_y, start_y + rows * cell_size], 
                color=product_color, linewidth=4.0, alpha=1.0)
        
        # Add thicker outer borders for the combined grids
        grid_width = 2 * cols * cell_size
        grid_height = rows * cell_size
        
        # Top border
        ax.plot([start_x, start_x + grid_width], 
                [start_y + grid_height, start_y + grid_height], 
                color=product_color, linewidth=2.5, alpha=1.0)
        # Bottom border
        ax.plot([start_x, start_x + grid_width], 
                [start_y, start_y], 
                color=product_color, linewidth=2.5, alpha=1.0)
        # Left border
        ax.plot([start_x, start_x], 
                [start_y, start_y + grid_height], 
                color=product_color, linewidth=2.5, alpha=1.0)
        # Right border
        ax.plot([start_x + grid_width, start_x + grid_width], 
                [start_y, start_y + grid_height], 
                color=product_color, linewidth=2.5, alpha=1.0)
        
        # Set plot limits with padding for two grids (no spacing)
        padding = 0.2
        total_width = 2 * cols * cell_size
        ax.set_xlim(start_x - padding, start_x + total_width + padding)
        ax.set_ylim(start_y - padding, start_y + rows * cell_size + padding)
        
    else:
        # Draw single grid (original logic for 0.a × 0.b)
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * cell_size
                y = start_y + (rows - 1 - row) * cell_size  # Start from top
                
                # Determine shading
                is_base_shaded = col < b  # Shade first b columns
                is_product_shaded = row < pattern_rows and is_base_shaded  # Pattern on first a rows, only where base shaded
                
                # Choose base color
                facecolor = base_color if is_base_shaded else "white"
                
                # Draw the base rectangle
                rect = Rectangle(
                    (x, y),
                    cell_size,
                    cell_size,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1.5,
                )
                ax.add_patch(rect)
                
                # Add diagonal pattern for product area
                if is_product_shaded:
                    _add_diagonal_pattern(ax, x, y, cell_size, product_color)
        
        # Add thicker outer borders for the single grid
        grid_width = cols * cell_size
        grid_height = rows * cell_size
        
        # Top border
        ax.plot([start_x, start_x + grid_width], 
                [start_y + grid_height, start_y + grid_height], 
                color=product_color, linewidth=2.5, alpha=1.0)
        # Bottom border
        ax.plot([start_x, start_x + grid_width], 
                [start_y, start_y], 
                color=product_color, linewidth=2.5, alpha=1.0)
        # Left border
        ax.plot([start_x, start_x], 
                [start_y, start_y + grid_height], 
                color=product_color, linewidth=2.5, alpha=1.0)
        # Right border
        ax.plot([start_x + grid_width, start_x + grid_width], 
                [start_y, start_y + grid_height], 
                color=product_color, linewidth=2.5, alpha=1.0)
        
        # Set plot limits with padding for single grid
        padding = 0.2
        ax.set_xlim(start_x - padding, start_x + cols * cell_size + padding)
        ax.set_ylim(start_y - padding, start_y + rows * cell_size + padding)
    
    plt.tight_layout()
    
    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/decimal_multiplication_{factor1}_{factor2}_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    
    return file_name


def _add_diagonal_pattern(ax, x, y, cell_size, product_color):
    """Helper function to add diagonal pattern to a cell."""
    line_spacing = cell_size / 8  # Density of diagonal lines
    for i in range(int(cell_size / line_spacing * 2)):
        # Diagonal lines from bottom-left to top-right
        x_start = x + i * line_spacing
        y_start = y
        x_end = x
        y_end = y + i * line_spacing
        
        if x_start <= x + cell_size and y_end <= y + cell_size:
            ax.plot([x_start, x_end], [y_start, y_end], 
                   color=product_color, linewidth=2, alpha=0.8)
        
        # Additional lines to fill the square completely
        x_start = x + cell_size
        y_start = y + i * line_spacing
        x_end = x + i * line_spacing
        y_end = y + cell_size
        
        if y_start <= y + cell_size and x_end >= x:
            ax.plot([x_start, x_end], [y_start, y_end], 
                   color=product_color, linewidth=2, alpha=0.8)
