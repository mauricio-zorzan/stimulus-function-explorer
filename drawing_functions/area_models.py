import random
import time

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.area_model import (
    AreaModel,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.unit_squares import (
    UnitSquareDecomposition,
)
from content_generators.settings import settings


@stimulus_function
def create_area_model(stim_desc: AreaModel):
    try:
        dimensions = stim_desc.dimensions
        columns = dimensions.columns
        rows = dimensions.rows

        column_headers = stim_desc.headers.columns
        row_headers = stim_desc.headers.rows
        data = stim_desc.data

        # Calculate dynamic cell dimensions based on content
        max_content_length = max(
            max(len(str(item)) for row in data for item in row),
            max(len(str(header)) for header in column_headers),
            max(len(str(header)) for header in row_headers) if row_headers else 0,
        )

        # Adjust cell dimensions based on content
        base_width = 0.8
        col_width = max(base_width, base_width * (max_content_length / 4))
        row_height = 0.6

        # Calculate figure size based on actual table needs
        figure_width = col_width * (columns + 1)  # +1 for row headers
        figure_height = row_height * (rows + 1)  # +1 for column headers

        plt.figure(figsize=(figure_width, figure_height))

        # Calculate column widths - make row header column narrower and add spacing
        row_header_width = col_width * 0.3  # Narrow row headers
        spacing_width = col_width * 0.4  # Empty space between headers and data
        data_col_width = col_width  # Normal width for data columns

        # Create colWidths array: [row_header_width] + [data_col_width] * columns
        col_widths = [row_header_width] + [data_col_width] * columns

        table = plt.table(
            cellText=data,
            colLabels=column_headers,
            rowLabels=row_headers,
            cellLoc="center",
            loc="center",
            bbox=[0.15, 0.1, 0.8, 0.8],  # Revert to reasonable positioning
            colWidths=col_widths,
        )
        plt.gca().set_axis_off()
        table.auto_set_font_size(False)
        # Adjust font size based on cell width
        font_size = min(16, max(8, int(col_width * 20)))
        table.set_fontsize(font_size)

        for key, cell in table.get_celld().items():
            cell.set_height(row_height)

            # Add extra spacing by adjusting cell positions
            if key[1] == -1:  # Row header cells
                cell.set_width(row_header_width)
                # Move row headers further left
                cell.set_x(cell.get_x() - spacing_width / 2)
            else:
                cell.set_width(data_col_width)
                # Move data cells further right
                cell.set_x(cell.get_x() + spacing_width / 2)

            cell.set_text_props(va="center", linespacing=0.01)
            cell.PAD = 0.05  # Standard padding for all cells

            if key[0] == 0 or key[1] == -1:
                cell.set_edgecolor("none")
            if isinstance(
                cell.get_text().get_text(), str
            ) and cell.get_text().get_text().upper() in {chr(i) for i in range(65, 91)}:
                cell.get_text().set_color("blue")

        plt.tight_layout(pad=0.2)  # Add some padding around the entire figure

        file_name = f"{settings.additional_content_settings.image_destination_folder}/area_model_{int(time.time_ns())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            pad_inches=0,
            dpi=800,
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()
        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def unit_square_decomposition(stim_desc: UnitSquareDecomposition) -> str:
    """
    Draw a unit square decomposition with a grid of squares and shaded areas.

    Creates a grid divided into unit squares (or rectangles) with some squares filled in.
    The filled squares always form a perfect rectangle (no leftover squares).
    The rectangle dimensions are randomly chosen from all possible factorizations
    of the filled_count that fit within the available space. The color of the filled
    squares is also randomly selected. Always leaves 2 rows at the bottom and
    2 columns at the right empty.

    Args:
        stim_desc: UnitSquareDecomposition stimulus description containing size, filled_count,
                  and optionally rectangle_tiles and height

    Returns:
        str: Path to the saved image file
    """
    grid_width = stim_desc.size
    grid_height = (
        stim_desc.height
        if stim_desc.rectangle_tiles and stim_desc.height
        else stim_desc.size
    )
    filled_count = stim_desc.filled_count

    # Ensure we always leave 2 rows at bottom and 2 columns at right
    max_fillable_width = grid_width - 2
    max_fillable_height = grid_height - 2
    max_fillable_area = max_fillable_width * max_fillable_height

    # Limit filled_count to the maximum fillable area
    filled_count = min(filled_count, max_fillable_area)

    # Find all possible rectangle dimensions that fit exactly
    possible_rectangles = []
    for width in range(1, max_fillable_width + 1):
        if filled_count % width == 0:  # Check if width divides evenly
            height = filled_count // width
            if height <= max_fillable_height:
                possible_rectangles.append((width, height))

    # If no perfect rectangles fit, find the largest rectangle that does fit
    if not possible_rectangles:
        # Find largest rectangle that fits in the fillable area
        best_area = 0
        best_rect = (1, 1)
        for width in range(1, max_fillable_width + 1):
            for height in range(1, max_fillable_height + 1):
                area = width * height
                if area <= filled_count and area > best_area:
                    best_area = area
                    best_rect = (width, height)
        possible_rectangles = [best_rect]
        filled_count = best_area  # Adjust filled_count to match the rectangle

    # Randomly choose between portrait and landscape orientations
    # Filter rectangles by orientation and randomly pick one
    portrait_rectangles = [
        (w, h) for w, h in possible_rectangles if h > w
    ]  # Tall rectangles
    landscape_rectangles = [
        (w, h) for w, h in possible_rectangles if w > h
    ]  # Wide rectangles
    square_rectangles = [
        (w, h) for w, h in possible_rectangles if w == h
    ]  # Square rectangles

    # Randomly choose orientation, preferring portrait/landscape over square
    available_orientations = []
    if portrait_rectangles:
        available_orientations.append(("portrait", portrait_rectangles))
    if landscape_rectangles:
        available_orientations.append(("landscape", landscape_rectangles))
    if (
        square_rectangles and not available_orientations
    ):  # Only use square if no other options
        available_orientations.append(("square", square_rectangles))

    if available_orientations:
        orientation, rectangles = random.choice(available_orientations)
        rect_width, rect_height = random.choice(rectangles)
    else:
        # Fallback to any rectangle if no orientation-specific ones exist
        rect_width, rect_height = random.choice(possible_rectangles)

    # Randomly choose a color for the filled squares
    colors = [
        "mediumpurple",
        "blue",
        "green",
        "lightcoral",
        "pink",
        "cyan",
        "salmon",
    ]
    fill_color = random.choice(colors)

    # Determine cell dimensions
    if stim_desc.rectangle_tiles:
        # Use fixed 1.5:1 aspect ratio for rectangular tiles
        cell_width = 1.5
        cell_height = 1.0
        # Adjust figure size based on grid dimensions and tile sizes
        total_width = grid_width * cell_width
        total_height = grid_height * cell_height
        fig_width = max(8, total_width * 0.8)
        fig_height = max(8, total_height * 0.8)
    else:
        cell_width = 1.0
        cell_height = 1.0
        fig_width = 8
        fig_height = 8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw the grid lines
    for i in range(grid_width + 1):
        # Vertical lines
        ax.plot(
            [i * cell_width, i * cell_width],
            [0, grid_height * cell_height],
            "black",
            linewidth=2,
        )
    for i in range(grid_height + 1):
        # Horizontal lines
        ax.plot(
            [0, grid_width * cell_width],
            [(grid_height - i) * cell_height, (grid_height - i) * cell_height],
            "black",
            linewidth=2,
        )

    # Fill the rectangle of squares (always a perfect rectangle)
    # Always start filling from position (0,0) - top-left corner
    filled_squares = set()
    for row in range(rect_height):
        for col in range(rect_width):
            filled_squares.add((row, col))

    # Draw the filled squares
    for row, col in filled_squares:
        x_left = col * cell_width
        x_right = (col + 1) * cell_width
        y_bottom = (grid_height - row - 1) * cell_height
        y_top = (grid_height - row) * cell_height
        ax.fill(
            [x_left, x_right, x_right, x_left],
            [y_bottom, y_bottom, y_top, y_top],
            color=fill_color,
            alpha=0.8,
        )

    # Add unit labels on top and left sides
    total_width = grid_width * cell_width
    total_height = grid_height * cell_height

    # Always use "1 unit" labels regardless of tile dimensions
    width_label = "1 unit"
    height_label = "1 unit"

    # Top label (shows tile width)
    ax.text(
        total_width / 2,
        total_height + 0.5,
        width_label,
        fontsize=18,
        color="black",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Left label (shows tile height)
    ax.text(
        -0.8,
        total_height / 2,
        height_label,
        fontsize=18,
        color="black",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Add bracket lines for the labels
    # Top bracket
    bracket_height = 0.2
    ax.plot(
        [0, 0],
        [total_height + bracket_height, total_height + 0.1],
        "black",
        linewidth=1.5,
    )
    ax.plot(
        [total_width, total_width],
        [total_height + bracket_height, total_height + 0.1],
        "black",
        linewidth=1.5,
    )
    ax.plot(
        [0, total_width],
        [total_height + bracket_height, total_height + bracket_height],
        "black",
        linewidth=1.5,
    )

    # Left bracket
    bracket_width = 0.2
    ax.plot([-bracket_width, -0.1], [0, 0], "black", linewidth=1.5)
    ax.plot(
        [-bracket_width, -0.1], [total_height, total_height], "black", linewidth=1.5
    )
    ax.plot([-bracket_width, -bracket_width], [0, total_height], "black", linewidth=1.5)

    # Set axis limits with some padding
    ax.set_xlim(-1, total_width + 1)
    ax.set_ylim(-0.5, total_height + 1)

    plt.tight_layout()

    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/unit_square_decomposition_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name
