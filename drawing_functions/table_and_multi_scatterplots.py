import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.table_and_multi_scatterplots import (
    DataPoint,
    TableAndMultiScatterplots,
)
from content_generators.settings import settings


@stimulus_function
def create_table_and_multi_scatterplots(data: TableAndMultiScatterplots) -> str:
    """
    Creates a combined visualization with a table and multiple scatterplots.
    Supports both correct and incorrect scatterplot options for assessment purposes.

    Args:
        data: TableAndMultiScatterplots containing table data and scatterplot configurations

    Returns:
        str: Path to the generated image file
    """
    # Validate data consistency
    _validate_table_scatterplot_consistency(data)

    num_scatterplots = len(data.scatterplots)

    # Determine layout configuration
    if data.layout == "horizontal":
        # Table on left, scatterplots on right
        if num_scatterplots == 1:
            fig, (ax_table, ax_scatter) = plt.subplots(1, 2, figsize=(16, 8))
            scatter_axes = [ax_scatter]
        else:
            # Calculate grid for scatterplots
            scatter_rows = int(np.ceil(num_scatterplots / 2))
            fig = plt.figure(figsize=(20, 6 * scatter_rows))

            # Create table subplot
            ax_table = plt.subplot2grid((scatter_rows, 3), (0, 0), rowspan=scatter_rows)

            # Create scatterplot subplots
            scatter_axes = []
            for i in range(num_scatterplots):
                row = i // 2
                col = (i % 2) + 1
                ax = plt.subplot2grid((scatter_rows, 3), (row, col))
                scatter_axes.append(ax)
    else:  # vertical layout
        # Table on top, scatterplots on bottom
        if num_scatterplots == 1:
            fig, (ax_table, ax_scatter) = plt.subplots(2, 1, figsize=(12, 12))
            scatter_axes = [ax_scatter]
        else:
            # Calculate grid for scatterplots
            scatter_cols = min(num_scatterplots, 2)
            scatter_rows = int(np.ceil(num_scatterplots / scatter_cols))

            plt.figure(figsize=(12, 8 + 6 * scatter_rows))

            # Create table subplot
            ax_table = plt.subplot2grid(
                (scatter_rows + 1, scatter_cols), (0, 0), colspan=scatter_cols
            )

            # Create scatterplot subplots
            scatter_axes = []
            for i in range(num_scatterplots):
                row = (i // scatter_cols) + 1
                col = i % scatter_cols
                ax = plt.subplot2grid((scatter_rows + 1, scatter_cols), (row, col))
                scatter_axes.append(ax)

    # Create the table
    _create_table(ax_table, data.table)

    # Create the scatterplots
    for i, scatter_data in enumerate(data.scatterplots):
        if i < len(scatter_axes):
            _create_scatterplot(scatter_axes[i], scatter_data)

    # Remove any unused scatter axes
    for i in range(len(data.scatterplots), len(scatter_axes)):
        scatter_axes[i].remove()

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/table_and_multi_scatterplots_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def _validate_table_scatterplot_consistency(data: TableAndMultiScatterplots) -> None:
    """
    Validate that the table data is consistent with the correct scatterplot(s).
    Only validates scatterplots marked as correct.
    """
    # Extract numeric data from table (assuming first two columns are x, y)
    if len(data.table.headers) < 2:
        return  # Skip validation if table doesn't have x, y columns

    table_points = []
    for row in data.table.rows:
        try:
            x = float(row[0])
            y = float(row[1])
            table_points.append((x, y))
        except (ValueError, IndexError):
            continue  # Skip non-numeric or incomplete rows

    # Validate correct scatterplots
    for scatter in data.scatterplots:
        if scatter.is_correct:
            scatter_points = [(point.x, point.y) for point in scatter.data_points]

            # Check if points match (allowing for floating point precision)
            if len(table_points) != len(scatter_points):
                continue  # Different number of points is allowed for some error types

            # Sort both lists for comparison
            table_sorted = sorted(table_points)
            scatter_sorted = sorted(scatter_points)

            for (tx, ty), (sx, sy) in zip(table_sorted, scatter_sorted):
                if abs(tx - sx) > 1e-6 or abs(ty - sy) > 1e-6:
                    break  # Points don't match exactly
            else:
                return  # Found a matching correct scatterplot

    # If we get here, no correct scatterplot matches the table data
    # This is allowed for assessment purposes where all options might be incorrect


def _generate_error_points(
    original_points: List[DataPoint], error_type: str
) -> List[DataPoint]:
    """
    Generate error points based on the specified error type.

    Args:
        original_points: The correct data points
        error_type: Type of error to introduce

    Returns:
        List of modified data points with the specified error
    """
    if error_type == "swapped_coordinates":
        return [DataPoint(x=point.y, y=point.x) for point in original_points]

    elif error_type == "missing_points":
        # Remove 1-2 random points
        points_copy = original_points.copy()
        num_to_remove = min(2, max(1, len(points_copy) // 3))
        for _ in range(num_to_remove):
            if points_copy:
                points_copy.pop(random.randint(0, len(points_copy) - 1))
        return points_copy

    elif error_type == "extra_points":
        # Add 1-2 random points within the same range
        points_copy = original_points.copy()
        if original_points:
            x_values = [p.x for p in original_points]
            y_values = [p.y for p in original_points]
            x_range = (min(x_values), max(x_values))
            y_range = (min(y_values), max(y_values))

            num_to_add = min(2, max(1, len(original_points) // 4))
            for _ in range(num_to_add):
                new_x = random.uniform(x_range[0], x_range[1])
                new_y = random.uniform(y_range[0], y_range[1])
                # Ensure new point doesn't duplicate existing points
                if not any(
                    abs(new_x - p.x) < 0.1 and abs(new_y - p.y) < 0.1
                    for p in points_copy
                ):
                    points_copy.append(DataPoint(x=new_x, y=new_y))
        return points_copy

    elif error_type == "shifted_points":
        # Shift all points by a small constant
        shift_x = random.choice([-1, -0.5, 0.5, 1])
        shift_y = random.choice([-1, -0.5, 0.5, 1])
        return [
            DataPoint(x=point.x + shift_x, y=point.y + shift_y)
            for point in original_points
        ]

    else:
        return original_points


def _create_table(ax, table_data):
    """Create a table visualization with enhanced styling."""
    ax.axis("off")
    ax.set_title("Data Table", fontsize=16, fontweight="bold", pad=20)

    # Prepare table data
    table_content = [table_data.headers] + table_data.rows

    # Create table
    table = ax.table(
        cellText=table_content[1:],  # Data rows
        colLabels=table_content[0],  # Headers
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Style header cells
    for i in range(len(table_data.headers)):
        table[(0, i)].set_facecolor("#E6E6FA")
        table[(0, i)].set_text_props(weight="bold")

    # Style data cells with alternating colors for better readability
    for i in range(1, len(table_content)):
        for j in range(len(table_data.headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#F8F8FF")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")


def _create_scatterplot(ax, scatter_data):
    """Create a single scatterplot with enhanced styling."""
    # Extract x and y values
    x_values = [point.x for point in scatter_data.data_points]
    y_values = [point.y for point in scatter_data.data_points]

    # Choose color based on correctness (subtle visual cue)
    point_color = "blue" if scatter_data.is_correct else "darkblue"

    # Create the scatter plot
    ax.scatter(
        x_values,
        y_values,
        s=60,
        alpha=0.8,
        color=point_color,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )

    # Set labels and title
    ax.set_xlabel(scatter_data.x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(scatter_data.y_label, fontsize=12, fontweight="bold")
    ax.set_title(scatter_data.title, fontsize=14, fontweight="bold", pad=15)

    # Set axis limits with some padding
    x_range = scatter_data.x_max - scatter_data.x_min
    y_range = scatter_data.y_max - scatter_data.y_min
    x_padding = x_range * 0.05
    y_padding = y_range * 0.05

    ax.set_xlim(scatter_data.x_min - x_padding, scatter_data.x_max + x_padding)
    ax.set_ylim(scatter_data.y_min - y_padding, scatter_data.y_max + y_padding)

    # Add grid for better coordinate reading
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Style the plot
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")
