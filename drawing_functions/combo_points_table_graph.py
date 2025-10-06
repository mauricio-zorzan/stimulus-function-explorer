import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.combo_points_table_graph import (
    ComboPointsTableGraph,
    GraphSpec,
    Point,
    TableData,
)
from content_generators.settings import settings


@stimulus_function
def draw_combo_points_table_graph(data: ComboPointsTableGraph) -> str:
    """
    Creates a combined visualization with table and graph using dynamic layout.
    Small tables are placed beside the graph, large tables above the graph.
    Supports function identification, proportional relationship comparison,
    and function property comparison for 8th grade standards.

    Args:
        data: ComboPointsTableGraph containing table, points, and graph specifications

    Returns:
        str: Path to the generated image file
    """
    if data.table:
        # Calculate table dimensions
        num_rows = len(data.table.rows) + 1  # +1 for header
        num_cols = len(data.table.headers)

        # Determine layout based on table size
        # Always use vertical layout to prevent table interference with graph
        use_side_layout = False  # Disabled side layout to prevent interference

        if use_side_layout:
            # Side-by-side layout: table on left, graph on right
            fig = plt.figure(figsize=(11, 7))  # Slightly smaller figure

            # Calculate width ratios - ensure table has enough space for content
            # Base width on number of columns and content length
            min_table_width = 0.15 + (num_cols * 0.08)  # More dynamic sizing
            table_width_ratio = min(
                0.35, max(0.18, min_table_width)
            )  # Ensure minimum space
            graph_width_ratio = 1 - table_width_ratio

            gs = fig.add_gridspec(
                1, 2, width_ratios=[table_width_ratio, graph_width_ratio], wspace=0.02
            )  # Minimal space between table and graph

            ax_table = fig.add_subplot(gs[0])
            ax_graph = fig.add_subplot(gs[1])
        else:
            # Vertical layout: table above graph
            fig = plt.figure(figsize=(10, 12))

            # Calculate height ratios - ensure table has enough space
            # Base height on number of rows and content
            min_table_height = 0.12 + (num_rows * 0.03)  # More dynamic sizing
            table_height_ratio = min(0.35, max(0.15, min_table_height))
            graph_height_ratio = 1 - table_height_ratio

            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[table_height_ratio, graph_height_ratio],
                hspace=0.05,  # Reduced spacing to prevent overlap
            )

            ax_table = fig.add_subplot(gs[0])
            ax_graph = fig.add_subplot(gs[1])

        # Create the table
        _create_table(ax_table, data.table, use_side_layout)
    else:
        # No table - just graph
        fig, ax_graph = plt.subplots(1, 1, figsize=(10, 8))

    # Create the graph
    _create_graph(ax_graph, data)

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/combo_points_table_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def _create_table(ax, table_data: TableData, use_side_layout: bool):
    """Create a table visualization sized according to content with minimal white space."""
    ax.axis("off")
    title = table_data.title if table_data.title else "Data Table"

    # Adjust title padding based on layout
    title_pad = 8 if use_side_layout else 5
    ax.set_title(title, fontsize=16, fontweight="bold", pad=title_pad)

    # Prepare table data
    table_content = [table_data.headers] + table_data.rows
    num_cols = len(table_data.headers)

    if use_side_layout:
        # Side layout: position table to be very close to graph
        table_width = 0.95  # Use almost all available width
        table_height = 0.7  # Compact height
        x_offset = 0.025  # Small left margin
        y_offset = 0.15  # Add slight space below table
        font_size = 10  # Slightly smaller font to fit content
        row_scale = 1.3  # Reduced row height for more compact table
    else:
        # Vertical layout: optimize for content containment
        # Calculate table width based on content length
        max_header_length = max(len(header) for header in table_data.headers)
        max_content_length = max(
            max(len(str(cell)) for cell in row) for row in table_data.rows
        )
        content_factor = max(max_header_length, max_content_length) / 10

        table_width = min(0.95, 0.1 * num_cols + 0.3 + (content_factor * 0.05))
        table_height = 0.75  # Reduced height to prevent overlap
        x_offset = (1 - table_width) / 2
        y_offset = 0.1  # Reduced top margin to prevent overlap
        font_size = 12  # Maintain readability
        row_scale = 1.4  # Adequate row height for content

    # Create table
    table = ax.table(
        cellText=table_content[1:],  # Data rows
        colLabels=table_content[0],  # Headers
        cellLoc="center",
        loc="center",
        bbox=[x_offset, y_offset, table_width, table_height],
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, row_scale)

    # Style header cells
    for i in range(len(table_data.headers)):
        table[(0, i)].set_facecolor("#E6E6FA")
        table[(0, i)].set_text_props(weight="bold")

    # Style data cells
    for i in range(1, len(table_content)):
        for j in range(len(table_data.headers)):
            table[(i, j)].set_facecolor("#F8F8FF")


def _create_graph(ax, data: ComboPointsTableGraph):
    """Create the graph with points, lines, and special features."""
    # For 8th grade coordinate planes, axes must always intersect at (0,0)
    # and show all four quadrants when appropriate

    # Determine the required range to include all data
    data_x_min = data.x_axis.min_value
    data_x_max = data.x_axis.max_value
    data_y_min = data.y_axis.min_value
    data_y_max = data.y_axis.max_value

    # Ensure axes always include the origin (0,0) for 8th grade standards
    # Extend the range to include 0 if it's not already included
    final_x_min = min(data_x_min, 0)
    final_x_max = max(data_x_max, 0)
    final_y_min = min(data_y_min, 0)
    final_y_max = max(data_y_max, 0)

    # Calculate ranges for padding
    x_range = final_x_max - final_x_min
    y_range = final_y_max - final_y_min

    # Add small padding to prevent legend overlap
    x_padding = 0.05 * x_range
    y_padding = 0.05 * y_range

    ax.set_xlim(final_x_min - x_padding, final_x_max + x_padding)
    ax.set_ylim(final_y_min - y_padding, final_y_max + y_padding)
    # Note: Axis labels will be added at the ends of the axes later

    if data.graph_title:
        ax.set_title(
            data.graph_title, fontsize=16, fontweight="bold", pad=15
        )  # Reduced to 15 for even more space

    # Set up ticks - use final ranges that include origin
    if data.x_axis.tick_interval:
        # Generate ticks across the full range including origin
        x_ticks = np.arange(
            final_x_min,
            final_x_max + data.x_axis.tick_interval,
            data.x_axis.tick_interval,
        )
        ax.set_xticks(x_ticks)

        # Remove the 0 tick mark since it clutters the origin intersection
        current_ticks = list(ax.get_xticks())
        if 0 in current_ticks:
            current_ticks.remove(0)
            ax.set_xticks(current_ticks)

    if data.y_axis.tick_interval:
        # Generate ticks across the full range including origin
        y_ticks = np.arange(
            final_y_min,
            final_y_max + data.y_axis.tick_interval,
            data.y_axis.tick_interval,
        )
        ax.set_yticks(y_ticks)

        # Remove the 0 tick mark since it clutters the origin intersection
        current_ticks = list(ax.get_yticks())
        if 0 in current_ticks:
            current_ticks.remove(0)
            ax.set_yticks(current_ticks)

    # Add grid
    if data.show_grid:
        ax.grid(True, alpha=0.6)

    # Note: Coordinate axes are now handled by repositioned spines below

    # Draw graphs/lines
    if data.graphs:
        for graph_spec in data.graphs:
            _draw_graph_spec(ax, graph_spec, final_x_min, final_x_max)

    # Plot points
    if data.points:
        _plot_points(ax, data.points, data.highlight_points)

    # Add legend if there are labeled elements with improved positioning
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Choose better legend position to avoid overlap
        legend_position = data.legend_position

        # If using default "upper right", try to position it better
        if legend_position == "upper right":
            # Check if we have space on the right side
            if data.x_axis.max_value > 3:  # If graph extends far right
                legend_position = "upper left"

        ax.legend(
            loc=legend_position,
            fontsize=10,
            framealpha=0.9,
            fancybox=True,
            shadow=True,
            bbox_to_anchor=(0.98, 0.98),
        )

    # Style the plot - hide all default spines since we draw custom axes
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Move tick marks to the coordinate axes (the cross) instead of plot edges
    # This creates the classic coordinate plane appearance
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    # Add axis labels at the ends of the axes (not in the middle)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Place "x" label at the right end of the x-axis
    ax.text(
        xlim[1] - 0.1,
        -0.3,
        "x",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
        transform=ax.transData,
    )

    # Place "y" label at the top end of the y-axis
    ax.text(
        0.1,
        ylim[1] - 0.1,
        "y",
        fontsize=14,
        fontweight="bold",
        ha="left",
        va="center",
        transform=ax.transData,
    )

    # Add a small dot at the origin to represent (0,0)
    ax.plot(0, 0, "ko", markersize=4, zorder=10)

    # Remove the default axis labels since we're adding custom ones
    ax.set_xlabel("")
    ax.set_ylabel("")


def _draw_graph_spec(ax, graph_spec: GraphSpec, x_min: float, x_max: float):
    """Draw a specific graph specification."""
    if graph_spec.type == "line":
        # Draw a line using slope and y-intercept
        if graph_spec.slope is not None and graph_spec.y_intercept is not None:
            x_values = np.linspace(x_min, x_max, 100)
            y_values = graph_spec.slope * x_values + graph_spec.y_intercept

            label = graph_spec.label
            if not label and graph_spec.equation:
                label = graph_spec.equation
            elif not label:
                label = f"y = {graph_spec.slope}x + {graph_spec.y_intercept}"

            ax.plot(
                x_values,
                y_values,
                color=graph_spec.color,
                linestyle=graph_spec.line_style,
                linewidth=graph_spec.line_width,
                label=label,
            )

    elif graph_spec.type == "scatter":
        # Draw scatter points
        if graph_spec.points:
            x_values = [point.x for point in graph_spec.points]
            y_values = [point.y for point in graph_spec.points]

            ax.scatter(
                x_values,
                y_values,
                color=graph_spec.color,
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                label=graph_spec.label,
            )

    elif graph_spec.type == "circle":
        # Draw circle: (x-h)² + (y-k)² = r²
        theta = np.linspace(0, 2 * np.pi, 100)
        x_values = graph_spec.center_x + graph_spec.radius * np.cos(theta)
        y_values = graph_spec.center_y + graph_spec.radius * np.sin(theta)

        label = (
            graph_spec.label or graph_spec.equation or f"Circle (r={graph_spec.radius})"
        )
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "sideways_parabola":
        # Draw sideways parabola: x = a(y-k)² + h
        y_values = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
        x_values = graph_spec.a * (y_values - graph_spec.k) ** 2 + graph_spec.h

        label = graph_spec.label or graph_spec.equation or "x = y²"
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "quadratic":
        # Draw quadratic: y = ax² + bx + c
        x_values = np.linspace(x_min, x_max, 100)
        y_values = graph_spec.a * x_values**2 + graph_spec.b * x_values + graph_spec.c

        label = (
            graph_spec.label
            or graph_spec.equation
            or f"y = {graph_spec.a}x² + {graph_spec.b}x + {graph_spec.c}"
        )
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "cubic":
        # Draw cubic: y = ax³ + bx² + cx + d
        x_values = np.linspace(x_min, x_max, 100)
        y_values = (
            graph_spec.a * x_values**3
            + graph_spec.b * x_values**2
            + graph_spec.c * x_values
            + graph_spec.k
        )

        label = graph_spec.label or graph_spec.equation or f"y = {graph_spec.a}x³"
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "sqrt":
        # Draw square root: y = a√(x-h) + k
        x_values = np.linspace(max(x_min, graph_spec.h), x_max, 100)
        y_values = graph_spec.a * np.sqrt(x_values - graph_spec.h) + graph_spec.k

        label = graph_spec.label or graph_spec.equation or "y = √x"
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "rational":
        # Draw rational function: y = a/x + k
        x_values = np.linspace(x_min, x_max, 1000)
        # Remove values near zero to avoid division by zero
        x_values = x_values[np.abs(x_values) > 0.01]
        y_values = graph_spec.a / x_values + graph_spec.k

        label = graph_spec.label or graph_spec.equation or f"y = {graph_spec.a}/x"
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "hyperbola":
        # Draw hyperbola: x² - y² = a²
        # Split into right and left branches to avoid discontinuity at x = ±a

        label = (
            graph_spec.label or graph_spec.equation or f"x² - y² = {graph_spec.a**2}"
        )

        # Right branch (x >= a)
        x_right = np.linspace(max(x_min, graph_spec.a), x_max, 100)
        if len(x_right) > 0:
            y_right_pos = np.sqrt(x_right**2 - graph_spec.a**2)
            y_right_neg = -np.sqrt(x_right**2 - graph_spec.a**2)

            ax.plot(
                x_right,
                y_right_pos,
                color=graph_spec.color,
                linestyle=graph_spec.line_style,
                linewidth=graph_spec.line_width,
                label=label,
            )
            ax.plot(
                x_right,
                y_right_neg,
                color=graph_spec.color,
                linestyle=graph_spec.line_style,
                linewidth=graph_spec.line_width,
            )

        # Left branch (x <= -a)
        x_left = np.linspace(x_min, min(x_max, -graph_spec.a), 100)
        if len(x_left) > 0:
            y_left_pos = np.sqrt(x_left**2 - graph_spec.a**2)
            y_left_neg = -np.sqrt(x_left**2 - graph_spec.a**2)

            ax.plot(
                x_left,
                y_left_pos,
                color=graph_spec.color,
                linestyle=graph_spec.line_style,
                linewidth=graph_spec.line_width,
            )
            ax.plot(
                x_left,
                y_left_neg,
                color=graph_spec.color,
                linestyle=graph_spec.line_style,
                linewidth=graph_spec.line_width,
            )

    elif graph_spec.type == "ellipse":
        # Draw ellipse: (x-h)²/a² + (y-k)²/b² = 1
        theta = np.linspace(0, 2 * np.pi, 100)
        x_values = graph_spec.center_x + graph_spec.a * np.cos(theta)
        y_values = graph_spec.center_y + graph_spec.b * np.sin(theta)

        label = (
            graph_spec.label
            or graph_spec.equation
            or f"x²/{graph_spec.a**2} + y²/{graph_spec.b**2} = 1"
        )
        ax.plot(
            x_values,
            y_values,
            color=graph_spec.color,
            linestyle=graph_spec.line_style,
            linewidth=graph_spec.line_width,
            label=label,
        )

    elif graph_spec.type == "curve":
        # For general curves, we'd need more specific parameters
        # This is a placeholder for future curve support
        pass


def _plot_points(ax, points: List[Point], highlight_points: Optional[List[str]] = None):
    """Plot and label points with simple styling and improved label positioning."""

    # Group points by x-coordinate to detect overlaps
    x_groups = {}
    for point in points:
        x_key = round(point.x, 3)  # Round to avoid floating point issues
        if x_key not in x_groups:
            x_groups[x_key] = []
        x_groups[x_key].append(point)

    # Get axis ranges for positioning
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    base_offset_x = 0.025 * x_range  # Reduced offset for closer labels
    base_offset_y = 0.025 * y_range

    # Check if there are already graph lines in the legend
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    has_graph_lines = len(existing_handles) > 0

    for x_coord, group_points in x_groups.items():
        # Sort points by y-coordinate for consistent ordering
        group_points.sort(key=lambda p: p.y)

        for j, point in enumerate(group_points):
            # Determine if this point should be highlighted
            is_highlighted = highlight_points and point.label in highlight_points

            # Use simple blue dots for all points, black for highlighted
            if is_highlighted:
                color = "black"
                marker_size = 100  # Larger for highlighted points
                edge_width = 3
                alpha = 1.0
            else:
                color = "blue"
                marker_size = 80  # Increased from 60 for better visibility
                edge_width = 2  # Increased edge width
                alpha = 0.9  # Slightly more opaque

            # Determine if this point should be added to legend
            # Only add to legend if:
            # 1. There are no graph lines (points-only plot), OR
            # 2. The point represents something meaningful (like person names, not just coordinates)
            point_label_for_legend = None
            if not has_graph_lines:
                # Points-only plot - only include meaningful names, not simple coordinates or letters
                if point.label and not any(
                    pattern in point.label.lower()
                    for pattern in ["(", ")", "a", "b", "c", "d", "e", "f"]
                ):
                    # Include points with meaningful names (like "Alice", "Bob") but not coordinate labels
                    if (
                        len(point.label) > 2
                        and point.label.isalpha()
                        and point.label not in ["A", "B", "C", "D", "E", "F"]
                    ):
                        point_label_for_legend = point.label
            elif point.label and not any(
                char in point.label.lower()
                for char in ["(", ")", "a", "b", "c", "d", "e", "f"]
            ):
                # Include points with meaningful names (like "Alice", "Bob") but not coordinate labels like "A", "B(2,4)"
                if (
                    len(point.label) > 2
                    and point.label.isalpha()
                    and point.label not in ["A", "B", "C", "D", "E", "F"]
                ):
                    point_label_for_legend = point.label

            # Plot the point with simple circular marker
            ax.scatter(
                point.x,
                point.y,
                color=color,
                marker="o",  # Always use circles
                s=marker_size,
                alpha=alpha,
                edgecolors="black",
                linewidth=edge_width,
                label=point_label_for_legend,  # Only add meaningful labels to legend
                zorder=5,
            )

            # Calculate label position with overlap avoidance
            if len(group_points) > 1:
                # Multiple points at same x-coordinate - use strategic positioning
                if j == 0:
                    # First point (lowest y) - place label to the left
                    offset_x = -base_offset_x * 1.8
                    ha = "right"
                elif j == len(group_points) - 1:
                    # Last point (highest y) - place label to the right
                    offset_x = base_offset_x * 1.8
                    ha = "left"
                else:
                    # Middle points - alternate left/right
                    if j % 2 == 0:
                        offset_x = -base_offset_x * 1.8
                        ha = "right"
                    else:
                        offset_x = base_offset_x * 1.8
                        ha = "left"

                # Vertical positioning based on point position in group
                if j < len(group_points) // 2:
                    offset_y = -base_offset_y * 1.2
                    va = "top"
                else:
                    offset_y = base_offset_y * 1.2
                    va = "bottom"
            else:
                # Single point at this x-coordinate - use standard positioning
                center_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
                center_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2

                if point.x < center_x:
                    offset_x = -base_offset_x * 1.5
                    ha = "right"
                else:
                    offset_x = base_offset_x * 1.5
                    ha = "left"

                if point.y < center_y:
                    offset_y = -base_offset_y * 1.5
                    va = "top"
                else:
                    offset_y = base_offset_y * 1.5
                    va = "bottom"

            # Place the label with larger font and better styling
            ax.annotate(
                point.label,
                (point.x, point.y),
                xytext=(point.x + offset_x, point.y + offset_y),
                fontsize=12,  # Larger font size
                fontweight="bold" if is_highlighted else "normal",
                color="black",
                ha=ha,
                va=va,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.95,
                    edgecolor="gray",
                    linewidth=0.8,
                ),
                zorder=6,
            )
