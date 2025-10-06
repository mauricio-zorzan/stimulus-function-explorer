import math
import os
import random
import textwrap
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.blank_coordinate_plane import (
    BlankCoordinatePlane,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.grouped_bar_chart import (
    GroupedBarChart,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.linear_diagram import (
    LinearDiagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.multi_graph import (
    BarGraphItem,
    CombinedGraphs,
    LineGraphItem,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.nonlinear_graph import (
    NonlinearGraph,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_line import (
    PlotLine,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_lines import (
    PlotLines,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_ploygon import (
    PlotPolygon,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_points import (
    Point as PlotPoint,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_points import (
    PointList,
    PointPlot,
    PointPlotWithContext,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    PolygonDilation,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.quantitative import (
    MultipleBarGraph,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.scatter_plot import (
    ScatterPlot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_scatterplot import (
    StatsScatterplot,
)
from content_generators.settings import settings
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

matplotlib.rcParams["font.family"] = "serif"


#####################
# Graphing - Points #
#####################
@stimulus_function
def plot_points(points: PointPlot):
    # Check for duplicate points
    seen = set()
    for point in points.points:
        key = (point.x, point.y)
        if key in seen:
            raise ValueError(f"Duplicate point found: ({point.x}, {point.y})")
        seen.add(key)

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d", "P", "X"]

    fig, ax = plt.subplots()

    x_values = [point.x for point in points.points]
    y_values = [point.y for point in points.points]

    # Check if all points lie in the first quadrant
    if all(x >= 0 for x in x_values) and all(y >= 0 for y in y_values):
        quadrant = "first"
    else:
        quadrant = "all"

    for i, point in enumerate(points.points):
        ax.scatter(
            point.x,
            point.y,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=point.label,
            linewidths=4,
            zorder=5,
            clip_on=False,
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if quadrant == "all":
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_zorder(2)
        ax.spines["bottom"].set_zorder(2)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.grid(True, linewidth=2)

    if quadrant == "first":
        ax.set_xlim(0, max(x_values) + 2)
        ax.set_ylim(0, max(y_values) + 2)
    else:
        ax.set_xlim(min(x_values) - 2, max(x_values) + 2)
        ax.set_ylim(min(y_values) - 2, max(y_values) + 2)

    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_graphing_points.{settings.additional_content_settings.stimulus_image_format}"
    plt.margins(0.05)
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()
    return file_name


@stimulus_function
def plot_lines(line_list: PlotLines):
    # Create a range of x values
    x = np.linspace(-10, 10, 400)

    # Set up a subplot with aspect ratio 1
    ax = plt.figure().add_subplot(1, 1, 1)

    # Move left y-axis and bottom x-axis to the middle
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")

    # Eliminate upper and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))

    # Setting the x and y axis limits
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_yticks(np.arange(-10, 11, 2))

    # Setting minor ticks for '1' mark
    ax.set_xticks(np.arange(-10, 11, 1), minor=True)
    ax.set_yticks(np.arange(-10, 11, 1), minor=True)
    ax.tick_params(axis="x", labelsize=12)  # Set font size for x-axis numbers
    ax.tick_params(axis="y", labelsize=12)  # Set font size for y-axis numbers

    ax.grid(True)  # Add Grid Lines

    # Cycle through the list of lines
    for line in line_list:
        y = (
            line.slope * x + line.y_intercept
        )  # Calculate y based on x, slope and intercept
        ax.plot(x, y, label=line.label, linewidth=2)

    # Add a legend
    ax.legend(fontsize=14)

    # Show the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_graphing_lines.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def plot_par_and_perp_lines(line_list: PlotLines):
    # Validate that all line labels are proper letter coordinates (exactly 2 letters)
    for line in line_list:
        label = line.label
        if not (len(label) == 2 and label.isalpha() and label.isupper()):
            raise ValueError(
                f"Line label '{label}' is not valid. All line labels must be exactly 2 uppercase letters "
                f"representing coordinate points (e.g., 'AB', 'CD', 'EF'). "
                f"Labels like 'Line 1', 'Line 2' are not acceptable."
            )

    # Create a range of x values for line segments (smaller range to show endpoints)
    x_range = 6  # Lines will extend from -6 to 6 instead of -10 to 10
    x = np.linspace(-x_range, x_range, 100)

    # Set up a square figure
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)

    # Remove all spines (no coordinate plane)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove grid
    ax.grid(False)

    # Set axis limits to match line range so endpoints are at the visual ends
    ax.set_xlim((-x_range, x_range))
    ax.set_ylim((-x_range, x_range))

    # Collect all letter positions first to avoid overlaps
    letter_positions = []

    # Plot each line with endpoints
    for line in line_list:
        y = line.slope * x + line.y_intercept

        # Calculate endpoint coordinates
        start_x, end_x = -x_range, x_range
        start_y = line.slope * start_x + line.y_intercept
        end_y = line.slope * end_x + line.y_intercept

        # Extract letters from validated line label (e.g., "AB" -> "A", "B")
        label = line.label
        start_letter = label[0]
        end_letter = label[1]

        # Plot the line segment first
        line_plot = ax.plot(x, y, linewidth=2, label=line.label)[0]
        line_color = line_plot.get_color()

        # Calculate arrow direction (unit vector along the line)
        dx = end_x - start_x
        dy = end_y - start_y
        length = np.sqrt(dx**2 + dy**2)
        unit_dx = dx / length
        unit_dy = dy / length

        # Draw arrows AT the endpoints (arrows are the endpoints)
        arrow_length = 0.4

        # Start arrow - arrow points from inside the line toward the start endpoint
        ax.annotate(
            "",
            xy=(start_x, start_y),
            xytext=(start_x + arrow_length * unit_dx, start_y + arrow_length * unit_dy),
            arrowprops=dict(arrowstyle="->", color=line_color, lw=2),
        )

        # End arrow - arrow points from inside the line toward the end endpoint
        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(end_x - arrow_length * unit_dx, end_y - arrow_length * unit_dy),
            arrowprops=dict(arrowstyle="->", color=line_color, lw=2),
        )

        # Calculate positions for black dots ON the line segments (not at endpoints)
        # Place dots at about 30% and 70% along each line to avoid arrows
        dot1_x = start_x + 0.3 * (end_x - start_x)
        dot1_y = start_y + 0.3 * (end_y - start_y)
        dot2_x = start_x + 0.7 * (end_x - start_x)
        dot2_y = start_y + 0.7 * (end_y - start_y)

        # Add black dots ON the line segments
        ax.plot(dot1_x, dot1_y, "ko", markersize=8, zorder=5)
        ax.plot(dot2_x, dot2_y, "ko", markersize=8, zorder=5)

        # Calculate initial letter positions
        perp_dx = -unit_dy  # Perpendicular direction
        perp_dy = unit_dx
        label_offset = 0.5

        # Store letter information for collision detection
        letter_positions.append(
            {
                "x": dot1_x + label_offset * perp_dx,
                "y": dot1_y + label_offset * perp_dy,
                "letter": start_letter,
                "dot_x": dot1_x,
                "dot_y": dot1_y,
                "perp_dx": perp_dx,
                "perp_dy": perp_dy,
            }
        )

        letter_positions.append(
            {
                "x": dot2_x + label_offset * perp_dx,
                "y": dot2_y + label_offset * perp_dy,
                "letter": end_letter,
                "dot_x": dot2_x,
                "dot_y": dot2_y,
                "perp_dx": perp_dx,
                "perp_dy": perp_dy,
            }
        )

    # Resolve label overlaps
    min_distance = 0.8  # Minimum distance between letters
    max_iterations = 10

    for iteration in range(max_iterations):
        moved = False
        for i in range(len(letter_positions)):
            for j in range(i + 1, len(letter_positions)):
                pos1 = letter_positions[i]
                pos2 = letter_positions[j]

                # Calculate distance between letters
                dist = np.sqrt(
                    (pos1["x"] - pos2["x"]) ** 2 + (pos1["y"] - pos2["y"]) ** 2
                )

                if dist < min_distance:
                    # Move letters apart - try different offset directions
                    offset_multiplier = 1.5

                    # Try moving in opposite perpendicular directions
                    letter_positions[i]["x"] = (
                        pos1["dot_x"] + offset_multiplier * 0.6 * pos1["perp_dx"]
                    )
                    letter_positions[i]["y"] = (
                        pos1["dot_y"] + offset_multiplier * 0.6 * pos1["perp_dy"]
                    )

                    letter_positions[j]["x"] = (
                        pos2["dot_x"] - offset_multiplier * 0.6 * pos2["perp_dx"]
                    )
                    letter_positions[j]["y"] = (
                        pos2["dot_y"] - offset_multiplier * 0.6 * pos2["perp_dy"]
                    )

                    moved = True

        if not moved:
            break

    # Place all letters with resolved positions
    for pos in letter_positions:
        ax.text(
            pos["x"],
            pos["y"],
            pos["letter"],
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    # Save the file directly without legend since letters are on the lines
    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_lines_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def create_scatterplot(data: ScatterPlot):
    # Extract data
    title = data.title
    x_label = data.x_axis.label
    y_label = data.y_axis.label
    points = data.points

    # Create lists for x and y values
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, zorder=2)

    # Set plot labels and title
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.title(title, fontsize=20)
    plt.tick_params(axis="both", labelsize=14)

    # Set limits for the axes
    plt.xlim(data.x_axis.min_value, data.x_axis.max_value)
    plt.ylim(data.y_axis.min_value, data.y_axis.max_value)

    # Add grid lines behind plot points
    plt.grid(True, which="both", color="gray", linestyle="-", linewidth=0.5, zorder=-1)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_scatterplot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def create_polygon(stimulus_description: PlotPolygon):
    """
    Draw a polygon on a coordinate grid.
    """
    x_range = stimulus_description.axes["x"].range
    y_range = stimulus_description.axes["y"].range
    points = stimulus_description.polygon.points
    coordinates = [p.coordinates for p in points]

    # Settings (can be tuned)
    MIN_FIG = 4.5
    MAX_FIG = 8.5
    UNIT_SCALE = 0.7  # inches per unit span (capped by min/max figure size)
    PAD_FRACTION = 0.015  # small % padding per side (ignored if strict_bounds)

    strict_bounds = getattr(stimulus_description, "strict_bounds", False)

    span_x = max(1e-9, x_range[1] - x_range[0])
    span_y = max(1e-9, y_range[1] - y_range[0])

    # Compute provisional figure size based on spans (proportional appearance)
    fig_w = min(MAX_FIG, max(MIN_FIG, span_x * UNIT_SCALE))
    fig_h = min(MAX_FIG, max(MIN_FIG, span_y * UNIT_SCALE))

    # Keep extreme ratios from becoming too skinny / wide (cap aspect distortion)
    ratio = fig_w / fig_h
    if ratio > 1.8:  # too wide
        fig_w = 1.8 * fig_h
    elif ratio < 0.55:  # too tall
        fig_h = fig_w / 0.55

    # Padding (slight) unless strict bounds requested
    def pad_limits(lo, hi):
        if strict_bounds:
            return lo, hi
        pad = (hi - lo) * PAD_FRACTION
        return lo - pad, hi + pad

    x_lo, x_hi = pad_limits(*x_range)
    y_lo, y_hi = pad_limits(*y_range)

    # 'Nice number' tick step chooser
    def nice_step(span, target_ticks=8):
        if span <= 0:
            return 1

        # Use integer-only candidates for cleaner axis labels
        candidates = [1, 2, 5, 10, 20, 25, 50]

        best_step = 1
        best_diff = float("inf")

        for step in candidates:
            ticks = span / step
            diff = abs(ticks - target_ticks)
            # Prefer steps that give 4-12 ticks total
            if 4 <= ticks <= 12 and diff < best_diff:
                best_step = step
                best_diff = diff

        return best_step

    step_x = nice_step(span_x)
    step_y = nice_step(span_y)

    # Build tick lists (integers where possible)
    def build_ticks(lo, hi, step):
        # Ensure inclusive end with floating tolerance
        ticks = []
        t = math.ceil(lo / step) * step
        while t <= hi + 1e-9:
            # Snap near-integers to int for clean labels
            if abs(t - round(t)) < 1e-8:
                t = int(round(t))
            ticks.append(t)
            t += step
        return ticks

    x_ticks = build_ticks(x_range[0], x_range[1], step_x)
    y_ticks = build_ticks(y_range[0], y_range[1], step_y)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(x_lo, x_hi)  # type: ignore
    ax.set_ylim(y_lo, y_hi)  # type: ignore
    ax.set_aspect("equal")  # Preserve unit square appearance

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Minor ticks (only if span moderate)
    if span_x <= 40:
        ax.set_xticks(
            build_ticks(x_range[0], x_range[1], max(step_x / 5, 1)), minor=True
        )
    if span_y <= 40:
        ax.set_yticks(
            build_ticks(y_range[0], y_range[1], max(step_y / 5, 1)), minor=True
        )

    # Remove origin label only
    def label_list(ticks):
        labs = []
        for t in ticks:
            if t == 0:
                labs.append("")
            else:
                # Avoid long floats
                labs.append(
                    str(int(t))
                    if isinstance(t, (int, np.integer)) or float(t).is_integer()
                    else str(t)
                )
        return labs

    ax.set_xticklabels(label_list(x_ticks))
    ax.set_yticklabels(label_list(y_ticks))

    # Grid styling
    ax.grid(
        which="major",
        color="#c3c3c3",
        linestyle="-",
        linewidth=0.7,
        alpha=0.75,
        zorder=0,
    )
    ax.grid(
        which="minor",
        color="#e2e2e2",
        linestyle="-",
        linewidth=0.4,
        alpha=0.5,
        zorder=0,
    )

    # Hide top/right
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Center axes ONLY if zero is inside respective ranges
    if x_range[0] <= 0 <= x_range[1]:
        ax.spines["left"].set_position("zero")
        ax.spines["left"].set_linewidth(1.4)
    else:
        ax.spines["left"].set_position(("data", x_range[0]))
        ax.spines["left"].set_linewidth(1.0)

    if y_range[0] <= 0 <= y_range[1]:
        ax.spines["bottom"].set_position("zero")
        ax.spines["bottom"].set_linewidth(1.4)
    else:
        ax.spines["bottom"].set_position(("data", y_range[0]))
        ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", labelsize=12, pad=3)
    ax.tick_params(axis="both", which="minor", length=3, labelsize=0)

    # Draw polygon edges
    polygon = coordinates + [coordinates[0]]
    xs, ys = zip(*polygon)
    ax.plot(xs, ys, "-", linewidth=2, color="#1f5a98", zorder=5)

    # Plot vertices
    any_calculated = any(getattr(p, "calculated", False) for p in points)
    for p in points:
        px, py = p.coordinates
        if getattr(p, "calculated", False):
            ax.scatter(
                px,
                py,
                s=100,
                facecolors="white",
                edgecolors="#d62728",
                linewidths=2.4,
                marker="o",
                zorder=8,
            )
        else:
            ax.scatter(
                px,
                py,
                s=36,
                color="#1f5a98",
                marker="o",
                linewidths=0,
                zorder=8,
            )

    # Label placement heuristic
    # Dynamic base offset (in axis units) so labels clear the marker
    # 1.5% of larger span, but never below 0.32 units, capped to avoid flying away on huge spans
    dynamic_base_offset = min(max(0.32, 0.015 * max(span_x, span_y)), 0.6)

    def offset(px, py):
        """
        Choose an (dx, dy) so label does not cover the point.
        Flips direction near right/top edges; also nudges off axis lines (x=0 or y=0) slightly.
        """
        dx = dynamic_base_offset
        dy = dynamic_base_offset

        # Flip horizontally if near right edge
        if px > x_range[1] - 0.9:
            dx = -dynamic_base_offset
        # Flip vertically if near top edge
        if py > y_range[1] - 0.9:
            dy = -dynamic_base_offset

        # If very close to left edge (not centered axis) push right
        if px < x_range[0] + 0.9 and not (x_range[0] <= 0 <= x_range[1]):
            dx = abs(dx)
        # If very close to bottom edge (not centered axis) push up
        if py < y_range[0] + 0.9 and not (y_range[0] <= 0 <= y_range[1]):
            dy = abs(dy)

        return dx, dy

    for p in points:
        px, py = p.coordinates
        dx, dy = offset(px, py)
        is_calc = getattr(p, "calculated", False)
        if is_calc:
            ax.annotate(
                f"{p.label}*",
                xy=(px, py),
                xytext=(px + dx * 1.7, py + dy * 1.7),
                textcoords="data",
                fontsize=13,
                fontweight="bold",
                color="#d62728",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor="white",
                    edgecolor="#d62728",
                    linewidth=1.3,
                    alpha=0.97,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#d62728",
                    linewidth=1.2,
                    shrinkA=4,
                    shrinkB=4,
                ),
                zorder=15,
            )
        else:
            # Slight extra radial offset (1.1) so box clears marker fully
            ax.text(
                px + dx * 1.1,
                py + dy * 1.1,
                p.label,
                fontsize=13,
                fontweight="bold",
                color="#1f3d66",
                ha="center",
                va="center",
                zorder=12,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="#c9d4df",
                    linewidth=0.55,
                    alpha=0.9,
                ),
            )

    # Footnote if any calculated points
    if any_calculated:
        ax.text(
            0.01,
            -0.04,
            "* indicates a calculated (missing) vertex",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
            color="#444444",
        )

    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"{int(time.time())}_graphing_polygon."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_name,
        dpi=650,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


@stimulus_function
def plot_line(stimulus_description: PlotLine):
    x_label = stimulus_description.x_axis.label
    x_range = stimulus_description.x_axis.range
    y_label = stimulus_description.y_axis.label
    y_range = stimulus_description.y_axis.range
    slope = stimulus_description.line.slope
    intercept = stimulus_description.line.intercept
    point = stimulus_description.point

    # LABEL_PADDING = 1

    # create array of x and y values
    x_values = np.linspace(x_range[0], x_range[1])
    y_values = slope * x_values + intercept
    y_range[1] = max(y_values)

    plt.figure()
    plt.plot(x_values, y_values)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    ax = plt.gca()

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Calculating offsets as a percentage of the axis limits
    x_range = plt.xlim()[1] - plt.xlim()[0]
    y_range = plt.ylim()[1] - plt.ylim()[0]

    x_ticks = list(ax.get_xticks())
    y_ticks = list(ax.get_yticks())

    # Set minor ticks and gridlines
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)
    ax.grid(which="major", color="black", linestyle="-", linewidth=1)

    # Check if the line passes through any grid intersection (major or minor)
    x_ticks = np.concatenate([ax.get_xticks(minor=False), ax.get_xticks(minor=True)])
    y_ticks = np.concatenate([ax.get_yticks(minor=False), ax.get_yticks(minor=True)])
    found_intersection = False
    tol = 1e-8
    for x in x_ticks:
        y = slope * x + intercept
        if np.any(np.isclose(y, y_ticks, atol=tol)):
            found_intersection = True
            break
    if not found_intersection:
        raise ValueError(
            "The line does not pass through any intersection of major or minor gridlines."
        )

    # Plot point if it is not an intercept
    if point and point[0] != 0 and point[1] != 0:
        ax.plot(point[0], point[1], "o", color="blue")

        # Determine label position based on slope
        if slope > 0:
            xytext = (0, -5)  # Place label below the point
            va = "top"
        else:
            xytext = (0, 5)  # Place label above the point
            va = "bottom"

        ax.annotate(
            f"({point[0]}, {point[1]})",
            xy=(point[0], point[1]),
            xytext=xytext,
            textcoords="offset points",
            ha="left",
            va=va,
        )

    plt.grid(True)
    plt.tight_layout()

    # save file
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_graphing_line.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def generate_coordinate_planes(points):
    max_x = max(point["x"] for point in points) + 1
    max_y = max(point["y"] for point in points) + 1

    fig, ax = plt.subplots()
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    if max_x <= 10:
        x_ticks = range(0, max_x + 1, 1)  # Ticks every 1 unit
    elif 10 < max_x <= 30:
        x_ticks = range(0, max_x + 1, 2)  # Ticks every 2 units
    else:
        x_ticks = range(0, max_x + 1, 5)  # Ticks every 5 units for larger values

    if max_y <= 10:
        y_ticks = range(0, max_y + 1, 1)  # Ticks every 1 unit
    elif 10 < max_y <= 30:
        y_ticks = range(0, max_y + 1, 2)  # Ticks every 2 units
    else:
        y_ticks = range(0, max_y + 1, 5)  # Ticks every 5 units for larger values

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    for point in points:
        x = point["x"]
        y = point["y"]
        label = point["label"]
        ax.scatter(x, y, label=f"{label} ({x}, {y})")
        ax.text(x + 0.1, y + 0.1, label, fontsize=12, color="blue")  # Label the point

    ax.grid(True)
    # plt.show()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_coordinate_planes_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


def add_points_four_quadrant(points: list[PlotPoint], label_points=True):
    # Use a fixed figure size to prevent huge images
    fig, ax = plt.subplots(figsize=(8, 8))
    AXIS_LIMIT = 20
    MIN_AXIS_LIMIT = 5

    # Check if any point is outside the supported range
    for point in points:
        if not (-AXIS_LIMIT <= point.x <= AXIS_LIMIT) or not (
            -AXIS_LIMIT <= point.y <= AXIS_LIMIT
        ):
            raise ValueError(
                f"Point ({point.x}, {point.y}) is outside the supported axis range [-{AXIS_LIMIT}, {AXIS_LIMIT}]."
            )

    # Compute min/max of data, with 1 unit padding
    x_min = min(point.x for point in points) - 1
    x_max = max(point.x for point in points) + 1
    y_min = min(point.y for point in points) - 1
    y_max = max(point.y for point in points) + 1

    # Clamp to [-20, 20]
    x_min = max(x_min, -AXIS_LIMIT)
    x_max = min(x_max, AXIS_LIMIT)
    y_min = max(y_min, -AXIS_LIMIT)
    y_max = min(y_max, AXIS_LIMIT)

    # Enforce minimum axis range of [-5, 5]
    if x_max - x_min < 10:
        x_center = (x_max + x_min) / 2
        x_min = min(x_min, x_center - 5)
        x_max = max(x_max, x_center + 5)
    if y_max - y_min < 10:
        y_center = (y_max + y_min) / 2
        y_min = min(y_min, y_center - 5)
        y_max = max(y_max, y_center + 5)

    # Make axes symmetric and equal about zero
    axis_limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    axis_limit = min(max(axis_limit, MIN_AXIS_LIMIT), AXIS_LIMIT)
    axis_limit = int(np.ceil(axis_limit))  # Always round up to nearest integer
    x_min = y_min = -axis_limit
    x_max = y_max = axis_limit

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Major ticks: every 5 if axis_limit == 20, else every 2; minor always every 1
    if axis_limit == AXIS_LIMIT:
        major_tick = 5
    else:
        major_tick = 2
    ax.set_xticks(np.arange(x_min, x_max + 1, major_tick))
    ax.set_yticks(np.arange(y_min, y_max + 1, major_tick))
    ax.set_xticks(np.arange(x_min, x_max + 1, 1), minor=True)
    ax.set_yticks(np.arange(y_min, y_max + 1, 1), minor=True)

    # Set axes to cross at (0,0) and style spines
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Make tick labels smaller and normal weight
    ax.tick_params(axis="both", which="major", labelsize=12, width=1, length=6)
    ax.tick_params(axis="both", which="minor", labelsize=10, width=1, length=3)
    for label in ax.get_xticklabels():
        label.set_fontweight("normal")
        label.set_fontsize(12)
    for label in ax.get_yticklabels():
        label.set_fontweight("normal")
        label.set_fontsize(12)

    # Draw points and emphasize point labels
    for point in points:
        label = point.label
        x = point.x
        y = point.y
        ax.scatter(x, y, label=label)
        if label_points:
            ax.text(
                x + 0.1,
                y + 0.1,
                label,
                fontsize=14,
                fontweight="normal",
                color="black",
                family="sans-serif",
            )

    # Configure spines to intersect at (0,0)
    ax.spines["left"].set_position("zero")  # Place y-axis at x=0
    ax.spines["bottom"].set_position("zero")  # Place x-axis at y=0
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.grid(True, which="both")
    return plt, fig, ax


@stimulus_function
def plot_polygon_dilation(data: PolygonDilation):
    """
    Plot polygon dilations on a four-quadrant coordinate plane.
    Shows preimage and image with dilation rays and clear center marking.

    Args:
        data: PolygonDilation containing preimage, image, scale factor, and center of dilation
    """
    # Extract data for dilation visualizations
    preimage = data.preimage
    image = data.image
    center = data.center_of_dilation
    show_center = data.show_center
    scale_factor = data.scale_factor

    # Use pastel colors instead of dark ones
    def get_pastel_color(color):
        """Convert standard colors to pastel versions."""
        pastel_map = {
            "blue": "#87CEEB",  # Sky blue (pastel)
            "red": "#FFB6C1",  # Light pink (pastel red)
            "green": "#98FB98",  # Pale green
            "purple": "#DDA0DD",  # Plum (pastel purple)
            "orange": "#FFDAB9",  # Peach puff (pastel orange)
        }
        return pastel_map.get(color, color)

    # Apply pastel colors
    preimage_color = get_pastel_color(data.preimage_color)
    image_color = get_pastel_color(data.image_color)
    center_color = get_pastel_color(data.center_color)

    # Convert polygon points to plot_points.Point format for add_points_four_quadrant
    # Only include polygon points, not the center point for grid creation
    plot_points = []
    for point in preimage.points + image.points:
        plot_points.append(PlotPoint(x=point.x, y=point.y, label=point.label))

    # Create the coordinate plane without showing center point in grid calculation
    plt, fig, ax = add_points_four_quadrant(plot_points, label_points=False)

    # Calculate bounds for zooming with better adaptive scaling
    all_points = preimage.points + image.points
    min_x = min(point.x for point in all_points)
    max_x = max(point.x for point in all_points)
    min_y = min(point.y for point in all_points)
    max_y = max(point.y for point in all_points)

    # Calculate coordinate range
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Add more padding for better zoom level
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding_x = max(x_range * 0.35, 1.5)  # 35% padding or minimum 1.5 units
    padding_y = max(y_range * 0.35, 1.5)  # 35% padding or minimum 1.5 units

    # Set axis limits with better zoom level
    new_xlim = (min_x - padding_x, max_x + padding_x)
    new_ylim = (min_y - padding_y, max_y + padding_y)
    ax.set_xlim(new_xlim)
    ax.set_ylim(new_ylim)

    # Reset ticks to align with integer boundaries after changing axis limits
    x_min, x_max = new_xlim
    y_min, y_max = new_ylim

    # Create ticks at integer boundaries within the new limits
    x_ticks = np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1)
    y_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Conditional grid styling based on scale factor
    # Check if scale factor >= 1.0 and has .5 decimal (like 1.5, 2.5)
    has_half_decimal_above_one = (
        scale_factor >= 1.0 and abs(scale_factor - round(scale_factor)) == 0.5
    )

    if scale_factor < 1.0 or has_half_decimal_above_one:
        # For reductions (scale < 1) OR enlargements with .5 (like 1.5, 2.5): Use detailed grid
        x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max) + 1, 0.5)
        y_minor_ticks = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 0.5)

        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_minor_ticks, minor=True)
        ax.grid(True, which="both", alpha=0.7)  # Show both major and minor grid lines
    else:
        # For whole number enlargements (scale >= 1.0, like 1.0, 2.0, 3.0): Use clean grid
        ax.set_xticks([], minor=True)  # Remove minor ticks
        ax.set_yticks([], minor=True)  # Remove minor ticks
        ax.grid(True, which="major", alpha=0.7)  # Show only major grid lines
        ax.grid(False, which="minor")  # Hide minor grid lines

    # Function to get lighter color for labels
    def get_label_color(polygon_color):
        """Get a slightly darker version of the pastel color for labels."""
        color_map = {
            "#87CEEB": "#4682B4",  # Steel blue for sky blue
            "#FFB6C1": "#DC143C",  # Crimson for light pink
            "#98FB98": "#32CD32",  # Lime green for pale green
            "#DDA0DD": "#9370DB",  # Medium slate blue for plum
            "#FFDAB9": "#FF8C00",  # Dark orange for peach puff
        }
        return color_map.get(polygon_color, polygon_color)

    # Draw dilation rays FIRST (lowest z-order) if showing center
    if show_center:
        for pre_point, img_point in zip(preimage.points, image.points):
            # Skip if the preimage point is at the center (no ray needed)
            if pre_point.x == center.x and pre_point.y == center.y:
                continue

            # Calculate direction vector from center to preimage point
            dx = pre_point.x - center.x
            dy = pre_point.y - center.y
            length = (dx**2 + dy**2) ** 0.5

            if length > 0:
                # Normalize direction vector
                dx_norm = dx / length
                dy_norm = dy / length

                # Calculate extension distance - start from center, go well beyond image point
                # Use the distance from center to image point as baseline for extension
                center_to_image_dist = (
                    (img_point.x - center.x) ** 2 + (img_point.y - center.y) ** 2
                ) ** 0.5
                extension = max(
                    1.5, center_to_image_dist * 0.4
                )  # At least 1.5 units or 40% of center-to-image distance

                # End point: extend beyond the furthest relevant point
                if scale_factor >= 1.0:
                    # For enlargements: extend beyond image point
                    end_x = img_point.x + extension * dx_norm
                    end_y = img_point.y + extension * dy_norm
                else:
                    # For reductions: extend beyond preimage point
                    end_x = pre_point.x + extension * dx_norm
                    end_y = pre_point.y + extension * dy_norm

                # Draw single ray from center through both points and beyond - much more visible
                ax.plot(
                    [center.x, end_x],
                    [center.y, end_y],
                    color="darkslategray",  # Darker, more visible color
                    linestyle="--",
                    alpha=0.9,  # Higher alpha for better visibility
                    linewidth=2.5,  # Thicker line
                    zorder=2,  # Higher z-order to be more prominent
                )

    # Collect all points from both polygons for comprehensive overlap detection
    all_dilation_points = list(preimage.points) + list(image.points)
    if show_center:
        all_dilation_points.append(center)

    # Global list to track ALL label positions across both polygons
    global_used_label_positions = []

    # Modified helper function to draw a polygon with global overlap awareness
    def draw_polygon_globally_aware(polygon, color, label_suffix=""):
        nonlocal global_used_label_positions
        label_color = get_label_color(color)

        # Plot polygon vertices
        for point in polygon.points:
            ax.scatter(point.x, point.y, color=color, s=20, zorder=5)

        # Smart labeling system to prevent overlaps - considers ALL dilation points
        def calculate_smart_label_position(point, all_points, used_positions):
            """
            Calculate optimal label position that avoids overlaps with other points and labels.
            Returns (x, y, ha, va) for label position and alignment.
            """
            # Calculate polygon center for reference
            polygon_center_x = sum(p.x for p in polygon.points) / len(polygon.points)
            polygon_center_y = sum(p.y for p in polygon.points) / len(polygon.points)

            # Base distance from point (respecting memory preference for appropriate distances)
            base_distance = 0.35

            # Try multiple positions around the point (8 cardinal and diagonal directions)
            candidate_positions = [
                (base_distance, 0, "left", "center"),  # Right
                (-base_distance, 0, "right", "center"),  # Left
                (0, base_distance, "center", "bottom"),  # Up
                (0, -base_distance, "center", "top"),  # Down
                (
                    base_distance * 0.7,
                    base_distance * 0.7,
                    "left",
                    "bottom",
                ),  # Top-right
                (
                    -base_distance * 0.7,
                    base_distance * 0.7,
                    "right",
                    "bottom",
                ),  # Top-left
                (
                    base_distance * 0.7,
                    -base_distance * 0.7,
                    "left",
                    "top",
                ),  # Bottom-right
                (
                    -base_distance * 0.7,
                    -base_distance * 0.7,
                    "right",
                    "top",
                ),  # Bottom-left
            ]

            # Score each position based on multiple criteria
            best_position = None
            best_score = -1

            for dx, dy, ha, va in candidate_positions:
                label_x = point.x + dx
                label_y = point.y + dy

                # Calculate score for this position
                score = 0

                # 1. Distance from ALL other points (preimage, image, center)
                min_distance_to_points = float("inf")
                for other_point in all_points:
                    if other_point != point:
                        dist = (
                            (label_x - other_point.x) ** 2
                            + (label_y - other_point.y) ** 2
                        ) ** 0.5
                        min_distance_to_points = min(min_distance_to_points, dist)

                # Add score based on distance from other points
                score += min(min_distance_to_points, 2.0) * 10  # Cap at 2.0 distance

                # 2. Distance from ALL other labels (both polygons)
                min_distance_to_labels = float("inf")
                for used_x, used_y in used_positions:
                    dist = ((label_x - used_x) ** 2 + (label_y - used_y) ** 2) ** 0.5
                    min_distance_to_labels = min(min_distance_to_labels, dist)

                if min_distance_to_labels != float("inf"):
                    # Heavy penalty for being too close to existing labels
                    if min_distance_to_labels < 0.6:  # Minimum label separation
                        score -= 50  # Heavy penalty for overlap risk
                    else:
                        score += min(min_distance_to_labels, 2.0) * 5

                # 3. Prefer positions that point away from polygon center (outward labeling)
                direction_from_center = (
                    (point.x - polygon_center_x) ** 2
                    + (point.y - polygon_center_y) ** 2
                ) ** 0.5
                if direction_from_center > 0:
                    point_to_center_x = (
                        point.x - polygon_center_x
                    ) / direction_from_center
                    point_to_center_y = (
                        point.y - polygon_center_y
                    ) / direction_from_center
                    label_direction_x = dx / base_distance if abs(dx) > 0.1 else 0
                    label_direction_y = dy / base_distance if abs(dy) > 0.1 else 0

                    # Dot product: positive means label points away from center (good)
                    outward_score = (
                        point_to_center_x * label_direction_x
                        + point_to_center_y * label_direction_y
                    )
                    score += outward_score * 3

                # 4. Prefer positions that keep labels within reasonable bounds
                if -6 <= label_x <= 6 and -6 <= label_y <= 6:
                    score += 5  # Bonus for staying in bounds
                else:
                    score -= 20  # Penalty for going out of bounds

                # 5. Slight preference for right/up positions for readability
                if dx > 0:  # Right side
                    score += 1
                if dy > 0:  # Top side
                    score += 1

                if score > best_score:
                    best_score = score
                    best_position = (label_x, label_y, ha, va)

            return (
                best_position
                if best_position
                else (point.x + 0.35, point.y + 0.35, "left", "bottom")
            )

        # For scale factors < 1.0, use simplified coordinate labeling to avoid overlap
        if scale_factor < 1.0:
            # Group points by coordinate and show simplified labels
            coordinate_groups = {}
            for point in polygon.points:
                coord_key = (int(round(point.x)), int(round(point.y)))
                if coord_key not in coordinate_groups:
                    coordinate_groups[coord_key] = []
                coordinate_groups[coord_key].append(point)

            # Show one coordinate label per unique coordinate with smart positioning
            for (x_coord, y_coord), points in coordinate_groups.items():
                if len(points) > 1:
                    # Multiple points at same coordinate - show just the coordinate
                    label_text = (
                        str(x_coord) if x_coord == y_coord else f"{x_coord},{y_coord}"
                    )
                else:
                    # Single point - show vertex label
                    label_text = points[0].label + label_suffix

                # Use smart positioning for the representative point
                rep_point = points[0]
                label_x, label_y, ha, va = calculate_smart_label_position(
                    rep_point, all_dilation_points, global_used_label_positions
                )
                global_used_label_positions.append((label_x, label_y))

                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    fontsize=9,
                    color=label_color,
                    fontweight="bold",
                    ha=ha,
                    va=va,
                    zorder=10,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.9,
                        linewidth=0.5,
                    ),
                )
        else:
            # For scale factors >= 1.0, use smart labeling for each point
            for point in polygon.points:
                label_x, label_y, ha, va = calculate_smart_label_position(
                    point, all_dilation_points, global_used_label_positions
                )
                global_used_label_positions.append((label_x, label_y))

                ax.text(
                    label_x,
                    label_y,
                    point.label + label_suffix,
                    fontsize=11,
                    color=label_color,
                    fontweight="bold",
                    ha=ha,
                    va=va,
                    zorder=10,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor=color,
                        alpha=0.9,
                        linewidth=0.5,
                    ),
                )

        # Draw polygon edges (lower z-order than axis)
        polygon_x = [point.x for point in polygon.points] + [polygon.points[0].x]
        polygon_y = [point.y for point in polygon.points] + [polygon.points[0].y]
        ax.plot(polygon_x, polygon_y, color=color, linewidth=2, zorder=3)

    # Draw preimage and image with global overlap awareness
    draw_polygon_globally_aware(preimage, preimage_color)
    draw_polygon_globally_aware(
        image, image_color, "'"
    )  # Add prime notation to image vertices

    # Draw center of dilation if requested
    if show_center:
        # Use a more prominent marker for the center
        ax.scatter(
            center.x,
            center.y,
            color=center_color,
            s=100,
            marker="o",
            edgecolors="black",
            linewidths=2,
            zorder=6,
        )

        # Label the center point with better spacing and clearer formatting
        if center.x == 0 and center.y == 0:
            center_label = center.label  # Just "O" for origin
        else:
            # Format coordinates nicely
            x_str = str(int(center.x)) if center.x == int(center.x) else str(center.x)
            y_str = str(int(center.y)) if center.y == int(center.y) else str(center.y)
            center_label = f"{center.label}({x_str}, {y_str})"

        # Use moderate offset for good spacing without being too far
        offset_x = 0.35 if center.x != 0 or center.y != 0 else 0.25
        offset_y = -0.35 if center.x != 0 or center.y != 0 else -0.3

        ax.text(
            center.x + offset_x,
            center.y + offset_y,
            center_label,
            fontsize=11,
            color=get_label_color(center_color),
            fontweight="bold",
            ha="left",
            va="top",
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=center_color,
                alpha=0.9,
                linewidth=0.5,
            ),
        )

        # Add scale factor annotation in top-right corner to avoid y-axis overlap
    ax.text(
        0.98,  # Move to right side
        0.98,  # Keep at top
        f"Scale Factor: {scale_factor}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="right",  # Right-align the text
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="lightyellow",
            edgecolor="orange",
            alpha=0.9,
        ),
        zorder=15,
    )

    # Set axis limits with better bounds checking - ALWAYS include (0,0)
    all_x = [p.x for p in preimage.points + image.points]
    all_y = [p.y for p in preimage.points + image.points]
    if show_center:
        all_x.append(center.x)
        all_y.append(center.y)

    # Calculate bounds with adaptive padding based on scale factor
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)

    # Use smaller padding for large scale factors to prevent unnecessary whitespace
    if scale_factor >= 3.0:
        padding = max(
            0.5, max(x_range, y_range) * 0.1
        )  # Minimal padding for large scale
    elif scale_factor >= 2.0:
        padding = max(
            0.8, max(x_range, y_range) * 0.12
        )  # Small padding for medium scale
    else:
        padding = max(
            1.0, max(x_range, y_range) * 0.15
        )  # Normal padding for small scale

    # Calculate limits but ALWAYS include (0,0) for proper axis intersection
    x_min = min(min(all_x) - padding, 0)  # Ensure x_min <= 0
    x_max = max(max(all_x) + padding, 0)  # Ensure x_max >= 0
    y_min = min(min(all_y) - padding, 0)  # Ensure y_min <= 0
    y_max = max(max(all_y) + padding, 0)  # Ensure y_max >= 0

    # For reasonable bounds, try to stay within (-10 to 10) but don't cut off content
    max_reasonable_bound = 10

    # Only apply bounds limits if the content actually fits within them
    if max(all_x) <= max_reasonable_bound and min(all_x) >= -max_reasonable_bound:
        x_min = max(x_min, -max_reasonable_bound)
        x_max = min(x_max, max_reasonable_bound)

    if max(all_y) <= max_reasonable_bound and min(all_y) >= -max_reasonable_bound:
        y_min = max(y_min, -max_reasonable_bound)
        y_max = min(y_max, max_reasonable_bound)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Configure spines - MUST be done after setting limits for proper centering
    ax.spines["left"].set_position("zero")  # Place y-axis at x=0
    ax.spines["bottom"].set_position("zero")  # Place x-axis at y=0
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Grid and ticks
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Set integer ticks for better readability
    x_ticks = range(int(x_min), int(x_max) + 1)
    y_ticks = range(int(y_min), int(y_max) + 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/polygon_dilation_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


@stimulus_function
def plot_points_four_quadrants(points: PointList):
    plt, fig, ax = add_points_four_quadrant(points.root, label_points=False)
    # Add grid for regular point plotting (not polygon dilation)
    ax.grid(True, which="both", alpha=0.7)

    # Draw each point and label it next to the point (not in the legend)
    for point in points.root:
        ax.scatter(
            point.x, point.y, s=80, zorder=5
        )  # zorder=5 ensures points are on top
        # Place label slightly offset from the point
        ax.text(point.x + 0.2, point.y + 0.2, point.label, fontsize=14, zorder=6)

    # Remove the legend (no ax.legend())

    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_four_coordinate_planes_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def plot_points_four_quadrants_with_label(points: PointList):
    plt, fig, ax = add_points_four_quadrant(points.root, label_points=False)
    # Add grid for regular point plotting (not polygon dilation)
    ax.grid(True, which="both", alpha=0.7)

    def format_coord(value: float) -> str:
        """Format a coordinate with at most one decimal (e.g., 2 -> 2, 2.0 -> 2, 2.34 -> 2.3)."""
        rounded = round(value * 10) / 10.0
        if abs(rounded - int(rounded)) < 1e-9:
            return str(int(rounded))
        return f"{rounded:.1f}"

    # Enforce at most 26 points and assign A..Z labels in order
    num_points = len(points.root)
    if num_points > 26:
        raise ValueError(
            "plot_points_four_quadrants_with_label supports at most 26 points (A-Z)."
        )

    # Draw each point and label it as "A (x, y)" above the point
    for idx, point in enumerate(points.root):
        ax.scatter(point.x, point.y, s=80, zorder=5)
        # Assigned alphabet label (A..Z)
        assigned_label = chr(ord("A") + idx)
        # Combined identifier + coordinates label
        coord_text = (
            f"{assigned_label} ({format_coord(point.x)}, {format_coord(point.y)})"
        )
        # Use pixel-based offset so label never overlaps the dot regardless of axis scale
        # Positive offset places label above; flip below if near top edge
        ymax = ax.get_ylim()[1]
        offset_points = 12 if point.y < ymax - 1 else -12
        va_setting = "bottom" if offset_points > 0 else "top"

        ax.annotate(
            coord_text,
            xy=(point.x, point.y),
            xytext=(0, offset_points),
            textcoords="offset points",
            fontsize=14,
            ha="center",
            va=va_setting,
            zorder=6,
            color="royalblue",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="black",
                alpha=1.0,
                linewidth=0.8,
            ),
        )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_four_coordinate_planes_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def plot_polygon_four_quadrants(points: PointList):
    plt, fig, ax = add_points_four_quadrant(points.root)

    polygon_points = points.root + [points.root[0]]
    polygon_x = [point.x for point in polygon_points]
    polygon_y = [point.y for point in polygon_points]
    ax.plot(polygon_x, polygon_y, linestyle="-", color="blue")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/polygon_four_coordinate_planes_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


def calc_step(axis_value):
    if axis_value < 100:
        return math.floor(axis_value / 10)
    return math.floor(axis_value / 100) * 10


@stimulus_function
def plot_points_quadrant_one_with_context(stimulus_description: PointPlotWithContext):
    fig, ax = plt.subplots()

    x_title = stimulus_description.x_title
    y_title = stimulus_description.y_title
    points = stimulus_description.points

    max_x_value = max(p.x for p in points)
    max_y_value = max(p.y for p in points)

    step_x = calc_step(max_x_value) or 1
    step_y = calc_step(max_y_value) or 1

    for point in points:
        label = point.label
        x = point.x
        y = point.y

        ax.scatter(x, y, label=label)
        ax.text(x + 0.1, y + 0.1, label, fontsize=14)  # Small offset for the labels

    x_max = max([point.x for point in points]) + max(4, step_x)
    y_max = max([point.y for point in points]) + max(4, step_y)

    ax.set_xticks(np.arange(0, x_max, step=step_x))
    ax.set_yticks(np.arange(0, y_max, step=step_y))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis="x", labelsize=14, rotation=90)
    ax.tick_params(axis="y", labelsize=14)

    ax.set_xlabel(x_title if x_title else "X-axis", fontsize=16)
    ax.set_ylabel(y_title if y_title else "Y-axis", fontsize=16)
    ax.grid(True)

    plt.legend(fontsize=16)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_coordinate_planes_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def plot_points_quadrant_one(stimulus_description: PointPlotWithContext):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_title = stimulus_description.x_title
    y_title = stimulus_description.y_title
    points = stimulus_description.points

    for point in points:
        label = point.label
        x = point.x
        y = point.y
        ax.scatter(x, y, label=label)
        ax.text(x + 0.1, y + 0.1, label, fontsize=14)

    x_max = max([point.x for point in points])
    y_max = max([point.y for point in points])

    # Use MaxNLocator for dynamic ticks to get a good interval
    temp_x_locator = MaxNLocator(
        nbins="auto", steps=[1, 2, 5, 10], integer=True, prune=None
    )
    temp_y_locator = MaxNLocator(
        nbins="auto", steps=[1, 2, 5, 10], integer=True, prune=None
    )

    # Temporarily set the locators to determine good intervals
    ax.xaxis.set_major_locator(temp_x_locator)
    ax.yaxis.set_major_locator(temp_y_locator)
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    def find_optimal_interval(coordinates, max_value, is_y_axis=False):
        """
        Find the interval from [1, 2, 5, 10, 50, 100] that:
        1. Ensures no point is more than 1 unit away from a grid line
        2. Balances having points on grid lines vs avoiding too many grid lines
        3. Never uses interval 1 for y-axis when max value > 20
        """
        possible_intervals = [1, 2, 5, 10, 50, 100]

        # Special constraint: never use interval 1 for y-axis when max > 20
        if is_y_axis and max_value > 20:
            possible_intervals = [2, 5, 10, 50, 100]

        best_interval = possible_intervals[
            0
        ]  # Use first available interval as fallback
        best_score = -float("inf")

        for interval in possible_intervals:
            # Check constraint: no point more than 1 unit from grid line
            max_distance = 0
            for coord in coordinates:
                nearest_grid = round(coord / interval) * interval
                distance = abs(coord - nearest_grid)
                max_distance = max(max_distance, distance)

            # Skip this interval if constraint is violated
            if max_distance > 1:
                continue

            # Count how many points fall exactly on grid lines
            points_on_grid = 0
            for coord in coordinates:
                if (
                    abs(coord % interval) < 1e-10
                ):  # Account for floating point precision
                    points_on_grid += 1

            # Calculate number of grid lines this would create
            num_grid_lines = max_value / interval

            # Score: balance points on grid vs grid density
            # Start with a base score that favors reasonable grid density
            score = 0

            # Reward points on grid lines
            score += points_on_grid * 3

            # Prefer intervals that create a reasonable number of grid lines (8-15 range)
            if 8 <= num_grid_lines <= 15:
                score += 10  # Bonus for good grid density
            elif num_grid_lines < 8:
                score -= (8 - num_grid_lines) * 2  # Penalty for too few lines
            else:  # num_grid_lines > 15
                score -= (num_grid_lines - 15) * 3  # Penalty for too many lines

            if score > best_score:
                best_score = score
                best_interval = interval

        return best_interval

    # Calculate optimal intervals for both axes
    x_coords = [point.x for point in points]
    y_coords = [point.y for point in points]

    x_interval = find_optimal_interval(x_coords, x_max, is_y_axis=False)
    y_interval = find_optimal_interval(y_coords, y_max, is_y_axis=True)

    # Create explicit tick locations that extend one interval beyond the max values
    x_ticks = list(range(0, int(x_max + x_interval) + 1, int(x_interval)))
    y_ticks = list(range(0, int(y_max + y_interval) + 1, int(y_interval)))

    # Set explicit tick locations
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Set axis limits to the last tick
    ax.set_xlim(0, max(x_ticks))
    ax.set_ylim(0, max(y_ticks))
    ax.tick_params(axis="x", labelsize=14, rotation=45)
    ax.tick_params(axis="y", labelsize=14)

    if x_title:
        ax.set_xlabel(x_title, fontsize=16, labelpad=10)
    if y_title:
        ax.set_ylabel(y_title, fontsize=16, labelpad=10)

    ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=16)
    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_coordinate_planes_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_stats_scatterplot(data: StatsScatterplot):
    points = data.points
    line_of_best_fit = data.line_of_best_fit

    # Extract x and y values from points
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]

    # Calculate the best fit line
    slope = line_of_best_fit.slope
    intercept = line_of_best_fit.intercept

    # Floating-point tolerance for determining if a point lies *exactly* on the line
    # A very small value is used to avoid false positives caused by rounding.
    tolerance = 1e-6

    # Separate points that are on the line from those that aren't
    points_on_line_x = []
    points_on_line_y = []
    points_off_line_x = []
    points_off_line_y = []

    for point in points:
        # Calculate expected y value for this x on the line
        expected_y = slope * point.x + intercept
        # Classify the point based on whether it is (within tolerance) on the line
        if abs(point.y - expected_y) <= tolerance:
            points_on_line_x.append(point.x)
            points_on_line_y.append(point.y)
        else:
            points_off_line_x.append(point.x)
            points_off_line_y.append(point.y)

    # Create the scatterplot with different colors
    if points_off_line_x:  # Plot points not on the line in black
        plt.scatter(
            points_off_line_x,
            points_off_line_y,
            color="black",
            s=80,
            alpha=0.7,
            zorder=3,
        )

    if points_on_line_x:  # Plot points on the line in red
        plt.scatter(
            points_on_line_x,
            points_on_line_y,
            color="red",
            s=100,
            alpha=0.9,
            zorder=3,
        )

    # Calculate the best fit line range
    best_fit_x = list(range(0, int(max(x_values)) + 5))
    best_fit_y = [slope * x + intercept for x in best_fit_x]

    # Plot the best fit line
    plt.plot(best_fit_x, best_fit_y, color="blue", linewidth=2, zorder=2)

    # Add darker major and minor gridlines behind data using low z-order
    plt.grid(
        which="major",
        color="gray",
        linestyle="-",
        linewidth=0.7,
        alpha=0.6,
        zorder=0,
    )
    plt.grid(
        which="minor",
        color="gray",
        linestyle=":",
        linewidth=0.5,
        alpha=0.4,
        zorder=0,
    )
    plt.minorticks_on()

    # Set x and y axis limits
    plt.xlim(0, max(x_values) + 2)
    plt.ylim(0, max(y_values) + 2)

    # Ensure every highlighted point lies exactly on the line; otherwise raise an error
    for x_pt, y_pt in zip(points_on_line_x, points_on_line_y):
        if abs(y_pt - (slope * x_pt + intercept)) > tolerance:
            raise ValueError(
                f"Highlighted point ({x_pt}, {y_pt}) does not lie on the line of best fit."
            )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/stats_scatter_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def draw_linear_diagram(stimulus_description: LinearDiagram):
    # Create a range of x values
    x = np.linspace(-10, 10, 400)

    # Set up a subplot with aspect ratio 1
    ax = plt.figure().add_subplot(1, 1, 1)

    # Move left y-axis and bottom x-axis to the middle
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")

    # Eliminate upper and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Set x and y axis major ticks (every 2 units)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_yticks(np.arange(-10, 11, 2))

    # Set x and y axis minor ticks (every 1 unit)
    ax.set_xticks(np.arange(-10, 11, 1), minor=True)
    ax.set_yticks(np.arange(-10, 11, 1), minor=True)

    ax.grid(True, which="both")  # Add Grid Lines

    # Cycle through the list of lines
    for line in stimulus_description.lines:
        y = line.slope * x + line.y_intercept
        ax.plot(x, y, color="black")

    # Plot the intersection point
    intersection_point = stimulus_description.intersection_point
    plt.scatter(intersection_point.x, intersection_point.y, color="black")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/linear_diagram_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def plot_nonlinear(stimulus_description: NonlinearGraph):
    equation_type = stimulus_description.equation_type
    parameters = stimulus_description.parameters

    # Create a range of x values
    x = np.linspace(-10, 10, 400)

    # Define the polynomial equations
    equations = {
        "quadratic": lambda p, x: (p.coef1 or 0) * x**2
        + (p.coef2 or 0) * x
        + (p.coef3 or 0),
        "cubic": lambda p, x: (p.coef1 or 0) * x**3
        + (p.coef2 or 0) * x**2
        + (p.coef3 or 0) * x
        + (p.coef4 or 0),
        "quartic": lambda p, x: (p.coef1 or 0) * x**4
        + (p.coef2 or 0) * x**3
        + (p.coef3 or 0) * x**2
        + (p.coef4 or 0) * x
        + (p.coef5 or 0),
        "quintic": lambda p, x: (p.coef1 or 0) * x**5
        + (p.coef2 or 0) * x**4
        + (p.coef3 or 0) * x**3
        + (p.coef4 or 0) * x**2
        + (p.coef5 or 0) * x
        + (p.coef6 or 0),
    }

    # Calculate y values based on the equation type
    y = equations[equation_type](parameters, x)

    # Set up a subplot with aspect ratio 1
    fig, ax = plt.subplots()
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.set_xticks(np.arange(-10, 11, 1))
    ax.set_yticks(np.arange(-10, 11, 1))
    ax.grid(True)

    # Plot the equation
    ax.plot(x, y)

    # Save the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/plot_nonlinear_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def multiple_bar_graph(stimulus_description: MultipleBarGraph):
    # Extracting data
    categories = [item.category for item in stimulus_description.data]
    x_data = [item.x_axis_subcategory_data for item in stimulus_description.data]
    subcategory_labels = stimulus_description.x_axis_subcategories
    title = stimulus_description.title
    x_axis_label = stimulus_description.x_axis_label
    y_axis_label = stimulus_description.y_axis_label

    num_subcategories = len(subcategory_labels)

    bar_width = 0.35
    space_between_groups = 0.5
    index = np.arange(len(categories)) * (
        num_subcategories * bar_width + space_between_groups
    )

    fig, ax = plt.subplots()

    # Plotting each subcategory data
    for i, category in enumerate(categories):
        ax.bar(index + i * bar_width, x_data[i], bar_width, label=category)

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (len(categories) - 1) / 2)
    ax.set_xticklabels(subcategory_labels)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Adding horizontal gridlines
    ax.grid(True, axis="y", color="black", linestyle="solid", linewidth=0.7, alpha=0.7)

    # Setting custom y-ticks to control the distance between the gridlines
    max_value = max(max(data) for data in x_data)
    step_size = (
        max_value // 10 if max_value > 100 else 10
    )  # Adjust step size dynamically
    ax.set_yticks(np.arange(0, max_value + step_size, step_size))

    # Slanting the x-axis labels
    plt.xticks(rotation=45)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/multiple_bar_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )

    fig.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def draw_grouped_bar_chart(stimulus_description: GroupedBarChart):
    # Access the list of dictionaries within the 'data' key
    data_list = stimulus_description.data

    # Extract data from the list of dictionaries, preserving original order of conditions
    groups = sorted(list(set([d.group for d in data_list])))
    conditions = []
    for d in data_list:  # Iterate through data_list
        if d.condition not in conditions:  # Check if condition is already in the list
            conditions.append(d.condition)  # Append if it's not in the list

    values = []
    errors = []  # Initialize errors list

    for group in groups:
        group_values = []
        group_errors = []
        for condition in conditions:
            entry = next(
                (d for d in data_list if d.group == group and d.condition == condition),
                None,
            )
            if entry:
                group_values.append(entry.value)
                # Use get() to handle missing 'error' key, defaulting to None
                group_errors.append(entry.error if hasattr(entry, "error") else None)
            else:
                group_values.append(0)
                group_errors.append(None)  # Append None for missing error values
        values.append(group_values)
        errors.append(group_errors)

    num_groups = len(groups)
    num_conditions = len(conditions)

    total_width = 0.8
    bar_width = total_width / num_groups
    condition_spacing = 0.2

    if num_conditions * (total_width + condition_spacing) > num_conditions:
        condition_spacing = (num_conditions - (num_conditions * total_width)) / (
            num_conditions - 1
        )

    x = np.arange(num_conditions) * (num_groups * bar_width + condition_spacing)

    fig, ax = plt.subplots(figsize=(10, 6))

    values = np.array(values)

    for i, group in enumerate(groups):
        x_coords = x + i * bar_width
        # Check if errors are present before plotting
        if errors[i] and all(err is not None for err in errors[i]):
            ax.bar(
                x_coords, values[i], bar_width, label=group, yerr=errors[i], capsize=5
            )
        else:
            ax.bar(x_coords, values[i], bar_width, label=group)

    # Use labels and title from stimulus_description
    ax.set_ylabel(stimulus_description.y_label)
    ax.set_xlabel(stimulus_description.x_label)
    ax.set_title(stimulus_description.title)

    offset = bar_width * (num_groups / 2)
    ax.set_xticks(x + offset - bar_width / 2)  # Center the labels within each group

    ax.set_xticklabels(conditions)

    ax.legend()

    plt.ylim(0, max([max(v) for v in values]) * 1.2)
    file_name = f"{settings.additional_content_settings.image_destination_folder}/grouped_bar_chart{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )

    return file_name


@stimulus_function
def draw_multi_line_graph(stimulus_description: LineGraphItem):
    plt.figure(figsize=(10, 6))

    for series in stimulus_description.data_series:
        marker = series.marker

        # Check if marker is valid
        valid_markers = ["o", "v", "<", ">", "s", "P", "*", "+", "x"]

        # Plot the line, using the marker only if valid
        if marker in valid_markers:
            plt.plot(
                series.x_values,
                series.y_values,
                label=series.label,
                marker=marker,
            )
        else:  # If marker is invalid or None, plot without marker
            plt.plot(series.x_values, series.y_values, label=series.label)

    plt.xlabel(stimulus_description.x_axis.label)
    plt.ylabel(stimulus_description.y_axis.label)

    plt.tight_layout()  # Adjust subplot parameters for a tight layout.
    plt.subplots_adjust(top=0.85)  # Adjust the top margin

    if stimulus_description.title:
        plt.title(stimulus_description.title)

    if stimulus_description.x_axis.range:
        plt.xlim(stimulus_description.x_axis.range)
    if stimulus_description.y_axis.range:
        plt.ylim(stimulus_description.y_axis.range)

    plt.grid(True)
    plt.legend()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_line_graph{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )

    plt.close()

    return file_name


@stimulus_function
def draw_combined_graphs(stimulus_description: CombinedGraphs):
    """
    Accepts a CombinedGraphs Pydantic model instance.
    Uses dot-style access to call the appropriate graphing methods.
    """
    graphs = stimulus_description.graphs

    # Adjust figure layout based on the number of graphs you expect
    # (example: 2 subplots if you always expect 2 graphs)
    fig, axes = plt.subplots(1, len(graphs), figsize=(20, 6))

    # If there's just one graph, axes won't be a list, so handle that:
    if len(graphs) == 1:
        axes = [axes]

    for i, graph_item in enumerate(graphs):
        ax = axes[i]
        if graph_item.graph_type == "bar_graph":
            draw_multi_bar_chart(graph_item, ax)
        elif graph_item.graph_type == "line_graph":
            draw_line_graphs(graph_item, ax)

    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"draw_combined_graphs{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )

    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )

    return file_name


@stimulus_function
def draw_multi_bar_chart(bar_graph: BarGraphItem, ax):
    """
    Accepts a BarGraphItem Pydantic model instead of a dict.
    """
    data_list = bar_graph.data

    groups = sorted({d.group for d in data_list})
    conditions = list(dict.fromkeys(d.condition for d in data_list))

    values = []
    errors = []

    # Collect data in parallel arrays
    for group in groups:
        group_values = []
        group_errors = []
        for condition in conditions:
            entry = next(
                (d for d in data_list if d.group == group and d.condition == condition),
                None,
            )
            if entry:
                group_values.append(entry.value)
                group_errors.append(entry.error or 0)
            else:
                group_values.append(0)
                group_errors.append(0)
        values.append(group_values)
        errors.append(group_errors)

    num_groups = len(groups)
    num_conditions = len(conditions)

    total_width = 0.8
    bar_width = total_width / num_groups

    # Handle spacing for single or multiple conditions
    if num_conditions > 1:
        condition_spacing = max(
            0.2, (1 - num_conditions * total_width) / (num_conditions - 1)
        )
    else:
        condition_spacing = 0.2

    x = np.arange(num_conditions) * (num_groups * bar_width + condition_spacing)

    values = np.array(values)
    colors = ["#1f77b4", "#aec7e8", "#2ca02c", "#ffbb78"]

    for i, group_label in enumerate(groups):
        x_coords = x + i * bar_width
        ax.bar(
            x_coords,
            values[i],
            bar_width,
            label=group_label,
            yerr=errors[i],
            capsize=5,
            color=colors[i % len(colors)],
        )

    ax.set_ylabel(bar_graph.y_label)
    ax.set_xlabel(bar_graph.x_label)

    # Wrap long titles if needed
    wrapped_title = "\n".join(textwrap.wrap(bar_graph.title, width=100))
    ax.set_title(wrapped_title, pad=20, loc="center")

    offset = bar_width * (num_groups / 2)
    ax.set_xticks(x + offset - bar_width / 2)
    ax.set_xticklabels(conditions)
    ax.legend()

    max_value_with_error = max(
        value + error
        for value_list, error_list in zip(values, errors)
        for value, error in zip(value_list, error_list)
    )
    ax.set_ylim(0, max_value_with_error * 1.2)


@stimulus_function
def draw_line_graphs(line_graph: LineGraphItem):
    """
    Accepts a LineGraphItem Pydantic model instead of a dict.
    Uses dot notation throughout.
    """
    # Create figure and axis internally
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use random colors for better visual variety
    # Define a larger palette of vibrant colors
    color_palette = [
        "#1f77b4",  # Bright blue
        "#ff7f0e",  # Bright orange
        "#2ca02c",  # Bright green
        "#d62728",  # Bright red
        "#9467bd",  # Bright purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#ff6b6b",  # Coral red
        "#4ecdc4",  # Teal
        "#45b7d1",  # Sky blue
        "#96ceb4",  # Mint green
        "#feca57",  # Golden yellow
        "#ff9ff3",  # Hot pink
        "#54a0ff",  # Electric blue
        "#5f27cd",  # Deep purple
        "#00d2d3",  # Turquoise
        "#ff9f43",  # Orange
    ]

    # Shuffle the color palette for random assignment
    random.shuffle(color_palette)

    for i, series in enumerate(line_graph.data_series):
        # Use modulo to cycle through shuffled colors
        color = color_palette[i % len(color_palette)]
        ax.plot(
            series.x_values,
            series.y_values,
            label=series.label if series.label else None,
            marker=series.marker if series.marker else "o",
            markersize=8,  # Slightly larger markers
            linewidth=3,  # Thicker lines for better visibility
            color=color,
            alpha=0.9,  # Slight transparency for depth
            markeredgecolor="white",  # White edge around markers
            markeredgewidth=1.5,  # Edge width for contrast
            markerfacecolor=color,  # Fill marker with line color
        )

    def use_latex(text):
        # Simple function to check if latex is used
        return "$" in text

    wrapped_title = "\n".join(textwrap.wrap(line_graph.title, width=100))
    ax.set_title(
        wrapped_title,
        pad=20,
        loc="center",
        usetex=use_latex(line_graph.title),
        fontsize=14,
        fontweight="bold",  # Make title more prominent
    )

    ax.set_xlabel(
        line_graph.x_axis.label,
        usetex=use_latex(line_graph.x_axis.label),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        line_graph.y_axis.label,
        usetex=use_latex(line_graph.y_axis.label),
        fontsize=12,
        fontweight="bold",
    )

    if line_graph.x_axis.range:
        ax.set_xlim(line_graph.x_axis.range)
    if line_graph.y_axis.range:
        ax.set_ylim(line_graph.y_axis.range)

    # NEW: Support categorical tick labels like ["Thursday", "Friday", ...]
    # Assumes all series share identical x positions; uses first series' x_values
    tick_labels = getattr(line_graph.x_axis, "tick_labels", None)
    if tick_labels:
        try:
            first_series = line_graph.data_series[0]
            xs = first_series.x_values
            # Only apply if lengths match to avoid ValueError
            if len(tick_labels) == len(xs):
                ax.set_xticks(xs)
                ax.set_xticklabels(tick_labels, rotation=0)
        except Exception:
            # Fail silently if structure isn't as expected; graph will still render
            pass

    # Enhance readability with better styling
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="both", which="major", labelsize=11, colors="black")

    # Set background color for better contrast
    ax.set_facecolor("#fafafa")

    # Add legend only if labels exist
    if any(s.label for s in line_graph.data_series):
        ax.legend(
            frameon=True,
            fontsize=10,
            fancybox=True,
            shadow=True,
            framealpha=0.9,
            edgecolor="black",
            facecolor="white",
        )

    # Save the file and return the filename
    file_name = f"{settings.additional_content_settings.image_destination_folder}/line_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_blank_coordinate_plane(stimulus: BlankCoordinatePlane):
    """
    Draw a blank coordinate plane with axes at edges (standard mathematical style).
    Dynamic scaling based on stimulus properties or defaults to [-10, 10].
    """
    # Get ranges with sensible defaults
    x_min = getattr(stimulus, "x_min", -10)
    x_max = getattr(stimulus, "x_max", 10)
    y_min = getattr(stimulus, "y_min", -10)
    y_max = getattr(stimulus, "y_max", 10)

    # Always ensure origin is included
    x_min = min(x_min, 0)
    x_max = max(x_max, 0)
    y_min = min(y_min, 0)
    y_max = max(y_max, 0)

    # Calculate ranges for better tick spacing
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Adjust figure size based on aspect ratio
    aspect_ratio = x_range / y_range
    if aspect_ratio > 1.5:
        fig_size = (10, 6)
    elif aspect_ratio < 0.67:
        fig_size = (6, 10)
    else:
        fig_size = (8, 8)

    fig, ax = plt.subplots(figsize=fig_size, dpi=150)

    # Set limits with small padding for better appearance
    padding_x = x_range * 0.02
    padding_y = y_range * 0.02
    ax.set_xlim(x_min - padding_x, x_max + padding_x)
    ax.set_ylim(y_min - padding_y, y_max + padding_y)

    # Smart tick spacing to avoid overcrowding
    x_ticks = list(range(x_min, x_max + 1))
    y_ticks = list(range(y_min, y_max + 1))

    # Thin out ticks if range is too large (>20 units)
    if x_range > 20:
        x_ticks = [x for x in x_ticks if x % 2 == 0]  # Every 2 units
    if y_range > 20:
        y_ticks = [y for y in y_ticks if y % 2 == 0]

    # If still too crowded (>30 units), use every 5
    if x_range > 30:
        x_ticks = [x for x in range(x_min, x_max + 1) if x % 5 == 0]
    if y_range > 30:
        y_ticks = [y for y in range(y_min, y_max + 1) if y % 5 == 0]

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # 4-quadrant style: axes through origin
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Style the axes
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Position ticks properly
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Light grid that doesn't overpower
    ax.grid(True, which="major", color="#d3d3d3", alpha=0.6, linewidth=0.8, zorder=0)

    # Clean tick styling
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        colors="black",
        direction="out",
        length=4,
        width=1,
    )

    # Square aspect ratio for geometric accuracy
    ax.set_aspect("equal", adjustable="box")

    # Only add titles if specifically provided and clean
    if hasattr(stimulus, "x_axis_title") and stimulus.x_axis_title:
        # Place title below the coordinate plane, not overlapping
        ax.text(
            0.5,
            -0.1,
            stimulus.x_axis_title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    if hasattr(stimulus, "y_axis_title") and stimulus.y_axis_title:
        # Place title to the left, rotated
        ax.text(
            -0.08,
            0.5,
            stimulus.y_axis_title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=11,
        )

    # Ensure origin is included
    if not (x_min <= 0 <= x_max and y_min <= 0 <= y_max):
        plt.close(fig)
        raise RuntimeError("Origin (0,0) must be within coordinate plane bounds")

    # Save with clean layout
    out_dir = settings.additional_content_settings.image_destination_folder
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(
        out_dir,
        f"blank_coordinate_plane_{time.time_ns()}.{settings.additional_content_settings.stimulus_image_format}",
    )

    plt.savefig(
        file_name,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=False,
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=600,
    )
    plt.close(fig)
    return file_name
