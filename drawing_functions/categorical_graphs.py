import time
from functools import reduce
from math import gcd

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.categorical_graph import (
    CategoricalGraphList,
    MultiGraphList,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_diagram import (
    StatsBarDiagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.tree_diagram import (
    TreeDiagram,
)
from content_generators.settings import settings
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator


def calculate_optimal_step(frequencies: list[float]):
    max_freq = max(frequencies)
    if max_freq <= 5:
        return 1

    # Define allowed step values in order of preference
    allowed_steps = [2, 3, 4, 5, 6, 8, 5, 10]

    # Add multiples of 10 for larger ranges
    if max_freq > 50:
        allowed_steps.extend([20, 50, 100, 200, 500, 1000])

    # Find the best step that gives reasonable number of ticks
    best_step = 1
    best_score = 0

    for step in allowed_steps:
        # Calculate how many ticks this step would create
        num_ticks = max_freq // step + 1

        # Prefer steps that create between 3 and 10 ticks
        if 3 <= num_ticks <= 10:
            # Score based on how many data points align with ticks
            tick_marks = np.arange(0, max_freq + 1, step)
            aligned_points = sum(frequency in tick_marks for frequency in frequencies)
            score = aligned_points - abs(num_ticks - 6)  # Prefer around 6 ticks

            if score > best_score:
                best_score = score
                best_step = step

    # If no good step found, use the smallest step that gives reasonable ticks
    if best_step == 1:
        for step in allowed_steps:
            num_ticks = max_freq // step + 1
            if 2 <= num_ticks <= 15:  # More lenient range
                best_step = step
                break

    return best_step


def calculate_minor_step(frequencies: list[float], major_step: int):
    major_ticks = np.arange(0, max(frequencies) + major_step, major_step)
    off_ticks = [f for f in frequencies if f not in major_ticks]

    if not off_ticks:
        return major_step

    possible_steps = []
    for freq in off_ticks:
        lower_tick = max([tick for tick in major_ticks if tick <= freq], default=0)  # type: ignore
        upper_tick = min(
            [tick for tick in major_ticks if tick >= freq],  # type: ignore
            default=max(major_ticks),  # type: ignore
        )
        distances = [freq - lower_tick, upper_tick - freq]
        for dist in distances:
            if dist != 0:
                # Convert to integer by multiplying by a factor to preserve precision
                # and then finding the GCD
                scaled_dist = int(
                    dist * 100
                )  # Scale by 100 to preserve 2 decimal places
                possible_steps.append(scaled_dist)

    if not possible_steps:
        return major_step

    # Calculate GCD of scaled distances, then scale back down
    minor_step_scaled = reduce(gcd, possible_steps)
    minor_step = minor_step_scaled / 100.0

    return minor_step


def smart_x_axis_labels(
    ax, categories: list[str], figure_width: float = 10, labelsize: int = 12
):
    """
    Intelligently handle x-axis label rotation and spacing based on:
    - Number of categories
    - Length of category names
    - Available figure width
    """
    num_categories = len(categories)
    max_label_length = max(len(cat) for cat in categories) if categories else 0

    # Calculate approximate space needed per character (in inches, assuming 12pt font)
    char_width_inches = 0.1  # Approximate character width in inches

    # Calculate total width needed for all labels if horizontal
    total_label_width = sum(len(cat) * char_width_inches for cat in categories)
    available_width = figure_width * 0.8  # Use 80% of figure width for labels

    # Decision logic for rotation
    if num_categories <= 3:
        # Few categories: always horizontal
        rotation = 0
        ha = "center"
    elif num_categories <= 5 and max_label_length <= 8:
        # Medium categories with short labels: horizontal
        rotation = 0
        ha = "center"
    elif total_label_width <= available_width:
        # Labels fit horizontally
        rotation = 0
        ha = "center"
    elif max_label_length <= 6:
        # Short labels: use 45° rotation
        rotation = 45
        ha = "right"
    else:
        # Long labels: use 90° rotation for maximum readability
        rotation = 90
        ha = "right"

    # Apply the rotation and alignment
    ax.tick_params(axis="x", labelsize=labelsize, rotation=rotation)
    if rotation > 0:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)

    return rotation


def create_bar_graph(
    categories: list[str],
    frequencies: list[float],
    title: str,
    x_label: str,
    y_label: str,
):
    major_step = calculate_optimal_step(frequencies)
    minor_step = calculate_minor_step(frequencies, major_step)

    # Dynamically adjust figure width based on number of categories and label length
    base_width = 10
    min_width = 6  # Minimum width to prevent overly wide single bars
    num_categories = len(categories)
    max_label_length = max(len(cat) for cat in categories) if categories else 0

    # Calculate width based on number of categories and label length
    if num_categories == 1:
        # For single bars, use a much smaller width that's not too wide
        figure_width = max(
            min_width, base_width * 0.1
        )  # 40% of base width for single bars
    elif num_categories > 6:
        figure_width = base_width + (num_categories - 6) * 0.8
    elif max_label_length > 10:
        figure_width = base_width + (max_label_length - 10) * 0.3
    else:
        figure_width = base_width

    # Ensure minimum width constraint
    figure_width = max(figure_width, min_width)

    fig, ax = plt.subplots(figsize=(figure_width, 6))

    # For single bars, adjust the bar width to look more proportional
    if num_categories == 1:
        # Use a much narrower bar width for single bars (default is 0.8)
        bar_width = 0.2
        ax.bar(
            categories, frequencies, color="skyblue", edgecolor="black", width=bar_width
        )
        # Set x-axis limits to create more space around the single bar
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.bar(categories, frequencies, color="skyblue", edgecolor="black")

    # Use smart x-axis label management instead of fixed 45° rotation
    smart_x_axis_labels(ax, categories, figure_width, labelsize=14)

    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.set_xlabel(x_label, fontweight="bold", fontsize=14)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=14)

    # Set major ticks with labels using major step
    major_ticks = np.arange(0, max(frequencies) + major_step, major_step)
    ax.set_yticks(major_ticks)

    # Set minor ticks without labels using minor step
    minor_ticks = np.arange(0, max(frequencies) + minor_step, minor_step)
    ax.set_yticks(minor_ticks, minor=True)

    # Show both major and minor grid lines for alignment
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.6, color="black")
    ax.grid(True, which="minor", axis="y", linestyle="--", linewidth=0.4, color="grey")

    ax.tick_params(axis="y", labelsize=18)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    fig.canvas.draw()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/categorical_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


def create_histogram(
    categories: list[str],
    frequencies: list[float],
    title: str,
    x_label: str,
    y_label: str,
):
    if sum(frequencies) > 120:
        raise ValueError("Data set is unacceptably large")
    if len(categories) > 6:
        raise ValueError("Too many bins")

    # category_map = {cat: i for i, cat in enumerate(sorted(set(categories)))}
    # category_labels = sorted(category_map, key=category_map.get)
    # values = [category_map[cat] for cat in categories]
    category_map = {cat: i for i, cat in enumerate(categories)}
    category_labels = list(category_map.keys())
    values = [category_map[cat] for cat in categories]

    # For histogram, we need to repeat values based on frequencies
    # Since frequencies can be floats, we'll use a different approach
    values_repeated = []
    for value, freq in zip(values, frequencies):
        # Add whole number of repetitions
        values_repeated.extend([value] * int(freq))
        # For half values, add one more with a special marker
        if freq != int(freq):
            values_repeated.append(value)

    major_step = calculate_optimal_step(frequencies)
    minor_step = calculate_minor_step(frequencies, major_step)
    bins = np.arange(min(values_repeated) - 0.5, max(values_repeated) + 1.5)
    # Dynamically adjust figure width for histograms too
    base_width = 10
    max_label_length = (
        max(len(cat) for cat in category_labels) if category_labels else 0
    )

    if len(category_labels) > 4:
        figure_width = base_width + (len(category_labels) - 4) * 0.8
    elif max_label_length > 10:
        figure_width = base_width + (max_label_length - 10) * 0.3
    else:
        figure_width = base_width

    fig, ax = plt.subplots(figsize=(figure_width, 6))
    ax.hist(
        values_repeated,
        bins=bins,
        color="skyblue",
        alpha=0.7,
        rwidth=1.0,
        edgecolor="black",
    )
    ax.set_xticks(np.arange(min(values_repeated), max(values_repeated) + 1))
    ax.set_xticklabels(category_labels)

    # Use smart x-axis label management for histograms
    smart_x_axis_labels(ax, category_labels, figure_width)

    ax.set_title(title, fontweight="bold", fontsize=16)
    ax.set_xlabel(x_label, fontweight="bold", fontsize=14)
    ax.set_ylabel(y_label, fontweight="bold", fontsize=14)
    ax.set_yticks(np.arange(0, max(frequencies) + 1, major_step))
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, color="grey")

    if major_step > 1:
        ax.yaxis.set_minor_locator(AutoMinorLocator(int(major_step / minor_step)))
        ax.grid(
            True, which="minor", axis="y", linestyle=":", linewidth=0.5, color="grey"
        )

    ax.tick_params(axis="y", labelsize=14)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    fig.canvas.draw()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/categorical_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


def create_picture_graph(
    categories: list[str],
    frequencies: list[float],
    title: str,
    star_value: int = 1,
    star_unit: str = "items",
    show_half_star_value: bool = False,
):
    """
    Creates and returns a filename for a picture graph in which each '★' character
    is replaced by a vector star marker (*) that is left-aligned inside its table cell,
    with spacing exactly matching the original text-based stars. Supports half stars.
    """

    # 1. Build "dummy" star strings so that column widths/font sizes match the old version
    # For half stars, we round up to determine the width needed
    star_representation = ["★" * int(freq + 0.5) for freq in frequencies]

    # 2. Estimate figure height (add a bit of vertical padding per row)
    fig_height = len(categories) * 0.5 + 2
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # 3. Hide axes, set the title
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=20, pad=1)

    # 4. Create the table with textual stars; we will immediately clear that text
    table_data = [
        [f"{cat}  ", rep] for cat, rep in zip(categories, star_representation)
    ]
    table = ax.table(cellText=table_data, colLabels=None, loc="center", cellLoc="left")

    # 5. Lock in the same font size & scale
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 3)

    # 6. Compute explicit column widths
    max_cat_length = (max(len(cat) for cat in categories) if categories else 0) + 1
    max_star_length = max(len(rep) for rep in star_representation) * 0.045

    # 7. Iterate all cells; set width/height and make background transparent
    for (row_idx, col_idx), cell in table.get_celld().items():
        if col_idx == 0:
            cell.set_width(max_cat_length * 0.03)
            cell.set_text_props(ha="right")
        elif col_idx == 1:
            cell.set_width(max_star_length)
            cell.get_text().set_text("")
        cell.set_height(0.2)
        cell.set_facecolor("none")  # Make cell background transparent

    # 8. Set the table's z-order so that it is above the white rectangle
    table.set_zorder(6)  # Rectangle=5, Table=6, Star edge=6/7, Whole stars=7

    # 9. Draw the canvas once so that get_window_extent(renderer) will work
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore

    # 10. Instead of measuring a dummy text-star, compute per-slot width directly:
    max_freq = max(
        int(freq + 0.5) for freq in frequencies
    )  # number of star slots in the largest row
    per_star_width = max_star_length / float(
        max_freq
    )  # divide the cell into max_freq equal slots

    # 11. For each row i, fetch its star-cell's bbox, then overlay N vector-stars at exactly
    #     (x0 + star_axes_width*(k + 0.5), y0 + height/2). That packs them flush-left inside the cell.
    for i, freq in enumerate(frequencies):
        if freq <= 0:
            continue

        cell = table[(i, 1)]  # row i, column 1 (the "star" column)
        disp_bbox = cell.get_window_extent(renderer)  # in display (pixels)
        axes_bbox = disp_bbox.transformed(
            ax.transAxes.inverted()
        )  # convert to axes (0…1)
        x0, y0 = axes_bbox.x0, axes_bbox.y0
        height = axes_bbox.height

        # Vertical center of the cell (axes coords)
        y_center = y0 + height / 2

        # Draw each of the freq stars. We place the k-th star at:
        #   x_center = x0 + per_star_width * (k + 0.5)
        # That means the left edge of the first star (k=0) sits at x0,
        # and each successive star is exactly one "per_star_width" further to the right.
        whole_stars = int(freq)
        has_half_star = freq != whole_stars

        # Draw whole stars
        for k in range(whole_stars):
            x_center = x0 + per_star_width * (k + 0.5)
            ax.scatter(
                x_center,
                y_center,
                marker="*",
                s=800,
                color="blue",
                edgecolors="black",
                linewidths=0.5,
                transform=ax.transAxes,
                zorder=7,  # Highest zorder for whole stars
            )
        # Draw half star if needed
        if has_half_star:
            x_center = x0 + per_star_width * (whole_stars + 0.5)
            # Draw blue fill for half star
            ax.scatter(
                x_center,
                y_center,
                marker="*",
                s=800,
                color="blue",
                edgecolors="none",
                transform=ax.transAxes,
                zorder=3,  # Above background, below rectangle
            )
            # Overlay a white rectangle to cover the right half
            rect_width = per_star_width * 0.52
            rect_height = height * 0.8
            ax.add_patch(
                Rectangle(
                    (x_center, y_center - rect_height / 2),
                    rect_width,
                    rect_height,
                    color="white",
                    transform=ax.transAxes,
                    zorder=4,  # Rectangle above fill, below table
                    linewidth=0,
                )
            )
            # Draw just the edge of the star over everything except whole stars
            ax.scatter(
                x_center,
                y_center,
                marker="*",
                s=800,
                color="none",
                edgecolors="black",
                linewidths=0.5,
                transform=ax.transAxes,
                zorder=6,  # Edge above table, below whole stars
            )

    # 12. Add legend showing what each star represents
    # Determine vertical positions based on whether half star legend is shown
    if show_half_star_value:
        full_star_y = 0.20  # Higher position when both legends are shown
        half_star_y = 0.07  # Higher position for half star to avoid cropping
    else:
        full_star_y = 0.05  # Standard position when only one legend

    # First, add the star marker at the legend position
    ax.scatter(
        0.02,
        full_star_y,
        marker="*",
        s=800,  # Same size as table stars
        color="blue",
        edgecolors="black",
        linewidths=0.5,
        transform=ax.transAxes,
        zorder=10,
    )

    # Add the legend text next to the star (format number properly)
    star_value_formatted = f"{star_value:g}"  # Remove .0 for whole numbers
    legend_text = f" = {star_value_formatted} {star_unit}"
    ax.text(
        0.035,  # Position text to the right of the star
        full_star_y,  # Same height as the star
        legend_text,
        transform=ax.transAxes,
        fontsize=16,  # Match table font size
        fontweight="bold",
        verticalalignment="center",
        horizontalalignment="left",
        zorder=10,
    )

    # Add half star legend if requested
    if show_half_star_value:
        half_value = star_value / 2

        # Draw half star for legend - simpler approach for clean cut
        x_center = 0.02
        y_center = half_star_y

        # Draw blue fill for half star
        ax.scatter(
            x_center,
            y_center,
            marker="*",
            s=800,  # Same size as table stars
            color="blue",
            edgecolors="none",
            transform=ax.transAxes,
            zorder=3,
        )

        ax.add_patch(
            Rectangle(
                (
                    x_center,
                    y_center - rect_height / 2,
                ),  # Start slightly right of center
                rect_width,
                rect_height,
                color="white",
                transform=ax.transAxes,
                zorder=4,
                linewidth=0,
            )
        )

        # Draw star outline
        ax.scatter(
            x_center,
            y_center,
            marker="*",
            s=800,  # Same size as table stars
            color="none",
            edgecolors="black",
            linewidths=0.5,
            transform=ax.transAxes,
            zorder=6,
        )

        # Add half star legend text (format number properly)
        half_value_formatted = f"{half_value:g}"  # Remove .0 for whole numbers
        half_legend_text = f" = {half_value_formatted} {star_unit}"
        ax.text(
            0.035,
            half_star_y,
            half_legend_text,
            transform=ax.transAxes,
            fontsize=16,  # Match table font size
            fontweight="bold",
            verticalalignment="center",
            horizontalalignment="left",
            zorder=10,
        )

    # 13. Final layout, save, and close
    plt.tight_layout()
    fig.canvas.draw()

    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"categorical_graph_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
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
def create_multi_bar_graph(data: MultiGraphList):
    """
    Creates a 2x2 grid of bar graphs in a single image.
    Args:
        data: A MultiGraphList containing exactly 4 CategoricalGraph objects
    """
    if len(data) != 4:
        raise ValueError("data must contain exactly 4 sets of data")

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for idx, graph_info in enumerate(data):
        categories = [item.category for item in graph_info.data]
        frequencies = [item.frequency for item in graph_info.data]
        title = graph_info.title
        x_label = graph_info.x_axis_label
        y_label = graph_info.y_axis_label

        major_step = calculate_optimal_step(frequencies)
        minor_step = calculate_minor_step(frequencies, major_step)

        ax = axes[idx]
        ax.bar(categories, frequencies, color="skyblue", edgecolor="black")

        # Use smart x-axis label management for each subplot
        smart_x_axis_labels(
            ax, categories, figure_width=10, labelsize=26
        )  # Each subplot gets 10 inches width

        ax.set_title(title, fontweight="bold", fontsize=26)
        ax.set_xlabel(x_label, fontweight="bold", fontsize=22)
        ax.set_ylabel(y_label, fontweight="bold", fontsize=22)

        # Calculate number of ticks with minor step
        max_freq = max(frequencies)
        minor_ticks = np.arange(0, max_freq + minor_step, minor_step)
        major_ticks = np.arange(0, max_freq + major_step, major_step)

        # Use major steps if there would be more than 20 minor ticks, otherwise use minor steps
        if len(minor_ticks) > 20:
            # Use major ticks for labels
            ax.set_yticks(major_ticks)
        else:
            # Use minor steps for y-axis ticks so all bars fall on labeled tick marks
            ax.set_yticks(minor_ticks)

        # Always add minor ticks at data values that don't align with major ticks
        data_values = set(frequencies)
        minor_tick_positions = []

        for value in data_values:
            # Check if this value is not already covered by a major tick
            if not any(abs(value - major_tick) < 0.01 for major_tick in major_ticks):
                minor_tick_positions.append(value)

        # Set minor ticks at data values (unlabeled)
        if minor_tick_positions:
            ax.set_yticks(minor_tick_positions, minor=True)
        ax.grid(
            True, which="major", axis="y", linestyle="--", linewidth=0.5, color="grey"
        )
        # Add minor grid lines for data values when we have minor tick positions
        if minor_tick_positions:
            ax.grid(
                True,
                which="minor",
                axis="y",
                linestyle=":",
                linewidth=0.3,
                color="lightgrey",
            )

        ax.tick_params(axis="y", labelsize=24)

    plt.tight_layout()
    fig.canvas.draw()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_categorical_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
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
def create_multi_picture_graph(data: MultiGraphList):
    """
    Creates a 2x2 grid of picture graphs in a single image.
    Args:
        data: A MultiGraphList containing exactly 4 CategoricalGraph objects
    """
    if len(data) != 4:
        raise ValueError("data must contain exactly 4 sets of data")

    # Calculate total height needed for all picture graphs
    total_height = 0
    for graph_info in data:
        categories = [item.category for item in graph_info.data]
        fig_height = len(categories) * 0.7 + 2
        total_height = max(total_height, fig_height)

    fig, axes = plt.subplots(2, 2, figsize=(16, total_height * 2))
    axes = axes.flatten()

    for idx, graph_info in enumerate(data):
        categories = [item.category for item in graph_info.data]
        frequencies = [item.frequency for item in graph_info.data]
        title = graph_info.title
        config = graph_info._picture_config
        star_value = config.star_value
        star_unit = config.star_unit
        show_half_star_value = config.show_half_star_value

        ax = axes[idx]

        # 1. Build "dummy" star strings so that column widths/font sizes match the old version
        # For half stars, we round up to determine the width needed
        star_representation = ["★" * int(freq + 0.5) for freq in frequencies]

        # 2. Hide axes, set the title
        ax.axis("off")
        # Add more padding to the title for graphs with 4 or more categories
        title_padding = 40 if len(categories) >= 4 else 10
        ax.set_title(title, fontweight="bold", fontsize=16, pad=title_padding)

        # 3. Create the table with textual stars; we will immediately clear that text
        table_data = [
            [f"{cat}  ", rep] for cat, rep in zip(categories, star_representation)
        ]
        table = ax.table(
            cellText=table_data, colLabels=None, loc="center", cellLoc="left"
        )

        # 4. Lock in the same font size & scale
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(1, 3.5)

        # 5. Compute explicit column widths
        max_cat_length = (max(len(cat) for cat in categories) if categories else 0) + 1
        max_star_length = max(len(rep) for rep in star_representation) * 0.045

        # 7. Iterate all cells; set width/height and make background transparent
        for (row_idx, col_idx), cell in table.get_celld().items():
            if col_idx == 0:
                cell.set_width(max_cat_length * 0.03)
                cell.set_text_props(ha="right")
            elif col_idx == 1:
                cell.set_width(max_star_length)
                cell.get_text().set_text("")
            cell.set_height(0.3)
            cell.set_facecolor("none")  # Make cell background transparent

        # 8. Set the table's z-order so that it is above the white rectangle
        table.set_zorder(6)  # Rectangle=5, Table=6, Star edge=6/7, Whole stars=7

        # 9. Draw the canvas once so that get_window_extent(renderer) will work
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore

        # 10. Instead of measuring a dummy text-star, compute per-slot width directly:
        max_freq = max(
            int(freq + 0.5) for freq in frequencies
        )  # number of star slots in the largest row
        per_star_width = max_star_length / float(
            max_freq
        )  # divide the cell into max_freq equal slots

        # 11. For each row i, fetch its star-cell's bbox, then overlay N vector-stars at exactly
        #     (x0 + star_axes_width*(k + 0.5), y0 + height/2). That packs them flush-left inside the cell.
        for i, freq in enumerate(frequencies):
            if freq <= 0:
                continue

            cell = table[(i, 1)]  # row i, column 1 (the "star" column)
            disp_bbox = cell.get_window_extent(renderer)  # in display (pixels)
            axes_bbox = disp_bbox.transformed(
                ax.transAxes.inverted()
            )  # convert to axes (0…1)
            x0, y0 = axes_bbox.x0, axes_bbox.y0
            height = axes_bbox.height

            # Vertical center of the cell (axes coords)
            y_center = y0 + height / 2

            # Draw each of the freq stars. We place the k-th star at:
            #   x_center = x0 + per_star_width * (k + 0.5)
            # That means the left edge of the first star (k=0) sits at x0,
            # and each successive star is exactly one "per_star_width" further to the right.
            whole_stars = int(freq)
            has_half_star = freq != whole_stars

            # Draw whole stars
            for k in range(whole_stars):
                x_center = x0 + per_star_width * (k + 0.5)
                ax.scatter(
                    x_center,
                    y_center,
                    marker="*",
                    s=800,
                    color="blue",
                    edgecolors="black",
                    linewidths=0.5,
                    transform=ax.transAxes,
                    zorder=7,  # Highest zorder for whole stars
                )
            # Draw half star if needed
            if has_half_star:
                x_center = x0 + per_star_width * (whole_stars + 0.5)
                # Draw blue fill for half star
                ax.scatter(
                    x_center,
                    y_center,
                    marker="*",
                    s=800,
                    color="blue",
                    edgecolors="none",
                    transform=ax.transAxes,
                    zorder=5,  # Above background, below rectangle
                )
                # Overlay a white rectangle to cover the right half
                rect_width = per_star_width * 0.52
                rect_height = height * 0.8
                ax.add_patch(
                    Rectangle(
                        (x_center, y_center - rect_height / 2),
                        rect_width,
                        rect_height,
                        color="white",
                        transform=ax.transAxes,
                        zorder=5,  # Rectangle above fill, below table
                        linewidth=0,
                    )
                )
                # Draw just the edge of the star over everything except whole stars
                ax.scatter(
                    x_center,
                    y_center,
                    marker="*",
                    s=800,
                    color="none",
                    edgecolors="black",
                    linewidths=0.5,
                    transform=ax.transAxes,
                    zorder=6,  # Edge above table, below whole stars
                )

        # --- Legend (key) outside the axes at left, non-clipping ---
        legend_x = (
            -0.06
        )  # just outside the axes; avoids overlap without hitting the figure edge
        if show_half_star_value:
            full_star_y = 0.10
            half_star_y = 0.02
        else:
            full_star_y = 0.06

        # full star
        ax.scatter(
            legend_x,
            full_star_y,
            marker="*",
            s=800,
            color="blue",
            edgecolors="black",
            linewidths=0.5,
            transform=ax.transAxes,
            zorder=10,
            clip_on=False,
        )

        # full star text
        star_value_formatted = f"{star_value:g}"
        ax.text(
            legend_x + 0.025,
            full_star_y,
            f" = {star_value_formatted} {star_unit}",
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=10,
            clip_on=False,
        )

        if show_half_star_value:
            half_value = star_value / 2
            ax.scatter(
                legend_x,
                half_star_y,
                marker="*",
                s=800,
                color="blue",
                edgecolors="none",
                transform=ax.transAxes,
                zorder=5,
                clip_on=False,
            )
            rect_width = 0.03
            rect_height = 0.06
            ax.add_patch(
                Rectangle(
                    (legend_x, half_star_y - rect_height / 2),
                    rect_width,
                    rect_height,
                    color="white",
                    transform=ax.transAxes,
                    zorder=6,
                    linewidth=0,
                    clip_on=False,
                )
            )
            ax.scatter(
                legend_x,
                half_star_y,
                marker="*",
                s=800,
                color="none",
                edgecolors="black",
                linewidths=0.5,
                transform=ax.transAxes,
                zorder=7,
                clip_on=False,
            )
            half_value_formatted = f"{half_value:g}"
            ax.text(
                legend_x + 0.025,
                half_star_y,
                f" = {half_value_formatted} {star_unit}",
                transform=ax.transAxes,
                fontsize=18,
                fontweight="bold",
                va="center",
                ha="left",
                zorder=10,
                clip_on=False,
            )

    plt.tight_layout()

    # Add more vertical spacing for tall graphs
    if any(len(item.data) >= 4 for item in data):
        plt.subplots_adjust(hspace=0.6)

    # Add left margin for the legend that's positioned outside the axes
    plt.subplots_adjust(left=0.12)

    fig.canvas.draw()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_picture_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.4,  # ensures the outside-left key is fully captured
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


@stimulus_function
def create_categorical_graph(data: CategoricalGraphList):
    """Determines the graph type and calls the appropriate function to create it."""
    try:
        graph_info = data[0]
        graph_type = graph_info.graph_type
        title = graph_info.title
        x_label = graph_info.x_axis_label
        y_label = graph_info.y_axis_label
        categories = [item.category for item in graph_info.data]
        frequencies = [item.frequency for item in graph_info.data]

        if graph_type == "bar_graph":
            file_name = create_bar_graph(
                categories, frequencies, title, x_label, y_label
            )
        elif graph_type == "histogram":
            file_name = create_histogram(
                categories, frequencies, title, x_label, y_label
            )
        elif graph_type == "picture_graph":
            config = graph_info._picture_config
            file_name = create_picture_graph(
                categories,
                frequencies,
                title,
                config.star_value,
                config.star_unit,
                config.show_half_star_value,
            )
        else:
            file_name = create_bar_graph(
                categories, frequencies, title, x_label, y_label
            )

        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def create_stats_diagram_bar(stimulus_description: StatsBarDiagram):
    x_axis_label = stimulus_description.x_axis_label
    y_axis_label = stimulus_description.y_axis_label
    x_data = stimulus_description.x_axis_data
    y_data = stimulus_description.y_axis_data

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.bar(x_data, y_data, color="gray", edgecolor="black")

    # Add grid lines
    plt.grid(axis="y", linestyle="-", linewidth=0.7)

    # Set the axis labels
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    max_y = max(y_data)
    rounded_max = round(max_y / 10) * 10

    if rounded_max > max_y:
        # Set the limits for y-axis
        plt.ylim(0, rounded_max)

        # Generate ticks at every 5 units
        plt.yticks(range(0, rounded_max + 1, 5))
    else:
        new_max = rounded_max + 5
        plt.ylim(0, new_max)
        plt.yticks(range(0, new_max + 1, 5))

    file_name = f"{settings.additional_content_settings.image_destination_folder}/stats_diagram_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_tree_diagram(stimulus: TreeDiagram):
    def get_depth(node):
        if node is None:
            return 0
        return 1 + max(get_depth(node.left), get_depth(node.right))

    def count_leaves(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    def assign_positions(node, x, y, dx, positions, node_map, level=0):
        if node is None:
            return
        node_id = id(node)
        positions[node_id] = (x, y)
        node_map[node_id] = node
        if node.left:
            assign_positions(
                node.left,
                x - dx / (2**level),
                y - 1,
                dx,
                positions,
                node_map,
                level + 1,
            )
        if node.right:
            assign_positions(
                node.right,
                x + dx / (2**level),
                y - 1,
                dx,
                positions,
                node_map,
                level + 1,
            )

    depth = get_depth(stimulus.root)
    leaves = count_leaves(stimulus.root)
    dx = max(2, leaves)
    positions = {}
    node_map = {}
    assign_positions(stimulus.root, 0, 0, dx, positions, node_map)

    fig, ax = plt.subplots(figsize=(dx * 1.5, depth * 2))
    ax.axis("off")

    branch_offset = 0.2  # vertical distance from node to branch point
    label_offset = 0.12

    # Draw edges with branch points
    for node_id, (x, y) in positions.items():
        node = node_map[node_id]
        has_left = node.left is not None
        has_right = node.right is not None
        if has_left or has_right:
            branch_y = y - branch_offset
            # Draw lines from branch point to each child
            for child in [node.left, node.right]:
                if child:
                    child_id = id(child)
                    x2, y2 = positions[child_id]
                    ax.add_line(
                        mlines.Line2D(
                            [x, x2], [branch_y, y2], color="gray", linewidth=2
                        )
                    )

    # Draw node labels: always at (x, y) for internal nodes, (x, y-label_offset) for leaves
    for node_id, (x, y) in positions.items():
        node = node_map[node_id]
        y_label = y - label_offset
        ax.text(
            x,
            y_label,
            node.label,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            zorder=3,
        )

    all_x = [x for (x, y) in positions.values()]
    all_y = [y for (x, y) in positions.values()]
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

    plt.tight_layout()
    fig.canvas.draw()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/tree_diagram_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
