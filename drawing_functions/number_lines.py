import random
import time
from functools import reduce
from math import gcd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.dual_stats_line import (
    DualStatsLinePlot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.inequality_number_line import (
    InequalityNumberLine,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.multi_inequality_number_line import (
    MultiInequalityNumberLine,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.number_line import (
    DecimalComparisonNumberLine,
    ExtendedUnitFractionNumberLine,
    FixedStepNumberLine,
    MultiExtendedUnitFractionNumberLine,
    MultiLabeledUnitFractionNumberLine,
    NumberLine,
    Point,
    Range,
    UnitFractionNumberLine,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stats_line import (
    StatsLinePlot,
)
from content_generators.settings import settings
from content_generators.utils import Lerp
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator


def convert_fraction_to_latex(fraction_str: str) -> str:
    """Convert a fraction string to LaTeX stacked format.

    Args:
        fraction_str: A string that can be either a whole number ("1", "2")
                     or a fraction in slash notation ("3/4", "8/4")

    Returns:
        LaTeX formatted string for stacked fractions or the original string if whole number
    """
    if "/" in fraction_str:
        parts = fraction_str.split("/")
        if len(parts) == 2:
            numerator = parts[0].strip()
            denominator = parts[1].strip()
            return f"$\\frac{{{numerator}}}{{{denominator}}}$"
    return fraction_str  # Return as-is if it's a whole number


def get_fraction_fontsize(base_fontsize: int) -> int:
    """Get appropriate font size for fraction labels.

    Args:
        base_fontsize: Base font size

    Returns:
        Enhanced font size for fractions
    """
    return max(base_fontsize + 4, 16)  # At least 16pt for fractions


def find_gcd(list_values):
    x = reduce(gcd, list_values)
    return x


def scale_points(points):
    # Scale points to remove decimals by multiplying to transform them into integers
    scaled = [
        int(p * (10**10)) for p in points
    ]  # scale factor of 10^10 for safety with decimals
    return scaled


def lookup_step(points, range_min, range_max):
    # Arrays of values mapped to their steps
    values_for_step_three = [0.33, 0.66]
    values_for_step_six = [0.16, 0.83]
    values_for_step_seven = [0.14, 0.28, 0.42, 0.57, 0.71, 0.857]
    values_for_step_nine = [0.11, 0.22, 0.44, 0.77, 0.88]

    # Mapping step sizes to their arrays
    step_mappings = {
        1 / 3: values_for_step_three,
        1 / 6: values_for_step_six,
        1 / 7: values_for_step_seven,
        1 / 9: values_for_step_nine,
    }

    steps_found = []

    # Check if the range difference equals 1 before checking points
    if range_max - range_min == 1:
        # Check each point against each group of step values
        for point in points:
            for step, values in step_mappings.items():
                if point in values:
                    steps_found.append(step)
                    break  # Stop checking once the first match is found for this point

    # If no special cases are matched, fall back to determining the step using GCD
    if not steps_found:
        scaled_point_values = scale_points(points)
        gcd_step = find_gcd(scaled_point_values) / (10**10)
        return gcd_step  # Return GCD based step

    # Return the largest step found for any matching special cases
    return max(steps_found)


def determine_optimal_step(points: list[Point], range_min: int, range_max: int):
    point_values = [point.value for point in points]
    total_range = range_max - range_min
    final_step = lookup_step(point_values, range_min, range_max)

    if total_range <= 15:
        # Ensure final_step is a factor of one
        while not (1 / final_step).is_integer():
            final_step /= 2

    if final_step < 0.05:
        return 1

    return final_step


@stimulus_function
def create_number_line(stim_desc: NumberLine):
    try:
        fig, ax = plt.subplots()
        original_min = stim_desc.range.min
        original_max = stim_desc.range.max
        buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

        range_min = original_min - buffer
        range_max = original_max + buffer

        ax.set_xlim(range_min, range_max)
        ax.set_ylim(-0.25, 0.25)

        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)
        major_ticks = []
        # Determine major ticks
        total_range = original_max - original_min

        # Special case for unit fraction number line (0 to 1)
        if original_min == 0 and original_max == 1:
            major_ticks = [0, 1]
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        else:

            def get_optimal_step_size(min_val, max_val):
                range_size = max_val - min_val
                if range_size <= 2:
                    return 0.5  # For small ranges, use half steps
                elif range_size <= 5:
                    return 1  # For medium ranges, use whole numbers
                elif range_size <= 10:
                    return 2  # For larger ranges, use multiples of 2
                elif range_size <= 50:
                    return 5  # For medium-large ranges, use multiples of 5
                elif range_size <= 100:
                    return 10  # For ranges up to 100, use multiples of 10
                elif range_size <= 200:
                    return 20  # For ranges up to 200, use multiples of 20
                elif range_size <= 1000:
                    return 100  # For ranges up to 1000, use multiples of 100
                else:
                    return 200  # For very large ranges, use multiples of 200

            step_size = get_optimal_step_size(original_min, original_max)
            start = np.floor(original_min / step_size) * step_size
            end = np.ceil(original_max / step_size) * step_size
            major_ticks = np.arange(start, end + step_size, step_size)
            ax.xaxis.set_major_locator(FixedLocator(major_ticks.tolist()))

            if total_range <= 15 and total_range > 1:
                major_ticks = np.arange(
                    np.floor(original_min), np.ceil(original_max) + 1
                )
                ax.xaxis.set_major_locator(FixedLocator(major_ticks.tolist()))
            elif total_range > 1:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
                major_ticks = ax.xaxis.get_major_locator().tick_values(
                    original_min, original_max
                )
                ax.xaxis.set_major_locator(FixedLocator(list(major_ticks)))
            else:
                ax.xaxis.set_major_locator(FixedLocator(major_ticks.tolist()))

        optimal_step = 1
        if total_range == 1 and stim_desc.minor_divisions is not None:
            optimal_step = 1 / stim_desc.minor_divisions
        elif total_range > 120:
            optimal_step = 10
        elif total_range >= 15:
            optimal_step = 1
        elif total_range > 1:
            optimal_step = determine_optimal_step(
                stim_desc.points, original_min, original_max
            )

        minimum = original_min
        maximum = original_max
        if len(major_ticks) > 0:
            minimum = min(original_min, major_ticks[0])
            maximum = max(original_max, major_ticks[-1])

        valid_minor_ticks = np.arange(minimum, maximum + optimal_step, optimal_step)
        ax.xaxis.set_minor_locator(FixedLocator(valid_minor_ticks.tolist()))

        # Define the value mappings
        value_replacements = {
            0.33: 0.33333333,
            0.66: 0.66666666,
            # Add more mappings as needed
        }

        texts = []
        base_offset = 0.02  # Base vertical offset
        horizontal_offset = 0.05  # Horizontal offset for overlapping labels

        # Plot points and create text objects in their initial positions
        point_values = []  # Store original point values for leader lines
        for point in stim_desc.points:
            original_value = point.value
            point.value = value_replacements.get(original_value, original_value)
            ax.plot(point.value, 0, "ko", markersize=7)

            # Create text object at initial position (above the point)
            text = ax.text(
                point.value,
                base_offset,
                point.label,
                ha="center",
                va="bottom",
                fontsize=14,
            )
            texts.append(text)
            point_values.append(point.value)

        # Force a draw so text objects have real extents
        fig.canvas.draw()

        # Do multiple passes to catch cascading overlaps
        for _ in range(3):
            for i, t1 in enumerate(texts):
                for t2 in texts[i + 1 :]:
                    b1 = t1.get_window_extent()
                    b2 = t2.get_window_extent()
                    if b1.overlaps(b2):
                        x1, y1 = t1.get_position()
                        x2, y2 = t2.get_position()
                        # Push them apart horizontally
                        if x1 < x2:
                            t1.set_x(x1 - horizontal_offset)
                            t2.set_x(x2 + horizontal_offset)
                        else:
                            t1.set_x(x1 + horizontal_offset)
                            t2.set_x(x2 - horizontal_offset)
            # Redraw so that extents update for the next pass
            fig.canvas.draw()

        # Manually draw each minor tick with variable length and color
        if original_min == 0 and original_max == 1 and len(valid_minor_ticks) >= 12:
            tick_length_scale = 0.008  # Base scale for tick length
            valid_minor_ticks = valid_minor_ticks[
                (valid_minor_ticks != 0) & (valid_minor_ticks != 1)
            ]
            for i, tick_position in enumerate(valid_minor_ticks):
                tick_color = "gray" if i % 2 == 0 else "#606060"
                line_width = 1.2 if i % 2 == 0 else 1.4
                tick_length = tick_length_scale * (1 if i % 2 == 0 else 1.3)
                ax.plot(
                    [tick_position, tick_position],
                    [0, -tick_length],
                    color=tick_color,
                    linewidth=line_width,
                )
        else:
            ax.tick_params(axis="x", which="minor", length=7, width=1.2, color="gray")

        ax.tick_params(axis="x", which="major", length=10, width=1.5, color="black")

        plt.tight_layout()
        file_name = f"{settings.additional_content_settings.image_destination_folder}/number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            dpi=800,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()
        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def create_inequality_number_line(stim_desc: InequalityNumberLine):
    try:
        range_min = stim_desc.range.min
        range_max = stim_desc.range.max
        full_range = range_max - range_min
        fig, ax = plt.subplots(figsize=(Lerp.linear(5, 8, full_range / 20), 1.5))
        buffer = 0.5
        x_lim_min = range_min - buffer
        x_lim_max = range_max + buffer

        ax.set_xlim(x_lim_min, x_lim_max)
        ax.set_ylim(-0.12, 0.1)

        ax.spines["bottom"].set_position("zero")
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position("bottom")

        # Set the zorder of the x-axis line to be behind the points
        ax.spines["bottom"].set_zorder(0.5)

        # Set the default major step value to 1
        ax.xaxis.set_major_locator(MultipleLocator(1))

        # Set tick mark size, thickness, and direction
        ax.tick_params(
            axis="x", which="major", labelsize=16, size=16, width=2, direction="inout"
        )

        # Remove tick marks and numbers in the buffer zone
        ax.set_xticks(range(range_min, range_max + 1))
        ax.set_xticklabels(
            [str(x) for x in range(range_min, range_max + 1)],
            fontsize=16 if full_range > 10 else 18,
        )

        # Draw lines first
        line = stim_desc.line
        min_value = line.min or range_min - buffer / 2
        max_value = line.max or range_max + buffer / 2

        ax.plot([min_value, max_value], [0, 0], linewidth=4, color="black", zorder=1)

        # Add arrow at the end of the line
        if line.min is None:
            ax.annotate(
                "",
                xy=(min_value, 0),
                xytext=(min_value - buffer / 2, 0),
                arrowprops=dict(arrowstyle="<-", color="black", linewidth=4),
                zorder=1,
            )
        elif line.max is None:
            ax.annotate(
                "",
                xy=(max_value + buffer / 2, 0),
                xytext=(max_value, 0),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=4),
                zorder=1,
            )

        # Draw points last
        for point in stim_desc.points:
            value = point.value
            fill = "black" if point.fill else "white"
            ax.plot(
                value,
                0,
                marker="o",
                markersize=10,
                color=fill,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=2,
                fillstyle="full",
            )

        plt.tight_layout()
        file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_number_line.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
            dpi=800,
        )
        plt.close()
        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def create_unit_fraction_number_line(stim_desc: UnitFractionNumberLine):
    # ensure range is 0 to 1

    number_line = NumberLine(
        **stim_desc.model_dump(),
        range=Range(min=0, max=1),
    )

    file = create_number_line(number_line)
    return file


@stimulus_function
def create_extended_unit_fraction_number_line(
    stim_desc: ExtendedUnitFractionNumberLine,
):
    """Create an extended unit fraction number line that goes past 1.
    Only labels 0 and the fraction endpoint, with minor divisions for counting."""

    try:
        fig, ax = plt.subplots()
        original_min = stim_desc.range.min  # Should be 0
        original_max = stim_desc.range.max  # Should be the fraction value
        buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

        range_min = original_min - buffer
        range_max = original_max + buffer

        ax.set_xlim(range_min, range_max)
        ax.set_ylim(-0.25, 0.25)

        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)

        # Calculate minor tick step based on minor_divisions
        minor_step = original_max / stim_desc.minor_divisions
        minor_ticks = np.arange(0, original_max + minor_step, minor_step)

        if stim_desc.show_all_tick_labels or stim_desc.labeled_fraction is not None:
            # Support single labeled fraction regardless of show_all_tick_labels
            ax.xaxis.set_major_locator(FixedLocator(minor_ticks.tolist()))
            ax.xaxis.set_minor_locator(FixedLocator([]))

            if stim_desc.labeled_fraction is not None:
                # Label only 0 and the provided fraction tick; do not label endpoint here
                labels = [""] * len(minor_ticks)
                labels[0] = "0"
                # Find index of labeled fraction tick
                labeled_ratio = eval(stim_desc.labeled_fraction)
                labeled_value = original_min + labeled_ratio * (
                    original_max - original_min
                )
                idx = int(round((labeled_value - original_min) / minor_step))
                idx = max(0, min(idx, len(minor_ticks) - 1))
                labels[idx] = convert_fraction_to_latex(stim_desc.labeled_fraction)
                ax.set_xticklabels(labels)
            else:
                # Default: label all ticks as before
                tick_labels = []
                for j, tick_pos in enumerate(minor_ticks):
                    if abs(tick_pos - original_min) < 1e-6:
                        tick_labels.append("0")
                    elif abs(tick_pos - original_max) < 1e-6:
                        tick_labels.append(
                            convert_fraction_to_latex(stim_desc.endpoint_fraction)
                        )
                    else:
                        if abs(original_max - 1) < 1e-6 and original_min == 0:
                            tick_labels.append(
                                "0"
                                if j == 0
                                else f"$\\frac{{{j}}}{{{stim_desc.minor_divisions}}}$"
                            )
                        else:
                            if "/" in stim_desc.endpoint_fraction:
                                ep_num, ep_den = map(
                                    int, stim_desc.endpoint_fraction.split("/")
                                )
                                numi = j * ep_num // stim_desc.minor_divisions
                                tick_labels.append(
                                    "0"
                                    if numi == 0
                                    else f"$\\frac{{{numi}}}{{{ep_den}}}$"
                                )
                            else:
                                tick_labels.append(
                                    "0"
                                    if j == 0
                                    else f"$\\frac{{{j}}}{{{stim_desc.minor_divisions}}}$"
                                )
                ax.set_xticklabels(tick_labels)

            # Configure tick marks for labeled mode
            # Use consistent font size regardless of endpoint fraction format
            tick_fontsize = 16  # Consistent size for all tick labels
            ax.tick_params(
                axis="x",
                which="major",
                length=10,
                width=1.5,
                color="black",
                labelsize=tick_fontsize,
            )
        else:
            # Original behavior: only show major ticks at 0 and the endpoint
            major_ticks = [0, original_max]
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            # Remove default tick labels
            ax.set_xticklabels(["", ""])
            ax.tick_params(
                axis="x",
                which="major",
                labelbottom=False,
                length=10,
                width=1.5,
                color="black",
            )

            ax.xaxis.set_minor_locator(FixedLocator(minor_ticks.tolist()))
            ax.tick_params(axis="x", which="minor", length=7, width=1.2, color="gray")

        # Plot the number line
        ax.plot([original_min, original_max], [0, 0], "k-", linewidth=2)

        # Add endpoint labels only if not showing all tick labels.
        # Additionally, if labeled_fraction_tick is set, we omit the endpoint label even when not showing all ticks.
        if not stim_desc.show_all_tick_labels and stim_desc.labeled_fraction is None:
            # Use consistent font size for both endpoints
            endpoint_fontsize = (
                get_fraction_fontsize(12) if "/" in stim_desc.endpoint_fraction else 12
            )
            ax.text(0, -0.02, "0", ha="center", va="top", fontsize=endpoint_fontsize)
            ax.text(
                original_max,
                -0.02,
                convert_fraction_to_latex(stim_desc.endpoint_fraction),
                ha="center",
                va="top",
                fontsize=endpoint_fontsize,
            )

        # If labeled_fraction_tick is provided, ensure we show only 0 (already handled above) and that fraction tick label
        if (
            stim_desc.labeled_fraction is not None
            and not stim_desc.show_all_tick_labels
        ):
            # Override ticks to include all minor ticks but only label 0 and the target fraction tick
            ax.xaxis.set_major_locator(FixedLocator(minor_ticks.tolist()))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            labels = [""] * len(minor_ticks)
            labels[0] = "0"
            labeled_value = eval(stim_desc.labeled_fraction)
            idx = int(round((labeled_value - original_min) / minor_step))
            idx = max(0, min(idx, len(minor_ticks) - 1))
            labels[idx] = convert_fraction_to_latex(stim_desc.labeled_fraction)
            ax.set_xticklabels(labels)

        # Plot the dot point
        ax.plot(stim_desc.dot_point.value, 0, "ko", markersize=8)
        ax.text(
            stim_desc.dot_point.value,
            0.02,
            stim_desc.dot_point.label,
            ha="center",
            va="bottom",
            fontsize=12,
        )

        # Adjust layout
        plt.tight_layout()

        file_name = f"{settings.additional_content_settings.image_destination_folder}/extended_unit_fraction_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name

    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise e


@stimulus_function
def create_dot_plot(stimulus_description: StatsLinePlot):
    """
    Create a single dot plot with enhanced visual appeal.

    Args:
        stimulus_description: StatsLinePlot containing data and title
    """
    # Define 10 distinct colors for the dots
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    # Randomly select a color for this plot generation
    selected_color = random.choice(colors)

    # Generate the counts for each unique value
    values, counts = np.unique(stimulus_description.data, return_counts=True)

    # Set figure size based on the number of unique values
    if len(values) <= 10:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate dot size based on figure size for better proportion - INCREASED SIZE
    dot_size = 180 if len(values) <= 10 else 160  # Increased from 120/100 to 180/160

    # Calculate vertical spacing for dots
    dot_spacing = 0.08  # Reduced spacing between vertically stacked dots

    # Plot dots with enhanced styling
    for value, count in zip(values, counts):
        # Create y-positions with tighter, more consistent spacing
        y_positions = np.arange(count) * dot_spacing + dot_spacing

        # Plot dots with enhanced styling
        ax.scatter(
            [value] * count,
            y_positions,
            s=dot_size,  # Use scatter for better size control
            color=selected_color,
            alpha=0.8,  # Slight transparency for better visual appeal
            edgecolors="white",  # White border around dots
            linewidth=0.5,  # Thin border
            zorder=3,  # Ensure dots are on top
        )

    # Enhanced plot configuration
    ax.set_xticks(np.arange(min(values), max(values) + 1))
    ax.tick_params(axis="x", labelsize=11, colors="#333333")
    ax.set_yticks([])  # Remove y-ticks
    ax.tick_params(axis="y", left=False)

    # Enhanced title styling - match with dot color
    ax.set_title(
        stimulus_description.title,
        color=selected_color,  # Match the dot color
        fontweight="bold",
        fontsize=14,
        pad=20,  # More space above title
    )

    # Enhanced spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["bottom"].set_position(("data", 0))  # Align with bottom of dots

    # Set better axis limits with padding
    x_padding = 0.3
    y_padding = max(counts) * dot_spacing * 0.1
    max_y = max(counts) * dot_spacing + dot_spacing
    ax.set_xlim(min(values) - x_padding, max(values) + x_padding)
    ax.set_ylim(-y_padding * 2, max_y + y_padding)

    # Add subtle grid for x-axis only
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5, color="#cccccc")
    ax.set_axisbelow(True)  # Put grid behind dots

    # Remove top and right margins
    ax.margins(0.02)

    # Enhanced layout
    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/dot_plot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        dpi=300,  # Higher resolution for crisp output
        format=settings.additional_content_settings.stimulus_image_format,
        facecolor="white",  # Ensure white background
    )
    plt.close()

    return file_name


@stimulus_function
def create_dual_dot_plot(stimulus_description: DualStatsLinePlot):
    """
    Create two dot plots side by side with enhanced visual appeal.

    Args:
        stimulus_description: DualStatsLinePlot containing top and bottom data and titles
    """
    # Define color pairs for alternating bins
    COLOR_PAIRS = [
        ("#28717D", "#DC3E26"),  # Green, Purple
        ("#3B53A5", "#E8785D"),  # Blue, Orange
        ("#E26274", "#023B20"),  # Magenta, Yellow
        ("#B2456E", "#035F5F"),  # Red, Blue
        ("#1A2A4D", "#77530B"),  # Dark Blue, White
        ("#0B922F", "#24013B"),  # Green, Beige
        ("#6A0DAD", "#CFA108EF"),  # Purple, Gold
        ("#800020", "#277802"),  # Burgundy, Beige
        ("#0047AB", "#C98003"),  # Blue, Amber/Orange
    ]

    # Randomly select a color pair for this plot generation
    selected_color_pair = random.choice(COLOR_PAIRS)
    left_color, right_color = selected_color_pair

    # Generate the counts for each unique value in both datasets
    left_values, left_counts = np.unique(
        stimulus_description.top_data, return_counts=True
    )
    right_values, right_counts = np.unique(
        stimulus_description.bottom_data, return_counts=True
    )

    # Limit to maximum 10 values per plot
    if len(left_values) > 10:
        left_values = left_values[:10]
        left_counts = left_counts[:10]
    if len(right_values) > 10:
        right_values = right_values[:10]
        right_counts = right_counts[:10]

    # Determine the overall range for consistent x-axis
    all_values = np.concatenate([left_values, right_values])
    min_value, max_value = min(all_values), max(all_values)

    # Set figure size based on the number of unique values - SIDE BY SIDE LAYOUT
    max_unique_values = max(len(left_values), len(right_values))
    if max_unique_values <= 10:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Side by side layout
    else:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6)
        )  # Larger side by side layout

    # Calculate dot size based on figure size for better proportion - INCREASED SIZE
    dot_size = (
        180 if max_unique_values <= 10 else 160
    )  # Increased from 120/100 to 180/160

    # Calculate vertical spacing for dots
    dot_spacing = 0.08  # Reduced spacing between vertically stacked dots

    # Plot left dot plot with enhanced styling
    for value, count in zip(left_values, left_counts):
        # Create y-positions with tighter, more consistent spacing
        y_positions = np.arange(count) * dot_spacing + dot_spacing

        # Plot dots with enhanced styling
        ax1.scatter(
            [value] * count,
            y_positions,
            s=dot_size,  # Use scatter for better size control
            color=left_color,
            alpha=0.8,  # Slight transparency for better visual appeal
            edgecolors="white",  # White border around dots
            linewidth=0.5,  # Thin border
            zorder=3,  # Ensure dots are on top
        )

    # Enhanced left plot configuration
    ax1.set_xticks(np.arange(min_value, max_value + 1))
    ax1.tick_params(axis="x", labelsize=11, colors="#333333")
    ax1.set_yticks([])  # Remove y-ticks
    ax1.tick_params(axis="y", left=False)

    # Enhanced title styling - match with dot color
    ax1.set_title(
        stimulus_description.top_title,
        color=left_color,  # Match the dot color
        fontweight="bold",
        fontsize=14,
        pad=20,  # More space above title
    )

    # Enhanced spine styling for left plot with border
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["left"].set_visible(True)
    ax1.spines["bottom"].set_visible(True)
    ax1.spines["top"].set_color("#666666")
    ax1.spines["right"].set_color("#666666")
    ax1.spines["left"].set_color("#666666")
    ax1.spines["bottom"].set_color("#666666")
    ax1.spines["top"].set_linewidth(1.5)
    ax1.spines["right"].set_linewidth(1.5)
    ax1.spines["left"].set_linewidth(1.5)
    ax1.spines["bottom"].set_linewidth(1.5)
    ax1.spines["bottom"].set_position(("data", 0))  # Align with bottom of dots

    # Set better axis limits with padding for left plot
    x_padding = 0.3
    y_padding = max(left_counts) * dot_spacing * 0.1
    max_y = max(left_counts) * dot_spacing + dot_spacing
    ax1.set_xlim(min_value - x_padding, max_value + x_padding)
    ax1.set_ylim(-y_padding * 2, max_y + y_padding)

    # Add subtle grid for x-axis only (left plot)
    ax1.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5, color="#cccccc")
    ax1.set_axisbelow(True)  # Put grid behind dots

    # Plot right dot plot with enhanced styling
    for value, count in zip(right_values, right_counts):
        # Create y-positions with tighter, more consistent spacing
        y_positions = np.arange(count) * dot_spacing + dot_spacing

        # Plot dots with enhanced styling
        ax2.scatter(
            [value] * count,
            y_positions,
            s=dot_size,  # Use scatter for better size control
            color=right_color,
            alpha=0.8,  # Slight transparency for better visual appeal
            edgecolors="white",  # White border around dots
            linewidth=0.5,  # Thin border
            zorder=3,  # Ensure dots are on top
        )

    # Enhanced right plot configuration
    ax2.set_xticks(np.arange(min_value, max_value + 1))
    ax2.tick_params(axis="x", labelsize=11, colors="#333333")
    ax2.set_yticks([])  # Remove y-ticks
    ax2.tick_params(axis="y", left=False)

    # Enhanced title styling - match with dot color
    ax2.set_title(
        stimulus_description.bottom_title,
        color=right_color,  # Match the dot color
        fontweight="bold",
        fontsize=14,
        pad=20,  # More space above title
    )

    # Enhanced spine styling for right plot with border
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_visible(True)
    ax2.spines["bottom"].set_visible(True)
    ax2.spines["top"].set_color("#666666")
    ax2.spines["right"].set_color("#666666")
    ax2.spines["left"].set_color("#666666")
    ax2.spines["bottom"].set_color("#666666")
    ax2.spines["top"].set_linewidth(1.5)
    ax2.spines["right"].set_linewidth(1.5)
    ax2.spines["left"].set_linewidth(1.5)
    ax2.spines["bottom"].set_linewidth(1.5)
    ax2.spines["bottom"].set_position(("data", 0))  # Align with bottom of dots

    # Set better axis limits with padding for right plot
    y_padding_right = max(right_counts) * dot_spacing * 0.1
    max_y_right = max(right_counts) * dot_spacing + dot_spacing
    ax2.set_xlim(min_value - x_padding, max_value + x_padding)
    ax2.set_ylim(-y_padding_right * 2, max_y_right + y_padding_right)

    # Add subtle grid for x-axis only (right plot)
    ax2.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5, color="#cccccc")
    ax2.set_axisbelow(True)  # Put grid behind dots

    # Remove top and right margins
    ax1.margins(0.02)
    ax2.margins(0.02)

    # Enhanced layout
    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/dual_dot_plot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        dpi=300,  # Higher resolution for crisp output
        format=settings.additional_content_settings.stimulus_image_format,
        facecolor="white",  # Ensure white background
    )
    plt.close()

    return file_name


@stimulus_function
def create_vertical_number_line(stim_desc: NumberLine):
    try:
        fig, ax = plt.subplots()
        original_min = stim_desc.range.min
        original_max = stim_desc.range.max
        buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

        range_min = original_min - buffer
        range_max = original_max + buffer

        # Set y-axis limits instead of x-axis for vertical orientation
        ax.set_ylim(range_min, range_max)
        ax.set_xlim(-0.25, 0.25)

        # Set up the vertical number line
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.get_xaxis().set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="y", labelsize=14)
        major_ticks = []

        # Determine major ticks
        total_range = original_max - original_min
        if total_range <= 15 and total_range > 1:
            major_ticks = np.arange(np.floor(original_min), np.ceil(original_max) + 1)
            ax.yaxis.set_major_locator(FixedLocator(major_ticks))
        elif total_range > 1:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
            major_ticks = ax.yaxis.get_major_locator().tick_values(
                original_min, original_max
            )
            ax.yaxis.set_major_locator(FixedLocator(major_ticks))
        else:
            ax.yaxis.set_major_locator(FixedLocator(major_ticks))

        optimal_step = 1
        if total_range == 1 and stim_desc.minor_divisions is not None:
            optimal_step = 1 / stim_desc.minor_divisions
        elif total_range > 120:
            optimal_step = 10
        elif total_range >= 15:
            optimal_step = 1
        elif total_range > 1:
            optimal_step = determine_optimal_step(
                stim_desc.points, original_min, original_max
            )

        minimum = original_min
        maximum = original_max
        if len(major_ticks) > 0:
            minimum = min(original_min, major_ticks[0])
            maximum = max(original_max, major_ticks[-1])

        valid_minor_ticks = np.arange(minimum, maximum + optimal_step, optimal_step)
        ax.yaxis.set_minor_locator(FixedLocator(valid_minor_ticks.tolist()))

        # Define the value mappings
        value_replacements = {
            0.33: 0.33333333,
            0.66: 0.66666666,
            # Add more mappings as needed
        }

        texts = []
        base_offset = 0.02  # Base vertical offset
        stack_offset = 0.08  # Additional offset for stacked labels

        # Plot points and adjust label positions
        for i, point in enumerate(stim_desc.points):
            original_value = point.value
            point.value = value_replacements.get(original_value, original_value)
            ax.plot(0, point.value, "ko", markersize=7)

            # Create text object
            text = ax.text(
                0.02,  # Fixed position to the right of the point
                point.value + base_offset,
                point.label,
                ha="left",  # Align text to the left of the point
                va="center",  # Center text vertically
                fontsize=14,
                rotation=0,  # Ensure text is horizontal
            )
            texts.append(text)

            # Check for overlaps with all previous labels
            if i > 0:
                current_extent = text.get_window_extent()
                for prev_text in texts[:-1]:
                    prev_extent = prev_text.get_window_extent()
                    if current_extent.overlaps(prev_extent):
                        # Move current label up
                        text.set_position(
                            (0.02, point.value + base_offset + stack_offset)
                        )
                        # Update extent after moving
                        current_extent = text.get_window_extent()
                        # If still overlapping, move previous label down
                        if current_extent.overlaps(prev_extent):
                            prev_pos = prev_text.get_position()
                            prev_text.set_position((0.02, prev_pos[1] - stack_offset))

        # Manually draw each minor tick with variable length and color
        if original_min == 0 and original_max == 1 and len(valid_minor_ticks) >= 12:
            tick_length_scale = 0.008  # Base scale for tick length
            valid_minor_ticks = valid_minor_ticks[
                (valid_minor_ticks != 0) & (valid_minor_ticks != 1)
            ]
            for i, tick_position in enumerate(valid_minor_ticks):
                tick_color = "gray" if i % 2 == 0 else "#606060"
                line_width = 1.2 if i % 2 == 0 else 1.4
                tick_length = tick_length_scale * (1 if i % 2 == 0 else 1.3)
                ax.plot(
                    [0, -tick_length],  # Draw ticks to the left
                    [tick_position, tick_position],
                    color=tick_color,
                    linewidth=line_width,
                )
        else:
            ax.tick_params(axis="y", which="minor", length=7, width=1.2, color="gray")

        ax.tick_params(axis="y", which="major", length=10, width=1.5, color="black")

        plt.tight_layout()
        file_name = f"{settings.additional_content_settings.image_destination_folder}/vertical_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            dpi=800,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()
        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def create_multi_inequality_number_line(stim_desc: MultiInequalityNumberLine):
    """Create a 2x2 grid of inequality number lines."""
    try:
        # Create a 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Adjust subplot spacing - decrease vertical space, increase horizontal space
        plt.subplots_adjust(
            hspace=0.01,  # Decrease vertical space between subplots
            wspace=0.3,  # Increase horizontal space between subplots
        )

        axes = axes.flatten()  # Flatten to make indexing easier

        # Process each inequality number line
        for i, inequality_line in enumerate(stim_desc.number_lines):
            ax = axes[i]

            # Add title (A, B, C, D) to each subplot
            title = chr(65 + i)  # Convert 0,1,2,3 to A,B,C,D
            ax.set_title(title, fontsize=16, fontweight="bold", pad=10)

            # Get the range and buffer for this line
            range_min = inequality_line.range.min
            range_max = inequality_line.range.max
            full_range = range_max - range_min
            buffer = 0.5
            x_lim_min = range_min - buffer
            x_lim_max = range_max + buffer

            # Set up the axes
            ax.set_xlim(x_lim_min, x_lim_max)
            ax.set_ylim(-0.12, 0.1)

            # Configure spines
            ax.spines["bottom"].set_position("zero")
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")

            # Set the zorder of the x-axis line to be behind the points
            ax.spines["bottom"].set_zorder(0.5)

            # Set the default major step value to 1
            ax.xaxis.set_major_locator(MultipleLocator(1))

            # Set tick mark size, thickness, and direction
            ax.tick_params(
                axis="x",
                which="major",
                labelsize=12,
                size=12,
                width=2,
                direction="inout",
            )

            # Set tick marks and labels
            ax.set_xticks(range(range_min, range_max + 1))
            ax.set_xticklabels(
                [str(x) for x in range(range_min, range_max + 1)],
                fontsize=12 if full_range > 10 else 14,
            )

            # Draw the line
            line = inequality_line.line
            min_value = line.min or range_min - buffer / 2
            max_value = line.max or range_max + buffer / 2

            ax.plot(
                [min_value, max_value], [0, 0], linewidth=3, color="black", zorder=1
            )

            # Add arrow at the end of the line
            if line.min is None:
                ax.annotate(
                    "",
                    xy=(min_value, 0),
                    xytext=(min_value - buffer / 2, 0),
                    arrowprops=dict(arrowstyle="<-", color="black", linewidth=3),
                    zorder=1,
                )
            elif line.max is None:
                ax.annotate(
                    "",
                    xy=(max_value + buffer / 2, 0),
                    xytext=(max_value, 0),
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=3),
                    zorder=1,
                )

            # Draw points
            for point in inequality_line.points:
                value = point.value
                fill = "black" if point.fill else "white"
                ax.plot(
                    value,
                    0,
                    marker="o",
                    markersize=8,
                    color=fill,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    zorder=2,
                    fillstyle="full",
                )

        # Save the figure
        file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_inequality_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
            dpi=800,
        )
        plt.close()
        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def create_multi_extended_unit_fraction_number_line_with_bar(
    stim_desc: MultiExtendedUnitFractionNumberLine,
):
    """Create multiple extended unit fraction number lines with bars stacked vertically.
    Each line is labeled with numbers 1, 2, 3, etc. (except when there's only one line)."""

    try:
        num_lines = len(stim_desc.number_lines)
        fig, axes = plt.subplots(num_lines, 1, figsize=(8, 1.2 * num_lines))

        # If there's only one line, axes won't be a list
        if num_lines == 1:
            axes = [axes]

        # Adjust spacing between subplots - remove extra space
        plt.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05)

        for i, (ax, line_desc) in enumerate(zip(axes, stim_desc.number_lines)):
            # Set up the axis for this number line
            original_min = line_desc.range.min  # Should be >= 0
            original_max = line_desc.range.max  # Should be the fraction value
            buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

            range_min = original_min - buffer
            range_max = original_max + buffer

            ax.set_xlim(range_min, range_max)
            ax.set_ylim(-0.25, 0.25)

            # Configure spines
            ax.spines["bottom"].set_position("zero")
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["bottom"].set_zorder(1)  # Put axis spine behind other elements
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", labelsize=16, direction="inout")

            # Only show major ticks at original_min and the endpoint
            major_ticks = [original_min, original_max]
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            # Remove default tick labels
            ax.set_xticklabels(["", ""])
            ax.tick_params(
                axis="x",
                which="major",
                labelbottom=False,
                length=18,
                width=2.0,
                color="black",
                direction="inout",
            )

            # Calculate minor tick step based on minor_divisions
            minor_step = (original_max - original_min) / line_desc.minor_divisions
            minor_ticks = np.arange(original_min, original_max + minor_step, minor_step)
            ax.xaxis.set_minor_locator(FixedLocator(minor_ticks.tolist()))
            ax.tick_params(
                axis="x",
                which="minor",
                length=14,
                width=1.8,
                color="gray",
                direction="inout",
            )

            # Use the bottom spine as the number line (keeps a uniform thin line across)
            # Previously an extra ax.plot added thickness in the middle; removed for consistency

            # Add triangles at the endpoints to make it look more like a number line
            triangle_size = 0.03
            triangle_width = 0.018
            # Place arrowheads relative to the subplot limits so they sit away from the ± endpoint labels
            # Use a margin that scales with the buffer so movement is visible
            arrow_margin = max(0.15 * (range_max - range_min), 10.0 * triangle_size)

            # Left triangle (arrowhead, pointing left)
            left_apex = range_min + arrow_margin
            left_triangle = mpatches.Polygon(
                [
                    [left_apex, 0],
                    [left_apex + triangle_size, triangle_width],
                    [left_apex + triangle_size, -triangle_width],
                ],
                closed=True,
                color="k",
                zorder=4,
            )
            ax.add_patch(left_triangle)

            # Right triangle (arrowhead, pointing right)
            right_apex = range_max - arrow_margin
            right_triangle = mpatches.Polygon(
                [
                    [right_apex, 0],
                    [right_apex - triangle_size, triangle_width],
                    [right_apex - triangle_size, -triangle_width],
                ],
                closed=True,
                color="k",
                zorder=4,
            )

            ax.add_patch(right_triangle)

            # Add endpoint labels (min and the fraction) with more offset for multi-line layout
            # Format the minimum value as integer if it's a whole number, otherwise as float
            min_label = (
                str(int(original_min))
                if original_min == int(original_min)
                else str(original_min)
            )
            ax.text(original_min, -0.12, min_label, ha="center", va="top", fontsize=14)
            ax.text(
                original_max,
                -0.12,
                convert_fraction_to_latex(line_desc.endpoint_fraction),
                ha="center",
                va="top",
                fontsize=get_fraction_fontsize(14)
                if "/" in line_desc.endpoint_fraction
                else 14,
            )

            # Calculate the start position of the blue bar based on dot_start_tick
            step = (original_max - original_min) / line_desc.minor_divisions
            bar_start_value = original_min + (line_desc.dot_point.dot_start_tick * step)
            # Calculate end position: start position + the value (length) of the bar
            bar_end_value = bar_start_value + line_desc.dot_point.value

            # Draw a thicker blue line from the start tick to the calculated end position (on top of axis spine)
            ax.plot(
                [bar_start_value, bar_end_value],
                [0, 0],
                "#6666ff",
                linewidth=4,
                zorder=2,
            )

            # Plot the dot at the end of the bar (red if specified, otherwise blue)
            dot_color = "#ff6666" if line_desc.dot_point.red else "#6666ff"
            ax.plot(
                bar_end_value,
                0,
                "o",
                color=dot_color,
                markersize=8,
                zorder=3,
            )

            # No label for the bar end point (removed as per requirement)
            # ax.text(
            #     bar_end_value,
            #     -0.13,
            #     line_desc.dot_point.label,
            #     ha="center",
            #     va="top",
            #     fontsize=14,
            # )

            # Add number label (1, 2, 3, etc.) to the left of each number line, but only if there's more than one line
            if num_lines > 1:
                ax.text(
                    range_min - (range_max - range_min) * 0.05,
                    0,
                    str(i + 1),
                    ha="right",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

        # Layout is handled by subplots_adjust above

        file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_extended_unit_fraction_number_line_with_bar_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name

    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise e


@stimulus_function
def create_multi_extended_unit_fraction_number_line_with_bar_v2(
    stim_desc: MultiExtendedUnitFractionNumberLine,
):
    """Create multiple extended unit fraction number lines with bars stacked vertically.
    Expects input range to be 0 to 2 and always labels 0, 1, and 2.
    Each line is labeled with numbers 1, 2, 3, etc. (except when there's only one line)."""

    try:
        num_lines = len(stim_desc.number_lines)
        fig, axes = plt.subplots(num_lines, 1, figsize=(8, 2 * num_lines))

        # If there's only one line, axes won't be a list
        if num_lines == 1:
            axes = [axes]

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.4)

        for i, (ax, line_desc) in enumerate(zip(axes, stim_desc.number_lines)):
            # Set up the axis for this number line - expects range to be 0 to 2
            original_min = line_desc.range.min
            original_max = line_desc.range.max
            buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

            range_min = original_min - buffer
            range_max = original_max + buffer

            ax.set_xlim(range_min, range_max)
            ax.set_ylim(-0.25, 0.25)

            # Configure spines
            ax.spines["bottom"].set_position("zero")
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["bottom"].set_zorder(1)  # Put axis spine behind other elements
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", labelsize=12)

            # Show major ticks at 0, 1, and 2
            major_ticks = [0, 1, 2]
            ax.xaxis.set_major_locator(FixedLocator(major_ticks))
            # Remove default tick labels
            ax.set_xticklabels(["", "", ""])
            ax.tick_params(
                axis="x",
                which="major",
                labelbottom=False,
                length=10,
                width=1.2,
                color="black",
            )

            # Calculate minor tick step based on minor_divisions
            minor_step = (original_max - original_min) / line_desc.minor_divisions
            minor_ticks = np.arange(original_min, original_max + minor_step, minor_step)
            ax.xaxis.set_minor_locator(FixedLocator(minor_ticks.tolist()))
            ax.tick_params(axis="x", which="minor", length=7, width=1.0, color="gray")

            # Plot the number line
            ax.plot([original_min, original_max], [0, 0], "k-", linewidth=2)

            # Add endpoint labels (0, 1, and 2) with more offset for multi-line layout
            ax.text(0, -0.05, "0", ha="center", va="top", fontsize=10)
            ax.text(1, -0.05, "1", ha="center", va="top", fontsize=10)
            ax.text(2, -0.05, "2", ha="center", va="top", fontsize=10)

            # Draw a thicker blue line from original_min to the dot point value (on top of axis spine)
            ax.plot(
                [original_min, line_desc.dot_point.value],
                [0, 0],
                "#6666ff",
                linewidth=4,
                zorder=2,
            )

            # Plot the blue dot point
            ax.plot(
                line_desc.dot_point.value,
                0,
                "o",
                color="#6666ff",
                markersize=8,
                zorder=3,
            )

            # Add number label (1, 2, 3, etc.) to the left of each number line, but only if there's more than one line
            if num_lines > 1:
                ax.text(
                    range_min - (range_max - range_min) * 0.05,
                    0,
                    str(i + 1),
                    ha="right",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

        # Adjust layout
        plt.tight_layout()

        file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_extended_unit_fraction_number_line_with_bar_v2_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name

    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise e


@stimulus_function
def create_multi_labeled_unit_fraction_number_line(
    stim_desc: MultiLabeledUnitFractionNumberLine,
):
    """Create multiple labeled unit fraction number lines stacked vertically.
    Each line labels all minor divisions as fractions (e.g., 1/3, 2/3).
    Each line is labeled with numbers 1, 2, 3, etc. (except when there's only one line)."""

    try:
        num_lines = len(stim_desc.number_lines)
        fig, axes = plt.subplots(num_lines, 1, figsize=(8, 2 * num_lines))

        # If there's only one line, axes won't be a list
        if num_lines == 1:
            axes = [axes]

        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.4)

        for i, (ax, line_desc) in enumerate(zip(axes, stim_desc.number_lines)):
            # Set up the axis for this number line
            original_min = line_desc.range.min  # Should be 0
            original_max = line_desc.range.max  # Should be the fraction value
            buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

            range_min = original_min - buffer
            range_max = original_max + buffer

            ax.set_xlim(range_min, range_max)
            ax.set_ylim(-0.25, 0.25)

            # Configure spines
            ax.spines["bottom"].set_position("zero")
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["bottom"].set_zorder(1)  # Put axis spine behind other elements
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", labelsize=10)

            # Calculate minor tick step based on minor_divisions
            minor_step = original_max / line_desc.minor_divisions
            minor_ticks = np.arange(0, original_max + minor_step, minor_step)

            # Set all ticks (both major and minor) at the same positions
            ax.xaxis.set_major_locator(FixedLocator(minor_ticks.tolist()))
            ax.xaxis.set_minor_locator(FixedLocator([]))  # No separate minor ticks

            # Create fraction labels for all tick positions
            tick_labels = []
            for j, tick_pos in enumerate(minor_ticks):
                if abs(tick_pos - 0) < 1e-6:  # At position 0
                    tick_labels.append("0")
                elif abs(tick_pos - original_max) < 1e-6:  # At the endpoint
                    tick_labels.append(
                        convert_fraction_to_latex(line_desc.endpoint_fraction)
                    )
                else:  # At minor divisions
                    # Calculate the fraction for this position
                    # If original_max is 1, then j out of minor_divisions
                    if abs(original_max - 1) < 1e-6:
                        # For unit fractions (0 to 1)
                        numerator = j
                        denominator = line_desc.minor_divisions
                        # Create the fraction string
                        if numerator == 0:
                            tick_labels.append("0")
                        elif numerator == denominator:
                            tick_labels.append("1")
                        else:
                            tick_labels.append(
                                f"$\\frac{{{numerator}}}{{{denominator}}}$"
                            )
                    else:
                        # For extended fractions, calculate based on the position
                        try:
                            endpoint_value = eval(line_desc.endpoint_fraction)
                            # Check if endpoint is a whole number greater than 1
                            if (
                                endpoint_value > 1
                                and abs(endpoint_value - round(endpoint_value)) < 1e-6
                            ):
                                # Don't simplify fractions for whole number endpoints > 1
                                # Calculate as (j * endpoint_value) / minor_divisions
                                numerator = j * int(round(endpoint_value))
                                denominator = line_desc.minor_divisions
                            else:
                                # For non-whole endpoints, calculate and simplify
                                fraction_value = (
                                    j * endpoint_value / line_desc.minor_divisions
                                )
                                from fractions import Fraction

                                frac = Fraction(fraction_value).limit_denominator()
                                numerator = frac.numerator
                                denominator = frac.denominator
                        except Exception:
                            # Fallback: use position-based fractions
                            numerator = j
                            denominator = line_desc.minor_divisions

                        # Create the fraction string (don't simplify for whole number endpoints > 1)
                        if numerator == 0:
                            tick_labels.append("0")
                        else:
                            tick_labels.append(
                                f"$\\frac{{{numerator}}}{{{denominator}}}$"
                            )

            # Set the labels
            ax.set_xticklabels(tick_labels)

            # Configure tick marks
            ax.tick_params(
                axis="x",
                which="major",
                length=8,
                width=1.0,
                color="black",
                labelsize=18,  # Increased from 14 for better fraction readability
            )

            # Plot the number line
            ax.plot([original_min, original_max], [0, 0], "k-", linewidth=2)

            # Add triangles at the endpoints to make it look more like a number line
            triangle_size = 0.03
            triangle_width = 0.018
            # Move arrowheads a bit farther from the endpoint tick labels by tying
            # the offset to the figure's buffer (proportional to range span)
            triangle_offset = max(3.4 * triangle_size, (range_max - range_min) * 0.05)
            # Left triangle (arrowhead, pointing left, further before the line)
            left_triangle = mpatches.Polygon(
                [
                    [original_min - triangle_offset, 0],
                    [original_min - triangle_offset + triangle_size, triangle_width],
                    [original_min - triangle_offset + triangle_size, -triangle_width],
                ],
                closed=True,
                color="k",
                zorder=4,
            )
            ax.add_patch(left_triangle)
            # Right triangle (arrowhead, pointing right, further after the line)
            right_triangle = mpatches.Polygon(
                [
                    [original_max + triangle_offset, 0],
                    [original_max + triangle_offset - triangle_size, triangle_width],
                    [original_max + triangle_offset - triangle_size, -triangle_width],
                ],
                closed=True,
                color="k",
                zorder=4,
            )
            ax.add_patch(right_triangle)

            # Add number label (1, 2, 3, etc.) to the left of each number line, but only if there's more than one line
            if num_lines > 1:
                ax.text(
                    range_min - (range_max - range_min) * 0.05,
                    0,
                    str(i + 1),
                    ha="right",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                )

        # Adjust layout
        plt.tight_layout()

        file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_labeled_unit_fraction_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name

    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise e


@stimulus_function
def create_multi_extended_unit_fraction_number_line_with_dots(
    stim_desc: MultiExtendedUnitFractionNumberLine,
):
    """Create multiple extended unit fraction number lines with colored dot points stacked vertically.
    Each line shows a colored dot point with label, without bars.
    Each line is labeled with numbers 1, 2, 3, etc. (except when there's only one line)."""

    try:
        num_lines = len(stim_desc.number_lines)
        fig, axes = plt.subplots(num_lines, 1, figsize=(8, 1.2 * num_lines))

        # If there's only one line, axes won't be a list
        if num_lines == 1:
            axes = [axes]

        # Adjust spacing between subplots - remove extra space
        plt.subplots_adjust(hspace=0.1, top=0.95, bottom=0.05)

        for i, (ax, line_desc) in enumerate(zip(axes, stim_desc.number_lines)):
            # Set up the axis for this number line
            original_min = line_desc.range.min  # Should be 0
            original_max = line_desc.range.max  # Should be the fraction value
            buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

            range_min = original_min - buffer
            range_max = original_max + buffer

            ax.set_xlim(range_min, range_max)
            ax.set_ylim(-0.25, 0.25)

            # Configure spines
            ax.spines["bottom"].set_position("zero")
            ax.spines["bottom"].set_linewidth(1)
            ax.spines["bottom"].set_zorder(1)  # Put axis spine behind other elements
            ax.spines["right"].set_color("none")
            ax.spines["top"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.get_yaxis().set_visible(False)
            ax.xaxis.set_ticks_position("bottom")
            ax.tick_params(axis="x", labelsize=16, direction="inout")

            # Calculate minor tick step based on minor_divisions
            minor_step = (original_max - original_min) / line_desc.minor_divisions
            minor_ticks = np.arange(original_min, original_max + minor_step, minor_step)

            if stim_desc.show_minor_division_labels:
                # Always keep subdivision ticks as minor ticks
                ax.xaxis.set_minor_locator(FixedLocator(minor_ticks.tolist()))

                # Integer-labelled majors when requested; otherwise label each subdivision as before
                if getattr(stim_desc, "show_integer_tick_labels", False):
                    # Major ticks on integers only within the visible range
                    int_start = int(np.ceil(original_min))
                    int_end = int(np.floor(original_max))
                    integer_ticks = np.arange(int_start, int_end + 1, 1, dtype=float)
                    ax.xaxis.set_major_locator(FixedLocator(integer_ticks.tolist()))
                    tick_labels = [str(int(t)) for t in integer_ticks]
                    ax.set_xticklabels(tick_labels)
                else:
                    # Label every subdivision (fraction-style) like the original behavior
                    ax.xaxis.set_major_locator(FixedLocator(minor_ticks.tolist()))
                    # Create fraction labels for all tick positions
                    tick_labels = []
                    for j, tick_pos in enumerate(minor_ticks):
                        if (
                            abs(tick_pos - original_min) < 1e-6
                        ):  # At position original_min
                            min_label = (
                                str(int(original_min))
                                if original_min == int(original_min)
                                else str(original_min)
                            )
                            tick_labels.append(min_label)
                        elif abs(tick_pos - original_max) < 1e-6:  # At the endpoint
                            # Check if endpoint fraction is a whole number
                            try:
                                endpoint_value = eval(line_desc.endpoint_fraction)
                                if abs(endpoint_value - round(endpoint_value)) < 1e-6:
                                    tick_labels.append(str(int(round(endpoint_value))))
                                else:
                                    tick_labels.append(
                                        convert_fraction_to_latex(
                                            line_desc.endpoint_fraction
                                        )
                                    )
                            except Exception:
                                tick_labels.append(
                                    convert_fraction_to_latex(
                                        line_desc.endpoint_fraction
                                    )
                                )
                        else:  # At minor divisions
                            # Calculate the fraction for this position
                            if abs(original_max - 1) < 1e-6 and original_min == 0:
                                # For unit fractions (0 to 1)
                                numerator = j
                                denominator = line_desc.minor_divisions
                                if numerator == 0:
                                    tick_labels.append("0")
                                elif numerator == denominator:
                                    tick_labels.append("1")
                                else:
                                    tick_labels.append(
                                        f"$\\frac{{{numerator}}}{{{denominator}}}$"
                                    )
                            else:
                                # For extended fractions, calculate based on the position
                                try:
                                    endpoint_value = eval(line_desc.endpoint_fraction)
                                    # Check if endpoint is a whole number greater than 1
                                    if (
                                        endpoint_value > 1
                                        and abs(endpoint_value - round(endpoint_value))
                                        < 1e-6
                                    ):
                                        # Don't simplify for whole number endpoints > 1
                                        numerator = j * int(round(endpoint_value))
                                        denominator = line_desc.minor_divisions
                                    else:
                                        # For non-whole endpoints, calculate and simplify
                                        fraction_value = (
                                            j
                                            * endpoint_value
                                            / line_desc.minor_divisions
                                        )
                                        from fractions import Fraction

                                        frac = Fraction(
                                            fraction_value
                                        ).limit_denominator()
                                        numerator = frac.numerator
                                        denominator = frac.denominator
                                except Exception:
                                    # Fallback: use position-based fractions
                                    numerator = j
                                    denominator = line_desc.minor_divisions

                                # Create the fraction string, but check if it's a whole number first
                                if numerator == 0:
                                    tick_labels.append("0")
                                elif denominator != 0 and numerator % denominator == 0:
                                    tick_labels.append(str(numerator // denominator))
                                else:
                                    tick_labels.append(
                                        f"$\\frac{{{numerator}}}{{{denominator}}}$"
                                    )
                # Apply labels configured above
                ax.set_xticklabels(tick_labels)

                # Configure tick marks
                ax.tick_params(
                    axis="x",
                    which="major",
                    length=16,
                    width=1.8,
                    color="black",
                    labelsize=20,  # Increased from 16 for better fraction readability
                    direction="inout",
                )
            else:
                # Original behavior: only show major ticks at original_min and the endpoint
                major_ticks = [original_min, original_max]
                ax.xaxis.set_major_locator(FixedLocator(major_ticks))
                # Remove default tick labels
                ax.set_xticklabels(["", ""])
                ax.tick_params(
                    axis="x",
                    which="major",
                    labelbottom=False,
                    length=18,
                    width=2.0,
                    color="black",
                    direction="inout",
                )

                ax.xaxis.set_minor_locator(FixedLocator(minor_ticks.tolist()))
                ax.tick_params(
                    axis="x",
                    which="minor",
                    length=14,
                    width=1.8,
                    color="gray",
                    direction="inout",
                )

            # Plot the number line
            ax.plot([original_min, original_max], [0, 0], "k-", linewidth=2)

            # Add triangles at the displayed ends so arrows appear at the line ends
            triangle_size = 0.03
            triangle_width = 0.018
            # Left triangle (arrowhead pointing left) at the left display limit
            left_triangle = mpatches.Polygon(
                [
                    [range_min, 0],
                    [range_min + triangle_size, triangle_width],
                    [range_min + triangle_size, -triangle_width],
                ],
                closed=True,
                color="k",
                zorder=4,
            )
            ax.add_patch(left_triangle)
            # Right triangle (arrowhead pointing right) at the right display limit
            right_triangle = mpatches.Polygon(
                [
                    [range_max, 0],
                    [range_max - triangle_size, triangle_width],
                    [range_max - triangle_size, -triangle_width],
                ],
                closed=True,
                color="k",
                zorder=4,
            )
            ax.add_patch(right_triangle)

            # Add endpoint labels only if not showing minor division labels
            if not stim_desc.show_minor_division_labels:
                # Format the minimum value as integer if it's a whole number, otherwise as float
                min_label = (
                    str(int(original_min))
                    if original_min == int(original_min)
                    else str(original_min)
                )
                ax.text(
                    original_min, -0.12, min_label, ha="center", va="top", fontsize=14
                )
                # Check if endpoint fraction is a whole number for consistent display
                try:
                    endpoint_value = eval(line_desc.endpoint_fraction)
                    if abs(endpoint_value - round(endpoint_value)) < 1e-6:
                        # Display as integer with same font as other numbers
                        endpoint_label = str(int(round(endpoint_value)))
                        endpoint_fontsize = 14
                    else:
                        # Display as fraction with appropriate font size
                        endpoint_label = convert_fraction_to_latex(
                            line_desc.endpoint_fraction
                        )
                        endpoint_fontsize = (
                            get_fraction_fontsize(14)
                            if "/" in line_desc.endpoint_fraction
                            else 14
                        )
                except Exception:
                    # Fallback to original behavior
                    endpoint_label = convert_fraction_to_latex(
                        line_desc.endpoint_fraction
                    )
                    endpoint_fontsize = (
                        get_fraction_fontsize(14)
                        if "/" in line_desc.endpoint_fraction
                        else 14
                    )

                ax.text(
                    original_max,
                    -0.12,
                    endpoint_label,
                    ha="center",
                    va="top",
                    fontsize=endpoint_fontsize,
                )

            # Plot the dot point (red if specified, otherwise blue)
            dot_color = "#ff6666" if line_desc.dot_point.red else "#6666ff"
            ax.plot(
                line_desc.dot_point.value,
                0,
                "o",
                color=dot_color,
                markersize=8,
                zorder=3,
            )

            # Add the label for the dot point above it
            ax.text(
                line_desc.dot_point.value,
                0.06,
                line_desc.dot_point.label,
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                color=dot_color,
            )

            # Add number label (1, 2, 3, etc.) to the left of each number line, but only if there's more than one line
            if num_lines > 1:
                ax.text(
                    range_min - (range_max - range_min) * 0.05,
                    0,
                    str(i + 1),
                    ha="right",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )

        # Layout is handled by subplots_adjust above

        file_name = f"{settings.additional_content_settings.image_destination_folder}/multi_extended_unit_fraction_number_line_with_dots_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name

    except Exception as e:
        plt.close()  # Ensure figure is closed even if error occurs
        raise e


@stimulus_function
def create_fixed_step_number_line(stim_desc: FixedStepNumberLine):
    """Create a number line with a fixed step size."""
    try:
        fig, ax = plt.subplots()
        original_min = stim_desc.range.min
        original_max = stim_desc.range.max
        buffer = (original_max - original_min) * 0.1  # Buffer is 10% of total range

        range_min = original_min - buffer
        range_max = original_max + buffer

        ax.set_xlim(range_min, range_max)
        ax.set_ylim(-0.25, 0.25)

        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelsize=14)

        # Determine major ticks based on the provided step_size
        start = np.floor(original_min / stim_desc.step_size) * stim_desc.step_size
        end = np.ceil(original_max / stim_desc.step_size) * stim_desc.step_size
        major_ticks = np.arange(start, end + stim_desc.step_size, stim_desc.step_size)
        ax.xaxis.set_major_locator(FixedLocator(major_ticks.tolist()))

        # Set minor ticks based on minor_divisions parameter
        minor_step = stim_desc.step_size / stim_desc.minor_divisions
        minimum = (
            min(original_min, major_ticks[0]) if len(major_ticks) > 0 else original_min
        )
        maximum = (
            max(original_max, major_ticks[-1]) if len(major_ticks) > 0 else original_max
        )
        minor_ticks = np.arange(minimum, maximum + minor_step, minor_step)
        ax.xaxis.set_minor_locator(FixedLocator(minor_ticks.tolist()))

        # Define the value mappings
        value_replacements = {
            0.33: 0.33333333,
            0.66: 0.66666666,
            # Add more mappings as needed
        }

        texts = []
        base_offset = 0.02  # Base vertical offset
        stack_offset = 0.03  # Additional offset for stacked labels

        # Plot points and adjust label positions
        for i, point in enumerate(stim_desc.points):
            original_value = point.value
            point.value = value_replacements.get(original_value, original_value)
            ax.plot(point.value, 0, "ko", markersize=7)

            # Create text object
            text = ax.text(
                point.value,
                base_offset,
                point.label,
                ha="center",
                va="bottom",
                fontsize=14,
            )
            texts.append(text)

            # Check for overlaps with all previous labels
            if i > 0:
                current_extent = text.get_window_extent()
                for prev_text in texts[:-1]:
                    prev_extent = prev_text.get_window_extent()
                    if current_extent.overlaps(prev_extent):
                        # Move current label up
                        text.set_position((point.value, base_offset + stack_offset))
                        # Update extent after moving
                        current_extent = text.get_window_extent()
                        # If still overlapping, move previous label down
                        if current_extent.overlaps(prev_extent):
                            prev_pos = prev_text.get_position()
                            prev_text.set_position(
                                (prev_pos[0], base_offset - stack_offset)
                            )

        # Set tick mark styles
        ax.tick_params(axis="x", which="minor", length=7, width=1.2, color="gray")
        ax.tick_params(axis="x", which="major", length=10, width=1.5, color="black")

        plt.tight_layout()
        file_name = f"{settings.additional_content_settings.image_destination_folder}/fixed_step_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            dpi=800,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()
        return file_name
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


@stimulus_function
def create_decimal_comparison_number_line(stim_desc: DecimalComparisonNumberLine):
    """Creates a decimal comparison number line with exactly 10 divisions.
    Supports either 0.1 or 0.01 increments based on the range span.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 2))

        range_min = stim_desc.range.min
        range_max = stim_desc.range.max
        range_span = range_max - range_min

        # Determine increment based on range span
        if abs(range_span - 1.0) < 1e-10:
            increment = 0.1
        else:  # range_span == 0.1
            increment = 0.01

        # Add buffer for display
        buffer = range_span * 0.1
        display_min = range_min - buffer
        display_max = range_max + buffer

        ax.set_xlim(display_min, display_max)
        ax.set_ylim(-0.3, 0.3)

        # Configure the axes
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["bottom"].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_ticks_position("bottom")

        # Create exactly 11 major ticks (0 to 10 divisions = 11 points)
        major_ticks = [range_min + i * increment for i in range(11)]
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))

        # Create custom labels based on label_interval
        tick_labels = []
        for i in range(11):  # 11 tick positions (0 to 10)
            tick_value = range_min + i * increment
            if i % stim_desc.label_interval == 0:  # Label at specified intervals
                if increment == 0.1:
                    # Format to 1 decimal place
                    tick_labels.append(f"{tick_value:.1f}")
                else:  # increment == 0.01
                    # Format to 2 decimal places
                    tick_labels.append(f"{tick_value:.2f}")
            else:
                tick_labels.append("")  # Empty string for unlabeled positions

        ax.set_xticklabels(tick_labels)

        # Style the ticks
        ax.tick_params(
            axis="x", which="major", length=12, width=2, color="black", labelsize=12
        )

        # Plot optional points if provided
        if stim_desc.points:
            for point in stim_desc.points:
                # Plot the point
                ax.plot(point.value, 0, "ro", markersize=8, zorder=5)

                # Add label above the point
                ax.text(
                    point.value,
                    0.15,
                    point.label,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                    zorder=6,
                )

        # Make the number line prominent with arrows at the ends
        arrow_space = range_span * 0.05  # 5% of range for arrow space
        line_start = range_min - arrow_space
        line_end = range_max + arrow_space

        # Draw the main number line extending all the way to the arrow positions
        ax.plot([line_start, line_end], [0, 0], color="black", linewidth=3, zorder=1)

        # Add arrow heads at both ends using markers
        arrow_size = 150  # Fixed arrow size in points

        # Left arrow head (pointing left)
        ax.plot(
            line_start,
            0,
            marker="<",
            markersize=arrow_size**0.5,
            color="black",
            markeredgecolor="black",
            markerfacecolor="black",
            zorder=2,
        )

        # Right arrow head (pointing right)
        ax.plot(
            line_end,
            0,
            marker=">",
            markersize=arrow_size**0.5,
            color="black",
            markeredgecolor="black",
            markerfacecolor="black",
            zorder=2,
        )

        plt.tight_layout()

        # Save the figure
        file_name = f"{settings.additional_content_settings.image_destination_folder}/decimal_comparison_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            dpi=800,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()
        return file_name

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
