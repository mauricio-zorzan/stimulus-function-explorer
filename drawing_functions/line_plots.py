import textwrap
import time
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.double_line_plot import (
    DoubleLinePlot,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.line_plots import (
    DataPoint,
    LinePlot,
    LinePlotList,
    SingleLinePlot,
)
from content_generators.settings import settings
from content_generators.utils.math import Math
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator


def validate_and_adjust_line_plot_ranges(
    line_plots: LinePlotList,
) -> tuple[LinePlotList, int]:
    # Extract the minimum and maximum values across all line plots
    all_values = list(
        set([dp.value for line_plot in line_plots for dp in line_plot.data_points])
    )

    # Get max frequency across plots
    all_frequencies = [
        dp.frequency for line_plot in line_plots for dp in line_plot.data_points
    ]
    max_frequency = max(all_frequencies)

    adjusted_line_plots = []
    for line_plot in line_plots:
        data_points = line_plot.data_points.copy()
        existing_values = [dp.value for dp in data_points]

        # Find missing values in the current line plot
        missing_values = set(all_values) - set(existing_values)

        # Add data points for each missing value
        for value in missing_values:
            data_points.append(DataPoint(value=value, frequency=0))

        adjusted_line_plot = LinePlot(
            title=line_plot.title,
            x_axis_label=line_plot.x_axis_label,
            data_points=data_points,
        )
        adjusted_line_plots.append(adjusted_line_plot)

    return adjusted_line_plots, max_frequency


def find_smallest_distance(points):
    # Sort points to calculate consecutive differences
    sorted_points = sorted(list(map(lambda x: round(x, 3), points)))
    differences = [
        sorted_points[i + 1] - sorted_points[i] for i in range(len(sorted_points) - 1)
    ]
    min_difference = min(differences)

    fractional_parts = [point % 1 for point in points if point % 1 != 0]
    if fractional_parts:
        smallest_fraction = min(fractional_parts)
    else:
        smallest_fraction = 1

    return min(min_difference, smallest_fraction)


@stimulus_function
def generate_stacked_line_plots(stimuli_descriptions: LinePlotList):
    stimuli_descriptions, max_frequency = validate_and_adjust_line_plot_ranges(
        stimuli_descriptions
    )
    y_max = max(max_frequency + 0.5, 4)
    # Create a figure and a set of subplots
    PADDING = 4.0
    plot_count = len(stimuli_descriptions)
    rows = Math.clamp(round(plot_count / 2), 1, 2)
    cols = Math.clamp(round(plot_count / rows), 1, 2)

    multiple = rows > 1 or cols > 1

    values = [
        point.value
        for stimuli_description in stimuli_descriptions
        for point in stimuli_description.data_points
    ]
    frequencies = [
        point.frequency
        for stimuli_description in stimuli_descriptions
        for point in stimuli_description.data_points
    ]

    max_frequency = max(frequencies)
    value_range = len(values) / len(stimuli_descriptions)

    x_size = Math.clamp(value_range / 2, 4, 10)
    y_size = Math.clamp(max_frequency / 4, 2, 4)

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * x_size + PADDING / 4, rows * y_size + PADDING / 4),
    )
    if multiple:
        axes = axes.flatten()  # Flatten the axes array if more than one row and column

    # Hide 4th plot if there are only 3 plots
    if len(stimuli_descriptions) == 3:
        axes[3].set_visible(False)

    for i, stimulus_description in enumerate(stimuli_descriptions):
        ax = axes
        if multiple:
            ax: Axes = axes[i]

        values = [point.value for point in stimulus_description.data_points]
        values_parsed = [
            point.value_parsed for point in stimulus_description.data_points
        ]

        frequencies = [point.frequency for point in stimulus_description.data_points]

        max_frequency = max(frequencies)
        value_range = len(values)

        # Setting the title and labels
        ax.set_title(stimulus_description.title, fontsize=16)
        ax.set_xlabel(stimulus_description.x_axis_label, fontsize=16)

        # Removing the y-axis completely
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Keep bottom spine visible as the horizontal line that ticks connect to
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("black")
        ax.spines["bottom"].set_linewidth(1)
        ax.spines["bottom"].set_position(("data", 0))

        # Check if all x-axis values are numerical
        if all(isinstance(value, (int, float)) for value in values_parsed):
            # Calculate the smallest distance between data points
            smallest_distance = find_smallest_distance(values_parsed)

            # Calculate all ticks for the entire range based on smallest distance
            all_ticks = []
            all_tick_labels = []

            # Get the range from axis limits
            x_min = min(values_parsed) - smallest_distance / 2
            x_max = max(values_parsed) + smallest_distance / 2

            # Generate all ticks at smallest_distance intervals
            current_tick = x_min
            while current_tick <= x_max:
                # Round to avoid floating point precision issues
                current_tick = (
                    round(current_tick / smallest_distance) * smallest_distance
                )
                all_ticks.append(current_tick)

                frac = Fraction(current_tick).limit_denominator()
                if frac.denominator == 1:
                    # Whole number
                    tick_label = str(frac.numerator)
                else:
                    # Convert to mixed number if improper fraction
                    if frac.numerator >= frac.denominator:
                        whole = frac.numerator // frac.denominator
                        remainder = frac.numerator % frac.denominator
                        if remainder == 0:
                            tick_label = str(whole)
                        else:
                            tick_label = f"{whole} {remainder}/{frac.denominator}"
                    else:
                        tick_label = f"{frac.numerator}/{frac.denominator}"

                # Convert to LaTeX format to match major tick formatting
                from content_generators.utils.string import String

                formatted_label = String.convert_string_with_optional_fraction_to_latex(
                    tick_label
                )
                all_tick_labels.append(formatted_label)

                current_tick += smallest_distance

            # Set all ticks as regular ticks (no major/minor distinction)
            ax.set_xticks(all_ticks)
            ax.set_xticklabels(all_tick_labels, fontsize=12)

            # Make all tick marks the same size
            ax.tick_params(
                axis="x",
                length=10,  # Increased from 8 to stretch upwards
                width=1,  # Decreased from 2 to make thinner
                direction="out",
                labelsize=12,
                pad=5,
                bottom=True,
                top=False,
            )

            # Set the x-axis limits based on the minimum and maximum values
            ax.set_xlim(
                min(values_parsed) - smallest_distance / 2,
                max(values_parsed) + smallest_distance / 2,
            )

            # Plotting the x marks at the specified values with the frequencies as y-values
            for point in stimulus_description.data_points:
                ax.scatter(
                    [point.value_parsed] * point.frequency,
                    range(1, point.frequency + 1),
                    color="black",
                    marker="x",
                )
        else:
            # Set the x-axis ticks to show all values in the range
            ax.set_xticks(range(0, value_range, 1))
            # Create labels for all positions based on the values list order
            # The labels should correspond to the same order as the values list
            all_labels = [
                point.value_label for point in stimulus_description.data_points
            ]
            ax.set_xticklabels(labels=all_labels)

            # Set the x-axis limits based on the value range
            ax.set_xlim(-0.5, value_range - 0.5)

            # Plotting the x marks at the specified values with the frequencies as y-values
            for point in stimulus_description.data_points:
                ax.scatter(
                    [values.index(point.value)] * point.frequency,
                    range(1, point.frequency + 1),
                    color="black",
                    marker="x",
                )

        ax.set_ylim(-0.5, y_max)
        ax.tick_params(axis="x", labelsize=14)

    plt.tight_layout(pad=PADDING)
    file_name = f"{settings.additional_content_settings.image_destination_folder}/line_plot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_double_line_plot(stimulus_description: DoubleLinePlot):
    plt.figure()

    # Drawing a horizontal line at y=0
    plt.axhline(y=0, color="black", linewidth=1)
    ax = plt.gca()
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    class_count = 0
    max_y = 0
    for dataset in stimulus_description.datasets:
        x_values = [point.value for point in dataset.data_points]
        y_values = [point.frequency for point in dataset.data_points]
        max_y = max(max_y, max(y_values))
        for x, y in zip(x_values, y_values):
            ax.scatter(
                [x] * y,
                range(1, y + 1) if class_count == 0 else np.arange(-1.5, -y - 1.5, -1),
                color="black",
                marker="x",
                clip_on=False,
            )
        class_count += 1

    # Set major ticks on all x-values
    ax.set_xticks(
        np.arange(stimulus_description.range.min, stimulus_description.range.max + 1)
    )

    plt.xlim(stimulus_description.range.min - 0.5, stimulus_description.range.max + 0.5)
    plt.ylim(-max_y - 0.5, max_y + 0.5)

    ax.spines["bottom"].set_position("center")
    ax.tick_params(axis="x", direction="inout", length=6)

    # Wrap the x-axis label text
    wrapped_label = "\n".join(
        textwrap.wrap(stimulus_description.x_axis_label, width=16)
    )
    num_lines = wrapped_label.count("\n") + 1  # Calculate the number of lines

    # Add x-axis label to the left and center it on y=0
    ax.text(
        -0.1,
        0,  # Position at y=0, slightly to the left of the y-axis
        wrapped_label,
        ha="right",
        va="center",
        rotation=0,
        transform=ax.get_yaxis_transform(),
    )

    # Add class 1 label to the left above x-axis label
    ax.text(
        -0.1,
        2 + 1 * num_lines / 2,  # Position at y=0, slightly to the left of the y-axis
        stimulus_description.datasets[0].title,
        ha="right",
        va="center",
        rotation=0,
        weight="bold",
        transform=ax.get_yaxis_transform(),
    )

    # Add class 2 label to the left below x-axis label
    ax.text(
        -0.1,
        -2 + -1 * num_lines / 2,
        stimulus_description.datasets[1].title,
        ha="right",
        va="center",
        rotation=0,
        weight="bold",
        transform=ax.get_yaxis_transform(),
    )

    # Adjust layout to add space on the left
    plt.subplots_adjust(left=0.3, bottom=0.1)
    file_name = f"{settings.additional_content_settings.image_destination_folder}/double_line_plot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_single_line_plot(stimulus_description: SingleLinePlot):
    # Reuse the logic from generate_stacked_line_plots for a single plot
    PADDING = 4.0
    y_max = max([dp.frequency for dp in stimulus_description.data_points] + [4]) + 0.5
    x_size = max(len(stimulus_description.data_points) / 2, 4)
    y_size = max(y_max / 4, 2)
    fig, ax = plt.subplots(figsize=(x_size + PADDING / 4, y_size + PADDING / 4))

    values = [point.value for point in stimulus_description.data_points]
    values_parsed = [point.value_parsed for point in stimulus_description.data_points]

    # Drawing a horizontal line at y=0
    ax.axhline(y=0, color="black", linewidth=1)
    ax.set_title(stimulus_description.title, fontsize=16)
    ax.set_xlabel(stimulus_description.x_axis_label, fontsize=16)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    if all(isinstance(value, (int, float)) for value in values_parsed):
        # Calculate the smallest distance between data points
        def find_smallest_distance(points):
            sorted_points = sorted(list(map(lambda x: round(x, 3), points)))
            differences = [
                sorted_points[i + 1] - sorted_points[i]
                for i in range(len(sorted_points) - 1)
            ]
            min_difference = min(differences) if differences else 1
            fractional_parts = [point % 1 for point in points if point % 1 != 0]
            smallest_fraction = min(fractional_parts) if fractional_parts else 1
            return min(min_difference, smallest_fraction)

        smallest_distance = find_smallest_distance(values_parsed)
        ax.set_xticks(
            values_parsed,
            labels=[point.value_label for point in stimulus_description.data_points],
        )
        minor_ticks = []
        for i in range(len(values_parsed) - 1):
            min_tick_step = (
                values_parsed[i + 1] - values_parsed[i]
            ) / smallest_distance
            if min_tick_step > 5:
                min_tick_step = 5
            for j in range(1, int(min_tick_step)):
                minor_ticks.append(values_parsed[i] + j * smallest_distance)
        ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
        ax.set_xlim(
            min(values_parsed) - smallest_distance / 2,
            max(values_parsed) + smallest_distance / 2,
        )
        for point in stimulus_description.data_points:
            ax.scatter(
                [point.value_parsed] * point.frequency,
                range(1, point.frequency + 1),
                color="black",
                marker="x",
            )
    else:
        ax.set_xticks(range(0, len(values), 1))
        ax.set_xticklabels(
            labels=[point.value_label for point in stimulus_description.data_points]
        )
        ax.set_xlim(-0.5, len(values) - 0.5)
        for point in stimulus_description.data_points:
            ax.scatter(
                [values.index(point.value)] * point.frequency,
                range(1, point.frequency + 1),
                color="black",
                marker="x",
            )
    ax.set_ylim(-0.5, y_max)
    ax.tick_params(axis="x", labelsize=14)
    plt.tight_layout(pad=PADDING)
    file_name = f"{settings.additional_content_settings.image_destination_folder}/single_line_plot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
