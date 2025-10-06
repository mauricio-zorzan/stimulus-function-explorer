import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.lines_of_best_fit_model import (
    Line,
    LinesOfBestFit,
)
from content_generators.settings import settings
from matplotlib.lines import Line2D


def generate_scatter_points(lines: List[Line], ax_range: int = 10) -> tuple:
    best_fit_line = next(line for line in lines if line.best_fit)

    # Generate evenly spaced x values within the 10x10 grid
    x = np.linspace(0, ax_range, 20)

    # Add small random offsets to x values for horizontal variation
    x += np.random.uniform(-0.3, 0.3, 20)

    # Calculate y values based on the best fit line
    y_base = best_fit_line.slope * x + best_fit_line.y_intercept

    # Generate noise and ensure equal points above and below the line
    noise = np.random.normal(0, 0.5, 20)
    noise[::2] = np.abs(noise[::2])  # Make every other point positive
    noise[1::2] = -np.abs(noise[1::2])  # Make every other point negative

    y = y_base + noise

    # Clip x and y values to stay within the ax_range
    x = np.clip(x, 0, ax_range)
    y = np.clip(y, 0, ax_range)

    # Remove points outside the specified bounds
    valid_points = (x >= 0) & (x <= ax_range) & (y >= 0) & (y <= ax_range)
    x = x[valid_points]
    y = y[valid_points]

    return x, y


@stimulus_function
def draw_lines_of_best_fit(stimulus_description: LinesOfBestFit):
    ax_range = 10  # Fixed 10x10 grid

    # Generate scatter points
    scatter_x, scatter_y = generate_scatter_points(stimulus_description.lines, ax_range)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    # Set up the coordinate plane (positive quadrant only)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Make axes black and thicker
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(2)

    # Set the range for x and y axes (10x10 grid)
    ax.set_xlim(0, ax_range)
    ax.set_ylim(0, ax_range)

    # Ensure equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Set ticks and increase font size
    ax.set_xticks(range(0, ax_range + 1))
    ax.set_yticks(range(0, ax_range + 1))
    ax.tick_params(
        axis="both", which="major", labelsize=16, color="black", width=2, length=6
    )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Define colors for the lines
    colors = ["r", "g", "b", "orange", "c", "m"]

    # Plot scatter points (increased size)
    ax.scatter(scatter_x, scatter_y, color="black", alpha=0.7, s=80)

    # Plot each line and prepare legend elements
    legend_elements = []
    x = np.linspace(0, ax_range, 100)
    for i, line in enumerate(stimulus_description.lines):
        color = colors[i % len(colors)]
        y = line.slope * x + line.y_intercept
        ax.plot(x, y, color=color, linestyle="-", alpha=0.7, linewidth=4.5)

        # Create legend entry
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                label=f"{line.label}",
                linewidth=3,
            )
        )

    # Add legend
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=18,
            handlelength=4,
            handleheight=4,
            labelspacing=0.2,
            borderaxespad=0.5,
            frameon=True,
        )

        # Customize the legend frame
        frame = legend.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(1.0)
        frame.set_facecolor("white")

        plt.subplots_adjust(right=0.75)  # Adjust right margin to accommodate legend
    else:
        plt.subplots_adjust(right=0.95)

    plt.tight_layout()

    # Save the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/lines_of_best_fit_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
        dpi=600,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name
