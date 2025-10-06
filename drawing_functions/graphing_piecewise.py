import random
import time

# Add this import for debugging
import traceback

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.graphing_piecewise_model import (
    GraphingPiecewise,
)
from content_generators.settings import settings


def setup_labeled_graph(ax, x_label, y_label):
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Thicken axes
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2)

    # Add arrowheads
    ax.plot(1, 0, ">k", ms=10, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", ms=10, transform=ax.get_xaxis_transform(), clip_on=False)

    # Set labels with larger, bold font
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")


def setup_unlabeled_graph(ax, x_min, x_max, y_min, y_max):
    # Set ticks to integers
    ax.set_xticks(range(x_min, x_max + 1))
    ax.set_yticks(range(y_min, y_max + 1))

    # Add arrows to the axes
    ax.plot(
        1,
        0,
        ls="",
        marker=">",
        ms=10,
        color="k",
        transform=ax.get_yaxis_transform(),
        clip_on=False,
    )
    ax.plot(
        0,
        1,
        ls="",
        marker="^",
        ms=10,
        color="k",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

    # Add grid
    ax.grid(True, color="gray", linestyle="-", linewidth=0.5)


@stimulus_function
def generate_piecewise_graph(stimulus_description: GraphingPiecewise):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))  # Changed back to rectangular figure

        # Calculate x and y ranges
        x_values = [
            x
            for segment in stimulus_description.segments
            for x, _ in [segment.start_coordinate, segment.end_coordinate]
        ]
        y_values = [
            y
            for segment in stimulus_description.segments
            for _, y in [segment.start_coordinate, segment.end_coordinate]
        ]
        x_min, x_max = min(x_values) - 1, max(x_values) + 1
        y_min, y_max = min(y_values) - 1, max(y_values) + 1

        # Set up the axes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        # Check if axes are labeled
        axes_labeled = (
            stimulus_description.x_axis_label and stimulus_description.y_axis_label
        )

        # Setup graph based on presence of labels
        if axes_labeled:
            setup_labeled_graph(
                ax, stimulus_description.x_axis_label, stimulus_description.y_axis_label
            )
        else:
            setup_unlabeled_graph(ax, x_min, x_max, y_min, y_max)

        # Draw segments
        for idx, segment in enumerate(stimulus_description.segments):
            x = np.array([segment.start_coordinate[0], segment.end_coordinate[0]])
            y = np.array([segment.start_coordinate[1], segment.end_coordinate[1]])

            if segment.linear:
                ax.plot(x, y, color="blue", linewidth=2)
            else:
                # Randomly choose between concave and convex
                curve_direction = random.choice([-1, 1])

                # Create a gentler curve for non-linear segments
                dy = y[1] - y[0]
                dx = x[1] - x[0]
                # Calculate midpoint
                mid_x, mid_y = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2

                # Calculate control point for quadratic Bezier curve
                # Adjust this factor to control curve severity (smaller = gentler)
                curve_factor = 0.2
                control_x = mid_x + curve_direction * curve_factor * dy
                control_y = mid_y - curve_direction * curve_factor * dx

                # Generate points along the quadratic Bezier curve
                t = np.linspace(0, 1, 100)
                x_curve = (
                    (1 - t) ** 2 * x[0] + 2 * (1 - t) * t * control_x + t**2 * x[1]
                )
                y_curve = (
                    (1 - t) ** 2 * y[0] + 2 * (1 - t) * t * control_y + t**2 * y[1]
                )

                ax.plot(x_curve, y_curve, color="blue", linewidth=2)

            # Add segment label only if axes are labeled
            if axes_labeled:
                label = chr(65 + idx)  # 65 is ASCII for 'A'
                mid_x = (x[0] + x[1]) / 2
                mid_y = (y[0] + y[1]) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    label,
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                )

        # Save the figure
        file_name = f"{settings.additional_content_settings.image_destination_folder}/piecewise_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name
    except Exception as e:
        print(f"Error in generate_piecewise_graph: {str(e)}")
        print(traceback.format_exc())
        raise
