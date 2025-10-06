import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.measurements_model import (
    Measurements,
)
from content_generators.settings import settings
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle


def convert_to_ml(value: float, unit: str) -> float:
    """Convert the measurement to milliliters."""
    return value * 1000 if unit == "liters" else value


def get_scale_parameters(measurement_ml: float) -> Tuple[int, int]:
    """Determine the appropriate scale and tick interval based on the measurement."""
    if measurement_ml <= 100:
        return 100, 20
    elif measurement_ml <= 500:
        return 500, 100
    elif measurement_ml < 1000:
        return 1000, 200
    elif measurement_ml <= 5000:
        return 5000, 1000
    elif measurement_ml <= 10000:
        return 10000, 2000
    else:
        return 100000, 10000  # Support up to 100 liters


def calculate_dimensions(max_scale: int) -> Tuple[float, float, float, float]:
    """Calculate cup dimensions based on the scale."""
    # For larger measurements, we'll scale the height more gradually
    if max_scale <= 1000:
        scale_factor = 5.2
    elif max_scale <= 5000:
        scale_factor = 4.0
    else:
        scale_factor = 3.0

    cup_height = max_scale / 1000 * scale_factor
    cup_width = cup_height * 0.88
    buffer = cup_height * 0.08
    total_height = cup_height + buffer
    return cup_height, cup_width, buffer, total_height


def draw_cup_and_liquid(
    ax: Axes,
    cup_width: float,
    cup_height: float,
    total_height: float,
    measurement: float,
    color: str | None,
) -> None:
    """Draw the measuring cup and the liquid inside it."""
    liquid_height = min(measurement, cup_height)
    liquid = Rectangle(
        (-cup_width / 2, 0),
        cup_width,
        liquid_height,
        color=color,
    )
    ax.add_patch(liquid)

    cup = Rectangle(
        (-cup_width / 2, 0),
        cup_width,
        total_height,
        fill=False,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(cup)


def draw_scale(
    ax: Axes,
    cup_width: float,
    cup_height: float,
    max_scale: int,
    major_tick_interval: int,
    unit: str,
) -> None:
    """Draw the measurement scale with major ticks, labels, and sub-ticks."""
    scale_width = cup_width * 0.3
    sub_scale_width = scale_width * 0.6

    # Draw major ticks and labels
    for i in range(0, max_scale + 1, major_tick_interval):
        y = i / max_scale * cup_height
        ax.plot(
            [-scale_width / 2, scale_width / 2], [y, y], color="black", linewidth=1.3
        )
        if i > 0:  # Skip label for 0
            label = f"{i} ml" if unit == "milliliters" else f"{i/1000} L"
            ax.text(
                -cup_width / 2 + 0.3,
                y,
                label,
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=20,
            )

    # Draw sub-ticks
    num_sub_ticks = 3
    for i in range(0, max_scale, major_tick_interval):
        for j in range(1, num_sub_ticks + 1):
            sub_y = (
                (i + j * major_tick_interval / (num_sub_ticks + 1))
                / max_scale
                * cup_height
            )
            ax.plot(
                [-sub_scale_width / 2, sub_scale_width / 2],
                [sub_y, sub_y],
                color="black",
                linewidth=1,
            )


def set_plot_parameters(ax: Axes, cup_width: float, total_height: float) -> None:
    """Set the plot limits and aspect ratio."""
    ax.set_xlim(-cup_width / 2 - 0.2, cup_width / 2 + 0.2)
    ax.set_ylim(0, total_height)
    ax.set_aspect("equal")
    ax.axis("off")


def save_figure(fig: Figure, file_name: str) -> None:
    """Save the figure to a file."""
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)


def draw_analog_scale(ax: Axes, measurement: float, unit: str) -> None:
    """Draw an analog scale for weight measurements."""
    # Set up the scale
    scale_radius = 0.85
    trapezoid_height = 2.0
    trapezoid_top_width = 1.7
    trapezoid_bottom_width = 2.1
    circle_center_y = 0.1

    # Draw trapezoid base (no fill, only outline)
    trapezoid = Polygon(
        [
            (-trapezoid_bottom_width / 2, -trapezoid_height / 2),
            (trapezoid_bottom_width / 2, -trapezoid_height / 2),
            (trapezoid_top_width / 2, trapezoid_height / 2),
            (-trapezoid_top_width / 2, trapezoid_height / 2),
        ],
        facecolor="none",
        edgecolor="black",
        linewidth=3,
    )
    ax.add_patch(trapezoid)

    # Draw circular scale
    circle = Circle((0, circle_center_y), scale_radius, fill=False, edgecolor="black")
    ax.add_patch(circle)

    # Set max value and intervals based on unit
    if unit == "grams":
        if measurement <= 100:
            max_value = 100
            major_interval = 10
            minor_interval = 2
        elif measurement <= 500:
            max_value = 500
            major_interval = 50
            minor_interval = 10
        else:
            max_value = 1000
            major_interval = 100
            minor_interval = 20
    else:  # kilograms
        if measurement <= 100:
            max_value = 100
            major_interval = 10
            minor_interval = 2
        elif measurement <= 500:
            max_value = 500
            major_interval = 50
            minor_interval = 10
        else:
            max_value = 1000
            major_interval = 100
            minor_interval = 20

    # Add scale markings and numbers
    for i in np.arange(0, max_value + minor_interval / 2, minor_interval):
        angle = 90 - i * 340 / max_value
        rad_angle = np.radians(angle)
        x = scale_radius * np.cos(rad_angle)
        y = scale_radius * np.sin(rad_angle) + circle_center_y

        if abs(i % major_interval) < 1e-6:
            # Major tick
            ax.plot(
                [0.85 * x, x],
                [0.85 * (y - circle_center_y) + circle_center_y, y],
                color="black",
                linewidth=1.5,
            )
            # Display integer labels for major ticks
            label = f"{int(i)}"
            ax.text(
                0.7 * x,
                0.7 * (y - circle_center_y) + circle_center_y,
                label,
                ha="center",
                va="center",
                fontsize=20,
            )
        else:
            # Minor tick
            ax.plot(
                [0.92 * x, x],
                [0.92 * (y - circle_center_y) + circle_center_y, y],
                color="black",
                linewidth=0.5,
            )

    # Draw the measurement hand
    start_angle = 90
    end_angle = -250  # 90 - 340
    angle = start_angle - (measurement / max_value) * (start_angle - end_angle)

    hand = Polygon(
        [
            (0, circle_center_y),
            (
                scale_radius * 0.05 * np.cos(np.radians(angle + 90)),
                scale_radius * 0.05 * np.sin(np.radians(angle + 90)) + circle_center_y,
            ),
            (
                scale_radius * 0.9 * np.cos(np.radians(angle)),
                scale_radius * 0.9 * np.sin(np.radians(angle)) + circle_center_y,
            ),
            (
                scale_radius * 0.05 * np.cos(np.radians(angle - 90)),
                scale_radius * 0.05 * np.sin(np.radians(angle - 90)) + circle_center_y,
            ),
        ],
        facecolor="red",
        edgecolor="none",
    )
    ax.add_patch(hand)

    # Add center dot
    center_dot = Circle(
        (0, circle_center_y), scale_radius * 0.05, facecolor="black", edgecolor="none"
    )
    ax.add_patch(center_dot)

    # Add unit label with capitalized first letter
    capitalized_unit = unit.capitalize()
    ax.text(
        0,
        -trapezoid_height / 2 + 0.1,
        capitalized_unit,
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
    )


def set_analog_scale_parameters(ax: Axes) -> None:
    """Set the plot limits and aspect ratio for analog scale."""
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


@stimulus_function
def draw_measurement(stimulus_description: Measurements) -> str:
    try:
        fig, ax = plt.subplots(figsize=(8, 8))

        if stimulus_description.units in ["milliliters", "liters"]:
            measurement_ml = convert_to_ml(
                stimulus_description.measurement, stimulus_description.units
            )
            max_scale, major_tick_interval = get_scale_parameters(measurement_ml)
            cup_height, cup_width, buffer, total_height = calculate_dimensions(
                max_scale
            )
            measurement = measurement_ml / max_scale * cup_height

            draw_cup_and_liquid(
                ax,
                cup_width,
                cup_height,
                total_height,
                measurement,
                stimulus_description.color,
            )
            draw_scale(
                ax,
                cup_width,
                cup_height,
                max_scale,
                major_tick_interval,
                stimulus_description.units,
            )
            set_plot_parameters(ax, cup_width, total_height)
        elif stimulus_description.units in ["grams", "kilograms"]:
            # Convert kilograms to grams if necessary
            measurement = stimulus_description.measurement
            draw_analog_scale(ax, measurement, stimulus_description.units)
            set_analog_scale_parameters(ax)
        else:
            raise ValueError(f"Unsupported unit: {stimulus_description.units}")

        file_name = f"{settings.additional_content_settings.image_destination_folder}/measurement_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        save_figure(fig, file_name)

        return file_name
    except Exception as e:
        print(f"Error in draw_measurement: {str(e)}")
        raise
