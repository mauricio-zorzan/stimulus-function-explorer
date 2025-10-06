import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.angle_on_circle import (
    AngleRange,
    CircleAngle,
)
from content_generators.settings import settings


def validate_angle_range_constraints(stimulus: CircleAngle):
    """Validate angle constraints based on range category."""
    angle = stimulus.angle_measure
    range_cat = stimulus.range_category

    # Basic range: 1-180 degrees
    if range_cat == AngleRange.BASIC and (angle <= 0 or angle > 180):
        raise ValueError("Basic range requires angles between 1 and 180 degrees")

    # Intermediate range: 181-359 degrees
    elif range_cat == AngleRange.INTERMEDIATE and (angle <= 180 or angle >= 360):
        raise ValueError(
            "Intermediate range requires angles between 181 and 359 degrees"
        )

    # Advanced range: no constraints yet (reserved for future use)
    return True


@stimulus_function
def draw_circle_angle_measurement(stimulus: CircleAngle):
    """
    Draw a circle with degree markings for angle measurement exercises.

    Creates a circle with:
    - Degree markings every 15°
    - Major labels at 0°, 90°, 180°, 270°
    - Shaded sector showing the angle to measure
    """
    # Validate range constraints in the drawing function
    validate_angle_range_constraints(stimulus)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    # Circle parameters
    center = (0, 0)
    radius = 1.0

    # Draw main circle
    circle = patches.Circle(center, radius, fill=False, edgecolor="gray", linewidth=3)
    ax.add_patch(circle)

    # Draw degree markings and labels
    draw_degree_markings(ax, center, radius)

    # Draw the angle sector
    draw_angle_sector(ax, center, radius, stimulus)

    # Draw rays forming the angle
    draw_angle_rays(ax, center, radius, stimulus)

    # Set plot limits with padding
    padding = 0.3
    ax.set_xlim(-radius - padding, radius + padding)
    ax.set_ylim(-radius - padding, radius + padding)

    plt.tight_layout()

    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/circle_angle_{stimulus.angle_measure}deg_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def draw_degree_markings(ax, center, radius):
    """Draw degree markings every 15° around the circle."""
    cx, cy = center

    # Draw tick marks every 15 degrees
    for angle_deg in range(0, 360, 15):
        angle_rad = np.deg2rad(angle_deg)

        # Calculate tick mark positions
        # Outer tick (on circle)
        x_outer = cx + radius * np.cos(angle_rad)
        y_outer = cy + radius * np.sin(angle_rad)

        # Inner tick (slightly inside circle)
        tick_length = 0.08
        x_inner = cx + (radius - tick_length) * np.cos(angle_rad)
        y_inner = cy + (radius - tick_length) * np.sin(angle_rad)

        # Draw tick mark
        ax.plot([x_inner, x_outer], [y_inner, y_outer], "k-", linewidth=2)

    # Add major degree labels at cardinal directions
    major_angles = [0, 90, 180, 270]
    label_radius = radius + 0.15

    for angle_deg in major_angles:
        angle_rad = np.deg2rad(angle_deg)

        # Calculate label position
        x_label = cx + label_radius * np.cos(angle_rad)
        y_label = cy + label_radius * np.sin(angle_rad)

        # Add degree label
        ax.text(
            x_label,
            y_label,
            f"{angle_deg}°",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )


def draw_angle_sector(ax, center, radius, stimulus: CircleAngle):
    """Draw the shaded sector representing the angle."""
    # Convert angles to matplotlib convention (0° = +x axis, counterclockwise positive)
    start_angle = stimulus.start_position
    end_angle = stimulus.start_position + stimulus.angle_measure

    # Create sector patch
    sector = patches.Wedge(
        center,
        radius,
        theta1=start_angle,
        theta2=end_angle,
        facecolor=stimulus.sector_color,
        alpha=0.6,
        edgecolor="darkgreen",
        linewidth=2,
    )
    ax.add_patch(sector)


def draw_angle_rays(ax, center, radius, stimulus: CircleAngle):
    """Draw the rays forming the angle."""
    cx, cy = center

    # Calculate ray endpoints
    start_rad = np.deg2rad(stimulus.start_position)
    end_rad = np.deg2rad(stimulus.start_position + stimulus.angle_measure)

    # Starting ray
    x1 = cx + radius * np.cos(start_rad)
    y1 = cy + radius * np.sin(start_rad)
    ax.plot([cx, x1], [cy, y1], "k-", linewidth=3)

    # Ending ray
    x2 = cx + radius * np.cos(end_rad)
    y2 = cy + radius * np.sin(end_rad)
    ax.plot([cx, x2], [cy, y2], "k-", linewidth=3)

    # Add center point
    ax.plot(cx, cy, "ko", markersize=6)


def get_angle_endpoints(center, radius, start_angle, angle_measure):
    """Helper function to get the endpoints of an angle's rays."""
    cx, cy = center

    start_rad = np.deg2rad(start_angle)
    end_rad = np.deg2rad(start_angle + angle_measure)

    start_point = (cx + radius * np.cos(start_rad), cy + radius * np.sin(start_rad))
    end_point = (cx + radius * np.cos(end_rad), cy + radius * np.sin(end_rad))

    return start_point, end_point


def calculate_arc_midpoint(center, radius, start_angle, angle_measure):
    """Helper function to calculate the midpoint of an arc."""
    cx, cy = center
    mid_angle = start_angle + angle_measure / 2
    mid_rad = np.deg2rad(mid_angle)

    mid_x = cx + radius * 0.7 * np.cos(mid_rad)  # Slightly inside the circle
    mid_y = cy + radius * 0.7 * np.sin(mid_rad)

    return mid_x, mid_y
