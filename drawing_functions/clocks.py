import os
import time

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.clock import (
    Clock,
)
from content_generators.settings import settings
from matplotlib.font_manager import FontProperties
from matplotlib.patches import ArrowStyle, FancyArrowPatch, FancyBboxPatch


def angle_to_position(angle_deg: float, radius: float) -> tuple[float, float]:
    angle_rad = np.deg2rad(angle_deg)
    return radius * np.cos(angle_rad), radius * np.sin(angle_rad)


def create_analog_clock(stim: Clock):
    # Configuration constants
    figsize = (5, 5)
    clock_radius = 1
    text_radius = 0.75
    hour_tick_radius = 0.9
    minute_tick_radius = 0.95
    hour_hand_length = 0.45
    minute_hand_length = 0.65
    hour_tick_width = 3
    minute_tick_width = 1
    font_size = 20
    center_dot_radius = 0.04

    # Arrow styles with increased tail width for thicker hands
    hour_arrow_style = ArrowStyle("Simple, tail_width=6, head_width=10, head_length=10")
    minute_arrow_style = ArrowStyle("Simple, tail_width=4, head_width=8, head_length=8")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw clock circle
    ax.add_patch(
        plt.Circle((0, 0), clock_radius, fill=False, color="black", linewidth=3)
    )

    # Draw tick marks
    for i in range(60):
        angle = i * 6 + 90
        outer_point = angle_to_position(angle, clock_radius)
        if i % 5 == 0:
            inner_point = angle_to_position(angle, hour_tick_radius)
            linewidth = hour_tick_width
        else:
            inner_point = angle_to_position(angle, minute_tick_radius)
            linewidth = minute_tick_width
        ax.plot(
            [inner_point[0], outer_point[0]],
            [inner_point[1], outer_point[1]],
            "k-",
            lw=linewidth,
        )

    # Draw numbers
    for i in range(1, 13):
        angle = 90 - i * 30
        xy_text = angle_to_position(angle, text_radius)
        ax.text(
            xy_text[0],
            xy_text[1],
            str(i),
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
        )

    # Draw hands with pointy ends and increased thickness
    hour_angle = 90 - ((stim.hour % 12 + stim.minute / 60) * 30)
    minute_angle = 90 - (stim.minute * 6)
    hour_end = angle_to_position(hour_angle, hour_hand_length)
    minute_end = angle_to_position(minute_angle, minute_hand_length)

    ax.add_patch(
        FancyArrowPatch(
            (0, 0),
            hour_end,
            connectionstyle="arc3",
            arrowstyle=hour_arrow_style,
            color="black",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0, 0),
            minute_end,
            connectionstyle="arc3",
            arrowstyle=minute_arrow_style,
            color="black",
        )
    )

    # Add a dot at the center
    ax.add_patch(plt.Circle((0, 0), center_dot_radius, color="black"))
    # Save figure
    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/clock_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    # plt.savefig(file_name, transparent=True, bbox_inches="tight")
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
    # plt.show()


def create_digital_clock(stim: Clock):
    # Configuration constants
    fig_width, fig_height = 4, 2  # Dimensions of the figure
    font_color = "white"  # Color of the numbers
    outline_color = "grey"  # Color of the outline
    font_size = 96
    outline_width = 3  # Width of the outline
    corner_radius = 0.05  # Radius of corners of the rectangle
    vertical_offset = 0  # Offset to adjust vertical positioning of text

    # Load the digital font
    font_path = os.path.join(os.path.dirname(__file__), "additional_fonts/DS-DIGI.TTF")
    digital_font = FontProperties(fname=font_path)

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # ax.set_facecolor((0, 0, 0, 0))  # RGBA tuple for transparency
    # fig.patch.set_alpha(0.0)  # Make background completely transparent

    # Remove axes
    ax.axis("off")

    # Time formatting
    hour = f"{stim.hour}"
    minute = f"{stim.minute:02}"
    time_str = f"{hour}:{minute}"

    # Draw the time text with adjusted vertical offset
    text = ax.text(
        0.5,
        0.5 + vertical_offset,
        time_str,
        color=font_color,
        fontsize=font_size,
        fontproperties=digital_font,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    # Get the bounding box of the text
    fig.canvas.draw()
    bbox = text.get_window_extent(fig.canvas.renderer).transformed(
        ax.transAxes.inverted()
    )

    # Calculate expanded bounding box
    width_exp_factor = 1.1  # Reduced expansion factor for width
    height_exp_factor = 1.1  # Keep the same expansion factor for height
    width = bbox.width * width_exp_factor
    height = bbox.height * height_exp_factor
    new_x0 = 0.5 - width / 2
    new_y0 = 0.5 - height / 2

    # Draw a rounded rectangle around the text
    rect = FancyBboxPatch(
        (new_x0, new_y0),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        ec=outline_color,
        fc="black",
        lw=outline_width,
        transform=ax.transAxes,
        clip_on=False,
        zorder=1,
    )

    ax.add_patch(rect)

    # Ensure text is on top of the rectangle
    text.set_zorder(2)

    # Save figure
    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/clock_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    # plt.savefig(file_name, transparent=True, bbox_inches="tight")
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
    # plt.show()


@stimulus_function
def create_clock(data: Clock):
    clock_type = data.type

    if clock_type == "digital":
        file_name = create_digital_clock(data)
    else:
        file_name = create_analog_clock(data)

    return file_name
