import random
import time

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.bar_model import (
    BarModel,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.box_plots import (
    BoxPlotData,
    BoxPlotDescription,
)
from content_generators.settings import settings
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import NullLocator

# Color palette with 10 distinct colors
COLOR_PALETTE = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Lavender
    "#85C1E9",  # Light Blue
]


def get_random_colors(num_colors: int) -> list[tuple[str, str]]:
    """Get random colors with their darker shades for box plots.

    Returns:
        List of tuples containing (light_color, dark_color) for each box plot
    """
    selected_colors = random.sample(COLOR_PALETTE, min(num_colors, len(COLOR_PALETTE)))

    # If we need more colors than available, cycle through the palette
    while len(selected_colors) < num_colors:
        selected_colors.extend(
            random.sample(
                COLOR_PALETTE,
                min(num_colors - len(selected_colors), len(COLOR_PALETTE)),
            )
        )

    # Create darker shades for borders and median lines
    color_pairs = []
    for color in selected_colors[:num_colors]:
        # Convert hex to RGB and darken
        hex_color = color.lstrip("#")
        rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        darker_rgb = tuple(
            max(0, int(c * 0.6)) for c in rgb
        )  # 60% of original brightness
        darker_hex = f"#{darker_rgb[0]:02x}{darker_rgb[1]:02x}{darker_rgb[2]:02x}"
        color_pairs.append((color, darker_hex))

    return color_pairs


@stimulus_function
def draw_bar_models(data: BarModel):
    labels = [item.label for item in data]
    lengths = [item.length for item in data]

    fig, ax = plt.subplots()

    # Creating horizontal bars
    ax.barh(labels, lengths, color=["skyblue", "lightgreen"])

    for label in ax.get_yticklabels():
        label.set_fontsize(20)

    # Removing the surrounding box
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Removing ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Removing x-axis ticks and labels
    ax.xaxis.set_major_locator(NullLocator())

    file_name = f"{settings.additional_content_settings.image_destination_folder}/bar_model_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
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
def draw_box_plots(box_plot_description: BoxPlotDescription):
    _, ax = plt.subplots(figsize=(10, 4))
    ax: Axes
    # Set the title and x-axis label
    display_title = box_plot_description.get_display_title()
    if display_title:
        ax.set_title(display_title, fontsize=16, fontweight="bold", pad=20)

    # Prepare data for plotting
    class_names = [data.class_name for data in box_plot_description.data]
    min_values = [int(data.min_value) for data in box_plot_description.data]
    q1_values = [int(data.q1) for data in box_plot_description.data]
    median_values = [int(data.median) for data in box_plot_description.data]
    q3_values = [int(data.q3) for data in box_plot_description.data]
    max_values = [int(data.max_value) for data in box_plot_description.data]

    # Get random colors for the box plots
    num_plots = len(class_names)
    color_pairs = get_random_colors(num_plots)

    # Plot each box plot with custom colors
    for i, class_name in enumerate(class_names):
        light_color, dark_color = color_pairs[i]

        # Create box plot with custom styling
        ax.bxp(
            [
                {
                    "med": median_values[i],
                    "q1": q1_values[i],
                    "q3": q3_values[i],
                    "whislo": min_values[i],
                    "whishi": max_values[i],
                    "fliers": [],
                }
            ],
            positions=[0.5 - (i * 0.5)],
            showfliers=False,
            vert=False,
            patch_artist=True,  # Enable custom coloring
            boxprops=dict(
                facecolor=light_color, edgecolor=dark_color, linewidth=2, alpha=0.8
            ),
            whiskerprops=dict(color=dark_color, linewidth=2),
            capprops=dict(color=dark_color, linewidth=2),
            medianprops=dict(
                color=dark_color,
                linewidth=3,  # Thicker median line for better visibility
                solid_capstyle="butt",
            ),
            flierprops=dict(
                marker="o",
                markerfacecolor=dark_color,
                markeredgecolor=dark_color,
                markersize=6,
            ),
        )

    # Set the y-ticks to the class names if they are set
    if None not in class_names:
        ax.set_yticks(np.arange(len(class_names)) * 0.5)
        bold_font = FontProperties(weight="bold")
        ax.set_yticklabels(
            [f"{name}" for name in reversed(class_names)],
            minor=False,
            fontproperties=bold_font,
        )
        ax.tick_params(axis="y", which="both", length=0)
    else:
        ax.set_yticks([])

    # Create regular scaled number line with appropriate intervals
    min_value = min(min_values)
    max_value = max(max_values)
    data_range = max_value - min_value

    # Determine the best interval (2, 5, or 10) based on data range
    if data_range <= 20:
        interval = 2
    elif data_range <= 50:
        interval = 5
    else:
        interval = 10

    # Calculate the start and end points for the number line
    # Round down to the nearest interval for start, round up for end
    start_value = (min_value // interval) * interval
    end_value = ((max_value + interval - 1) // interval) * interval

    # Create major ticks with the determined interval
    major_ticks = list(range(start_value, end_value + 1, interval))
    ax.set_xticks(major_ticks)

    # Set larger font size for x-axis labels and make ticks more prominent
    ax.tick_params(axis="x", which="major", length=8, width=1.5, labelsize=12)

    # Create minor ticks for values between major ticks
    minor_ticks = []
    for i in range(len(major_ticks) - 1):
        minor_ticks.extend(range(major_ticks[i] + 1, major_ticks[i + 1]))

    if minor_ticks:
        ax.set_xticks(minor_ticks, minor=True)
        ax.tick_params(axis="x", which="minor", length=5, width=1)

    # Remove grid background
    ax.grid(False)

    # Remove the surrounding box (spines) except for the x-axis spine
    for spine_name, spine in ax.spines.items():
        if spine_name != "bottom":  # 'bottom' corresponds to the x-axis spine
            spine.set_visible(False)

    plt.subplots_adjust(left=0.15, right=0.9, top=0.85, bottom=0.15)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/box_plot_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


# Example usage
if __name__ == "__main__":
    example_data = BoxPlotDescription(
        title="Test Scores",
        data=[
            BoxPlotData(
                class_name="Class A",
                min_value=60,
                q1=70,
                median=80,
                q3=90,
                max_value=100,
            ),
            BoxPlotData(
                class_name="Class B",
                min_value=65,
                q1=75,
                median=85,
                q3=95,
                max_value=100,
            ),
        ],
    )
    j = example_data.model_dump_json()
    print(j)
    draw_box_plots(j)
