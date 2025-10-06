import time

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.number_line_clock_model import (
    NumberLineClockStimulus,
)
from content_generators.settings import settings


@stimulus_function
def create_clock_number_line(stim_desc: NumberLineClockStimulus):
    try:
        fig, ax = plt.subplots(figsize=(10, 2))

        range_min = stim_desc.range.min
        range_max = stim_desc.range.max

        # Handle crossing 12 o'clock
        if range_max < range_min:
            range_max += 12

        # Add a 15-minute buffer on each side
        buffer = 0.25  # 15 minutes = 0.25 hours
        ax.set_xlim(range_min - buffer, range_max + buffer)
        ax.set_ylim(-0.5, 0.5)

        # Set up the number line
        ax.spines["bottom"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.get_yaxis().set_visible(False)
        ax.xaxis.set_visible(False)  # Hide default ticks

        # Set the bounds of the bottom spine
        ax.spines["bottom"].set_bounds(range_min, range_max)

        # Draw 5-minute ticks
        minor_ticks_5 = np.arange(np.ceil(range_min * 12) / 12, range_max, 1 / 12)
        for tick in minor_ticks_5:
            ax.plot([tick, tick], [0, -0.05], color="gray", linewidth=0.8, zorder=1)

        # Draw 15-minute ticks
        minor_ticks_15 = np.arange(np.ceil(range_min * 4) / 4, range_max, 0.25)
        for tick in minor_ticks_15:
            ax.plot([tick, tick], [0, -0.08], color="black", linewidth=1.3, zorder=2)

        # Draw hour ticks and labels
        major_ticks = np.arange(np.ceil(range_min), np.floor(range_max) + 1)
        for tick in major_ticks:
            ax.plot([tick, tick], [0, -0.11], color="black", linewidth=1.7, zorder=3)
            ax.text(
                tick,
                -0.2,
                f"{int((tick-1)%12+1):d}:00",
                ha="center",
                va="top",
                fontsize=14,
            )

        # Plot points
        for point in stim_desc.points:
            x_value = point.hour + point.minute / 60
            if x_value < range_min:
                x_value += 12
            ax.plot(x_value, 0, "ko", markersize=8, zorder=4)
            ax.annotate(
                f"{point.label}",
                (x_value, 0),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                zorder=5,
            )

        plt.tight_layout()

        file_name = f"{settings.additional_content_settings.image_destination_folder}/clock_number_line_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            transparent=False,
            bbox_inches="tight",
            dpi=600,
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()
        return file_name

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
