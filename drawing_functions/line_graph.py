# drawing_functions/line_graph.py

import time

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.line_graph_description import (
    LineGraphList,
)
from content_generators.settings import settings


@stimulus_function
def create_line_graph(data: LineGraphList):
    """
    Takes a LineGraphList object and generates a line graph image from it.
    The graph can contain one or more colored lines.
    """
    try:
        # Extract the single graph definition from the list
        graph_info = data[0]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # --- Plot each line series ---
        for series in graph_info.data_series:
            # Unpack the x and y coordinates from the data points
            x_values = [point.x for point in series.data_points]
            y_values = [point.y for point in series.data_points]

            ax.plot(
                x_values,
                y_values,
                label=series.label,
                color=series.color or "blue",  # Use specified color or default to blue
                marker="o",  # Add markers to clearly show data points
                linestyle="-",  # Solid line
                linewidth=2,
            )

        # --- Customize the graph's appearance ---
        ax.set_title(graph_info.title, fontweight="bold", fontsize=16)
        ax.set_xlabel(graph_info.x_axis_label, fontweight="bold", fontsize=14)
        ax.set_ylabel(graph_info.y_axis_label, fontweight="bold", fontsize=14)

        ax.grid(True, which="major", linestyle="--", linewidth=0.7, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Show legend only if there are labels defined
        if any(s.label for s in graph_info.data_series):
            ax.legend()

        # Adjust layout to prevent labels from being cut off
        plt.tight_layout()

        # --- Save the file ---
        file_name = f"{settings.additional_content_settings.image_destination_folder}/line_graph_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.savefig(
            file_name,
            dpi=600,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name

    except Exception as e:
        print(f"An error occurred while creating the line graph: {str(e)}")
        raise
