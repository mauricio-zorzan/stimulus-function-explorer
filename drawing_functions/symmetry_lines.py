import logging
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.symmetry_identification_model import (
    SymmetryIdentification,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.symmetry_lines_model import (
    LinesOfSymmetry,
)
from content_generators.settings import settings
from matplotlib.lines import Line2D


@stimulus_function
def generate_lines_of_symmetry(stimulus_description: LinesOfSymmetry):
    logging.info(stimulus_description)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")

    # Plot the shape
    shape_coords = stimulus_description.shape_coordinates + [
        stimulus_description.shape_coordinates[0]
    ]
    x, y = zip(*shape_coords)
    ax.plot(x, y, "k-", linewidth=2)

    # Calculate the limits for the plot
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    padding = max((x_max - x_min), (y_max - y_min)) * 0.05  # Reduced padding
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Define a list of colors for the lines
    colors = [
        "r",
        "g",
        "b",
        "orange",
        "c",
        "m",
    ]  # Changed 'y' to 'orange' for better distinction

    # Plot the lines and prepare legend elements
    legend_elements = []
    for i, line in enumerate(stimulus_description.lines):
        color = colors[i % len(colors)]  # Cycle through colors
        if line.slope is None:
            ax.axvline(
                x=line.intercept, color=color, linestyle="--", alpha=0.7, linewidth=3
            )
        else:
            x_vals = np.array([x_min - padding, x_max + padding])
            y_vals = line.slope * x_vals + line.intercept
            ax.plot(x_vals, y_vals, color=color, linestyle="--", alpha=0.7, linewidth=3)

        # Create legend entry
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                label=f"Line {line.label}",
                linewidth=3,
            )
        )

    ax.set_aspect("equal", "box")

    # Remove axis
    ax.axis("off")

    # Only show legend if there are lines
    if legend_elements:
        # Calculate the position for the legend
        y_max_shape = max(y)  # Highest y-coordinate of the shape
        legend_y = (y_max_shape - y_min) / (y_max - y_min)

        # Add legend without title, aligned with the highest point of the shape
        legend = ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1, legend_y),
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
        plt.subplots_adjust(
            right=0.95
        )  # No need for extra space on the right if there's no legend

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/lines_of_symmetry_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.1,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_symmetry_identification_task(stimulus_description: SymmetryIdentification):
    """
    Generate a clean symmetry identification task similar to ixl
    Shows a single shape with one dashed line for students to identify if it's a line of symmetry.
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")

    # Define various object types that can be generated
    shape_type = stimulus_description.shape_type

    if shape_type == "flower":
        # Draw a symmetrical flower with proper petal shapes
        center = (0, 0)
        petal_count = 8

        # Draw petals as elongated ellipses radiating from center
        angles = np.linspace(0, 2 * np.pi, petal_count, endpoint=False)
        for angle in angles:
            # Create petal shape - elongated ellipse
            petal_length = 0.6  # Length of the petal (radial direction)
            petal_width = 0.2  # Width of the petal (perpendicular to radial)

            # Position petal center away from flower center
            petal_distance = (
                0.2 + petal_length / 2
            )  # 0.2 is the radius of center circle
            petal_center_x = center[0] + petal_distance * np.cos(angle)
            petal_center_y = center[1] + petal_distance * np.sin(angle)

            petal = patches.Ellipse(
                (petal_center_x, petal_center_y),
                petal_length,  # the first parameter (width along x-axis)
                petal_width,  # the second parameter (height along y-axis)
                angle=np.degrees(angle),  # rotate to point radially outward
                facecolor="pink",
                alpha=0.9,
                edgecolor="deeppink",
                linewidth=1.5,
            )
            ax.add_patch(petal)

        # Add flower center (yellow circle)
        flower_center = patches.Circle(
            center, 0.2, facecolor="yellow", alpha=1.0, edgecolor="orange", linewidth=2
        )
        ax.add_patch(flower_center)

        plot_bounds = (-1.0, 1.0, -1.0, 1.0)

    elif shape_type == "sun":
        # Draw a sun shape
        center = (0, 0)
        ray_count = 12
        inner_radius = 0.4
        outer_radius = 1.0

        # Draw rays
        angles = np.linspace(0, 2 * np.pi, ray_count, endpoint=False)
        for angle in angles:
            ray_x = [
                center[0] + inner_radius * np.cos(angle),
                center[0] + outer_radius * np.cos(angle),
            ]
            ray_y = [
                center[1] + inner_radius * np.sin(angle),
                center[1] + outer_radius * np.sin(angle),
            ]
            ax.plot(ray_x, ray_y, color="gold", linewidth=8, solid_capstyle="round")

        # Center circle - Fix: use patches.Circle instead of plt.Circle
        circle = patches.Circle(
            center,
            inner_radius,
            color="yellow",
            alpha=0.9,
            edgecolor="orange",
            linewidth=3,
        )
        ax.add_patch(circle)

        plot_bounds = (-1.3, 1.3, -1.3, 1.3)

    elif shape_type == "diamond":
        # Draw a diamond shape
        diamond_coords = [[0, 1], [0.8, 0], [0, -1], [-0.8, 0]]
        x, y = zip(*diamond_coords + [diamond_coords[0]])
        ax.fill(x, y, color="gold", alpha=0.8, edgecolor="darkorange", linewidth=3)

        plot_bounds = (-1.2, 1.2, -1.5, 1.5)

    elif shape_type == "heart":
        # Draw a heart shape
        t = np.linspace(0, 2 * np.pi, 200)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        x = x / 20  # Scale down
        y = y / 20
        ax.fill(x, y, color="pink", alpha=0.8, edgecolor="red", linewidth=3)

        plot_bounds = (-1.2, 1.2, -1.0, 1.2)

    elif shape_type == "house":
        # Draw a house shape
        # Base rectangle
        base_x = [-0.6, 0.6, 0.6, -0.6, -0.6]
        base_y = [-0.5, -0.5, 0.3, 0.3, -0.5]
        ax.fill(
            base_x, base_y, color="lightblue", alpha=0.8, edgecolor="blue", linewidth=3
        )

        # Roof triangle
        roof_x = [-0.8, 0, 0.8, -0.8]
        roof_y = [0.3, 0.8, 0.3, 0.3]
        ax.fill(
            roof_x, roof_y, color="brown", alpha=0.8, edgecolor="#654321", linewidth=3
        )

        # Door
        door_x = [-0.1, 0.1, 0.1, -0.1, -0.1]
        door_y = [-0.5, -0.5, 0.0, 0.0, -0.5]
        ax.fill(door_x, door_y, color="darkred", alpha=0.9)

        plot_bounds = (-1.2, 1.2, -0.8, 1.2)

    elif shape_type == "wheel":
        # Draw a ship's wheel / steering wheel
        center = (0, 0)
        outer_radius = 1.0
        inner_radius = 0.25
        spoke_count = 8

        # Draw outer rim
        outer_circle = patches.Circle(
            center,
            outer_radius,
            facecolor="none",
            edgecolor="#B8860B",  # Goldenrod
            linewidth=12,
        )
        ax.add_patch(outer_circle)

        # Inner rim line for depth (concentric, maintains symmetry)
        inner_rim = patches.Circle(
            center,
            outer_radius - 0.08,
            facecolor="none",
            edgecolor="#8B4513",  # Saddle brown
            linewidth=2,
            alpha=0.5,
        )
        ax.add_patch(inner_rim)

        # Draw spokes with perfect radial symmetry
        spoke_angles = np.linspace(0, 2 * np.pi, spoke_count, endpoint=False)
        for angle in spoke_angles:
            spoke_start_x = center[0] + inner_radius * 0.8 * np.cos(angle)
            spoke_start_y = center[1] + inner_radius * 0.8 * np.sin(angle)
            spoke_end_x = center[0] + (outer_radius - 0.08) * np.cos(angle)
            spoke_end_y = center[1] + (outer_radius - 0.08) * np.sin(angle)

            # Main spoke - single color for perfect symmetry
            ax.plot(
                [spoke_start_x, spoke_end_x],
                [spoke_start_y, spoke_end_y],
                color="#8B4513",
                linewidth=8,
                solid_capstyle="round",
            )

        # Draw center hub with concentric circles
        # Outer hub ring
        hub_outer = patches.Circle(
            center,
            inner_radius,
            facecolor="#654321",
            edgecolor="#8B4513",
            linewidth=3,
        )
        ax.add_patch(hub_outer)

        # Inner hub detail (concentric)
        hub_inner = patches.Circle(
            center,
            inner_radius * 0.6,
            facecolor="#8B4513",
            edgecolor="#654321",
            linewidth=2,
        )
        ax.add_patch(hub_inner)

        # Center bolt/cap (concentric)
        center_cap = patches.Circle(
            center,
            inner_radius * 0.3,
            facecolor="#4B2F1B",  # Very dark brown
            edgecolor="#654321",
            linewidth=1,
        )
        ax.add_patch(center_cap)

        # Add symmetric grip marks
        grip_count = 16
        grip_angles = np.linspace(0, 2 * np.pi, grip_count, endpoint=False)
        grip_radius = outer_radius - 0.06
        for angle in grip_angles:
            grip_x = center[0] + grip_radius * np.cos(angle)
            grip_y = center[1] + grip_radius * np.sin(angle)
            grip_mark = patches.Circle(
                (grip_x, grip_y),
                0.03,
                facecolor="#8B4513",
                alpha=0.3,
            )
            ax.add_patch(grip_mark)

        plot_bounds = (-1.2, 1.2, -1.2, 1.2)

    elif shape_type == "football":
        # Draw an American football shape
        center = (0, 0)

        # Create football shape
        football_length = 1.2  # Length (tip to tip) - horizontal
        football_width = 0.6  # Width (top to bottom) - vertical

        # Draw main football body
        football_body = patches.Ellipse(
            center,
            football_length,
            football_width,
            facecolor="#8B4513",  # Saddle brown
            alpha=1.0,
            edgecolor="#654321",  # Dark brown
            linewidth=2,
        )
        ax.add_patch(football_body)

        # Add centered highlight for 3D effect (maintains symmetry)
        highlight = patches.Ellipse(
            center,  # Keep centered for symmetry
            football_length * 0.85,
            football_width * 0.85,
            facecolor="#A0522D",  # Lighter brown
            alpha=0.3,  # More subtle
            edgecolor=None,
        )
        ax.add_patch(highlight)

        # Draw realistic laces - HORIZONTAL for a horizontal football
        lace_length = 0.4  # Length of the lacing area

        # Main lace line (horizontal)
        ax.plot(
            [-lace_length / 2, lace_length / 2],
            [0, 0],  # Centered at y=0
            color="white",
            linewidth=2,
            solid_capstyle="round",
        )

        # Cross laces (vertical lines)
        lace_count = 8
        lace_positions = np.linspace(
            -lace_length / 2 + 0.05, lace_length / 2 - 0.05, lace_count
        )

        for x_pos in lace_positions:
            # Draw vertical cross laces
            lace_height = 0.06
            ax.plot(
                [x_pos, x_pos],
                [-lace_height / 2, lace_height / 2],
                color="white",
                linewidth=2.5,
                solid_capstyle="round",
            )

        # Add symmetric seams (curved lines)
        # Top seam
        theta = np.linspace(-np.pi / 3, np.pi / 3, 50)
        seam_x_top = football_length / 3 * np.sin(theta)
        seam_y_top = football_width / 3 * np.ones_like(theta)
        ax.plot(seam_x_top, seam_y_top, color="#654321", linewidth=1.5, alpha=0.7)

        # Bottom seam (mirror of top)
        ax.plot(seam_x_top, -seam_y_top, color="#654321", linewidth=1.5, alpha=0.7)

        plot_bounds = (-1.2, 1.2, -0.8, 0.8)

    else:
        # Default to the provided polygon coordinates
        shape_coords = stimulus_description.shape_coordinates + [
            stimulus_description.shape_coordinates[0]
        ]
        x, y = zip(*shape_coords)
        ax.fill(
            x, y, color="lightsteelblue", alpha=0.7, edgecolor="darkblue", linewidth=3
        )

        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        padding = max((x_max - x_min), (y_max - y_min)) * 0.2
        plot_bounds = (
            x_min - padding,
            x_max + padding,
            y_min - padding,
            y_max + padding,
        )

    # Set plot limits
    ax.set_xlim(plot_bounds[0], plot_bounds[1])
    ax.set_ylim(plot_bounds[2], plot_bounds[3])

    # Draw the symmetry line (or potential symmetry line)
    if stimulus_description.lines:
        line = stimulus_description.lines[0]  # Only use first line

        if line.slope is None:
            # Vertical line
            y_range = plot_bounds[3] - plot_bounds[2]
            ax.plot(
                [line.intercept, line.intercept],
                [plot_bounds[2] - y_range * 0.1, plot_bounds[3] + y_range * 0.1],
                color="black",
                linestyle="--",
                linewidth=4,
                alpha=0.8,
            )
        else:
            # Angled line
            x_range = plot_bounds[1] - plot_bounds[0]
            x_vals = np.array(
                [plot_bounds[0] - x_range * 0.1, plot_bounds[1] + x_range * 0.1]
            )
            y_vals = line.slope * x_vals + line.intercept
            ax.plot(
                x_vals,
                y_vals,
                color="black",
                linestyle="--",
                linewidth=4,
                alpha=0.8,
            )

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/symmetry_identification_{shape_type}_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=300,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name
