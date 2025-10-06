import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stepwise_dot_pattern import (
    StepwiseShapePattern,
)
from content_generators.settings import settings


@stimulus_function
def draw_stepwise_shape_pattern(stimulus: StepwiseShapePattern):
    steps = stimulus.steps
    n_steps = len(steps)

    # Validate that steps have labels when there are more than 3 steps
    if n_steps > 3:
        for idx, step in enumerate(steps):
            if not step.label:
                raise ValueError(
                    f"Step {idx + 1} must have a label when there are more than 3 steps total. "
                    f"Please provide a label for all steps in the pattern."
                )

    shape_size = stimulus.shape_size
    # spacing = stimulus.spacing
    # max_cols = max(step.columns for step in steps)
    steps_per_row = 3  # Show 3 steps per row
    n_rows = (n_steps + steps_per_row - 1) // steps_per_row  # Ceiling division

    # Calculate base font size and adjust based on number of rows
    base_font_size = 24
    # Only increase font size if we have more than 3 steps
    font_size = base_font_size if n_steps <= 3 else base_font_size + (n_rows - 1) * 10

    # Calculate dynamic spacing based on shape size and pattern dimensions
    def calculate_step_width(step):
        # Base width is the number of columns
        width = step.columns
        # Adjust for shape size
        width *= shape_size
        # Add extra space for triangles and squares which need more room
        if step.shape in ["triangle", "square"]:
            width *= 1
        return width

    # Calculate spacing based on width of quarter column (reduced from half)
    column_width = shape_size  # Width of a single column
    dynamic_spacing = (
        column_width * 0.25
    )  # Space between steps is width of quarter column

    # Create a temporary figure to measure text width
    temp_fig, temp_ax = plt.subplots()
    temp_ax.axis("off")

    # Calculate maximum width for each step considering both pattern and label
    step_widths = []
    for step in steps:
        pattern_width = calculate_step_width(step)
        # Measure label width if it exists
        if step.label:
            text = temp_ax.text(0, 0, step.label, fontsize=font_size)
            text_width = text.get_window_extent().width / temp_fig.dpi
            # Convert to data coordinates (approximate)
            label_width = text_width * 2  # Scale factor to match data coordinates
            step_widths.append(max(pattern_width, label_width))
        else:
            step_widths.append(pattern_width)

    plt.close(temp_fig)

    # Calculate figure dimensions
    fig_width = steps_per_row * (max(step_widths) + dynamic_spacing)

    # --- FLEXBOX-LIKE LAYOUT LOGIC ---
    label_height = 0.7  # Space for label below each pattern
    step_padding = 0.5  # Padding inside each step box
    row_padding = 1.0  # Padding between rows

    # 1. Calculate step box heights for each step
    step_box_heights = []
    for step in steps:
        pattern_height = step.rows * shape_size
        box_height = pattern_height + label_height + step_padding
        step_box_heights.append(box_height)

    # 2. For each row, get the max step box height
    row_heights = []
    for row in range(n_rows):
        start_idx = row * steps_per_row
        end_idx = min((row + 1) * steps_per_row, n_steps)
        row_box_heights = step_box_heights[start_idx:end_idx]
        row_heights.append(max(row_box_heights))

    # 3. Compute y-offsets for each row (from top to bottom)
    row_y_offsets = []
    y_cursor = 0
    for h in row_heights:
        row_y_offsets.append(y_cursor)
        y_cursor -= h + row_padding

    # 4. For each row, compute the label y-position (bottom of the row's box)
    row_label_y = []
    for row in range(n_rows):
        # The label line for this row is at the bottom of the row's box
        row_label_y.append(row_y_offsets[row] - row_heights[row] + label_height)

    # 5. Figure height
    fig_height = sum(row_heights) + (n_rows - 1) * row_padding + 2

    # 6. Figure width (same as before)
    fig_width = steps_per_row * (max(step_widths) + dynamic_spacing)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_aspect("equal")
    ax.axis("off")

    step_centers_x = np.linspace(0.5, fig_width - 0.5, steps_per_row)

    overall_leftmost = float("inf")
    overall_rightmost = float("-inf")
    overall_top = float("-inf")
    overall_bottom = float("inf")

    for idx, step in enumerate(steps):
        row = idx // steps_per_row
        col = idx % steps_per_row
        box_height = step_box_heights[idx]
        pattern_height = step.rows * shape_size
        # y_offset = row_y_offsets[row]
        # Bottom-align the pattern in the box
        pattern_base_y = row_label_y[row] + label_height
        for step_row in range(step.rows):
            for step_col in range(step.columns):
                if step.columns == 1:
                    x = step_centers_x[col]
                else:
                    start_x = step_centers_x[col] - (step.columns - 1) / 2
                    x = start_x + step_col
                y = pattern_base_y + (step.rows - step_row - 1) * shape_size
                if step.shape == "circle":
                    ax.plot(
                        x,
                        y,
                        "o",
                        color=step.color,
                        markersize=30 * shape_size,
                        markeredgewidth=1,
                        markeredgecolor="black",
                        alpha=0.7,
                    )
                elif step.shape == "square":
                    size = 0.8 * shape_size
                    square = patches.Rectangle(
                        (x - size / 2, y - size / 2),
                        size,
                        size,
                        linewidth=1,
                        edgecolor="black",
                        facecolor=step.color,
                        alpha=0.7,
                    )
                    ax.add_patch(square)
                elif step.shape == "triangle":
                    rot_deg = getattr(step, "rotation", 0.0) or 0.0
                    size = 0.8 * shape_size
                    base_triangle = [
                        (0, size / 2),
                        (-size / 2, -size / 2),
                        (size / 2, -size / 2),
                    ]
                    theta = np.radians(-rot_deg)
                    rot_matrix = np.array(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)],
                        ]
                    )
                    rotated = [
                        tuple(np.array([x, y]) + rot_matrix @ np.array([px, py]))
                        for (px, py) in base_triangle
                    ]
                    triangle = patches.Polygon(
                        rotated,
                        closed=True,
                        linewidth=1,
                        edgecolor="black",
                        facecolor=step.color,
                        alpha=0.7,
                    )
                    ax.add_patch(triangle)
                overall_leftmost = min(overall_leftmost, x - 0.5)
                overall_rightmost = max(overall_rightmost, x + 0.5)
                overall_top = max(overall_top, y + 0.5)
                overall_bottom = min(overall_bottom, y - 0.5)
        # Label below the pattern, aligned for the row
        ax.text(
            step_centers_x[col],
            row_label_y[row],
            step.label or f"step {idx + 1}",
            ha="center",
            va="top",
            fontsize=font_size,
        )

    ax.set_xlim(overall_leftmost - 0.5, overall_rightmost + 0.5)
    ax.set_ylim(overall_bottom - 0.5, overall_top + 0.5)
    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/stepwise_shape_pattern_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=300,
    )
    plt.close()
    return file_name
