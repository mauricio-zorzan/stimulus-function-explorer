import random
import time

import matplotlib
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.base_ten_block import (
    BaseTenBlock,
    BaseTenBlockDivisionStimulus,
    BaseTenBlockGridStimulus,
    BaseTenBlockStimulus,
)
from content_generators.settings import settings
from matplotlib.patches import Rectangle


def generate_random_palette():
    """
    Generate a random color palette with 3D shading effect.
    Returns a palette with top (lightest), front (medium), and right (darkest) shades.
    """
    # Generate a random base hue (0-360 degrees)
    hue = random.randint(0, 360)

    # High saturation and varying lightness for 3D effect
    saturation = random.randint(60, 90)  # Keep saturation high for vibrant colors

    # Three lightness levels for 3D shading
    top_lightness = random.randint(75, 90)  # Lightest (top face)
    front_lightness = random.randint(50, 70)  # Medium (front face)
    right_lightness = random.randint(25, 45)  # Darkest (right face)

    def hsl_to_hex(h, s, l):
        """Convert HSL to hex color"""
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0

        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        if s == 0:
            r = g = b = l  # achromatic
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return {
        "top": hsl_to_hex(hue, saturation, top_lightness),
        "front": hsl_to_hex(hue, saturation, front_lightness),
        "right": hsl_to_hex(hue, saturation, right_lightness),
    }


def _draw_single_block(
    ax,
    value,
    x_offset: float = 0,
    y_offset: float = 0,
    align_from_top: bool = False,
    max_height: float = 0,
):
    """
    Helper function to draw a single base ten block representation.

    Args:
        ax: Matplotlib axes object to draw on
        value: The numerical value to represent
        x_offset: Horizontal offset for positioning
        y_offset: Vertical offset for positioning
        align_from_top: Whether to align from top (True) or bottom (False)
        max_height: Maximum height for top alignment

    Returns:
        tuple: (max_width, total_height) of the drawn block
    """
    hundreds = value // 100
    tens = (value % 100) // 10
    ones = value % 10

    max_width = 0

    # Block size - increased from 0.1 to 0.15
    block_size = 0.15

    # Use consistent spacing throughout
    internal_spacing = 0.03  # Small gap between individual elements within same type
    between_type_spacing = 0.2  # Gap between different block types

    # Calculate heights for positioning
    ones_height = block_size if ones > 0 else 0
    tens_height = (
        tens * (block_size + internal_spacing) if tens > 0 else 0
    )  # Small gap between ten strips
    hundreds_height = (
        hundreds * (10 * block_size + internal_spacing) if hundreds > 0 else 0
    )  # 10x10 grid + small gap

    # Count the number of non-zero block types for spacing
    non_zero_types = sum([ones > 0, tens > 0, hundreds > 0])
    total_spacing = between_type_spacing * max(0, non_zero_types - 1)
    total_height = ones_height + tens_height + hundreds_height + total_spacing

    # For top alignment, start from the top and work downward
    if align_from_top:
        # Start from the top of the canvas - this is the reference point for ALL blocks
        top_y = y_offset + max_height
    else:
        # Start from the bottom and work upward (default behavior)
        current_y = y_offset

    # When aligning from top, we draw in reverse order: hundreds, tens, ones
    if align_from_top:
        # All blocks start from the same top_y position

        # Draw hundreds (cyan squares) at the top first
        if hundreds > 0:
            for h in range(hundreds):
                # Draw each hundred block as a 10x10 grid
                for j in range(10):
                    for k in range(10):
                        rect = Rectangle(
                            (
                                x_offset + j * block_size,
                                top_y
                                - (
                                    h * (10 * block_size + internal_spacing)
                                    + (k + 1) * block_size
                                ),
                            ),
                            block_size,
                            block_size,
                            facecolor="cyan",
                            edgecolor="black",
                            linewidth=1,
                        )
                        ax.add_patch(rect)
            hundreds_width = 10 * block_size
            max_width = max(max_width, hundreds_width)

        # Draw tens (lime rectangles) - also starting from top_y
        if tens > 0:
            # Calculate position relative to top_y, accounting for hundreds above if they exist
            tens_start_y = top_y - (
                hundreds_height + (between_type_spacing if hundreds > 0 else 0)
            )

            for t in range(tens):
                # Draw each ten as a horizontal strip (10 blocks wide, 1 block tall)
                for j in range(10):
                    rect = Rectangle(
                        (
                            x_offset + j * block_size,
                            tens_start_y
                            - block_size
                            - t * (block_size + internal_spacing),
                        ),
                        block_size,
                        block_size,
                        facecolor="lime",
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax.add_patch(rect)
            tens_width = 10 * block_size
            max_width = max(max_width, tens_width)

        # Draw ones (orange single blocks) - also starting from top_y
        if ones > 0:
            # Calculate position relative to top_y, accounting for hundreds and tens above
            ones_start_y = top_y - (
                hundreds_height
                + (between_type_spacing if hundreds > 0 else 0)
                + tens_height
                + (between_type_spacing if tens > 0 else 0)
            )

            for i in range(ones):
                rect = Rectangle(
                    (
                        x_offset + i * (block_size + internal_spacing),
                        ones_start_y - block_size,
                    ),
                    block_size,
                    block_size,
                    facecolor="orange",
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)
            ones_width = ones * (block_size + internal_spacing)
            max_width = max(max_width, ones_width)
    else:
        # Original bottom-up drawing logic for single blocks
        # Draw ones (orange single blocks) at the bottom
        if ones > 0:
            for i in range(ones):
                rect = Rectangle(
                    (x_offset + i * (block_size + internal_spacing), current_y),
                    block_size,
                    block_size,
                    facecolor="orange",
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)
            ones_width = ones * (block_size + internal_spacing)
            max_width = max(max_width, ones_width)
            current_y += ones_height

            # Add spacing if there are more block types above
            if tens > 0 or hundreds > 0:
                current_y += between_type_spacing

        # Draw tens (lime rectangles) above ones
        if tens > 0:
            for t in range(tens):
                # Draw each ten as a horizontal strip (10 blocks wide, 1 block tall)
                for j in range(10):
                    rect = Rectangle(
                        (
                            x_offset + j * block_size,
                            current_y + t * (block_size + internal_spacing),
                        ),
                        block_size,
                        block_size,
                        facecolor="lime",
                        edgecolor="black",
                        linewidth=1,
                    )
                    ax.add_patch(rect)
            tens_width = 10 * block_size
            max_width = max(max_width, tens_width)
            current_y += tens_height

            # Add spacing if there are hundreds above
            if hundreds > 0:
                current_y += between_type_spacing

        # Draw hundreds (cyan squares) at the top
        if hundreds > 0:
            for h in range(hundreds):
                # Draw each hundred block as a 10x10 grid
                for j in range(10):
                    for k in range(10):
                        rect = Rectangle(
                            (
                                x_offset + j * block_size,
                                current_y
                                + h * (10 * block_size + internal_spacing)
                                + k * block_size,
                            ),
                            block_size,
                            block_size,
                            facecolor="cyan",
                            edgecolor="black",
                            linewidth=1,
                        )
                        ax.add_patch(rect)
            hundreds_width = 10 * block_size
            max_width = max(max_width, hundreds_width)

    return max_width, total_height  # Return width and total height


@stimulus_function
def draw_base_ten_blocks(stimulus: BaseTenBlockStimulus):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # ---------- helpers ----------
    def cube_faces(i, j, k):
        v0 = (i, j, k)
        v1 = (i + 1, j, k)
        v2 = (i + 1, j + 1, k)
        v4 = (i, j, k + 1)
        v5 = (i + 1, j, k + 1)
        v6 = (i + 1, j + 1, k + 1)
        v7 = (i, j + 1, k + 1)
        return {
            "top": [v4, v5, v6, v7],  # +z
            "right": [v1, v2, v6, v5],  # +x
            "front": [v0, v1, v5, v4],  #  y toward viewer
        }

    # Generate random color palettes for each block group
    num_blocks = len(stimulus.blocks)
    palettes = [generate_random_palette() for _ in range(num_blocks)]
    edge_color = "black"
    edge_width = 0.4

    def draw_cube(ax, i, j, k, pal):
        faces = cube_faces(i, j, k)
        for key in ("top", "right", "front"):
            ax.add_collection3d(
                Poly3DCollection(
                    [faces[key]],
                    facecolors=pal[key],
                    edgecolors=edge_color,
                    linewidths=edge_width,
                )
            )

    def format_value(block: BaseTenBlock) -> str:
        if not block.display_as_decimal:
            return str(block.value)
        decimal_value = (
            ((block.value // 100) % 10) * 1.0
            + ((block.value // 10) % 10) * 0.1
            + (block.value % 10) * 0.01
        )
        return f"{decimal_value:.2f}"

    # draw one number (hundreds, tens, ones) starting at x_offset
    def draw_number_group(ax, value: int, x0: float, pal) -> dict:
        hundreds = (value // 100) % 10
        tens = (value // 10) % 10
        ones = value % 10

        # layout constants (in "cube" units)
        gap_between_types = 1  # horizontal gap between 100s | 10s | 1s
        rod_y_position = 12  # rods behind flats
        ones_y_position = 14  # ones furthest back

        # extra spacing constants
        hundred_z_gap = 1  # space between hundred-flats
        ten_x_gap = 1  # space between rods
        one_z_gap = 1  # space between unit cubes

        # ---- HUNDREDS: stack flats with vertical gap
        if hundreds > 0:
            for h in range(hundreds):
                z = h * (1 + hundred_z_gap)  # 1 is cube height
                for i in range(10):
                    for j in range(10):
                        draw_cube(ax, x0 + i, j, z, pal)
        width_hundreds = 10 if hundreds > 0 else 0

        # ---- TENS: side-by-side with horizontal gap
        x_tens = x0 + width_hundreds + (gap_between_types if tens > 0 else 0)
        if tens > 0:
            for t in range(tens):
                rod_x = x_tens + t * (1 + ten_x_gap)  # 1 is cube width
                for k in range(10):
                    draw_cube(ax, rod_x, rod_y_position, k, pal)
        width_tens = tens * (1 + ten_x_gap) - ten_x_gap if tens > 0 else 0

        # ---- ONES: vertical column with vertical gap
        x_ones = x_tens + width_tens + (gap_between_types if ones > 0 else 0)
        if ones > 0:
            for k in range(ones):
                z = k * (1 + one_z_gap)
                draw_cube(ax, x_ones, ones_y_position, z, pal)
        width_ones = 1 if ones > 0 else 0

        total_width = (
            width_hundreds
            + (gap_between_types if width_hundreds and tens else 0)
            + width_tens
            + (gap_between_types if (width_hundreds or tens) and ones else 0)
            + width_ones
        )

        # bounding box extents for this group
        x_min = x0
        x_max = x0 + max(1, total_width)
        y_min = -1
        y_max = max(10, rod_y_position + 1, ones_y_position + 1)
        z_min = 0
        z_max = max(hundreds, 10 if tens else 0, ones, 1)

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "center_x": (x_min + x_max) / 2.0,
        }

    # ---------- figure & axis ----------
    fig = plt.figure(figsize=(12, 8))  # Increased from (8, 5) to make image bigger
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()

    # ---------- render groups ----------
    if num_blocks not in (1, 2):
        raise ValueError("BaseTenBlockStimulus must contain either 1 or 2 blocks.")

    group_spacing = 12  # horizontal space between numbers
    x_cursor = 0.0
    group_boxes = []

    for idx, block in enumerate(stimulus.blocks):
        pal = palettes[idx % len(palettes)]
        box = draw_number_group(ax, block.value, x_cursor, pal)
        group_boxes.append((box, block))
        x_cursor = box["x_max"] + group_spacing

    # ---------- labels / operation ----------
    if stimulus.show_values:
        label_y = -2.0
        label_z = -0.8
        for box, block in group_boxes:
            ax.text(
                box["center_x"],
                label_y,
                label_z,
                format_value(block),
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )

    # ---------- tidy limits & save ----------
    x_min = min(b["x_min"] for b, _ in group_boxes) - 1  # Reduced padding from -2 to -1
    x_max = max(b["x_max"] for b, _ in group_boxes) + 1  # Reduced padding from +2 to +1
    y_min = min(b["y_min"] for b, _ in group_boxes) - 1  # Reduced padding from -2 to -1
    y_max = max(b["y_max"] for b, _ in group_boxes) + 1
    z_min = -0.5  # Reduced from -1.0 to -0.5
    z_max = max(b["z_max"] for b, _ in group_boxes) + 0.5  # Reduced from +1 to +0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, (z_max - z_min) * 0.8))

    file_name = f"{settings.additional_content_settings.image_destination_folder}/base_ten_blocks_3d_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


@stimulus_function
def draw_base_ten_blocks_grid(stimulus: BaseTenBlockGridStimulus):
    """
    Draw a grid of identical base ten blocks arranged in 2 columns.
    Maximum of 6 blocks (3 rows x 2 columns).
    No labels are shown - just the visual blocks.
    """
    fig, ax = plt.subplots(figsize=(12, 10))  # Larger figure for grid layout
    ax.set_aspect("equal")
    ax.axis("off")

    # Calculate grid layout
    num_blocks = stimulus.count
    columns = 2
    rows = (num_blocks + columns - 1) // columns  # Ceiling division

    # Calculate block dimensions for spacing
    def calculate_block_dimensions(value):
        hundreds = value // 100
        tens = (value % 100) // 10
        ones = value % 10

        block_size = 0.15
        internal_spacing = 0.03
        between_type_spacing = 0.2

        ones_height = block_size if ones > 0 else 0
        tens_height = tens * (block_size + internal_spacing) if tens > 0 else 0
        hundreds_height = (
            hundreds * (10 * block_size + internal_spacing) if hundreds > 0 else 0
        )

        non_zero_types = sum([ones > 0, tens > 0, hundreds > 0])
        total_spacing = between_type_spacing * max(0, non_zero_types - 1)
        total_height = ones_height + tens_height + hundreds_height + total_spacing

        # Width is determined by the widest block type
        max_width = 0
        if ones > 0:
            max_width = max(max_width, ones * (block_size + internal_spacing))
        if tens > 0 or hundreds > 0:
            max_width = max(max_width, 10 * block_size)

        return max_width, total_height

    block_width, block_height = calculate_block_dimensions(stimulus.block_value)

    # Grid spacing
    horizontal_spacing = 3.0  # Space between columns
    vertical_spacing = 1.0  # Space between rows

    # Calculate total dimensions
    total_width = columns * block_width + (columns - 1) * horizontal_spacing
    total_height = rows * block_height + (rows - 1) * vertical_spacing

    # Draw blocks in grid
    for i in range(num_blocks):
        row = i // columns
        col = i % columns

        x_offset = col * (block_width + horizontal_spacing)
        y_offset = (rows - 1 - row) * (
            block_height + vertical_spacing
        )  # Start from top

        # Draw the block
        actual_width, actual_height = _draw_single_block(
            ax, stimulus.block_value, x_offset, y_offset
        )

    # Set plot limits with padding
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-0.5, total_height + 0.5)

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/base_ten_blocks_grid_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_base_ten_blocks_division(stimulus: BaseTenBlockDivisionStimulus):
    """
    Draw base ten blocks arranged to show division layout.
    The dividend is represented by base ten blocks distributed across groups
    according to the divisor, showing the division process visually.

    For example, 165 ÷ 11 would show blocks representing 165 arranged in 11 groups.
    """
    fig, ax = plt.subplots(
        figsize=(20, 14)
    )  # Larger figure for division layout with more height
    ax.set_aspect("equal")
    ax.axis("off")

    dividend = stimulus.dividend
    divisor = stimulus.divisor

    # Break down the dividend into base ten components
    hundreds = dividend // 100
    tens = (dividend % 100) // 10
    ones = dividend % 10

    # Create a list of all individual blocks to distribute
    blocks = []

    # Add hundreds blocks (each worth 100)
    for _ in range(hundreds):
        blocks.extend([100] * 100)  # Each hundred block = 100 unit blocks

    # Add tens blocks (each worth 10)
    for _ in range(tens):
        blocks.extend([10] * 10)  # Each ten block = 10 unit blocks

    # Add ones blocks (each worth 1)
    blocks.extend([1] * ones)

    # Now we have a list representing all unit blocks
    # Distribute them into groups
    groups = [[] for _ in range(divisor)]

    # Distribute blocks round-robin style to show equal distribution
    for i, block in enumerate(blocks):
        group_index = i % divisor
        groups[group_index].append(block)

    # Calculate layout parameters - adjust size based on presence of hundreds blocks
    hundreds = dividend // 100
    if hundreds > 0:
        block_size = (
            0.4  # Smaller blocks when hundreds are present to keep image manageable
        )
        internal_spacing = 0.15  # Increased spacing between blocks of same scale
        between_scale_spacing = 0.5  # Reduced spacing between different scales
    else:
        block_size = (
            1.2  # Much bigger blocks when no hundreds present for better visibility
        )
        internal_spacing = 0.25  # Increased spacing between blocks of same scale
        between_scale_spacing = 1.5  # Increased spacing between different scales
    group_spacing = 3.0  # Space between groups

    # Adjust line width based on block size
    line_width = 1.0 if hundreds > 0 else 2.0

    # Generate a single random color for all blocks (not too dark or too light)
    import random

    # Generate HSL color with controlled lightness (40-70% for good visibility)
    hue = random.randint(0, 360)
    saturation = random.randint(60, 90)  # Good saturation for vibrant color
    lightness = random.randint(40, 70)  # Not too dark (>40%) or too light (<70%)

    def hsl_to_hex(h, s, l):
        """Convert HSL to hex color"""
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0

        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        if s == 0:
            r = g = b = l  # achromatic
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1 / 3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1 / 3)

        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    block_color = hsl_to_hex(hue, saturation, lightness)

    # Always use 2-row layout with 2x2, 3x2, or 4x2 arrangements
    rows_of_groups = 2  # Always 2 rows
    max_groups_per_row = (
        divisor + 1
    ) // 2  # Ceiling division to distribute across 2 rows
    # Limit to maximum 4 columns (4x2 = 8 groups max)
    max_groups_per_row = min(max_groups_per_row, 4)

    # Draw each group
    for group_idx, group in enumerate(groups):
        if not group:  # Skip empty groups
            continue

        # Calculate group position
        row = group_idx // max_groups_per_row
        col = group_idx % max_groups_per_row

        # Calculate maximum width needed for a group (hundreds + tens + ones + spacing)
        max_group_width = 0
        if hundreds > 0:
            max_group_width += (
                10 * block_size + between_scale_spacing
            )  # hundreds width + spacing

        # Calculate max tens width (vertical strips next to each other with increased spacing)
        max_tens_in_group = (
            max(len([b for b in group if b == 10]) for group in groups)
            if any(groups)
            else 0
        )
        tens_horizontal_spacing = (
            internal_spacing * 2
        )  # Same increased spacing as in drawing
        max_tens_width = (
            max_tens_in_group * (block_size + tens_horizontal_spacing)
            if max_tens_in_group > 0
            else block_size
        )
        max_group_width += (
            max_tens_width + between_scale_spacing
        )  # tens width + spacing
        # Calculate ones width accounting for multiple stacks (max 5 per stack)
        max_ones_in_group = (
            max(len([b for b in group if b == 1]) for group in groups)
            if any(groups)
            else 0
        )
        ones_stacks_needed = (
            max_ones_in_group + 4
        ) // 5  # Ceiling division for number of stacks
        ones_vertical_spacing = internal_spacing * 2
        ones_width = (
            ones_stacks_needed * (block_size + ones_vertical_spacing)
            if ones_stacks_needed > 0
            else block_size
        )
        max_group_width += ones_width

        group_x = col * (
            max_group_width + group_spacing
        )  # Base x position for this group
        group_y = (rows_of_groups - 1 - row) * (
            12 * block_size + group_spacing
        )  # Base y position (height for hundreds)

        # Count different block types in this group
        hundreds_in_group = group.count(100)
        tens_in_group = group.count(10)
        ones_in_group = group.count(1)

        # Convert back to visual representation
        visual_hundreds = hundreds_in_group // 100
        remaining_after_hundreds = hundreds_in_group % 100
        visual_tens = (tens_in_group + remaining_after_hundreds) // 10
        visual_ones = (tens_in_group + remaining_after_hundreds) % 10 + ones_in_group

        # Draw the visual blocks for this group arranged horizontally
        current_x = group_x

        # Draw hundreds (cyan squares) on the left
        if visual_hundreds > 0:
            for h in range(visual_hundreds):
                # Draw each hundred block as a 10x10 grid
                for j in range(10):
                    for k in range(10):
                        rect = Rectangle(
                            (
                                current_x + j * block_size,
                                group_y
                                + k * block_size
                                + h * (10 * block_size + internal_spacing),
                            ),
                            block_size,
                            block_size,
                            facecolor=block_color,
                            edgecolor="black",
                            linewidth=line_width,
                        )
                        ax.add_patch(rect)
            # Move to the right for tens
            current_x += 10 * block_size + between_scale_spacing

        # Draw tens (lime rectangles) in the middle - vertical strips next to each other
        if visual_tens > 0:
            # Increased spacing between tens blocks
            tens_horizontal_spacing = (
                internal_spacing * 2
            )  # Double the spacing between tens blocks
            for t in range(visual_tens):
                # Draw each ten as a vertical strip (1 block wide, 10 blocks tall)
                ten_x = current_x + t * (block_size + tens_horizontal_spacing)
                for j in range(10):
                    rect = Rectangle(
                        (
                            ten_x,
                            group_y + j * block_size,
                        ),
                        block_size,
                        block_size,
                        facecolor=block_color,
                        edgecolor="black",
                        linewidth=line_width,
                    )
                    ax.add_patch(rect)
            # Move to the right for ones
            current_x += (
                visual_tens * (block_size + tens_horizontal_spacing)
                + between_scale_spacing
            )

        # Draw ones (orange single blocks) on the right - stacked vertically with max 5 per stack
        if visual_ones > 0:
            # Increased vertical spacing between ones blocks (same as tens horizontal spacing)
            ones_vertical_spacing = (
                internal_spacing * 2
            )  # Same increased spacing as tens blocks
            max_stack_height = 5  # Maximum 5 blocks per vertical stack

            for i in range(visual_ones):
                stack_number = i // max_stack_height  # Which stack (0, 1, 2, ...)
                position_in_stack = (
                    i % max_stack_height
                )  # Position within the stack (0-4)

                ones_x = current_x + stack_number * (
                    block_size + ones_vertical_spacing
                )  # Horizontal offset for each stack
                ones_y = group_y + position_in_stack * (
                    block_size + ones_vertical_spacing
                )  # Vertical position within stack

                rect = Rectangle(
                    (ones_x, ones_y),
                    block_size,
                    block_size,
                    facecolor=block_color,
                    edgecolor="black",
                    linewidth=line_width,
                )
                ax.add_patch(rect)

        # Removed group labels - no numbers should be shown

    # Calculate plot limits based on horizontal layout
    # Calculate maximum width needed for a group (hundreds + tens + ones + spacing)
    max_group_width = 0
    if hundreds > 0:
        max_group_width += (
            10 * block_size + between_scale_spacing
        )  # hundreds width + spacing

    # Calculate max tens width across all groups (vertical strips next to each other with increased spacing)
    max_tens_in_any_group = (
        max(len([b for b in group if b == 10]) for group in groups)
        if any(groups)
        else 0
    )
    tens_horizontal_spacing = (
        internal_spacing * 2
    )  # Same increased spacing as in drawing
    max_tens_width = (
        max_tens_in_any_group * (block_size + tens_horizontal_spacing)
        if max_tens_in_any_group > 0
        else block_size
    )
    max_group_width += max_tens_width + between_scale_spacing  # tens width + spacing

    # Calculate ones width accounting for multiple stacks (max 5 per stack)
    max_ones_in_any_group = (
        max(len([b for b in group if b == 1]) for group in groups) if any(groups) else 0
    )
    ones_stacks_needed = (
        max_ones_in_any_group + 4
    ) // 5  # Ceiling division for number of stacks
    ones_vertical_spacing = internal_spacing * 2
    ones_width = (
        ones_stacks_needed * (block_size + ones_vertical_spacing)
        if ones_stacks_needed > 0
        else block_size
    )
    max_group_width += ones_width

    total_width = max_groups_per_row * (max_group_width + group_spacing)

    # Calculate height accounting for increased ones spacing
    ones_vertical_spacing = internal_spacing * 2  # Same as in drawing
    max_ones_in_any_group = (
        max(len([b for b in group if b == 1]) for group in groups) if any(groups) else 0
    )
    max_ones_height = (
        max_ones_in_any_group * (block_size + ones_vertical_spacing)
        if max_ones_in_any_group > 0
        else 0
    )

    # Use the maximum of hundreds height (12 * block_size) and ones height
    max_group_height = max(
        12 * block_size, max_ones_height, 10 * block_size
    )  # 10 * block_size for tens height
    total_height = rows_of_groups * (max_group_height + group_spacing)

    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-1.0, total_height - 2.0)  # Reduced top white space

    # Removed equation labels - no numbers should be shown

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/base_ten_blocks_division_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_base_ten_blocks_division_grid(stimulus: BaseTenBlockDivisionStimulus):
    """
    Base-ten division grid for d in [11..19], N in [100..199], N % d == 0.
    Rows = divisor (10 + r). Columns = quotient q = N // d. No remainder.

    Uses canonical tiling pattern with proper base-ten blocks:
    - One 10×10 hundred block at columns 0-9, rows 0-9
    - r horizontal tens spanning columns 0-9 at rows 10 to 10+r-1
    - For extra columns 10 to q-1: vertical tens at rows 0-9, ones at rows 10 to 10+r-1

    This represents exactly N = q(10+r) = qd units with perfect structure.

    If incorrect_tiling=True, generates a different but valid division for multiple choice questions.
    """
    original_N = stimulus.dividend
    original_d = stimulus.divisor

    # Generate incorrect tiling if requested
    if getattr(stimulus, "incorrect_tiling", False):
        import random

        # Generate alternative dividend and divisor within constraints
        attempts = 0
        max_attempts = 50

        while attempts < max_attempts:
            # Try different dividend (±20) and divisor (±2)
            dividend_offset = random.choice(
                [
                    -20,
                    -19,
                    -18,
                    -17,
                    -16,
                    -15,
                    -14,
                    -13,
                    -12,
                    -11,
                    -10,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                ]
            )
            divisor_offset = random.choice([-2, -1, 1, 2])

            alt_N = original_N + dividend_offset
            alt_d = original_d + divisor_offset

            # Check if alternative meets all constraints
            if (
                11 <= alt_d <= 19
                and 100 <= alt_N <= 199
                and alt_N % alt_d == 0
                and alt_N // alt_d >= 10
                and (alt_N != original_N or alt_d != original_d)
            ):  # Must be different
                N = alt_N
                d = alt_d
                break
            attempts += 1
        else:
            # Fallback if no valid alternative found
            N = original_N
            d = original_d
    else:
        N = original_N
        d = original_d

    assert 11 <= d <= 19, "Divisor must be between 11 and 19."
    assert 100 <= N <= 199, "Dividend must be between 100 and 199."
    assert N % d == 0, "This visualization requires N divisible by d (no remainder)."

    q = N // d  # columns (quotient)
    r = d - 10  # extra rows (1..9)

    # Use same layout parameters as draw_base_ten_blocks_division
    hundreds = N // 100
    if hundreds > 0:
        block_size = 0.4  # Smaller blocks when hundreds are present
        internal_spacing = 0.15
        between_scale_spacing = 0.5
    else:
        block_size = 1.2  # Much bigger blocks when no hundreds present
        internal_spacing = 0.25
        between_scale_spacing = 1.5

    line_width = 1.0 if hundreds > 0 else 2.0

    # Generate a single random color for all blocks (matching draw_base_ten_blocks_division)
    import random

    hue = random.randint(0, 360)
    saturation = random.randint(60, 90)
    lightness = random.randint(40, 70)

    def hsl_to_hex(h, s, l):
        """Convert HSL to hex color"""
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0

        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        if s == 0:
            r = g = b = l  # achromatic
        else:
            q_color = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q_color
            r = hue_to_rgb(p, q_color, h + 1 / 3)
            g = hue_to_rgb(p, q_color, h)
            b = hue_to_rgb(p, q_color, h - 1 / 3)

        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    block_color = hsl_to_hex(hue, saturation, lightness)

    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_aspect("equal")
    ax.axis("off")

    # Calculate positions for the canonical layout
    # Position everything with hundred block and vertical tens at the top
    hundred_x = 0
    top_y = (
        r * (block_size + internal_spacing) + between_scale_spacing
    )  # Start from top, accounting for ones below

    # Draw the hundred block as a 10×10 grid (matching draw_base_ten_blocks_division style)
    for j in range(10):
        for k in range(10):
            rect = Rectangle(
                (hundred_x + j * block_size, top_y + k * block_size),
                block_size,
                block_size,
                facecolor=block_color,
                edgecolor="black",
                linewidth=line_width,
            )
            ax.add_patch(rect)

    # Draw r horizontal tens below the hundred block
    tens_y = 0  # Put horizontal tens at the bottom
    for k in range(r):
        for j in range(10):
            rect = Rectangle(
                (
                    hundred_x + j * block_size,
                    tens_y + k * (block_size + internal_spacing),
                ),
                block_size,
                block_size,
                facecolor=block_color,
                edgecolor="black",
                linewidth=line_width,
            )
            ax.add_patch(rect)

    # Draw extra columns (q-10) to the right
    extra_cols_x = hundred_x + 10 * block_size + between_scale_spacing
    for col_idx in range(q - 10):
        col_x = extra_cols_x + col_idx * (block_size + between_scale_spacing)

        # Draw vertical ten (10 blocks tall) for this column - aligned with hundred block at top
        for j in range(10):
            rect = Rectangle(
                (col_x, top_y + j * block_size),
                block_size,
                block_size,
                facecolor=block_color,
                edgecolor="black",
                linewidth=line_width,
            )
            ax.add_patch(rect)

        # Draw r ones below the vertical ten - aligned with horizontal tens at bottom
        for k in range(r):
            rect = Rectangle(
                (col_x, tens_y + k * (block_size + internal_spacing)),
                block_size,
                block_size,
                facecolor=block_color,
                edgecolor="black",
                linewidth=line_width,
            )
            ax.add_patch(rect)

    # Calculate bounds - account for the new positioning
    # Width: hundred block (10 * block_size) + spacing + extra columns
    if q > 10:
        total_width = (
            10 * block_size
            + between_scale_spacing
            + (q - 10) * (block_size + between_scale_spacing)
        )
    else:
        total_width = 10 * block_size

    # Height: from bottom (horizontal tens + ones) to top (hundred block + vertical tens)
    # Bottom part: r horizontal tens with spacing
    bottom_height = (
        r * (block_size + internal_spacing) - internal_spacing if r > 0 else 0
    )
    # Gap between bottom and top
    gap_height = between_scale_spacing
    # Top part: 10 blocks for hundred/vertical tens
    top_height = 10 * block_size

    total_height = bottom_height + gap_height + top_height

    # Set limits with padding to ensure everything is visible
    padding = 1.0  # Increased padding to ensure visibility
    ax.set_xlim(-padding, total_width + padding)
    ax.set_ylim(-padding, total_height + padding)

    plt.tight_layout()
    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"base_ten_blocks_division_grid_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


# Example usage
if __name__ == "__main__":
    # Test with two blocks
    double_block = BaseTenBlockStimulus(
        blocks=[BaseTenBlock(value=556), BaseTenBlock(value=400)], operation="addition"
    ).model_dump()
    file_path = draw_base_ten_blocks(double_block)
    matplotlib.use("TkAgg")
    img = plt.imread(file_path)
    # Display the image (optional, for immediate viewing)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # Test with grid of blocks (e.g., for division: 440÷4 = 110 each)
    grid_stimulus = BaseTenBlockGridStimulus(
        block_value=110, count=4, display_as_decimal=False
    ).model_dump()
    grid_file_path = draw_base_ten_blocks_grid(grid_stimulus)
    grid_img = plt.imread(grid_file_path)
    plt.figure()
    plt.imshow(grid_img)
    plt.axis("off")
    plt.show()

    # Test with canonical division grid (e.g., 132 ÷ 12 = 11, showing structured base-ten blocks)
    division_grid_stimulus = BaseTenBlockDivisionStimulus(
        dividend=132, divisor=12
    ).model_dump()
    division_grid_file_path = draw_base_ten_blocks_division_grid(division_grid_stimulus)
    division_grid_img = plt.imread(division_grid_file_path)
    plt.figure()
    plt.imshow(division_grid_img)
    plt.axis("off")
    plt.show()
