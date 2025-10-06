import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.prism_net import (
    CubePrismNet,
    CustomTriangularPrismNet,
    DualNetsShapeType,
    DualPrismNets,
    EPrismType,
    Position,
    PrismNet,
    PyramidPrismNet,
    RectangularPrismNet,
    RectangularPyramidPrismNet,
    RegularRectangularPrismNet,
    SquarePyramidPrismNet,
    TriangularPrismNet,
)
from content_generators.settings import settings
from matplotlib.patches import Polygon, Rectangle

FONT_SIZE_FACTOR = 0.7
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 20
# Fill color for all shapes
FILL_COLOR = "#6699cc"  # Medium blue shade


@dataclass
class Label:
    """Represents a text label with position and dimensions."""

    x: float
    y: float
    text: str
    fontsize: float
    ha: str = "center"
    va: str = "center"
    width: float = 0
    height: float = 0
    alpha: float = 0.1  # Default opacity for standard labels
    positioning_params: Optional[dict] = (
        None  # Store positioning parameters for recalculation
    )


def measure_text_dimensions(text: str, fontsize: float, ax) -> Tuple[float, float]:
    """Measure the width and height of text in data coordinates with robust fallbacks."""
    try:
        # Create a temporary text object to measure dimensions
        temp_text = ax.text(0, 0, text, fontsize=fontsize, alpha=0)

        # Force a draw to ensure the figure is rendered
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        bbox = temp_text.get_window_extent(renderer)

        try:
            # Convert from display coordinates to data coordinates
            bbox_data = ax.transData.inverted().transform_bbox(bbox)
            width = bbox_data.width
            height = bbox_data.height

            # Sanity check: if dimensions are impossibly small, coordinate system isn't established
            if width < 0.001 or height < 0.001 or width > 1000 or height > 1000:
                raise ValueError(
                    "Invalid coordinate transformation - coordinate system not established"
                )

            # Remove the temporary text and return valid dimensions
            temp_text.remove()
            return width, height

        except (ValueError, AttributeError):
            # Coordinate system not properly established - use display coordinates with conversion
            display_width = bbox.width
            display_height = bbox.height

            # Get figure dimensions for conversion
            fig_width, fig_height = ax.figure.get_size_inches()
            dpi = ax.figure.dpi

            # Get axes limits to calculate conversion factor
            try:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                axes_data_width = xlim[1] - xlim[0]
                axes_data_height = ylim[1] - ylim[0]

                # Convert display pixels to data coordinates using actual axes dimensions
                # Display size in pixels
                display_width_pixels = fig_width * dpi
                display_height_pixels = fig_height * dpi

                # Calculate conversion factors
                pixel_to_data_x = axes_data_width / display_width_pixels
                pixel_to_data_y = axes_data_height / display_height_pixels

                width = display_width * pixel_to_data_x
                height = display_height * pixel_to_data_y

            except (AttributeError, ValueError):
                # If axes limits aren't set, use improved estimation
                # Convert display pixels to approximate data coordinates
                pixel_to_data_scale = 0.02  # Increased from 0.01 for better accuracy
                width = display_width * pixel_to_data_scale
                height = display_height * pixel_to_data_scale

            # Remove the temporary text
            temp_text.remove()
            return width, height

    except Exception as e:
        print(f"Error measuring text dimensions: {e}")
        # Final fallback: use improved estimation based on fontsize
        # These factors are more realistic for typical matplotlib text
        estimated_width = (
            fontsize * 0.6 * len(text) / 72
        )  # 72 points per inch, more realistic scaling
        estimated_height = fontsize * 1.2 / 72  # More realistic height estimation
        return estimated_width, estimated_height


def get_label_bounds(label: Label) -> Tuple[float, float, float, float]:
    """Get the bounding box coordinates (left, right, bottom, top) for a label."""
    # Adjust position based on alignment
    if label.ha == "center":
        left = label.x - label.width / 2
        right = label.x + label.width / 2
    elif label.ha == "left":
        left = label.x
        right = label.x + label.width
    else:  # right
        left = label.x - label.width
        right = label.x

    if label.va == "center":
        bottom = label.y - label.height / 2
        top = label.y + label.height / 2
    elif label.va == "bottom":
        bottom = label.y
        top = label.y + label.height
    else:  # top
        bottom = label.y - label.height
        top = label.y

    return left, right, bottom, top


def labels_overlap(label1: Label, label2: Label, buffer: float = 0.1) -> bool:
    """Check if two labels overlap."""
    left1, right1, bottom1, top1 = get_label_bounds(label1)
    left2, right2, bottom2, top2 = get_label_bounds(label2)

    # Check for overlap with buffer
    return not (
        right1 + buffer < left2
        or right2 + buffer < left1
        or top1 + buffer < bottom2
        or top2 + buffer < bottom1
    )


def label_overlaps_with_shape_edge(
    label: Label, shape_edges: List[Tuple[float, float, float, float]]
) -> bool:
    """Check if a label overlaps with any shape edge (line segment)."""
    left, right, bottom, top = get_label_bounds(label)
    buffer = 0.05

    for x1, y1, x2, y2 in shape_edges:
        # Check if label bounds intersect with line segment (with buffer)
        # Expand line to a thin rectangle
        line_left = min(x1, x2) - buffer
        line_right = max(x1, x2) + buffer
        line_bottom = min(y1, y2) - buffer
        line_top = max(y1, y2) + buffer

        # Check for overlap
        if not (
            right < line_left
            or left > line_right
            or top < line_bottom
            or bottom > line_top
        ):
            return True
    return False


def find_optimal_label_position(
    base_x: float,
    base_y: float,
    text: str,
    fontsize: float,
    ax,
    shape_edges: List[Tuple[float, float, float, float]],
    other_labels: List[Label],
    preferred_offsets: List[Tuple[float, float, str, str]],
) -> Tuple[float, float, str, str]:
    """Find optimal position for a label by trying different offsets."""

    for offset_x, offset_y, ha, va in preferred_offsets:
        test_label = Label(
            x=base_x + offset_x,
            y=base_y + offset_y,
            text=text,
            fontsize=fontsize,
            ha=ha,
            va=va,
        )
        test_label.width, test_label.height = measure_text_dimensions(
            text, fontsize, ax
        )

        # Check for overlaps with other labels
        overlaps_with_labels = any(
            labels_overlap(test_label, other_label) for other_label in other_labels
        )

        # Check for overlaps with shape edges
        overlaps_with_shapes = label_overlaps_with_shape_edge(test_label, shape_edges)

        if not overlaps_with_labels and not overlaps_with_shapes:
            return test_label.x, test_label.y, ha, va

    # If no position works, return the first preferred position
    return (
        base_x + preferred_offsets[0][0],
        base_y + preferred_offsets[0][1],
        preferred_offsets[0][2],
        preferred_offsets[0][3],
    )


def recalculate_label_position(label: Label, ax) -> None:
    """Recalculate label position based on updated text dimensions."""
    if not label.positioning_params:
        return

    params = label.positioning_params

    # Check if this is a height label (different parameter structure)
    if params.get("is_height_label", False):
        # For height labels, we don't recalculate position since they use
        # complex collision detection that would be expensive to re-run
        # Just update the text dimensions
        text_width, text_height = measure_text_dimensions(
            label.text, label.fontsize, ax
        )
        label.width = text_width
        label.height = text_height
        return

    # Handle regular edge labels
    rect_x = params["rect_x"]
    rect_y = params["rect_y"]
    rect_width = params["rect_width"]
    rect_height = params["rect_height"]
    edge = params["edge"]
    margin = params["margin"]
    is_inside = params.get("is_inside", False)

    # Measure current text dimensions
    text_width, text_height = measure_text_dimensions(label.text, label.fontsize, ax)
    label.width = text_width
    label.height = text_height

    if is_inside:
        # Inside positioning logic
        if edge == "top":
            label.x = rect_x + rect_width / 2
            label.y = rect_y + rect_height - margin - text_height / 2
            label.ha, label.va = "center", "center"
        elif edge == "bottom":
            label.x = rect_x + rect_width / 2
            label.y = rect_y + margin + text_height / 2
            label.ha, label.va = "center", "center"
        elif edge == "left":
            label.x = rect_x + margin
            label.y = rect_y + rect_height / 2
            label.ha, label.va = "left", "center"
        elif edge == "right":
            label.x = rect_x + rect_width - margin
            label.y = rect_y + rect_height / 2
            label.ha, label.va = "right", "center"
    else:
        # Outside positioning logic
        padding = 0.03

        if edge == "top":
            label.x = rect_x + rect_width / 2
            label.y = rect_y + rect_height + padding + text_height / 2 + margin
            label.ha, label.va = "center", "center"
        elif edge == "bottom":
            label.x = rect_x + rect_width / 2
            label.y = rect_y - padding - text_height / 2 - margin
            label.ha, label.va = "center", "center"
        elif edge == "left":
            label.x = rect_x - padding - margin
            label.y = rect_y + rect_height / 2
            label.ha, label.va = "right", "center"
        elif edge == "right":
            label.x = rect_x + rect_width + padding + margin
            label.y = rect_y + rect_height / 2
            label.ha, label.va = "left", "center"


def optimize_label_sizes(
    labels: List[Label],
    ax,
    initial_fontsize: float,
    reduction_factor: float = 0.9,
    overlap_buffer: float = 0.1,
) -> float:
    """
    Optimize font sizes to prevent overlaps.
    Returns the final font size used.
    """
    # Update all label dimensions with initial font size
    for label in labels:
        label.fontsize = initial_fontsize
        label.width, label.height = measure_text_dimensions(
            label.text, label.fontsize, ax
        )

    # Check for overlaps and reduce font size if needed
    current_fontsize = initial_fontsize
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations and current_fontsize > MIN_FONT_SIZE:
        # Check if any labels overlap
        has_overlap = False
        overlapping_pairs = []
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i + 1 :], i + 1):
                if labels_overlap(label1, label2, overlap_buffer):
                    has_overlap = True
                    overlapping_pairs.append((i, j))
            if has_overlap:
                break

        if not has_overlap:
            break

        # Reduce font size and update dimensions
        current_fontsize = max(current_fontsize * reduction_factor, MIN_FONT_SIZE)

        for label in labels:
            label.fontsize = current_fontsize
            # Recalculate position based on new font size if positioning parameters are available
            recalculate_label_position(label, ax)

        iteration += 1

    return current_fontsize


def get_rectangle_edges(
    x: float, y: float, width: float, height: float
) -> List[Tuple[float, float, float, float]]:
    """Get edges of a rectangle as line segments."""
    return [
        (x, y, x + width, y),  # Bottom edge
        (x + width, y, x + width, y + height),  # Right edge
        (x + width, y + height, x, y + height),  # Top edge
        (x, y + height, x, y),  # Left edge
    ]


def get_triangle_edges(
    x: float, y: float, base: float, height: float, orientation: str
) -> List[Tuple[float, float, float, float]]:
    """Get edges of a triangle as line segments."""
    if orientation == "up":
        return [
            (x, y, x + base, y),  # Base
            (x + base, y, x + base / 2, y + height),  # Right side
            (x + base / 2, y + height, x, y),  # Left side
        ]
    elif orientation == "down":
        return [
            (x, y, x + base, y),  # Base
            (x + base, y, x + base / 2, y - height),  # Right side
            (x + base / 2, y - height, x, y),  # Left side
        ]
    elif orientation == "right":
        return [
            (x, y, x, y + height),  # Left side (base)
            (x, y + height, x + base, y + height / 2),  # Top side
            (x + base, y + height / 2, x, y),  # Bottom side
        ]
    elif orientation == "left":
        return [
            (x, y, x, y + height),  # Right side (base)
            (x, y + height, x - base, y + height / 2),  # Top side
            (x - base, y + height / 2, x, y),  # Bottom side
        ]
    return []


def create_smart_height_label(
    base_x: float,
    base_y: float,
    height_value: float,
    unit_label: str,
    fontsize: float,
    ax,
    shape_edges: List[Tuple[float, float, float, float]],
    existing_labels: List[Label],
    orientation: str = "up",
    alpha: float = 0.7,  # Higher opacity for height labels
) -> Label:
    """Create a height label with smart positioning to avoid overlaps."""
    text = (
        f"{height_value:.1f} {unit_label}"
        if height_value != int(height_value)
        else f"{int(height_value)} {unit_label}"
    )

    # First measure text dimensions to calculate proper offset using robust measurement
    text_width, text_height = measure_text_dimensions(text, fontsize, ax)

    # Calculate clearance to prevent background box overlap with dashed lines
    # Background box padding (reduced for closer positioning)
    background_padding = 0.03
    # Small margin to prevent touching (reduced)
    margin = 0.02

    # Calculate clearance based on text dimensions and background box
    if orientation in ["up", "down"]:
        # Vertical dashed line - use minimal clearance for closer positioning
        base_clearance = (
            background_padding + margin
        )  # No text width factor for closer positioning
        secondary_clearance = text_height / 2 + background_padding + margin

        preferred_offsets = [
            (
                base_clearance,
                0.0,
                "left",
                "center",
            ),  # To right of dashed line (left-aligned)
            (
                -base_clearance,
                0.0,
                "right",
                "center",
            ),  # To left of dashed line (right-aligned)
            (0.0, secondary_clearance, "center", "center"),  # Above center
            (0.0, -secondary_clearance, "center", "center"),  # Below center
            (base_clearance * 2, 0.0, "left", "center"),  # Further right
            (-base_clearance * 2, 0.0, "right", "center"),  # Further left
        ]
    elif orientation in ["left", "right"]:
        # Horizontal dashed line - use minimal clearance for closer positioning
        base_clearance = (
            background_padding + margin
        )  # No text height factor for closer positioning
        secondary_clearance = text_width / 2 + background_padding + margin

        preferred_offsets = [
            (0.0, base_clearance, "center", "bottom"),  # Above dashed line
            (0.0, -base_clearance, "center", "top"),  # Below dashed line
            (secondary_clearance, 0.0, "center", "center"),  # To right
            (-secondary_clearance, 0.0, "center", "center"),  # To left
            (0.0, base_clearance * 2, "center", "bottom"),  # Further above
            (0.0, -base_clearance * 2, "center", "top"),  # Further below
        ]
    else:
        # Default fallback - use minimal clearance
        base_clearance = background_padding + margin
        preferred_offsets = [(base_clearance, 0.0, "left", "center")]

    # Use improved collision detection instead of bypassing it
    x, y, ha, va = find_optimal_label_position(
        base_x,
        base_y,
        text,
        fontsize,
        ax,
        shape_edges,
        existing_labels,
        preferred_offsets,
    )

    # Create final label with measured dimensions
    label = Label(x=x, y=y, text=text, fontsize=fontsize, ha=ha, va=va, alpha=alpha)
    label.width = text_width
    label.height = text_height

    # Store positioning parameters for potential recalculation after font optimization
    label.positioning_params = {
        "base_x": base_x,
        "base_y": base_y,
        "height_value": height_value,
        "unit_label": unit_label,
        "orientation": orientation,
        "shape_edges": shape_edges,
        "is_height_label": True,  # Mark this as a height label
    }

    return label


def calculate_tight_bounds(faces, triangles, labels):
    """Calculate tight bounds for all drawn elements including shapes and labels."""
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    # Check rectangular faces
    for (x, y), width, height, label in faces:
        min_x = min(min_x, x)
        max_x = max(max_x, x + width)
        min_y = min(min_y, y)
        max_y = max(max_y, y + height)

    # Check triangular faces
    for (x, y), base, height, label, orientation in triangles:
        if orientation == "up":
            min_x = min(min_x, x)
            max_x = max(max_x, x + base)
            min_y = min(min_y, y)
            max_y = max(max_y, y + height)
        elif orientation == "down":
            min_x = min(min_x, x)
            max_x = max(max_x, x + base)
            min_y = min(min_y, y - height)
            max_y = max(max_y, y)
        elif orientation == "right":
            min_x = min(min_x, x)
            max_x = max(max_x, x + base)
            min_y = min(min_y, y)
            max_y = max(max_y, y + height)
        elif orientation == "left":
            min_x = min(min_x, x - base)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y + height)

    # Check label bounds (including background boxes)
    for label in labels:
        left, right, bottom, top = get_label_bounds(label)
        padding = 0.1
        min_x = min(min_x, left - padding)
        max_x = max(max_x, right + padding)
        min_y = min(min_y, bottom - padding)
        max_y = max(max_y, top + padding)

    # Add small margin
    margin = 0.5
    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def draw_optimized_labels(
    labels: List[Label], ax, object_dimensions: Tuple[float, float, float] | None = None
):
    """Draw all labels with optimized sizing and background boxes."""
    if not labels:
        return

    # Calculate initial font size based on figure dimensions and number of labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    figure_area = (xlim[1] - xlim[0]) * (ylim[1] - ylim[0])

    # Base font size on figure area and number of labels
    base_fontsize = min(
        max(
            (figure_area / max(len(labels), 1)) ** 0.5 * FONT_SIZE_FACTOR, MIN_FONT_SIZE
        ),
        MAX_FONT_SIZE,
    )

    # If object dimensions are provided, adjust font size based on object size
    if object_dimensions:
        height, width, length = object_dimensions
        min_dimension = min(height, width, length)

        # Cap font size based on smallest object dimension
        # Use a more generous factor that ensures text fits reasonably within the object scale
        # Balance readability with proportional sizing for educational content
        if min_dimension <= 2:
            # For very small objects (like 2x2x2 cubes), ensure minimum readable size
            # Increased from 12 to 16 for better readability in educational content
            max_object_fontsize = max(
                MIN_FONT_SIZE + 8, min_dimension * 8.0
            )  # At least 16
        elif min_dimension <= 5:
            # For small-medium objects, use a generous multiplier
            max_object_fontsize = max(min_dimension * 3.8, MIN_FONT_SIZE)
        elif min_dimension <= 12:
            # For medium-large objects, use moderate multiplier with small minimum boost
            max_object_fontsize = max(min_dimension * 2.8, MIN_FONT_SIZE + 3)
        else:
            # For large objects, use conservative multiplier but maintain readability
            max_object_fontsize = max(min_dimension * 2.4, MIN_FONT_SIZE + 4)

        # Keep the 12x12x12 cube in the readability-prioritized category
        if min_dimension <= 3:
            # Very small objects: prioritize readability
            base_fontsize = max(base_fontsize, max_object_fontsize)
        elif min_dimension <= 12:
            # Small-medium-large objects: use the larger value for better readability
            base_fontsize = max(base_fontsize, max_object_fontsize)
        else:
            # Very large objects: apply size constraints to prevent oversized labels
            base_fontsize = min(base_fontsize, max_object_fontsize)

        # keep global cap
        base_fontsize = min(base_fontsize, MAX_FONT_SIZE)
        # only shrink relative to size for medium/large objects
        if min_dimension > 6:
            base_fontsize = min(base_fontsize, min_dimension * 2.0)

    # Optimize font sizes to prevent overlaps
    # Balance readability with proper spacing for educational content
    if object_dimensions and min(object_dimensions) <= 2:
        # For very small objects (like 2x2x2 cubes), prioritize readability
        reduction_factor = 0.95  # Very gentle reduction
        overlap_buffer = 0.03  # Tighter tolerance for overlaps
    elif object_dimensions and min(object_dimensions) <= 5:
        reduction_factor = 0.92  # Gentle reduction
        overlap_buffer = 0.05
    elif object_dimensions and min(object_dimensions) <= 10:
        # For medium objects, balanced approach
        reduction_factor = 0.91
        overlap_buffer = 0.07
    else:
        # For larger objects, more standard overlap reduction
        reduction_factor = 0.9
        overlap_buffer = 0.1
    optimize_label_sizes(labels, ax, base_fontsize, reduction_factor, overlap_buffer)

    # Draw background boxes first, then text labels
    padding = 0.03  # Reduced padding for closer positioning to match label positioning

    # Draw all background boxes
    for label in labels:
        left, right, bottom, top = get_label_bounds(label)

        # Add padding to the background box
        box_left = left - padding
        box_right = right + padding
        box_bottom = bottom - padding
        box_top = top + padding

        # Create background rectangle
        from matplotlib.patches import Rectangle

        background_box = Rectangle(
            (box_left, box_bottom),
            box_right - box_left,
            box_top - box_bottom,
            facecolor="white",
            alpha=label.alpha,  # Use label's custom opacity
            edgecolor="none",  # Remove border
            zorder=10,  # High z-order to be above all shape elements
        )
        ax.add_patch(background_box)

    # Draw all text labels on top of background boxes
    for label in labels:
        ax.text(
            label.x,
            label.y,
            label.text,
            ha=label.ha,
            va=label.va,
            fontsize=label.fontsize,
            zorder=11,  # Highest z-order to be on top of everything
        )


def draw_triangle(ax, x, y, base, height, label, orientation="up"):
    vertices = []
    if orientation == "up":
        vertices = [(x, y), (x + base, y), (x + base / 2, y + height)]
    elif orientation == "down":
        vertices = [(x, y), (x + base, y), (x + base / 2, y - height)]
    elif orientation == "left":
        vertices = [(x, y), (x, y + height), (x - base, y + height / 2)]
    elif orientation == "right":
        vertices = [(x, y), (x, y + height), (x + base, y + height / 2)]
    ax.add_patch(Polygon(vertices, fill=False))
    if settings.debug:
        ax.text(
            x
            + (
                base / 2
                if orientation in ["up", "down"]
                else (base if orientation == "right" else -base / 2)
            ),
            y
            + (
                height / 2
                if orientation == "up"
                else (-height / 2 if orientation == "down" else height / 2)
            ),
            label,
            ha="center",
            va="center",
        )
    ax.add_patch(Polygon(vertices, fill=True, facecolor=FILL_COLOR, edgecolor="black"))


def draw_right_angle_indicator(
    ax, corner_x, corner_y, direction_x, direction_y, size=0.3
):
    """Draw a small right angle indicator."""
    # Create right angle symbol
    ax.plot(
        [corner_x, corner_x + direction_x * size],
        [corner_y, corner_y],
        linestyle="-",
        color="black",
        linewidth=1,
    )
    ax.plot(
        [corner_x, corner_x],
        [corner_y, corner_y + direction_y * size],
        linestyle="-",
        color="black",
        linewidth=1,
    )
    ax.plot(
        [corner_x + direction_x * size, corner_x + direction_x * size],
        [corner_y, corner_y + direction_y * size],
        linestyle="-",
        color="black",
        linewidth=1,
    )
    ax.plot(
        [corner_x, corner_x + direction_x * size],
        [corner_y + direction_y * size, corner_y + direction_y * size],
        linestyle="-",
        color="black",
        linewidth=1,
    )


def draw_pyramid_triangle_height_indicator(
    ax,
    base_corner_x,
    base_corner_y,
    triangle_x,
    triangle_y,
    base,
    height,
    orientation="up",
):
    """Draw simplified height indicator: dashed line from base to apex with right angle indicator."""
    # Calculate appropriate size for right angle marker based on shape dimensions
    marker_size = min(base, height) * 0.15  # Scale with smallest dimension
    marker_size = max(marker_size, 0.3)  # Ensure minimum visibility
    marker_size = min(marker_size, 1.0)  # Cap maximum size

    if orientation == "up":
        apex_x, apex_y = triangle_x + base / 2, triangle_y + height
        height_base_x = apex_x  # Height drops straight down from apex
        height_base_y = triangle_y
        # Dashed line from base to apex
        ax.plot(
            [height_base_x, apex_x],
            [height_base_y, apex_y],
            linestyle="--",
            color="#666666",  # Light gray for better differentiation from edges
            linewidth=2,
        )
        # Right angle indicator at base (horizontal base, vertical height)
        draw_right_angle_indicator(
            ax, height_base_x, height_base_y, marker_size, marker_size, size=marker_size
        )

    elif orientation == "down":
        apex_x, apex_y = triangle_x + base / 2, triangle_y - height
        height_base_x = apex_x  # Height goes straight up from apex
        height_base_y = triangle_y
        # Dashed line from base to apex
        ax.plot(
            [height_base_x, apex_x],
            [height_base_y, apex_y],
            linestyle="--",
            color="#666666",  # Light gray for better differentiation from edges
            linewidth=2,
        )
        # Right angle indicator at base (horizontal base, vertical height)
        draw_right_angle_indicator(
            ax,
            height_base_x,
            height_base_y,
            marker_size,
            -marker_size,
            size=marker_size,
        )

    elif orientation == "right":
        apex_x, apex_y = triangle_x + base, triangle_y + height / 2
        height_base_x = triangle_x
        height_base_y = apex_y  # Height goes horizontally from apex
        # Dashed line from base to apex
        ax.plot(
            [height_base_x, apex_x],
            [height_base_y, apex_y],
            linestyle="--",
            color="#666666",  # Light gray for better differentiation from edges
            linewidth=2,
        )
        # Right angle indicator at base (vertical base, horizontal height)
        draw_right_angle_indicator(
            ax, height_base_x, height_base_y, marker_size, marker_size, size=marker_size
        )

    elif orientation == "left":
        apex_x, apex_y = triangle_x - base, triangle_y + height / 2
        height_base_x = triangle_x
        height_base_y = apex_y  # Height goes horizontally from apex
        # Dashed line from base to apex
        ax.plot(
            [height_base_x, apex_x],
            [height_base_y, apex_y],
            linestyle="--",
            color="#666666",  # Light gray for better differentiation from edges
            linewidth=2,
        )
        # Right angle indicator at base (vertical base, horizontal height)
        draw_right_angle_indicator(
            ax,
            height_base_x,
            height_base_y,
            -marker_size,
            marker_size,
            size=marker_size,
        )


def draw_triangle_height_indicator(ax, x, y, base, height, orientation="up"):
    """Draw simplified height indicator: dashed line from base to apex with right angle indicator."""
    # Calculate appropriate size for right angle marker based on shape dimensions
    marker_size = min(base, height) * 0.15  # Scale with smallest dimension
    marker_size = max(marker_size, 0.3)  # Ensure minimum visibility
    marker_size = min(marker_size, 1.0)  # Cap maximum size

    if orientation == "up":
        apex_x, apex_y = x + base / 2, y + height
        height_base_x = apex_x  # Height drops straight down from apex
        height_base_y = y
        # Dashed line from base to apex
        ax.plot(
            [height_base_x, apex_x],
            [height_base_y, apex_y],
            linestyle="--",
            color="#666666",  # Light gray for better differentiation from edges
            linewidth=2,
        )
        # Right angle indicator at base (horizontal base, vertical height)
        draw_right_angle_indicator(
            ax, height_base_x, height_base_y, marker_size, marker_size, size=marker_size
        )

    elif orientation == "down":
        apex_x, apex_y = x + base / 2, y - height
        height_base_x = apex_x  # Height goes straight up from apex
        height_base_y = y
        # Dashed line from base to apex
        ax.plot(
            [height_base_x, apex_x],
            [height_base_y, apex_y],
            linestyle="--",
            color="#666666",  # Light gray for better differentiation from edges
            linewidth=2,
        )
        # Right angle indicator at base (horizontal base, vertical height)
        draw_right_angle_indicator(
            ax,
            height_base_x,
            height_base_y,
            marker_size,
            -marker_size,
            size=marker_size,
        )


def generate_comprehensive_pyramid_labels(faces, triangles, h, w, l, unit_label, ax):
    """Generate comprehensive labels for pyramid net edges, avoiding shared edge duplicates."""
    labels = []

    # Pyramid has 1 rectangular base and 4 triangular faces
    # Base: (0,0), w×l
    # Triangular faces share edges with the base, so base edges should be labeled strategically

    all_edge_labels = []

    # Base rectangle - label all outer edges
    base_x, base_y = faces[0][0][0], faces[0][0][1]
    base_width, base_height = faces[0][1], faces[0][2]

    # Label base edges using precise positioning
    all_edge_labels.extend(
        [
            (0, "top", base_x, base_y, base_width, base_height, f"{w} {unit_label}"),
            (0, "bottom", base_x, base_y, base_width, base_height, f"{w} {unit_label}"),
            (0, "left", base_x, base_y, base_width, base_height, f"{l} {unit_label}"),
            (0, "right", base_x, base_y, base_width, base_height, f"{l} {unit_label}"),
        ]
    )

    # For pyramids, the triangular faces don't share edges with each other,
    # only with the base. So no edge deduplication needed for triangle-triangle edges.
    # The triangle-base edge sharing is handled by positioning labels away from shared edges.

    # Generate final labels using precise positioning
    for (
        face_idx,
        edge_side,
        rect_x,
        rect_y,
        rect_width,
        rect_height,
        text,
    ) in all_edge_labels:
        # Check if this is the center face (Base face = index 0)
        if face_idx == 0:  # Center face - use inside edge positioning
            label = create_inside_edge_positioned_label(
                text, 16, rect_x, rect_y, rect_width, rect_height, edge_side, ax
            )
        else:  # Non-center face - use edge positioning
            label = create_edge_positioned_label(
                text, 16, rect_x, rect_y, rect_width, rect_height, edge_side, ax
            )
        labels.append(label)

    return labels


def generate_comprehensive_triangular_labels(faces, triangles, h, w, l, unit_label, ax):
    """Generate comprehensive labels for triangular prism edges, avoiding shared edge duplicates."""
    labels = []

    # Triangular prism has 3 rectangular faces and 2 triangular faces
    # Rectangular faces: Front(0,0), Right(w,0), Left(-side_w,0)
    # Triangular faces: Top(0,l), Bottom(0,0)

    all_edge_labels = []

    # Face 0: Front rectangular face (0,0), w×l
    front_x, front_y = faces[0][0][0], faces[0][0][1]
    front_width, front_height = faces[0][1], faces[0][2]
    all_edge_labels.extend(
        [
            (
                0,
                "top",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{w} {unit_label}",
            ),
            (
                0,
                "bottom",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{w} {unit_label}",
            ),
            (
                0,
                "left",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{l} {unit_label}",
            ),
            (
                0,
                "right",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{l} {unit_label}",
            ),
        ]
    )

    # Face 1: Right rectangular face (w,0), side_w×l
    right_x, right_y = faces[1][0][0], faces[1][0][1]
    right_width, right_height = faces[1][1], faces[1][2]  # side_w×l
    side_w = right_width  # calculated side width
    all_edge_labels.extend(
        [
            (
                1,
                "top",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{side_w:.1f} {unit_label}",
            ),
            (
                1,
                "bottom",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{side_w:.1f} {unit_label}",
            ),
            (
                1,
                "left",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{l} {unit_label}",
            ),
            (
                1,
                "right",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{l} {unit_label}",
            ),
        ]
    )

    # Face 2: Left rectangular face (-side_w,0), side_w×l
    left_x, left_y = faces[2][0][0], faces[2][0][1]
    left_width, left_height = faces[2][1], faces[2][2]
    all_edge_labels.extend(
        [
            (
                2,
                "top",
                left_x,
                left_y,
                left_width,
                left_height,
                f"{side_w:.1f} {unit_label}",
            ),
            (
                2,
                "bottom",
                left_x,
                left_y,
                left_width,
                left_height,
                f"{side_w:.1f} {unit_label}",
            ),
            (2, "left", left_x, left_y, left_width, left_height, f"{l} {unit_label}"),
            (2, "right", left_x, left_y, left_width, left_height, f"{l} {unit_label}"),
        ]
    )

    # For triangular faces, we'll handle them with the existing smart height label system
    # since they need special positioning logic for the triangle edges

    # Define shared edges for triangular prism
    shared_edges = [
        # Front-Right shared: Front right edge = Right left edge
        ((0, "right"), (1, "left")),
        # Front-Left shared: Front left edge = Left right edge
        ((0, "left"), (2, "right")),
        # The triangular faces share edges with rectangles, but those are handled by triangle labeling
    ]

    # Remove shared edge duplicates
    edges_to_skip = set()
    for edge_pair in shared_edges:
        face1_idx, edge1 = edge_pair[0]
        face2_idx, edge2 = edge_pair[1]
        if face1_idx < face2_idx:
            edges_to_skip.add((face2_idx, edge2))
        else:
            edges_to_skip.add((face1_idx, edge1))

    # Generate final labels for rectangular faces only using precise positioning
    for (
        face_idx,
        edge_side,
        rect_x,
        rect_y,
        rect_width,
        rect_height,
        text,
    ) in all_edge_labels:
        if (face_idx, edge_side) not in edges_to_skip:
            # Check if this is the center face (Front face = index 0)
            if face_idx == 0:  # Center face - use inside edge positioning for ALL edges
                label = create_inside_edge_positioned_label(
                    text, 16, rect_x, rect_y, rect_width, rect_height, edge_side, ax
                )
            else:  # Non-center face - use edge positioning
                label = create_edge_positioned_label(
                    text, 16, rect_x, rect_y, rect_width, rect_height, edge_side, ax
                )
            labels.append(label)

    return labels


def generate_comprehensive_rectangular_labels(faces, h, w, l, unit_label, ax):
    """Generate comprehensive labels for all edges, avoiding shared edge duplicates."""
    labels = []

    # Face layout: [Front, Top, Left, Right, Back, Bottom]
    # Face positions: [(0,l), (0,0), (-h,0), (w,0), (h+w,0), (0,-h)]

    # Define all potential edge labels for each face
    all_edge_labels = []

    # Face 0: Front (0, l), w×h
    front_x, front_y = faces[0][0][0], faces[0][0][1]
    front_width, front_height = faces[0][1], faces[0][2]
    all_edge_labels.extend(
        [
            (
                0,
                "top",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{w} {unit_label}",
            ),
            (
                0,
                "bottom",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{w} {unit_label}",
            ),
            (
                0,
                "left",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{h} {unit_label}",
            ),
            (
                0,
                "right",
                front_x,
                front_y,
                front_width,
                front_height,
                f"{h} {unit_label}",
            ),
        ]
    )

    # Face 1: Top (0, 0), w×l
    top_x, top_y = faces[1][0][0], faces[1][0][1]
    top_width, top_height = faces[1][1], faces[1][2]
    all_edge_labels.extend(
        [
            (1, "top", top_x, top_y, top_width, top_height, f"{w} {unit_label}"),
            (1, "bottom", top_x, top_y, top_width, top_height, f"{w} {unit_label}"),
            (1, "left", top_x, top_y, top_width, top_height, f"{l} {unit_label}"),
            (1, "right", top_x, top_y, top_width, top_height, f"{l} {unit_label}"),
        ]
    )

    # Face 2: Left (-h, 0), h×l
    left_x, left_y = faces[2][0][0], faces[2][0][1]
    left_width, left_height = faces[2][1], faces[2][2]
    all_edge_labels.extend(
        [
            (2, "top", left_x, left_y, left_width, left_height, f"{h} {unit_label}"),
            (2, "bottom", left_x, left_y, left_width, left_height, f"{h} {unit_label}"),
            (2, "left", left_x, left_y, left_width, left_height, f"{l} {unit_label}"),
            (2, "right", left_x, left_y, left_width, left_height, f"{l} {unit_label}"),
        ]
    )

    # Face 3: Right (w, 0), h×l
    right_x, right_y = faces[3][0][0], faces[3][0][1]
    right_width, right_height = faces[3][1], faces[3][2]
    all_edge_labels.extend(
        [
            (
                3,
                "top",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{h} {unit_label}",
            ),
            (
                3,
                "bottom",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{h} {unit_label}",
            ),
            (
                3,
                "left",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{l} {unit_label}",
            ),
            (
                3,
                "right",
                right_x,
                right_y,
                right_width,
                right_height,
                f"{l} {unit_label}",
            ),
        ]
    )

    # Face 4: Back (h+w, 0), w×l
    back_x, back_y = faces[4][0][0], faces[4][0][1]
    back_width, back_height = faces[4][1], faces[4][2]
    all_edge_labels.extend(
        [
            (4, "top", back_x, back_y, back_width, back_height, f"{w} {unit_label}"),
            (4, "bottom", back_x, back_y, back_width, back_height, f"{w} {unit_label}"),
            (4, "left", back_x, back_y, back_width, back_height, f"{l} {unit_label}"),
            (4, "right", back_x, back_y, back_width, back_height, f"{l} {unit_label}"),
        ]
    )

    # Face 5: Bottom (0, -h), w×h
    bottom_x, bottom_y = faces[5][0][0], faces[5][0][1]
    bottom_width, bottom_height = faces[5][1], faces[5][2]
    all_edge_labels.extend(
        [
            (
                5,
                "top",
                bottom_x,
                bottom_y,
                bottom_width,
                bottom_height,
                f"{w} {unit_label}",
            ),
            (
                5,
                "bottom",
                bottom_x,
                bottom_y,
                bottom_width,
                bottom_height,
                f"{w} {unit_label}",
            ),
            (
                5,
                "left",
                bottom_x,
                bottom_y,
                bottom_width,
                bottom_height,
                f"{h} {unit_label}",
            ),
            (
                5,
                "right",
                bottom_x,
                bottom_y,
                bottom_width,
                bottom_height,
                f"{h} {unit_label}",
            ),
        ]
    )

    # Define shared edges based on actual geometric adjacency in the net
    # Face layout: [Front(0,l), Top(0,0), Left(-h,0), Right(w,0), Back(h+w,0), Bottom(0,-h)]
    shared_edges = [
        # Front-Top shared: Front bottom edge = Top top edge (both at y=l, x∈[0,w])
        ((0, "bottom"), (1, "top")),
        # Top-Left shared: Top left edge = Left right edge (both at x=0, y∈[0,l])
        ((1, "left"), (2, "right")),
        # Top-Right shared: Top right edge = Right left edge (both at x=w, y∈[0,l])
        ((1, "right"), (3, "left")),
        # Top-Bottom shared: Top bottom edge = Bottom top edge (both at y=0, x∈[0,w])
        ((1, "bottom"), (5, "top")),
        # Right-Back shared: Right right edge = Back left edge (both at x=w+h, y∈[0,l])
        ((3, "right"), (4, "left")),
    ]

    # Remove shared edge duplicates (keep the edge from the face with lower index)
    edges_to_skip = set()
    for edge_pair in shared_edges:
        face1_idx, edge1 = edge_pair[0]
        face2_idx, edge2 = edge_pair[1]
        # Keep the edge from the face with higher priority (lower index)
        if face1_idx < face2_idx:
            edges_to_skip.add((face2_idx, edge2))
        else:
            edges_to_skip.add((face1_idx, edge1))

    # Generate final labels using precise positioning, skipping shared edge duplicates
    for (
        face_idx,
        edge_side,
        rect_x,
        rect_y,
        rect_width,
        rect_height,
        text,
    ) in all_edge_labels:
        if (face_idx, edge_side) not in edges_to_skip:
            # Check if this is the center face (Top face = index 1)
            if face_idx == 1:  # Center face - use inside edge positioning for ALL edges
                label = create_inside_edge_positioned_label(
                    text, 16, rect_x, rect_y, rect_width, rect_height, edge_side, ax
                )
            else:  # Non-center face - use edge positioning
                label = create_edge_positioned_label(
                    text, 16, rect_x, rect_y, rect_width, rect_height, edge_side, ax
                )
            labels.append(label)

    return labels


@stimulus_function
def draw_rectangular_prism_net(net: RectangularPrismNet):
    fig, ax = setup_prism_figure()

    h, w, l = net.height, net.width, net.length

    faces = [
        ((0, l), w, h, "Front"),
        ((0, 0), w, l, "Top"),
        ((-h, 0), h, l, "Left"),
        ((w, 0), h, l, "Right"),
        ((h + w, 0), w, l, "Back"),
        ((0, -h), w, h, "Bottom"),
    ]

    # Draw face rectangles
    for (x, y), width, height, label in faces:
        ax.add_patch(Rectangle((x, y), width, height, fill=False))

    # Draw debug labels if enabled
    draw_debug_labels(ax, faces, [], h, w, l)

    # Establish preliminary axes limits early
    setup_coordinate_system(ax, faces, [])

    # Create main dimension labels for rectangular prism
    labels = []

    # Only add labels if this is not a blank net
    if not net.blank_net:
        if not net.label_all_sides:
            # For rectangular prism, we want to show the three main dimensions clearly:
            # Front face (0, l) is w×h, Top face (0, 0) is w×l
            # 1. Width (w) - on front face top/bottom edge
            # 2. Height (h) - on front face left/right edge
            # 3. Length (l) - on top face left/right edge

            front_face = faces[0]  # ((0, l), w, h, "Front")
            top_face = faces[1]  # ((0, 0), w, l, "Top")

            # Width label on top edge of front face
            width_label = create_edge_positioned_label(
                f"{w} {net.unit_label}",
                20,
                front_face[0][0],
                front_face[0][1],
                front_face[1],
                front_face[2],
                "top",
                ax,
            )
            labels.append(width_label)

            # Height label on left edge of front face
            height_label = create_edge_positioned_label(
                f"{h} {net.unit_label}",
                20,
                front_face[0][0],
                front_face[0][1],
                front_face[1],
                front_face[2],
                "left",
                ax,
            )
            labels.append(height_label)

            # Length label INSIDE the center top face, adjacent to left edge (since it has adjacent faces on all sides)
            length_label = create_inside_edge_positioned_label(
                f"{l} {net.unit_label}",
                20,
                top_face[0][0],
                top_face[0][1],
                top_face[1],
                top_face[2],
                "left",  # Position inside, adjacent to left edge
                ax,
            )
            labels.append(length_label)
        else:
            # If label_all_sides is True, add comprehensive labeling
            comprehensive_labels = generate_comprehensive_rectangular_labels(
                faces, h, w, l, net.unit_label, ax
            )
            labels.extend(comprehensive_labels)

    # Finalize and save
    return finalize_and_save_figure(
        fig, ax, faces, [], labels, "rectangular_prism", (h, w, l)
    )


def draw_rectangular_prism_net_for_dual_display(
    ax, net: RegularRectangularPrismNet | CubePrismNet, offset_x: float = 0
) -> list:
    """Draw a rectangular prism net for dual display (no measurements, shifted by offset).

    Args:
        ax: The matplotlib axes to draw on
        net: The rectangular prism net configuration
        offset_x: Horizontal offset to shift the net (for positioning in dual display)

    Returns:
        List of faces for the net
    """
    h, w, l = net.height, net.width, net.length

    # Apply horizontal offset to all face positions
    faces = [
        ((0 + offset_x, l), w, h, "Front"),
        ((0 + offset_x, 0), w, l, "Top"),
        ((-h + offset_x, 0), h, l, "Left"),
        ((w + offset_x, 0), h, l, "Right"),
        ((h + w + offset_x, 0), w, l, "Back"),
        ((0 + offset_x, -h), w, h, "Bottom"),
    ]

    # Draw face rectangles
    for (x, y), width, height, label in faces:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=True,
                facecolor=FILL_COLOR,
                edgecolor="black",
            )
        )

    # Draw debug labels if enabled (but not measurements)
    draw_debug_labels(ax, faces, [], h, w, l)

    # Establish preliminary axes limits early
    setup_coordinate_system(ax, faces, [])

    ax.set_aspect("equal")
    return faces


@stimulus_function
def draw_triangular_prism_net(net: TriangularPrismNet):
    fig, ax = setup_prism_figure()

    h, w, l = net.height, net.width, net.length
    side_w = (h**2 + (w / 2) ** 2) ** 0.5

    faces = [
        ((0, 0), w, l, "Front"),  # Rectangular front face
        ((w, 0), side_w, l, "Right"),  # Rectangular right face
        ((-side_w, 0), side_w, l, "Left"),  # Rectangular back face
    ]
    triangles = [
        ((0, l), w, h, "Top", "up"),  # Triangular top face
        ((0, 0), w, h, "Bottom", "down"),  # Triangular bottom face
    ]

    # Draw face rectangles
    for (x, y), width, height, label in faces:
        ax.add_patch(Rectangle((x, y), width, height, fill=False))

    # Draw triangles
    for (x, y), base, height, label, orientation in triangles:
        draw_triangle(ax, x, y, base, height, label, orientation)

    # Draw height indicator lines for triangles (only if not a blank net)
    if not net.blank_net:
        triangle_x, triangle_y = triangles[0][0]
        draw_triangle_height_indicator(
            ax, triangle_x, triangle_y, triangles[0][1], triangles[0][2], "up"
        )

        bottom_triangle_x, bottom_triangle_y = triangles[1][0]
        draw_triangle_height_indicator(
            ax,
            bottom_triangle_x,
            bottom_triangle_y,
            triangles[1][1],
            triangles[1][2],
            "down",
        )

    # Draw debug labels if enabled
    draw_debug_labels(ax, faces, triangles, h, w, l, side_w)

    # Establish preliminary axes limits early
    setup_coordinate_system(ax, faces, triangles)

    # Collect all shape edges for collision detection
    shape_edges = []
    # Add rectangular face edges
    for (x, y), width, height, label in faces:
        shape_edges.extend(get_rectangle_edges(x, y, width, height))
    # Add triangular face edges
    for (x, y), base, height, label, orientation in triangles:
        shape_edges.extend(get_triangle_edges(x, y, base, height, orientation))

    # Create main dimension labels for triangular prism
    labels = []

    # Only add labels if this is not a blank net
    if not net.blank_net:
        if not net.label_all_sides:
            # For triangular prism, we want to show the three main dimensions clearly:
            # 1. Width (w) - on the triangle base
            # 2. Height (h) - as triangle height
            # 3. Length (l) - on a rectangular face

            # Length label INSIDE the center front rectangular face, adjacent to left edge (since it has adjacent shapes on all sides)
            front_face = faces[0]  # ((0, 0), w, l, "Front")
            face_x, face_y = front_face[0]
            face_width, face_height = front_face[1], front_face[2]

            length_label = create_inside_edge_positioned_label(
                f"{l} {net.unit_label}",
                20,
                face_x,
                face_y,
                face_width,
                face_height,
                "left",  # Position inside, adjacent to left edge
                ax,
            )
            labels.append(length_label)

            # Width label on triangle base (below the triangle base line)
            triangle_base_label = create_precisely_positioned_label(
                f"{w} {net.unit_label}",
                20,
                (triangle_x, triangle_y),
                (triangle_x + w, triangle_y),
                "below",  # Position below the triangle base line
                ax,
            )
            labels.append(triangle_base_label)

            # Height label on triangle (only one)
            height_label = create_smart_height_label(
                triangle_x + triangles[0][1] / 2,
                triangle_y + triangles[0][2] / 2,
                h,
                net.unit_label,
                24,
                ax,
                shape_edges,
                labels,
                "up",
            )
            labels.append(height_label)
        else:
            # If label_all_sides is True, add comprehensive labeling
            comprehensive_labels = generate_comprehensive_triangular_labels(
                faces, triangles, h, w, l, net.unit_label, ax
            )
            labels.extend(comprehensive_labels)

            # Add height labels for both triangles in comprehensive mode
            # Top triangle height
            height_label_1 = create_smart_height_label(
                triangle_x + triangles[0][1] / 2,
                triangle_y + triangles[0][2] / 2,
                h,
                net.unit_label,
                22,
                ax,
                shape_edges,
                labels,
                "up",
            )
            labels.append(height_label_1)

            # Bottom triangle height
            height_label_2 = create_smart_height_label(
                bottom_triangle_x + triangles[1][1] / 2,
                bottom_triangle_y - triangles[1][2] / 2,
                h,
                net.unit_label,
                22,
                ax,
                shape_edges,
                labels,
                "down",
            )
            labels.append(height_label_2)

    # Finalize and save
    return finalize_and_save_figure(
        fig, ax, faces, triangles, labels, "triangular_prism", (h, w, l)
    )


def draw_triangular_prism_net_for_dual_display(
    ax, net: TriangularPrismNet, offset_x: float = 0
) -> tuple[list, list]:
    """Draw a triangular prism net for dual display (no measurements, shifted by offset).

    Args:
        ax: The matplotlib axes to draw on
        net: The triangular prism net configuration
        offset_x: Horizontal offset to shift the net (for positioning in dual display)

    Returns:
        Tuple of (faces list, triangles list) for the net
    """
    h, w, l = net.height, net.width, net.length
    side_w = (h**2 + (w / 2) ** 2) ** 0.5

    # Apply horizontal offset to all face positions
    faces = [
        ((0 + offset_x, 0), w, l, "Front"),  # Rectangular front face
        ((w + offset_x, 0), side_w, l, "Right"),  # Rectangular right face
        ((-side_w + offset_x, 0), side_w, l, "Left"),  # Rectangular back face
    ]
    triangles = [
        ((0 + offset_x, l), w, h, "Top", "up"),  # Triangular top face
        ((0 + offset_x, 0), w, h, "Bottom", "down"),  # Triangular bottom face
    ]

    # Draw face rectangles with alternating colors
    for i, ((x, y), width, height, label) in enumerate(faces):
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=True,
                facecolor=FILL_COLOR,
                edgecolor="black",
            )
        )

    # Draw triangles
    for (x, y), base, height, label, orientation in triangles:
        draw_triangle(ax, x, y, base, height, label, orientation)

    # Draw debug labels if enabled (but not measurements)
    draw_debug_labels(ax, faces, triangles, h, w, l, side_w)

    # Establish preliminary axes limits early
    setup_coordinate_system(ax, faces, triangles)

    ax.set_aspect("equal")
    return faces, triangles


@stimulus_function
def draw_pyramid_net(net: PyramidPrismNet):
    fig, ax = setup_prism_figure()

    h, w, l = net.height, net.width, net.length

    faces = [
        ((0, 0), w, l, "Base"),  # Rectangular base (not necessarily a square)
    ]
    triangles = [
        ((0, l), w, h, "Front", "up"),  # Triangular front face
        ((w, 0), h, l, "Right", "right"),  # Triangular right face
        ((0, 0), w, h, "Back", "down"),  # Triangular back face
        ((0, 0), h, l, "Left", "left"),  # Triangular left face
    ]

    # Draw face rectangles
    for (x, y), width, height, label in faces:
        ax.add_patch(Rectangle((x, y), width, height, fill=False))

    # Draw triangles
    for (x, y), base, height, label, orientation in triangles:
        draw_triangle(ax, x, y, base, height, label, orientation)

    # Draw height indicator lines for pyramid (only if not a blank net)
    if not net.blank_net:
        square_top_right_x = faces[0][0][0] + faces[0][1]
        square_top_right_y = faces[0][0][1] + faces[0][2]
        # Use simplified height indicator for front triangle
        draw_pyramid_triangle_height_indicator(
            ax,
            square_top_right_x,
            square_top_right_y,
            triangles[0][0][0],
            triangles[0][0][1],
            triangles[0][1],
            triangles[0][2],
            "up",
        )

    # Draw debug labels if enabled
    draw_debug_labels(ax, faces, triangles, h, w, l)

    # Establish preliminary axes limits early
    setup_coordinate_system(ax, faces, triangles)

    # Collect all shape edges for collision detection
    shape_edges = []
    # Add rectangular face edges
    for (x, y), width, height, label in faces:
        shape_edges.extend(get_rectangle_edges(x, y, width, height))
    # Add triangular face edges
    for (x, y), base, height, label, orientation in triangles:
        shape_edges.extend(get_triangle_edges(x, y, base, height, orientation))

    # Create main dimension labels for pyramid
    labels = []

    # Only add labels if this is not a blank net
    if not net.blank_net:
        if not net.label_all_sides:
            # For pyramid, the base face is the center and should have inside labels
            base_face = faces[0]  # ((0, 0), w, l, "Base")

            # Width and Length labels INSIDE the center base face, adjacent to their respective edges
            width_label = create_inside_edge_positioned_label(
                f"{w} {net.unit_label}",
                20,
                base_face[0][0],
                base_face[0][1],
                base_face[1],
                base_face[2],
                "top",  # Position inside, adjacent to top edge (width)
                ax,
            )
            labels.append(width_label)

            length_label = create_inside_edge_positioned_label(
                f"{l} {net.unit_label}",
                20,
                base_face[0][0],
                base_face[0][1],
                base_face[1],
                base_face[2],
                "left",  # Position inside, adjacent to left edge (length)
                ax,
            )
            labels.append(length_label)

            # Add smart height label for front triangle (only when NOT doing comprehensive labeling)
            height_label = create_smart_height_label(
                triangles[0][0][0] + triangles[0][1] / 2,
                triangles[0][0][1] + triangles[0][2] / 2,
                h,
                net.unit_label,
                24,  # Increased font size
                ax,
                shape_edges,
                labels,
                "up",
            )
            labels.append(height_label)
        else:
            # If label_all_sides is True, add comprehensive labeling
            comprehensive_labels = generate_comprehensive_pyramid_labels(
                faces, triangles, h, w, l, net.unit_label, ax
            )
            labels.extend(comprehensive_labels)

            # Comprehensive labeling case - add height indicators and labels for all triangular faces
            # Draw height indicators for all triangular faces with proper base corner alignment
            # Base rectangle corners: (0,0) bottom-left, (w,0) bottom-right, (0,l) top-left, (w,l) top-right
            base_corners = [
                (w, l),  # Front triangle aligns with top-right corner
                (w, 0),  # Right triangle aligns with bottom-right corner
                (0, 0),  # Back triangle aligns with bottom-left corner
                (0, l),  # Left triangle aligns with top-left corner
            ]

            for i, ((x, y), base, height, label, orientation) in enumerate(triangles):
                if i > 0:  # Skip front triangle (already drawn above)
                    corner_x, corner_y = base_corners[i]
                    draw_pyramid_triangle_height_indicator(
                        ax, corner_x, corner_y, x, y, base, height, orientation
                    )

            # Add smart height labels for all triangular faces
            for i, ((x, y), base, height, label, orientation) in enumerate(triangles):
                # Calculate the center point of the height line for each triangle
                if orientation == "up":
                    center_x, center_y = x + base / 2, y + height / 2
                elif orientation == "right":
                    center_x, center_y = x + base / 2, y + height / 2
                elif orientation == "down":
                    center_x, center_y = x + base / 2, y - height / 2
                elif orientation == "left":
                    center_x, center_y = x - base / 2, y + height / 2
                else:
                    center_x, center_y = x + base / 2, y + height / 2

                height_label = create_smart_height_label(
                    center_x,
                    center_y,
                    h,
                    net.unit_label,
                    22,  # Increased font size
                    ax,
                    shape_edges,
                    labels,
                    orientation,
                )
                labels.append(height_label)

    # Finalize and save
    return finalize_and_save_figure(
        fig, ax, faces, triangles, labels, "pyramid", (h, w, l)
    )


def draw_pyramid_net_for_dual_display(
    ax, net: SquarePyramidPrismNet | RectangularPyramidPrismNet, offset_x: float = 0
) -> tuple[list, list]:
    """Draw a pyramid net for dual display (no measurements, shifted by offset).

    Args:
        ax: The matplotlib axes to draw on
        net: The pyramid net configuration
        offset_x: Horizontal offset to shift the net (for positioning in dual display)

    Returns:
        Tuple of (faces list, triangles list) for the net
    """
    h, w, l = net.height, net.width, net.length

    # Apply horizontal offset to all face positions
    faces = [
        (
            (0 + offset_x, 0),
            w,
            l,
            "Base",
        ),  # Rectangular base (not necessarily a square)
    ]
    triangles = [
        ((0 + offset_x, l), w, h, "Front", "up"),  # Triangular front face
        ((w + offset_x, 0), h, l, "Right", "right"),  # Triangular right face
        ((0 + offset_x, 0), w, h, "Back", "down"),  # Triangular back face
        ((0 + offset_x, 0), h, l, "Left", "left"),  # Triangular left face
    ]

    # Draw face rectangles with alternating colors
    for i, ((x, y), width, height, label) in enumerate(faces):
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=True,
                facecolor=FILL_COLOR,
                edgecolor="black",
            )
        )

    # Draw triangles
    for (x, y), base, height, label, orientation in triangles:
        draw_triangle(ax, x, y, base, height, label, orientation)

    # Draw debug labels if enabled (but not measurements)
    draw_debug_labels(ax, faces, triangles, h, w, l)

    # Establish preliminary axes limits early
    setup_coordinate_system(ax, faces, triangles)

    ax.set_aspect("equal")
    return faces, triangles


@stimulus_function
def draw_prism_net(stimulus: PrismNet):
    if stimulus.net_type == EPrismType.RECTANGULAR:
        return draw_rectangular_prism_net(stimulus)  # type: ignore
    elif stimulus.net_type == EPrismType.TRIANGULAR:
        return draw_triangular_prism_net(stimulus)  # type: ignore
    elif stimulus.net_type == EPrismType.PYRAMIDAL:
        return draw_pyramid_net(stimulus)  # type: ignore


@stimulus_function
def draw_custom_triangular_prism_net(net: CustomTriangularPrismNet):
    fig, ax = setup_prism_figure()

    h, w, l = net.height, net.width, net.length
    side_w = net.side_w

    # Define rectangular faces - no fill
    faces = [
        ((0, 0), w, l, "Front"),  # Rectangular front face
        ((w, 0), side_w, l, "Right"),  # Rectangular right face
        ((-side_w, 0), side_w, l, "Left"),  # Rectangular left face
    ]

    # Define triangular faces - no fill
    triangle_x = 0
    triangle_y = l
    triangles = [
        ((triangle_x, triangle_y), w, h, "Top", "up"),
        ((triangle_x, 0), w, h, "Bottom", "down"),
    ]

    # Draw faces
    for (x, y), width, height, label in faces:
        rect = patches.Rectangle(
            (x, y), width, height, fill=False, edgecolor="black", linewidth=1
        )
        ax.add_patch(rect)

    # Draw triangles
    for (x, y), base, height, label, orientation in triangles:
        points = np.array(
            [
                [x, y],
                [x + base, y],
                [x + base / 2, y + height if orientation == "up" else y - height],
                [x, y],
            ]
        )
        ax.plot(points[:, 0], points[:, 1], "black", linewidth=1)

    # Add labels if not blank net
    if not net.blank_net:
        labels = []
        # Get shape edges for label collision detection
        shape_edges = []
        for (x, y), width, height, _ in faces:
            shape_edges.extend(get_rectangle_edges(x, y, width, height))
        for (x, y), base, height, _, orientation in triangles:
            shape_edges.extend(get_triangle_edges(x, y, base, height, orientation))

        if not net.label_all_sides:
            # Basic labels (length, width, height)
            length_label = create_inside_edge_positioned_label(
                f"{l} {net.unit_label}",
                20,
                0,
                0,
                w,
                l,
                "left",
                ax,
            )
            labels.append(length_label)

            # Width label on triangle base
            triangle_base_label = create_precisely_positioned_label(
                f"{w} {net.unit_label}",
                20,
                (triangle_x, triangle_y),
                (triangle_x + w, triangle_y),
                "below",
                ax,
            )
            labels.append(triangle_base_label)

            # Height label on triangle
            height_label = create_smart_height_label(
                triangle_x + triangles[0][1] / 2,
                triangle_y + triangles[0][2] / 2,
                h,
                net.unit_label,
                24,
                ax,
                shape_edges,
                labels,
                "up",
            )
            labels.append(height_label)
        else:
            # Comprehensive labeling - all sides
            comprehensive_labels = generate_comprehensive_triangular_labels(
                faces, triangles, h, w, l, net.unit_label, ax
            )
            labels.extend(comprehensive_labels)

            # # Add height labels for both triangles
            height_label_1 = create_smart_height_label(
                triangle_x + triangles[0][1] / 2,  # Use triangles array for coordinates
                triangle_y + triangles[0][2] / 2,  # Add half height to y position
                h,
                net.unit_label,
                24,  # Keep consistent font size
                ax,
                shape_edges,  # Use actual shape edges
                labels,
                "up",
            )
            labels.append(height_label_1)

            # height_label_2 = create_smart_height_label(
            #     triangle_x + triangles[1][1] / 2,  # Use triangles array for coordinates
            #     triangles[1][2] / 2,  # Half height for bottom triangle
            #     h,
            #     net.unit_label,
            #     24,  # Keep consistent font size
            #     ax,
            #     shape_edges,  # Use actual shape edges
            #     labels,
            #     "down",
            # )
            # labels.append(height_label_2)

    # Draw height lines and right angle markers for triangles
    draw_triangle_height_indicator(ax, triangle_x, triangle_y, w, h, "up")
    draw_triangle_height_indicator(ax, triangle_x, 0, w, h, "down")

    # Setup and save
    setup_coordinate_system(ax, faces, triangles)

    max_rect_w = max(w, side_w)
    ratio = (l / max_rect_w) if max_rect_w else 1.0
    dim_hint = (min(h, w, side_w, l),) * 3 if ratio < 0.9 else (h, w, l)

    return finalize_and_save_figure(
        fig, ax, faces, triangles, labels, "custom_triangular", dim_hint
    )


@stimulus_function
def draw_dual_nets(stimulus: DualPrismNets) -> str:
    """Draw two nets side by side."""

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(20, 14),
        tight_layout=False,  # Disable tight layout
        constrained_layout=False,  # Disable constrained layout
    )  # Keep width at 20, increase height from 8 to 10 for labels
    ax.set_aspect("equal")

    # Create both nets with same dimensions
    def create_net(net_type: DualNetsShapeType) -> PrismNet:
        """Create a net of the specified type with default dimensions."""
        common_params = {
            "unit_label": "",
            "blank_net": True,
        }

        match net_type:
            case DualNetsShapeType.CUBE:
                return CubePrismNet(
                    height=4,
                    width=4,
                    length=4,
                    **common_params,
                )
            case DualNetsShapeType.RECTANGULAR_PRISM:
                return RegularRectangularPrismNet(
                    height=4,
                    width=4,
                    length=6,
                    **common_params,
                )
            case DualNetsShapeType.TRIANGULAR_PRISM:
                return TriangularPrismNet(
                    height=4,
                    width=4,
                    length=4,
                    **common_params,
                )
            case DualNetsShapeType.SQUARE_PYRAMID:
                return SquarePyramidPrismNet(
                    height=4,
                    width=4,
                    length=4,
                    **common_params,
                )
            case DualNetsShapeType.RECTANGULAR_PYRAMID:
                return RectangularPyramidPrismNet(
                    height=4,
                    width=4,
                    length=6,
                    **common_params,
                )

    # Create correct and incorrect nets
    correct_net = create_net(stimulus.correct_shape_type)
    incorrect_net = create_net(stimulus.incorrect_shape_type)

    # Place nets based on specified position
    if stimulus.correct_shape_position == Position.LEFT:
        first_net = correct_net  # Figure 1 (left) shows correct net
        second_net = incorrect_net  # Figure 2 (right) shows incorrect net
    else:
        first_net = incorrect_net  # Figure 1 (left) shows incorrect net
        second_net = correct_net  # Figure 2 (right) shows correct net

    # Set offsets for spacing - extra wide gap to prevent any overlap
    left_offset = -9  # Move left net further left
    right_offset = 9  # Move right net further right

    # Draw nets and collect faces/triangles
    all_faces = []
    all_triangles = []

    # Draw first net (left)
    if isinstance(first_net, (RegularRectangularPrismNet, CubePrismNet)):
        faces1 = draw_rectangular_prism_net_for_dual_display(ax, first_net, left_offset)
        triangles1 = []
    elif isinstance(first_net, TriangularPrismNet):
        faces1, triangles1 = draw_triangular_prism_net_for_dual_display(
            ax, first_net, left_offset
        )
    else:  # Square or Rectangular Pyramid
        assert isinstance(
            first_net, (SquarePyramidPrismNet, RectangularPyramidPrismNet)
        )  # Help type checker
        faces1, triangles1 = draw_pyramid_net_for_dual_display(
            ax, first_net, left_offset
        )

    # Draw second net (right)
    if isinstance(second_net, (RegularRectangularPrismNet, CubePrismNet)):
        faces2 = draw_rectangular_prism_net_for_dual_display(
            ax, second_net, right_offset
        )
        triangles2 = []
    elif isinstance(second_net, TriangularPrismNet):
        faces2, triangles2 = draw_triangular_prism_net_for_dual_display(
            ax, second_net, right_offset
        )
    else:  # Square or Rectangular Pyramid
        assert isinstance(
            second_net, (SquarePyramidPrismNet, RectangularPyramidPrismNet)
        )  # Help type checker
        faces2, triangles2 = draw_pyramid_net_for_dual_display(
            ax, second_net, right_offset
        )

    # Collect all faces and triangles
    all_faces.extend(faces1)
    all_faces.extend(faces2)
    all_triangles.extend(triangles1)
    all_triangles.extend(triangles2)

    # After drawing all nets but BEFORE adding labels, fix the plot limits
    ax.set_xlim(-15, 15)  # Fixed horizontal limits
    ax.set_ylim(-6, 6)  # Reduce range to trim whitespace

    # Calculate the center and highest point of each net
    left_net_center = left_offset + first_net.width / 2
    right_net_center = right_offset + second_net.width / 2

    # Calculate the highest point of each net (all nets have highest point at l + h)
    left_net_highest = first_net.length + first_net.height
    right_net_highest = second_net.length + second_net.height

    # Position labels above each net with consistent spacing
    label_spacing = 1.0  # Space above the highest point
    label_positions = [
        (left_net_center, "Figure 1", left_net_highest + label_spacing),
        (right_net_center, "Figure 2", right_net_highest + label_spacing),
    ]
    for x, text, y in label_positions:
        ax.text(
            x,
            y,  # Use individual height for each label
            text,
            ha="center",
            va="center",  # Center vertically
            fontsize=16,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.8",  # More padding in the box
                fc="white",
                ec="black",
                lw=1,
                alpha=1,
            ),
            zorder=10,  # Ensure labels are always on top
        )

    # Hide axes
    ax.axis("off")

    # Finalize and save
    return finalize_and_save_dual_nets_figure(
        fig, ax, all_faces, all_triangles, [], "dual_nets", (4, 4, 4)
    )


def finalize_and_save_dual_nets_figure(
    fig, ax, faces, triangles, labels, net_type_name, dimensions
):
    """Finalize dual nets figure with labels and save to file."""
    # Draw all labels with adaptive sizing
    draw_optimized_labels(labels, ax, dimensions)

    # Set final tight bounds to eliminate blank space
    min_x, max_x, min_y, max_y = calculate_tight_bounds(faces, triangles, labels)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y + 1)

    plt.tight_layout()

    # Generate filename and save with bbox_inches=None for dual nets
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{net_type_name}_net_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


def create_precisely_positioned_label(
    text: str,
    fontsize: float,
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    position_side: str,  # "above", "below", "left", "right"
    ax,
    margin: float = 0.02,  # Small margin to prevent touching
) -> Label:
    """Create a label positioned precisely adjacent to a line without overlapping."""

    # Measure text dimensions first
    text_width, text_height = measure_text_dimensions(text, fontsize, ax)

    # Calculate line properties
    line_center_x = (line_start[0] + line_end[0]) / 2
    line_center_y = (line_start[1] + line_end[1]) / 2

    # Determine line orientation
    is_horizontal = abs(line_end[1] - line_start[1]) < abs(line_end[0] - line_start[0])

    # Background box padding (must match the padding in draw_optimized_labels)
    padding = 0.05

    # Calculate label position based on side and line orientation
    if position_side == "above":
        if is_horizontal:
            # Horizontal line, position above
            label_x = line_center_x
            label_y = (
                max(line_start[1], line_end[1]) + padding + text_height / 2 + margin
            )
            ha, va = "center", "center"
        else:
            # Vertical line, position to the right
            label_x = (
                max(line_start[0], line_end[0]) + padding + text_width / 2 + margin
            )
            label_y = line_center_y
            ha, va = "center", "center"

    elif position_side == "below":
        if is_horizontal:
            # Horizontal line, position below
            label_x = line_center_x
            label_y = (
                min(line_start[1], line_end[1]) - padding - text_height / 2 - margin
            )
            ha, va = "center", "center"
        else:
            # Vertical line, position to the left
            label_x = (
                min(line_start[0], line_end[0]) - padding - text_width / 2 - margin
            )
            label_y = line_center_y
            ha, va = "center", "center"

    elif position_side == "left":
        if is_horizontal:
            # Horizontal line, position to the left
            label_x = (
                min(line_start[0], line_end[0]) - padding - text_width / 2 - margin
            )
            label_y = line_center_y
            ha, va = "center", "center"
        else:
            # Vertical line, position below
            label_x = line_center_x
            label_y = (
                min(line_start[1], line_end[1]) - padding - text_height / 2 - margin
            )
            ha, va = "center", "center"

    elif position_side == "right":
        if is_horizontal:
            # Horizontal line, position to the right
            label_x = (
                max(line_start[0], line_end[0]) + padding + text_width / 2 + margin
            )
            label_y = line_center_y
            ha, va = "center", "center"
        else:
            # Vertical line, position above
            label_x = line_center_x
            label_y = (
                max(line_start[1], line_end[1]) + padding + text_height / 2 + margin
            )
            ha, va = "center", "center"
    else:
        # Default fallback
        label_x = line_center_x
        label_y = line_center_y
        ha, va = "center", "center"

    # Create and return the label
    label = Label(x=label_x, y=label_y, text=text, fontsize=fontsize, ha=ha, va=va)
    label.width = text_width
    label.height = text_height

    return label


def create_edge_positioned_label(
    text: str,
    fontsize: float,
    rect_x: float,
    rect_y: float,
    rect_width: float,
    rect_height: float,
    edge: str,  # "top", "bottom", "left", "right"
    ax,
    margin: float = 0.01,  # Reduced margin for closer positioning
) -> Label:
    """Create a label positioned precisely adjacent to a rectangle edge without overlapping."""

    # Measure text dimensions first
    text_width, text_height = measure_text_dimensions(text, fontsize, ax)

    # Background box padding (reduced for closer positioning)
    padding = 0.03

    # Calculate label position based on edge
    if edge == "top":
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height + padding + text_height / 2 + margin
        ha, va = "center", "center"
    elif edge == "bottom":
        label_x = rect_x + rect_width / 2
        label_y = rect_y - padding - text_height / 2 - margin
        ha, va = "center", "center"
    elif edge == "left":
        # For vertical lines, align text edge closer to the shape edge instead of centering
        label_x = rect_x - padding - margin
        label_y = rect_y + rect_height / 2
        ha, va = "right", "center"  # Right-align text to bring it closer to the edge
    elif edge == "right":
        # For vertical lines, align text edge closer to the shape edge instead of centering
        label_x = rect_x + rect_width + padding + margin
        label_y = rect_y + rect_height / 2
        ha, va = "left", "center"  # Left-align text to bring it closer to the edge
    else:
        # Default fallback
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height / 2
        ha, va = "center", "center"

    # Create and return the label
    label = Label(x=label_x, y=label_y, text=text, fontsize=fontsize, ha=ha, va=va)
    label.width = text_width
    label.height = text_height

    # Store positioning parameters for potential recalculation after font optimization
    label.positioning_params = {
        "rect_x": rect_x,
        "rect_y": rect_y,
        "rect_width": rect_width,
        "rect_height": rect_height,
        "edge": edge,
        "margin": margin,
    }

    return label


def setup_prism_figure():
    """Set up a standard figure for prism net drawing."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_debug_labels(ax, faces, triangles, h, w, l, side_w=None):
    """Draw debug labels if settings.debug is True."""
    if not settings.debug:
        return

    # Debug labels for rectangular faces
    for (x, y), width, height, label in faces:
        if side_w:
            debug_fontsize = max(
                max(side_w + w + side_w, h * 2 + l) * FONT_SIZE_FACTOR, 12
            )
        else:
            debug_fontsize = max(max(h * 2 + w + w, h * 2 + l) * FONT_SIZE_FACTOR, 12)
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=debug_fontsize,
        )


def setup_coordinate_system(ax, faces, triangles):
    """Establish preliminary axes limits early to ensure proper coordinate system."""
    preliminary_bounds = calculate_tight_bounds(faces, triangles, [])
    ax.set_xlim(preliminary_bounds[0], preliminary_bounds[1])
    ax.set_ylim(preliminary_bounds[2], preliminary_bounds[3])


def detect_triangle_base_conflicts(faces, triangles):
    """Detect which face edges would conflict with triangle dashed height lines."""
    conflicts = set()

    for triangle_idx, ((tri_x, tri_y), base, height, label, orientation) in enumerate(
        triangles
    ):
        # For each triangle, determine which face edge corresponds to its base
        # and mark that edge as conflicted (should not have labels on the triangle side)

        if orientation == "up":
            # Triangle points up, base is at bottom
            # Base line goes from (tri_x, tri_y) to (tri_x + base, tri_y)
            base_y = tri_y
            base_x_start, base_x_end = tri_x, tri_x + base

            # Find which face this base edge belongs to
            for face_idx, ((face_x, face_y), width, height, face_label) in enumerate(
                faces
            ):
                # Check if triangle base aligns with face top edge
                if (
                    abs(base_y - (face_y + height)) < 0.01
                    and base_x_start >= face_x
                    and base_x_end <= face_x + width
                ):
                    conflicts.add((face_idx, "top"))
                # Check if triangle base aligns with face bottom edge
                elif (
                    abs(base_y - face_y) < 0.01
                    and base_x_start >= face_x
                    and base_x_end <= face_x + width
                ):
                    conflicts.add((face_idx, "bottom"))

        elif orientation == "down":
            # Triangle points down, base is at top
            # Base line goes from (tri_x, tri_y) to (tri_x + base, tri_y)
            base_y = tri_y
            base_x_start, base_x_end = tri_x, tri_x + base

            # Find which face this base edge belongs to
            for face_idx, ((face_x, face_y), width, height, face_label) in enumerate(
                faces
            ):
                # Check if triangle base aligns with face top edge
                if (
                    abs(base_y - (face_y + height)) < 0.01
                    and base_x_start >= face_x
                    and base_x_end <= face_x + width
                ):
                    conflicts.add((face_idx, "top"))
                # Check if triangle base aligns with face bottom edge
                elif (
                    abs(base_y - face_y) < 0.01
                    and base_x_start >= face_x
                    and base_x_end <= face_x + width
                ):
                    conflicts.add((face_idx, "bottom"))

        elif orientation == "left":
            # Triangle points left, base is on right side
            # Base line goes from (tri_x, tri_y) to (tri_x, tri_y + height)
            base_x = tri_x
            base_y_start, base_y_end = tri_y, tri_y + height

            # Find which face this base edge belongs to
            for face_idx, ((face_x, face_y), width, height, face_label) in enumerate(
                faces
            ):
                # Check if triangle base aligns with face left edge
                if (
                    abs(base_x - face_x) < 0.01
                    and base_y_start >= face_y
                    and base_y_end <= face_y + height
                ):
                    conflicts.add((face_idx, "left"))
                # Check if triangle base aligns with face right edge
                elif (
                    abs(base_x - (face_x + width)) < 0.01
                    and base_y_start >= face_y
                    and base_y_end <= face_y + height
                ):
                    conflicts.add((face_idx, "right"))

        elif orientation == "right":
            # Triangle points right, base is on left side
            # Base line goes from (tri_x, tri_y) to (tri_x, tri_y + height)
            base_x = tri_x
            base_y_start, base_y_end = tri_y, tri_y + height

            # Find which face this base edge belongs to
            for face_idx, ((face_x, face_y), width, height, face_label) in enumerate(
                faces
            ):
                # Check if triangle base aligns with face left edge
                if (
                    abs(base_x - face_x) < 0.01
                    and base_y_start >= face_y
                    and base_y_end <= face_y + height
                ):
                    conflicts.add((face_idx, "left"))
                # Check if triangle base aligns with face right edge
                elif (
                    abs(base_x - (face_x + width)) < 0.01
                    and base_y_start >= face_y
                    and base_y_end <= face_y + height
                ):
                    conflicts.add((face_idx, "right"))

    return conflicts


def create_main_dimension_labels(net, faces, ax, triangles=None):
    """Create main dimension labels for any net type."""
    labels = []

    if not net.label_all_sides:
        # Get the primary face (always first face)
        primary_face = faces[0]
        face_x, face_y = primary_face[0]
        face_width, face_height = primary_face[1], primary_face[2]

        # Determine label positioning based on net type
        # Check the actual type of the net object
        net_type_name = type(net).__name__

        # Detect conflicts with triangle dashed height lines
        conflicts = set()
        if triangles:
            conflicts = detect_triangle_base_conflicts(faces, triangles)

        if net_type_name == "RectangularPrismNet":
            # Rectangular: width on top, length on left
            width_edge, length_edge = "top", "left"
        elif net_type_name == "TriangularPrismNet":
            # Triangular: need to avoid conflict with triangle dashed lines
            # Default positioning
            width_edge, length_edge = "bottom", "left"

            # Check for conflicts and adjust positioning
            if (0, "bottom") in conflicts:
                # Bottom edge conflicts with triangle, try top
                if (0, "top") not in conflicts:
                    width_edge = "top"
                else:
                    # Both top and bottom conflict, use right edge
                    width_edge = "right"

            if (0, "left") in conflicts:
                # Left edge conflicts with triangle, try right
                if (0, "right") not in conflicts:
                    length_edge = "right"
                else:
                    # Both left and right conflict, use bottom (if width not using it)
                    if width_edge != "bottom":
                        length_edge = "bottom"
                    else:
                        length_edge = "top"

        elif net_type_name == "PyramidPrismNet":
            # Pyramid: width on top, length on left, but check for conflicts
            width_edge, length_edge = "top", "left"

            # Check for conflicts and adjust positioning
            if (0, "top") in conflicts:
                width_edge = "bottom"
            if (0, "left") in conflicts:
                length_edge = "right"
        else:
            # Default fallback
            width_edge, length_edge = "top", "left"

        # Create width label
        width_label = create_edge_positioned_label(
            f"{net.width} {net.unit_label}",
            20,
            face_x,
            face_y,
            face_width,
            face_height,
            width_edge,
            ax,
        )
        labels.append(width_label)

        # Create length label
        length_label = create_edge_positioned_label(
            f"{net.length} {net.unit_label}",
            20,
            face_x,
            face_y,
            face_width,
            face_height,
            length_edge,
            ax,
        )
        labels.append(length_label)

    return labels


def finalize_and_save_figure(
    fig, ax, faces, triangles, labels, net_type_name, dimensions
):
    """Finalize figure with labels and save to file."""
    # Draw all labels with adaptive sizing
    draw_optimized_labels(labels, ax, dimensions)

    # Set final tight bounds to eliminate blank space
    min_x, max_x, min_y, max_y = calculate_tight_bounds(faces, triangles, labels)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.tight_layout()

    # Generate filename and save
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{net_type_name}_net_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


def create_inside_positioned_label(
    text: str,
    fontsize: float,
    rect_x: float,
    rect_y: float,
    rect_width: float,
    rect_height: float,
    ax,
    position: str = "center",  # "center", "upper", "lower", "left", "right"
    margin: float = 0.1,
) -> Label:
    """Create a label positioned inside a rectangle with specific positioning."""

    # Measure text dimensions first
    text_width, text_height = measure_text_dimensions(text, fontsize, ax)

    # Calculate position based on positioning parameter
    if position == "center":
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height / 2
    elif position == "upper":
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height * 0.7  # Upper portion
    elif position == "lower":
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height * 0.3  # Lower portion
    elif position == "left":
        label_x = rect_x + rect_width * 0.3  # Left portion
        label_y = rect_y + rect_height / 2
    elif position == "right":
        label_x = rect_x + rect_width * 0.7  # Right portion
        label_y = rect_y + rect_height / 2
    else:
        # Default to center
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height / 2

    # Create and return the label
    label = Label(
        x=label_x, y=label_y, text=text, fontsize=fontsize, ha="center", va="center"
    )
    label.width = text_width
    label.height = text_height

    return label


def identify_center_face_index(net_type_name: str) -> int:
    """Identify which face index is the center face for each net type."""
    if net_type_name == "RectangularPrismNet":
        return 1  # Top face at (0, 0) is center
    elif net_type_name == "TriangularPrismNet":
        return 0  # Front rectangular face at (0, 0) is center
    elif net_type_name == "PyramidPrismNet":
        return 0  # Base face at (0, 0) is center
    else:
        return -1  # No center face identified


def create_inside_edge_positioned_label(
    text: str,
    fontsize: float,
    rect_x: float,
    rect_y: float,
    rect_width: float,
    rect_height: float,
    edge: str,  # "top", "bottom", "left", "right"
    ax,
    margin: float = 0.05,  # Reduced distance from the edge, inside the shape
) -> Label:
    """Create a label positioned inside a rectangle, adjacent to the specified edge."""

    # Measure text dimensions first
    text_width, text_height = measure_text_dimensions(text, fontsize, ax)

    # Calculate label position based on edge, but positioned INSIDE the shape
    if edge == "top":
        label_x = rect_x + rect_width / 2
        label_y = (
            rect_y + rect_height - margin - text_height / 2
        )  # Inside, near top edge
        ha, va = "center", "center"
    elif edge == "bottom":
        label_x = rect_x + rect_width / 2
        label_y = rect_y + margin + text_height / 2  # Inside, near bottom edge
        ha, va = "center", "center"
    elif edge == "left":
        # For vertical edges, position closer to the edge
        label_x = rect_x + margin  # Just margin from edge, not centered
        label_y = rect_y + rect_height / 2
        ha, va = "left", "center"  # Left-align to bring text closer to edge
    elif edge == "right":
        # For vertical edges, position closer to the edge
        label_x = rect_x + rect_width - margin  # Just margin from edge, not centered
        label_y = rect_y + rect_height / 2
        ha, va = "right", "center"  # Right-align to bring text closer to edge
    else:
        # Default fallback to center
        label_x = rect_x + rect_width / 2
        label_y = rect_y + rect_height / 2
        ha, va = "center", "center"

    # Create and return the label
    label = Label(x=label_x, y=label_y, text=text, fontsize=fontsize, ha=ha, va=va)
    label.width = text_width
    label.height = text_height

    # Store positioning parameters for potential recalculation after font optimization
    label.positioning_params = {
        "rect_x": rect_x,
        "rect_y": rect_y,
        "rect_width": rect_width,
        "rect_height": rect_height,
        "edge": edge,
        "margin": margin,
        "is_inside": True,  # Mark this as an inside label
    }

    return label


if __name__ == "__main__":
    """
    I use debug mode to debug print the face names.
    """
    # Testing function
    settings.debug = True
    net = PrismNet(
        height=5,
        width=8,
        length=8,
        net_type=EPrismType.TRIANGULAR,
        unit_label="in",
    )
    # Generate the image
    file_path = draw_prism_net(net)
    img = plt.imread(file_path)
    # Display the image (optional, for immediate viewing)
    matplotlib.use("TkAgg")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
