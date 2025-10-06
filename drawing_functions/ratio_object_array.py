import random
import time
from typing import Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.ratio_object_array import (
    RatioObjectArray,
    RatioObjectShape,
)
from content_generators.settings import settings
from matplotlib import colors as mcolors

# 15 colors similar to the image (blue, green, orange variations)
COLOR_PALETTE = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
]


def _get_random_color() -> str:
    """Get a random color from the palette."""
    return random.choice(COLOR_PALETTE)


def _darken_color(color: str, factor: float = 0.6) -> Tuple[float, float, float, float]:
    """Return a darker RGBA for the given color. factor in (0,1); lower => darker."""
    try:
        r, g, b, a = mcolors.to_rgba(color)
    except (ValueError, TypeError):
        # Fallback to a default color if parsing fails
        r, g, b, a = mcolors.to_rgba("#000000")
    return (r * factor, g * factor, b * factor, a)


def _validate_grid(
    objects: Sequence[Sequence[object]], rows: int, columns: int
) -> None:
    """Ensure the 2D grid matches rows x columns (Sequence so len() is valid)."""
    if len(objects) != rows:
        raise ValueError(f"objects has {len(objects)} rows but rows={rows}.")
    for i, row in enumerate(objects):
        if len(row) != columns:
            raise ValueError(f"Row {i} has {len(row)} columns but columns={columns}.")


def _star_vertices(
    cx: float, cy: float, outer_r: float, inner_ratio: float = 0.5, points: int = 5
):
    """
    Build a 5-point star polygon with outer radius = outer_r.
    inner_ratio controls 'thickness' (0.45–0.55 looks good). Point-up orientation.
    """
    inner_r = outer_r * inner_ratio
    verts = []
    # Start at -90° so one tip points upward; step by 36° (2 * pi / (2*points))
    theta0 = -np.pi / 2
    step = np.pi / points
    for k in range(points * 2):
        r = outer_r if k % 2 == 0 else inner_r
        theta = theta0 + k * step
        verts.append((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    return verts


@stimulus_function
def draw_ratio_object_array(stimulus: RatioObjectArray) -> str:
    """
    Draw a grid of shapes (circle, square, triangle, star, hexagon).

    Enhancements:
      - allows 2, 3, or 4 DISTINCT SHAPES in one image (colors may vary)
      - adds HEXAGON
      - outlines use a darker version of fill color
      - STAR now drawn as a polygon sized to match circle/square footprint
      - Random color assignment from 15-color palette - ONE COLOR PER SHAPE TYPE
      - Limited to 2 rows maximum
      - Configurable shape size (auto-adjusts for single row)
    """
    rows = stimulus.rows
    columns = stimulus.columns
    objects = stimulus.objects

    _validate_grid(objects, rows, columns)

    # Validate number of distinct SHAPES (not (shape,color) pairs)
    unique_shapes = {cell.shape for row in objects for cell in row}
    if not (2 <= len(unique_shapes) <= 4):
        raise ValueError(
            f"Image must contain between 2 and 4 distinct shapes; found {len(unique_shapes)}."
        )

    # Get configurable shape size
    shape_size = stimulus.get_effective_shape_size()
    base = 0.7 * shape_size  # circle diameter == square side == hex flat-to-flat

    # Assign ONE COLOR PER SHAPE TYPE
    shape_colors = {}
    for shape in unique_shapes:
        shape_colors[shape] = _get_random_color()

    # Adjust figure size based on shape size and number of rows
    fig_width = max(6, columns * 1.2)
    fig_height = max(
        4, rows * 1.2 * (shape_size / 0.8)
    )  # Scale height based on shape size

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, columns)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    for r in range(rows):
        for c in range(columns):
            cell = objects[r][c]
            cx = c + 0.5
            cy = rows - r - 0.5

            # Use the same color for all shapes of this type
            face = shape_colors[cell.shape]
            edge = _darken_color(face, factor=0.6)

            if cell.shape == RatioObjectShape.CIRCLE:
                patch = patches.Circle(
                    (cx, cy),
                    radius=base / 2,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.8,
                )
                ax.add_patch(patch)

            elif cell.shape == RatioObjectShape.SQUARE:
                patch = patches.Rectangle(
                    (cx - base / 2, cy - base / 2),
                    base,
                    base,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.8,
                )
                ax.add_patch(patch)

            elif cell.shape == RatioObjectShape.TRIANGLE:
                size = 0.8 * shape_size
                pts = np.array(
                    [
                        [cx, cy + size / 2],
                        [cx - size / 2, cy - size / 2],
                        [cx + size / 2, cy - size / 2],
                    ]
                )
                patch = patches.Polygon(
                    pts.tolist(),
                    closed=True,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.8,
                )
                ax.add_patch(patch)

            elif cell.shape == RatioObjectShape.STAR:
                # Match outer width to 'base' by setting outer radius = base/2
                verts = _star_vertices(
                    cx, cy, outer_r=base / 2, inner_ratio=0.5, points=5
                )
                patch = patches.Polygon(
                    verts,
                    closed=True,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.8,
                    joinstyle="round",
                )
                ax.add_patch(patch)

            elif cell.shape == RatioObjectShape.HEXAGON:
                # flat-to-flat = base  => R = base / √3
                R = base / np.sqrt(3.0)
                patch = patches.RegularPolygon(
                    (cx, cy),
                    numVertices=6,
                    radius=R,
                    orientation=np.pi / 6.0,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.8,
                )
                ax.add_patch(patch)

            else:
                raise ValueError(f"Unsupported shape: {cell.shape}")

    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"ratio_object_array_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
