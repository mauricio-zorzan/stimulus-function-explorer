import math
import random
import time
from enum import Enum
from typing import List, Tuple

import matplotlib.path as mpath
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from content_generators.settings import settings
from matplotlib import patches as mpatches
from pydantic import Field, field_validator


class TriangleMode(str, Enum):
    DRAW_ONLY = "draw_only"
    COMPUTE_AREA = "compute_area"
    COMPUTE_HEIGHT = "compute_height"
    COMPUTE_BASE = "compute_base"


class TriangleDecomposition(StimulusDescription):
    """Minimal schema for triangle-only decomposition/labeling with decimal-only display."""

    title: str = Field("", description="Optional title displayed above the figure")
    units: str = Field("cm", description="Units: cm, m, in, ft")
    # Exactly 3 vertices, each vertex is [x, y]
    vertices: List[List[float]] = Field(
        ..., description="Exactly 3 vertices of the triangle"
    )
    shaded: bool = Field(
        False, description="If true, fill the triangle with a gray shade"
    )
    mode: TriangleMode = Field(
        TriangleMode.DRAW_ONLY, description="Drawing/labeling mode"
    )
    show_missing_placeholder: bool = Field(
        False, description="Show placeholder for unknown dimension"
    )
    placeholder_text: str = Field(
        "?", max_length=4, description="Placeholder text for unknown dimension"
    )

    @field_validator("vertices")
    @classmethod
    def _check_vertices(cls, v: List[List[float]]) -> List[List[float]]:
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError("vertices must contain exactly 3 points")
        for p in v:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError("each vertex must be a [x, y] list of length 2")
        # Enforce: base = bottom horizontal side; apex strictly above base
        tol = 1e-6
        y_vals = [float(p[1]) for p in v]
        pairs = [(0, 1), (1, 2), (0, 2)]
        base_pair = None
        for i, j in pairs:
            if abs(y_vals[i] - y_vals[j]) < tol:
                k = 3 - i - j
                # base y must be the minimum and apex above
                if (
                    y_vals[k] > y_vals[i] + tol
                    and y_vals[i]
                    <= min(y_vals[(3 - k) % 3], y_vals[(3 - k + 1) % 3]) + tol
                ):
                    base_pair = (i, j, k)
                    break
        if base_pair is None:
            raise ValueError(
                "Two vertices must share the same lowest y (horizontal base) and the third must be higher."
            )
        return v

    @field_validator("units")
    @classmethod
    def _check_units(cls, v: str) -> str:
        if not v:
            return "cm"
        if len(v) > 3:
            raise ValueError("Units must be abbreviated to ≤ 3 characters")
        return v


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _project_point_to_line(
    point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float]:
    ax, ay = a
    bx, by = b
    px, py = point
    vx, vy = bx - ax, by - ay
    v_len_sq = vx * vx + vy * vy
    if v_len_sq == 0:
        return a
    t = ((px - ax) * vx + (py - ay) * vy) / v_len_sq
    # Projection on the infinite line; for altitude we want perpendicular to the side segment
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return (proj_x, proj_y)


def _project_point_to_line_with_t(
    point: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[Tuple[float, float], float]:
    ax, ay = a
    bx, by = b
    px, py = point
    vx, vy = bx - ax, by - ay
    v_len_sq = vx * vx + vy * vy
    if v_len_sq == 0:
        return (a, 0.0)
    t = ((px - ax) * vx + (py - ay) * vy) / v_len_sq
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return (proj_x, proj_y), t


def _decimal_label(value: float, units: str) -> str:
    if abs(value - int(value)) < 1e-9:
        return f"{int(round(value))} {units}"
    formatted = f"{value:.1f}".rstrip("0").rstrip(".")
    if "." not in formatted:
        formatted = str(int(float(formatted)))
    return f"{formatted} {units}"


def _contains_point(
    shape: List[Tuple[float, float]], point: Tuple[float, float]
) -> bool:
    return mpath.Path(shape).contains_point(point)


def _draw_clipped_grid(ax, shape: List[Tuple[float, float]]):
    path = mpath.Path(shape)
    min_x = min(p[0] for p in shape)
    max_x = max(p[0] for p in shape)
    min_y = min(p[1] for p in shape)
    max_y = max(p[1] for p in shape)

    for x in range(int(math.floor(min_x)) + 1, int(math.ceil(max_x))):
        vertical = [
            (x, y)
            for y in [min_y + i * 0.02 for i in range(int((max_y - min_y) / 0.02) + 1)]
        ]
        clipped = [p for p in vertical if path.contains_point(p)]
        if clipped:
            ys = [p[1] for p in clipped]
            ax.plot(
                [x, x],
                [min(ys), max(ys)],
                color="black",
                linewidth=1,
                linestyle="-",
                zorder=3,
            )

    for y in range(int(math.floor(min_y)) + 1, int(math.ceil(max_y))):
        horizontal = [
            (x, y)
            for x in [min_x + i * 0.02 for i in range(int((max_x - min_x) / 0.02) + 1)]
        ]
        clipped = [p for p in horizontal if path.contains_point(p)]
        if clipped:
            xs = [p[0] for p in clipped]
            ax.plot(
                [min(xs), max(xs)],
                [y, y],
                color="black",
                linewidth=1,
                linestyle="-",
                zorder=3,
            )


def _double_arrow(
    ax,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    color: str = "black",
    lw: float = 1.5,
):
    ax.annotate(
        "", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="|-|", color=color, lw=lw)
    )


def _draw_right_angle_marker(
    ax,
    foot: Tuple[float, float],
    base_dir_unit: Tuple[float, float],
    alt_dir_unit: Tuple[float, float],
    size: float = 0.3,
    color: str = "darkblue",
    flip_base_leg: bool = False,
):
    # Draw a small right-angle square aligned to the base direction and a base-perpendicular direction
    fx, fy = foot
    # Optionally flip the base leg so the square appears on the left of the altitude
    if flip_base_leg:
        bx, by = -base_dir_unit[0], -base_dir_unit[1]
    else:
        bx, by = base_dir_unit
    # Perpendicular to base
    px, py = -by, bx
    # Ensure perpendicular points toward the altitude direction for consistent orientation
    if px * alt_dir_unit[0] + py * alt_dir_unit[1] < 0:
        px, py = -px, -py
    # Normalize perpendicular (base_dir_unit is already unit-length)
    plen = math.hypot(px, py)
    if plen > 0:
        px, py = px / plen, py / plen
    p1 = (fx, fy)
    p2 = (fx + bx * size, fy + by * size)
    p4 = (fx + px * size, fy + py * size)
    p3 = (p2[0] + px * size, p2[1] + py * size)
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=color,
        linewidth=2.2,
        linestyle="-",
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=6,
    )
    ax.plot(
        [p2[0], p3[0]],
        [p2[1], p3[1]],
        color=color,
        linewidth=2.2,
        linestyle="-",
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=6,
    )
    ax.plot(
        [p1[0], p4[0]],
        [p1[1], p4[1]],
        color=color,
        linewidth=2.2,
        linestyle="-",
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=6,
    )
    # Overlay an explicit square polygon (edges only) to guarantee orientation
    square = mpatches.Polygon(
        [p1, p2, p3, p4],
        closed=True,
        fill=False,
        edgecolor=color,
        linewidth=2.2,
        zorder=7,
    )
    ax.add_patch(square)


@stimulus_function
def create_triangle_decomposition_decimal_only(data: TriangleDecomposition) -> str:
    """Render a single triangle with decimal-only measurement labels and optional modes.

    Behavior:
    - Always expects exactly 3 vertices.
    - Fills the triangle (gray if shaded=True), draws outline, optional clipped gridlines.
    - Chooses the longest side as base; computes altitude from opposite vertex.
    - Decimal-only labels (≤ 1 dp) for base and height.
      * mode=draw_only, compute_area: show both base and height values
      * mode=compute_height: base labeled with value, height shows placeholder if enabled
      * mode=compute_base: height labeled with value, base shows placeholder if enabled
    - Returns the saved image filename.
    """
    # Extract vertices
    v0 = (float(data.vertices[0][0]), float(data.vertices[0][1]))
    v1 = (float(data.vertices[1][0]), float(data.vertices[1][1]))
    v2 = (float(data.vertices[2][0]), float(data.vertices[2][1]))

    # Determine base as bottom horizontal side
    tol = 1e-6
    pts = [v0, v1, v2]
    y_vals = [p[1] for p in pts]
    candidates = [(0, 1, 2), (1, 2, 0), (0, 2, 1)]
    base_i = base_j = apex_idx = -1
    for i, j, k in candidates:
        if abs(y_vals[i] - y_vals[j]) < tol and y_vals[k] > y_vals[i] + tol:
            base_i, base_j, apex_idx = i, j, k
            break
    a = pts[base_i]
    b = pts[base_j]
    if a[0] <= b[0]:
        base_end_a, base_end_b = a, b
    else:
        base_end_a, base_end_b = b, a
    base_len = abs(base_end_b[0] - base_end_a[0])
    apex = pts[apex_idx]

    # Vertical altitude
    base_y = base_end_a[1]
    foot = (apex[0], base_y)
    left_x, right_x = base_end_a[0], base_end_b[0]
    if right_x - left_x < tol:
        t_on_base = 0.0
    else:
        t_on_base = (apex[0] - left_x) / (right_x - left_x)
    height_len = abs(apex[1] - base_y)

    # Figure setup
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")

    shape = [v0, v1, v2, v0]

    # Outline
    ax.plot([p[0] for p in shape], [p[1] for p in shape], color="black")

    # Fill with random color from pastel palette (Gelato Days)
    pastel_palette = [
        "#FFCBE1",  # pink
        "#D6E5BD",  # green
        "#F9E1A8",  # yellow
        "#BCD8EC",  # blue
        "#DCCCEC",  # purple
        "#FFDAB4",  # peach
    ]
    fill_color = random.choice(pastel_palette)
    ax.fill(
        [p[0] for p in [v0, v1, v2]], [p[1] for p in [v0, v1, v2]], color=fill_color
    )

    # Gridlines are not supported for triangle decomposition

    # Build base label content
    base_text = _decimal_label(base_len, data.units)
    height_text = _decimal_label(height_len, data.units)

    # Apply mode overrides: missing length labels become "?" explicitly
    if data.mode == TriangleMode.COMPUTE_HEIGHT:
        height_text = "?"
    if data.mode == TriangleMode.COMPUTE_BASE:
        base_text = "?"

    # Draw measurement arrows and labels
    # For base, place an offset parallel to the base outside the triangle
    # Compute outward normal
    vx, vy = base_end_b[0] - base_end_a[0], base_end_b[1] - base_end_a[1]
    nlen = math.hypot(vx, vy)
    nx, ny = (0.0, -1.0)
    if nlen > 0:
        nx, ny = -vy / nlen, vx / nlen

    # Decide which side of base is outside the triangle
    # Check midpoint on both sides
    midx = (base_end_a[0] + base_end_b[0]) / 2.0
    midy = (base_end_a[1] + base_end_b[1]) / 2.0
    outside_plus = not _contains_point([v0, v1, v2], (midx + nx * 0.5, midy + ny * 0.5))
    outside_minus = not _contains_point(
        [v0, v1, v2], (midx - nx * 0.5, midy - ny * 0.5)
    )
    direction = 1.0 if outside_plus or not outside_minus else -1.0
    # Place the measurement line a few more pixels below the base
    offset = max(0.2, min(0.5, 0.06 * base_len))
    base_p1 = (
        base_end_a[0] + direction * nx * offset,
        base_end_a[1] + direction * ny * offset,
    )
    base_p2 = (
        base_end_b[0] + direction * nx * offset,
        base_end_b[1] + direction * ny * offset,
    )
    _double_arrow(ax, base_p1, base_p2, color="black", lw=1.5)
    # Place base label a few pixels further below than the measurement line
    base_mid_x = (base_p1[0] + base_p2[0]) / 2.0
    base_mid_y = (base_p1[1] + base_p2[1]) / 2.0
    base_text_dx = direction * nx * 0.3
    base_text_dy = direction * ny * 0.3
    ax.text(
        base_mid_x + base_text_dx,
        base_mid_y + base_text_dy,
        base_text,
        fontsize=14,
        color="black",
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        ),
    )

    # Height indicator along the altitude from apex to foot (dashed; with right-angle marker)
    height_p1 = apex
    height_p2 = foot
    # Right triangle if apex x aligns with either base endpoint x
    is_right_triangle = (
        abs(apex[0] - base_end_a[0]) < 1e-6 or abs(apex[0] - base_end_b[0]) < 1e-6
    )
    if not is_right_triangle:
        ax.plot(
            [height_p1[0], height_p2[0]],
            [height_p1[1], height_p2[1]],
            color="darkblue",
            linestyle="--",
            linewidth=2,
        )

    # Right-angle marker at the foot
    # Base unit vector and altitude unit vector
    if nlen > 0:
        base_unit = (vx / nlen, vy / nlen)
    else:
        base_unit = (1.0, 0.0)
    if height_len > 0:
        alt_unit = (
            (height_p1[0] - height_p2[0]) / height_len,
            (height_p1[1] - height_p2[1]) / height_len,
        )
    else:
        alt_unit = (0.0, 1.0)
    # If apex is to the far right of the base or exactly equals right base x (right triangle),
    # draw the square on the left side of the altitude
    flip_leg = apex[0] >= max(base_end_a[0], base_end_b[0])
    _draw_right_angle_marker(
        ax, height_p2, base_unit, alt_unit, size=0.25, flip_base_leg=flip_leg
    )

    # If vertical foot is outside base segment, draw dashed horizontal connector to base
    if t_on_base < 0.0 or t_on_base > 1.0:
        clamp_x = min(max(foot[0], left_x), right_x)
        ax.plot(
            [foot[0], clamp_x],
            [base_y, base_y],
            color="darkblue",
            linestyle="--",
            linewidth=2,
        )

    if is_right_triangle:
        # Draw an external vertical measurement line like the base rule
        # If apex x equals the lowest base x, draw on the left; otherwise draw on the right
        left_base_x = min(base_end_a[0], base_end_b[0])
        out_dir = -1.0 if abs(apex[0] - left_base_x) < 1e-6 else 1.0
        vert_offset = max(0.2, min(0.5, 0.06 * base_len))
        xline = apex[0] + out_dir * vert_offset
        v_p1 = (xline, base_y)
        v_p2 = (xline, apex[1])
        _double_arrow(ax, v_p1, v_p2, color="black", lw=1.5)
        vmx = (v_p1[0] + v_p2[0]) / 2.0
        vmy = (v_p1[1] + v_p2[1]) / 2.0
        ax.text(
            vmx,
            vmy,
            height_text,
            fontsize=14,
            color="black",
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            ),
        )
    else:
        # Place height text centered on the altitude (no lateral offset)
        hx = (height_p1[0] + height_p2[0]) / 2.0
        hy = (height_p1[1] + height_p2[1]) / 2.0
        ax.text(
            hx,
            hy,
            height_text,
            fontsize=14,
            color="black",
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            ),
        )

    # Axis limits with buffer
    xs = [p[0] for p in [v0, v1, v2]]
    ys = [p[1] for p in [v0, v1, v2]]
    buffer = 0.8
    ax.set_xlim(min(xs) - buffer, max(xs) + buffer)
    ax.set_ylim(min(ys) - buffer, max(ys) + buffer)

    # Save
    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/triangle_decomp_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
