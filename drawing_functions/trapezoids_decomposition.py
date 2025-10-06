import random
import time
from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from content_generators.settings import settings
from pydantic import Field, field_validator


class TrapezoidMode(str, Enum):
    DRAW_ONLY = "draw_only"
    COMPUTE_AREA = "compute_area"
    COMPUTE_HEIGHT = "compute_height"
    COMPUTE_BASE = "compute_base"


class TrapezoidDecomposition(StimulusDescription):
    """Input schema dedicated to trapezoid-only rendering.

    Vertices are v0, v1 (top, left->right) and v2, v3 (bottom, left->right).
    Required properties:
      - Exactly 4 vertices.
      - Top (v0,v1) and bottom (v2,v3) are horizontal and parallel.
      - Bottom is strictly longer than top: |v3_x - v2_x| > |v1_x - v0_x|.
      - The interval [v0_x, v1_x] is contained within [v2_x, v3_x].
      - No titles; no gridlines supported.
    """

    units: str = Field("cm", description="Units: cm, m, in, ft (≤3 chars)")
    vertices: List[List[float]] = Field(
        ..., description="[[v0x,v0y],[v1x,v1y],[v2x,v2y],[v3x,v3y]]"
    )
    shaded: bool = Field(False, description="If true, fills with pastel color")
    mode: TrapezoidMode = Field(
        TrapezoidMode.DRAW_ONLY, description="Drawing/labeling mode"
    )
    show_missing_placeholder: bool = Field(
        False, description="Show placeholder for hidden dimension"
    )
    placeholder_text: str = Field("?", max_length=4, description="Placeholder text")

    @field_validator("units")
    @classmethod
    def _check_units(cls, v: str) -> str:
        if not v:
            return "cm"
        if len(v) > 3:
            raise ValueError("Units must be abbreviated to ≤ 3 characters")
        return v

    @field_validator("vertices")
    @classmethod
    def _check_vertices(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4:
            raise ValueError("Trapezoid requires exactly 4 vertices [v0,v1,v2,v3]")
        for p in v:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError("Each vertex must be a [x,y] list")

        v0, v1, v2, v3 = v
        # Ensure top and bottom are horizontal
        if abs(v0[1] - v1[1]) > 1e-6 or abs(v2[1] - v3[1]) > 1e-6:
            raise ValueError("Top and bottom sides must be horizontal")
        # Ensure bottom is below top
        if not (v2[1] < v0[1] and v3[1] < v1[1]):
            raise ValueError("Bottom must be lower than top")
        # Order by x left->right for both sides
        if v0[0] > v1[0]:
            v0, v1 = v1, v0
        if v2[0] > v3[0]:
            v2, v3 = v3, v2
        top_len = abs(v1[0] - v0[0])
        bot_len = abs(v3[0] - v2[0])
        if not (bot_len > top_len):
            raise ValueError("Bottom side must be strictly longer than the top side")
        # Containment rule
        if not (
            v2[0] <= v0[0] + 1e-6 and v1[0] <= v3[0] + 1e-6 and v0[0] >= v2[0] - 1e-6
        ):
            raise ValueError("[v0_x, v1_x] must be contained within [v2_x, v3_x]")
        # Save possibly reordered vertices
        return [v0, v1, v2, v3]


def _decimal_label(value: float, units: str) -> str:
    iv = int(round(value))
    if abs(value - iv) < 1e-9:
        return f"{iv} {units}"
    return f"{value:.1f}".rstrip("0").rstrip(".") + f" {units}"


def _draw_right_angle_marker(
    ax,
    foot: Tuple[float, float],
    size: float = 0.25,
    color: str = "darkblue",
):
    fx, fy = foot
    bx, by = 1.0, 0.0
    px, py = 0.0, 1.0
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
        zorder=6,
    )
    ax.plot(
        [p2[0], p3[0]],
        [p2[1], p3[1]],
        color=color,
        linewidth=2.2,
        linestyle="-",
        zorder=6,
    )
    ax.plot(
        [p1[0], p4[0]],
        [p1[1], p4[1]],
        color=color,
        linewidth=2.2,
        linestyle="-",
        zorder=6,
    )
    from matplotlib import patches as mpatches

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
def create_trapezoid_decomposition_decimal_only(data: TrapezoidDecomposition) -> str:
    """Draw a trapezoid with decimal-only labels and pastel fill.

    - No titles and no gridlines.
    - Bottom base labeled below; top base labeled above.
    - Colors chosen from the triangle pastel palette.
    - Self-contained; no shared internal renderer.
    """

    v0 = (float(data.vertices[0][0]), float(data.vertices[0][1]))
    v1 = (float(data.vertices[1][0]), float(data.vertices[1][1]))
    v2 = (float(data.vertices[2][0]), float(data.vertices[2][1]))
    v3 = (float(data.vertices[3][0]), float(data.vertices[3][1]))

    # Derived dimensions
    top_len = abs(v1[0] - v0[0])
    bot_len = abs(v3[0] - v2[0])
    units = data.units

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.axis("off")

    # Outline
    poly = [v0, v1, v3, v2, v0]
    ax.plot([p[0] for p in poly], [p[1] for p in poly], color="black")

    # Pastel palette fill
    pastel_palette = [
        "#FFCBE1",
        "#D6E5BD",
        "#F9E1A8",
        "#BCD8EC",
        "#DCCCEC",
        "#FFDAB4",
    ]
    fill_color = random.choice(pastel_palette)
    ax.fill(
        [v0[0], v1[0], v3[0], v2[0]], [v0[1], v1[1], v3[1], v2[1]], color=fill_color
    )

    # Base rulers: draw measurement lines slightly outside
    # Bottom ruler
    offset = max(0.2, min(0.5, 0.06 * bot_len))
    b_p1 = (v2[0], v2[1] - offset)
    b_p2 = (v3[0], v3[1] - offset)
    ax.annotate(
        "",
        xy=b_p2,
        xytext=b_p1,
        arrowprops=dict(arrowstyle="|-|", color="black", lw=1.5),
    )
    b_mid = ((b_p1[0] + b_p2[0]) / 2.0, (b_p1[1] + b_p2[1]) / 2.0 - 0.3)
    bottom_label_text = _decimal_label(bot_len, units)
    if data.mode == TrapezoidMode.COMPUTE_BASE:
        bottom_label_text = "?"
    ax.text(
        b_mid[0],
        b_mid[1],
        bottom_label_text,
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

    # Top ruler
    toffset = max(0.2, min(0.5, 0.06 * top_len))
    t_p1 = (v0[0], v0[1] + toffset)
    t_p2 = (v1[0], v1[1] + toffset)
    ax.annotate(
        "",
        xy=t_p2,
        xytext=t_p1,
        arrowprops=dict(arrowstyle="|-|", color="black", lw=1.5),
    )
    t_mid = ((t_p1[0] + t_p2[0]) / 2.0, (t_p1[1] + t_p2[1]) / 2.0 + 0.3)
    top_label_text = _decimal_label(top_len, units)
    ax.text(
        t_mid[0],
        t_mid[1],
        top_label_text,
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

    # Altitude/height rendering
    base_y = v2[1]
    altitude_top = (v0[0], v0[1])
    altitude_foot = (v0[0], base_y)

    # Special case: left side vertical (v0_x == v2_x). No dashed altitude.
    if abs(v0[0] - v2[0]) < 1e-6:
        # Draw external vertical measurement line to the LEFT of the trapezoid
        v_offset = max(0.2, min(0.5, 0.06 * bot_len))
        xline = v2[0] - v_offset
        v_p1 = (xline, base_y)
        v_p2 = (xline, v0[1])
        ax.annotate(
            "",
            xy=v_p2,
            xytext=v_p1,
            arrowprops=dict(arrowstyle="|-|", color="black", lw=1.5),
        )
        vmx = (v_p1[0] + v_p2[0]) / 2.0
        vmy = (v_p1[1] + v_p2[1]) / 2.0
        height_label_text = _decimal_label(abs(v0[1] - base_y), units)
        if data.mode == TrapezoidMode.COMPUTE_HEIGHT:
            height_label_text = "?"
        ax.text(
            vmx - 0.15,
            vmy,
            height_label_text,
            fontsize=14,
            color="black",
            ha="right",
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
        # Add right-angle marker at the interior bottom-left corner
        _draw_right_angle_marker(ax, (v2[0], base_y), size=0.25, color="darkblue")
    else:
        # Default: dashed altitude and right-angle marker at bottom-right
        ax.plot(
            [altitude_top[0], altitude_foot[0]],
            [altitude_top[1], altitude_foot[1]],
            color="darkblue",
            linestyle="--",
            linewidth=2,
        )
        _draw_right_angle_marker(ax, altitude_foot, size=0.25, color="darkblue")
        h_len = abs(v0[1] - base_y)
        hx = altitude_top[0]
        hy = (altitude_top[1] + altitude_foot[1]) / 2.0
        height_label_text = _decimal_label(h_len, units)
        if data.mode == TrapezoidMode.COMPUTE_HEIGHT:
            height_label_text = "?"
        ax.text(
            hx,
            hy,
            height_label_text,
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

    # Axis limits
    xs = [v0[0], v1[0], v2[0], v3[0]]
    ys = [v0[1], v1[1], v2[1], v3[1]]
    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/trapezoid_decomp_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name
