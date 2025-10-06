# content_generators/additional_content/stimulus_image/drawing_functions/equation_tape_diagram.py
# IMPORTANT: do NOT add: from __future__ import annotations


import json
import os
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.equation_tape_diagram import (
    AdditionDiagram,
    ComparisonDiagram,
    DivisionDiagram,
    EqualGroupsDiagram,
    EquationTapeDiagram,
    EquationTapeDiagramWrapper,
    FractionStripNew,
    MultiplicationDiagram,
    SubtractionDiagram,
)
from content_generators.settings import settings
from matplotlib.patches import FancyBboxPatch

# -------------------- visuals --------------------
PALETTES = [
    ("#B7E4C7", "#2D6A4F", "#6C757D"),
    ("#CDB4DB", "#5A189A", "#6C757D"),
    ("#FEC89A", "#CC5803", "#6C757D"),
    ("#A8D8FF", "#1D4ED8", "#6C757D"),
    ("#FFE9A8", "#A16207", "#6C757D"),
]


def _pick_palette():
    i = np.random.randint(0, len(PALETTES))
    return PALETTES[i]


def _bar(ax, x, y, w, h, fill, edge, label, *, fs=18):
    """Rectangle bar with centered label."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="square,pad=0",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=fill,
        antialiased=True,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        str(label),
        ha="center",
        va="center",
        fontsize=fs,
        color="#1f1f20",
        zorder=3,
    )


def _cap(ax, x0, x1, y, text, accent, *, fs=16, z=6):
    """
    Draw a 'cap' line with small end ticks. y is the y-position of the line
    (either midline of the bar, or slightly above it). High z-order so it sits
    in front of bars and remains visible.
    """
    if x1 < x0:
        x0, x1 = x1, x0
    tick = 0.05
    ax.plot(
        [x0, x1], [y, y], color=accent, linewidth=5, zorder=z, solid_capstyle="butt"
    )
    ax.plot([x0, x0], [y - tick, y + tick], color=accent, linewidth=5, zorder=z)
    ax.plot([x1, x1], [y - tick, y + tick], color=accent, linewidth=5, zorder=z)
    if text not in (None, ""):
        # put text under the line so it never obscures bar labels
        _smart_cap_label(ax, x0, x1, y - tick - 0.03, str(text), base_fs=fs)


# --- utils: data<->pixel helpers ---------------------------------------------
def _px_to_data_x(ax, px: float) -> float:
    inv = ax.transData.inverted()
    x0, _ = inv.transform((0, 0))
    x1, _ = inv.transform((px, 0))
    return x1 - x0


def _data_width_px(ax, x0: float, x1: float, y: float) -> float:
    (X0, _), (X1, _) = ax.transData.transform([(x0, y), (x1, y)])
    return abs(X1 - X0)


# --- smart cap label ----------------------------------------------------------
def _smart_cap_label(
    ax,
    x0: float,
    x1: float,
    y: float,
    text: str,
    base_fs: int = 12,
    min_fs: int = 8,
    pad_px: int = 8,
    outside_offset_px: int = 6,
):
    """
    Center text inside [x0,x1] at y if it fits; else shrink; else place outside.
    Only triggers on cramped cases (e.g., '.25' in a very small pill).
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure renderer exists
    renderer = fig.canvas.get_renderer()

    cap_px = _data_width_px(ax, x0, x1, y)

    # Measure at base font size (invisible throwaway text)
    t = ax.text(0, 0, text, fontsize=base_fs, transform=ax.transData, visible=False)
    fig.canvas.draw()
    txt_px = t.get_window_extent(renderer=renderer).width + pad_px
    t.remove()

    if txt_px <= cap_px:
        ax.text((x0 + x1) / 2, y, text, fontsize=base_fs, ha="center", va="center")
        return

    # Try a scaled-down font size (but not below min_fs)
    scale = max(cap_px / max(txt_px, 1), 0.0)
    fs = max(int(base_fs * scale), min_fs)

    # Re-measure at shrunk size
    t = ax.text(0, 0, text, fontsize=fs, transform=ax.transData, visible=False)
    fig.canvas.draw()
    txt_px2 = t.get_window_extent(renderer=renderer).width + pad_px
    t.remove()

    if txt_px2 <= cap_px:
        ax.text((x0 + x1) / 2, y, text, fontsize=fs, ha="center", va="center")
        return

    # Final fallback: put it just outside the pill to the right
    dx = _px_to_data_x(ax, outside_offset_px)
    ax.text(x1 + dx, y, text, fontsize=base_fs, ha="left", va="center")


def _render_single_diagram(ax, stimulus, fill_color, edge_color, accent_color):
    """Helper function to render a single diagram on a given axis."""
    # --- layout ---
    H = 0.16
    LEFT = 0.08
    RIGHT = 0.92
    TOP_Y = 0.70
    BOT_Y = 0.40

    # cap y-locations and a little horizontal gap so the line doesn't
    # touch rounded corners, Keep only the two positions we actually use.
    # CAP_Y_MID = BOT_Y + H * 0.5
    # CAP_Y_ABOVE = BOT_Y + H + 0.022

    CAP_Y_BELOW = BOT_Y - 0.12  # Further below for addition diagrams
    CAP_Y_BESIDE = BOT_Y + H * 0.5  # Beside the box for subtraction diagrams
    CAP_GAP_X = (RIGHT - LEFT) * 0.012
    SUBTRACTION_GAP_X = (RIGHT - LEFT) * 0.025  # Larger gap for subtraction diagrams

    def width_for(val: float, whole: float) -> float:
        span = RIGHT - LEFT
        return span * (max(val, 0.0) / float(max(whole, 1.0)))

    def ensure_precise_widths(
        part1: float, part2: float, whole: float
    ) -> tuple[float, float]:
        """Ensure that part widths add up exactly to the total width to prevent overlap."""
        total_width = width_for(whole, whole)
        w1 = width_for(part1, whole)
        w2 = width_for(part2, whole)

        # If there's a small difference due to floating point precision, adjust w2
        if abs((w1 + w2) - total_width) > 0.001:
            w2 = total_width - w1

        return w1, w2

    def ensure_precise_equal_groups(n: int, group_size: float, whole: float) -> float:
        """Ensure that n equal group widths add up exactly to the total width to prevent overlap."""
        total_width = width_for(whole, whole)
        seg_w = width_for(group_size, whole)

        # If there's a small difference due to floating point precision, adjust seg_w
        if abs((n * seg_w) - total_width) > 0.001:
            seg_w = total_width / n

        return seg_w

    # ---------------- Addition ----------------
    if isinstance(stimulus, AdditionDiagram):
        if stimulus.unknown == "total":
            if stimulus.part1 is None or stimulus.part2 is None:
                raise ValueError(
                    "part1 and part2 must be provided when unknown is 'total'"
                )
            # Type assertion after validation
            part1 = float(stimulus.part1)
            part2 = float(stimulus.part2)
            whole = part1 + part2
        else:
            if stimulus.total is None:
                raise ValueError("total must be provided when unknown is not 'total'")
            whole = float(stimulus.total)

        # top (total) bar
        _bar(
            ax,
            LEFT,
            TOP_Y,
            width_for(whole, whole),
            H,
            fill_color,
            edge_color,
            stimulus.total if stimulus.unknown != "total" else stimulus.variable_symbol,
        )

        # bottom row
        if stimulus.unknown == "total":
            # both parts known; cap goes BELOW both bars to show total
            w1, w2 = ensure_precise_widths(part1, part2, whole)
            # Add a small gap between bars for visual clarity
            gap = (RIGHT - LEFT) * 0.010  # Small gap (0.5% of total width)

            # Ensure bars are placed side by side with a small gap
            _bar(ax, LEFT, BOT_Y, w1, H, fill_color, edge_color, part1)
            _bar(ax, LEFT + w1 + gap, BOT_Y, w2, H, fill_color, edge_color, part2)
            # Use the total width for the cap to ensure perfect alignment
            total_width = width_for(whole, whole)
            _cap(
                ax,
                LEFT + CAP_GAP_X,
                LEFT + total_width - CAP_GAP_X,
                CAP_Y_BELOW,  # Well below the bottom bar to avoid overlap
                stimulus.variable_symbol,
                accent_color,
            )
        else:
            # one part known, the other is the variable; cap inline with midline
            known = stimulus.part1 if stimulus.unknown == "part2" else stimulus.part2
            w_known = width_for(known, whole)
            _bar(ax, LEFT, BOT_Y, w_known, H, fill_color, edge_color, known)
            _cap(
                ax,
                LEFT + w_known + SUBTRACTION_GAP_X,
                LEFT + width_for(whole, whole) - CAP_GAP_X,
                CAP_Y_BESIDE,  # Inline with the box (midline), like subtraction
                stimulus.variable_symbol,
                accent_color,
            )

    # ---------------- Subtraction ----------------
    elif isinstance(stimulus, SubtractionDiagram):
        whole = (
            stimulus.start
            if stimulus.unknown != "start"
            else (stimulus.result + stimulus.change)
        )

        # top (start) bar
        _bar(
            ax,
            LEFT,
            TOP_Y,
            width_for(whole, whole),
            H,
            fill_color,
            edge_color,
            stimulus.start if stimulus.unknown != "start" else stimulus.variable_symbol,
        )

        # bottom row (always cap on the midline, padded horizontally)
        if stimulus.unknown == "result":
            w_change = width_for(stimulus.change, whole)
            _cap(
                ax,
                LEFT + SUBTRACTION_GAP_X,
                LEFT + width_for(whole, whole) - w_change - SUBTRACTION_GAP_X,
                CAP_Y_BESIDE,  # Beside the box for subtraction diagrams
                stimulus.variable_symbol,
                accent_color,
            )
            _bar(
                ax,
                LEFT + width_for(whole, whole) - w_change,
                BOT_Y,
                w_change,
                H,
                fill_color,
                edge_color,
                stimulus.change,
            )
        elif stimulus.unknown == "change":
            w_result = width_for(stimulus.result, whole)
            _bar(ax, LEFT, BOT_Y, w_result, H, fill_color, edge_color, stimulus.result)
            _cap(
                ax,
                LEFT + w_result + SUBTRACTION_GAP_X,
                LEFT + width_for(whole, whole) - CAP_GAP_X,
                CAP_Y_BESIDE,  # Beside the box for subtraction diagrams
                stimulus.variable_symbol,
                accent_color,
            )
        else:  # unknown start
            w_result = width_for(stimulus.result, whole)
            _bar(ax, LEFT, BOT_Y, w_result, H, fill_color, edge_color, stimulus.result)
            _cap(
                ax,
                LEFT + w_result + SUBTRACTION_GAP_X,
                LEFT + w_result + width_for(stimulus.change, whole) - CAP_GAP_X,
                CAP_Y_BESIDE,  # Beside the box for subtraction diagrams
                stimulus.change,
                accent_color,
            )

    # ---------------- Equal Groups ----------------
    elif isinstance(stimulus, EqualGroupsDiagram):
        if stimulus.unknown == "groups":
            if stimulus.total is None or stimulus.group_size in (None, 0):
                raise ValueError(
                    "When unknown='groups', total and non-zero group_size are required"
                )
            n = int(stimulus.total / stimulus.group_size)
            whole = stimulus.total
            top_label = stimulus.total
        elif stimulus.unknown == "total":
            n = int(stimulus.groups)
            whole = n * stimulus.group_size
            top_label = stimulus.variable_symbol
        else:
            n = int(stimulus.groups)
            whole = stimulus.total
            top_label = stimulus.total

        # top bar
        _bar(
            ax,
            LEFT,
            TOP_Y,
            width_for(whole, whole),
            H,
            fill_color,
            edge_color,
            top_label,
        )

        # bottom: n equal groups with appropriate cap placement
        if stimulus.unknown == "total":
            seg_w = ensure_precise_equal_groups(int(n), stimulus.group_size, whole)
            for i in range(int(n)):
                _bar(
                    ax,
                    LEFT + i * seg_w,
                    BOT_Y,
                    seg_w,
                    H,
                    fill_color,
                    edge_color,
                    stimulus.group_size,
                )
            _cap(
                ax,
                LEFT + CAP_GAP_X,
                LEFT + int(n) * seg_w - CAP_GAP_X,
                CAP_Y_BELOW,  # below the bars to avoid overlap
                stimulus.variable_symbol,
                accent_color,
            )
        else:
            seg_val = whole / n
            seg_w = ensure_precise_equal_groups(int(n), seg_val, whole)
            for i in range(int(n)):
                _bar(
                    ax,
                    LEFT + i * seg_w,
                    BOT_Y,
                    seg_w,
                    H,
                    fill_color,
                    edge_color,
                    stimulus.variable_symbol,
                )
            _cap(
                ax,
                LEFT + CAP_GAP_X,
                LEFT + int(n) * seg_w - CAP_GAP_X,
                CAP_Y_BELOW,  # below the bars to avoid overlap
                "",
                accent_color,
            )

    # ---------------- Fraction Strip New (Simple) ----------------
    elif isinstance(stimulus, FractionStripNew):
        # Simple layout: equal parts with bottom total line
        # Center the diagram vertically by using a single row at middle position
        CENTER_Y = 0.55  # Center the single row of boxes

        # Calculate box dimensions with gaps
        n = stimulus.total_parts
        gap = (RIGHT - LEFT) * 0.01
        total_width = RIGHT - LEFT
        total_gap_width = (n - 1) * gap
        width_for_boxes = total_width - total_gap_width
        seg_w = width_for_boxes / n

        # Draw the equal parts (boxes)
        for i in range(n):
            _bar(
                ax,
                LEFT + i * (seg_w + gap),
                CENTER_Y,
                seg_w,
                H,
                fill_color,
                edge_color,
                stimulus.part_value,
            )

        # Draw the bottom total line with label
        # Make the line slightly wider than the boxes to encapsulate them
        total_boxes_width = n * seg_w + (n - 1) * gap
        line_extension = total_boxes_width * 0.005  # Extend line by 2% on each side
        _cap(
            ax,
            LEFT - line_extension,
            LEFT + total_boxes_width + line_extension,
            CENTER_Y - 0.12,  # Below the boxes
            stimulus.total_value,
            accent_color,
        )

    # ---------------- Multiplication (Simplified) ----------------
    elif isinstance(stimulus, MultiplicationDiagram):
        # Convert to simple fraction strip format
        if stimulus.unknown == "product":
            # factor × factor2 = product (unknown product)
            n = int(stimulus.factor)  # number of parts
            part_value = str(stimulus.factor2)  # what goes in each part
            total_value = stimulus.variable_symbol  # total is unknown
        elif stimulus.unknown == "factor":
            # Handle both cases: factor unknown or factor2 unknown
            if stimulus.factor is None:
                # factor is unknown, factor2 and product are known
                n = int(
                    stimulus.product / stimulus.factor2
                )  # calculate number of parts
                part_value = str(stimulus.factor2)  # what goes in each part
            else:
                # factor2 is unknown, factor and product are known
                n = int(stimulus.factor)  # number of parts
                part_value = (
                    stimulus.variable_symbol
                )  # what goes in each part (unknown)
            total_value = str(stimulus.product)  # total is known
        else:  # unknown factor2
            n = int(stimulus.factor)  # number of parts
            part_value = (
                stimulus.variable_symbol
            )  # what goes in each part (unknown factor2)
            total_value = str(stimulus.product)  # total is known

        # Simple layout: equal parts with bottom total line
        CENTER_Y = 0.55  # Center the single row of boxes

        # Calculate box dimensions with gaps
        gap = (RIGHT - LEFT) * 0.01
        total_width = RIGHT - LEFT
        total_gap_width = (n - 1) * gap
        width_for_boxes = total_width - total_gap_width
        seg_w = width_for_boxes / n

        # Draw the equal parts (boxes)
        for i in range(n):
            _bar(
                ax,
                LEFT + i * (seg_w + gap),
                CENTER_Y,
                seg_w,
                H,
                fill_color,
                edge_color,
                part_value,
            )

        # Draw the bottom total line with label
        # Make the line slightly wider than the boxes to encapsulate them
        total_boxes_width = n * seg_w + (n - 1) * gap
        line_extension = total_boxes_width * 0.02  # Extend line by 2% on each side
        _cap(
            ax,
            LEFT - line_extension,
            LEFT + total_boxes_width + line_extension,
            CENTER_Y - 0.12,  # Below the boxes
            total_value,
            accent_color,
        )

    # ---------------- Division (Simplified) ----------------
    elif isinstance(stimulus, DivisionDiagram):
        # Convert to simple fraction strip format
        if stimulus.unknown == "quotient":
            # dividend ÷ divisor = quotient (unknown quotient)
            n = int(stimulus.divisor)  # number of parts (divisor)
            part_value = (
                stimulus.variable_symbol
            )  # what goes in each part (unknown quotient)
            total_value = str(stimulus.dividend)  # total is known (dividend)
        elif stimulus.unknown == "divisor":
            # dividend ÷ divisor = quotient (unknown divisor)
            n = int(stimulus.dividend / stimulus.quotient)  # calculate number of parts
            part_value = str(stimulus.quotient)  # what goes in each part
            total_value = str(stimulus.dividend)  # total is known (dividend)
        else:  # unknown dividend
            n = int(stimulus.divisor)  # number of parts
            part_value = str(stimulus.quotient)  # what goes in each part
            total_value = stimulus.variable_symbol  # total is unknown (dividend)

        # Simple layout: equal parts with bottom total line
        CENTER_Y = 0.55  # Center the single row of boxes

        # Calculate box dimensions with gaps
        gap = (RIGHT - LEFT) * 0.01
        total_width = RIGHT - LEFT
        total_gap_width = (n - 1) * gap
        width_for_boxes = total_width - total_gap_width
        seg_w = width_for_boxes / n

        # Draw the equal parts (boxes)
        for i in range(n):
            _bar(
                ax,
                LEFT + i * (seg_w + gap),
                CENTER_Y,
                seg_w,
                H,
                fill_color,
                edge_color,
                part_value,
            )

        # Draw the bottom total line with label
        # Make the line slightly wider than the boxes to encapsulate them
        total_boxes_width = n * seg_w + (n - 1) * gap
        line_extension = total_boxes_width * 0.02  # Extend line by 2% on each side
        _cap(
            ax,
            LEFT - line_extension,
            LEFT + total_boxes_width + line_extension,
            CENTER_Y - 0.12,  # Below the boxes
            total_value,
            accent_color,
        )

    else:
        raise ValueError("Unsupported stimulus description for comparison")


def _coerce(obj):
    # Already top-level instances
    if isinstance(obj, (EquationTapeDiagram, EquationTapeDiagramWrapper)):
        return obj.root
    # JSON string -> dict
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception as e:
            raise TypeError(
                "stimulus must be a dict or JSON string matching the schema"
            ) from e
    # Dict -> validate top-level -> unwrap
    if isinstance(obj, dict):
        top = EquationTapeDiagramWrapper.model_validate({"root": obj})
        return top.root
    # Concrete submodel already
    return obj


@stimulus_function
def draw_equation_tape_diagram(stimulus: EquationTapeDiagram) -> str:
    # keep your original rendering logic intact; just normalize first
    stimulus = _coerce(stimulus)

    # --- layout ---
    H = 0.16
    LEFT = 0.08
    RIGHT = 0.92
    TOP_Y = 0.70
    BOT_Y = 0.40

    # cap y-locations and a little horizontal gap so the line doesn’t
    # touch rounded corners, Keep only the two positions we actually use.
    # CAP_Y_MID = BOT_Y + H * 0.5
    # CAP_Y_ABOVE = BOT_Y + H + 0.022

    CAP_Y_BELOW = BOT_Y - 0.12  # Further below for addition diagrams
    CAP_Y_BESIDE = BOT_Y + H * 0.5  # Beside the box for subtraction diagrams
    CAP_GAP_X = (RIGHT - LEFT) * 0.012
    SUBTRACTION_GAP_X = (RIGHT - LEFT) * 0.025  # Larger gap for subtraction diagrams

    FILL, EDGE, ACCENT = _pick_palette()

    fig = plt.figure(figsize=(8.2, 4.2))
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def width_for(val: float, whole: float) -> float:
        span = RIGHT - LEFT
        return span * (max(val, 0.0) / float(max(whole, 1.0)))

    def ensure_precise_widths(
        part1: float, part2: float, whole: float
    ) -> tuple[float, float]:
        """Ensure that part widths add up exactly to the total width to prevent overlap."""
        total_width = width_for(whole, whole)
        w1 = width_for(part1, whole)
        w2 = width_for(part2, whole)

        # If there's a small difference due to floating point precision, adjust w2
        if abs((w1 + w2) - total_width) > 0.001:
            w2 = total_width - w1

        return w1, w2

    def ensure_precise_equal_groups(n: int, group_size: float, whole: float) -> float:
        """Ensure that n equal group widths add up exactly to the total width to prevent overlap."""
        total_width = width_for(whole, whole)
        seg_w = width_for(group_size, whole)

        # If there's a small difference due to floating point precision, adjust seg_w
        if abs((n * seg_w) - total_width) > 0.001:
            seg_w = total_width / n

        return seg_w

    # ---------------- Addition ----------------
    if isinstance(stimulus, AdditionDiagram):
        if stimulus.unknown == "total":
            if stimulus.part1 is None or stimulus.part2 is None:
                raise ValueError(
                    "part1 and part2 must be provided when unknown is 'total'"
                )
            # Type assertion after validation
            part1 = float(stimulus.part1)
            part2 = float(stimulus.part2)
            whole = part1 + part2
        else:
            if stimulus.total is None:
                raise ValueError("total must be provided when unknown is not 'total'")
            whole = float(stimulus.total)

        # top (total) bar
        _bar(
            ax,
            LEFT,
            TOP_Y,
            width_for(whole, whole),
            H,
            FILL,
            EDGE,
            stimulus.total if stimulus.unknown != "total" else stimulus.variable_symbol,
        )

        # bottom row
        if stimulus.unknown == "total":
            # both parts known; cap goes BELOW both bars to show total
            w1, w2 = ensure_precise_widths(part1, part2, whole)
            # Add a small gap between bars for visual clarity
            gap = (RIGHT - LEFT) * 0.010  # Small gap (0.5% of total width)

            # Ensure bars are placed side by side with a small gap
            _bar(ax, LEFT, BOT_Y, w1, H, FILL, EDGE, part1)
            _bar(ax, LEFT + w1 + gap, BOT_Y, w2, H, FILL, EDGE, part2)
            # Use the total width for the cap to ensure perfect alignment
            total_width = width_for(whole, whole)
            _cap(
                ax,
                LEFT + CAP_GAP_X,
                LEFT + total_width - CAP_GAP_X,
                CAP_Y_BELOW,  # Well below the bottom bar to avoid overlap
                stimulus.variable_symbol,
                ACCENT,
            )
        else:
            # one part known, the other is the variable; cap inline with midline
            known = stimulus.part1 if stimulus.unknown == "part2" else stimulus.part2
            w_known = width_for(known, whole)
            _bar(ax, LEFT, BOT_Y, w_known, H, FILL, EDGE, known)
            _cap(
                ax,
                LEFT + w_known + SUBTRACTION_GAP_X,
                LEFT + width_for(whole, whole) - CAP_GAP_X,
                CAP_Y_BESIDE,  # Inline with the box (midline), like subtraction
                stimulus.variable_symbol,
                ACCENT,
            )

    # ---------------- Subtraction ----------------
    elif isinstance(stimulus, SubtractionDiagram):
        whole = (
            stimulus.start
            if stimulus.unknown != "start"
            else (stimulus.result + stimulus.change)
        )

        # top (start) bar
        _bar(
            ax,
            LEFT,
            TOP_Y,
            width_for(whole, whole),
            H,
            FILL,
            EDGE,
            stimulus.start if stimulus.unknown != "start" else stimulus.variable_symbol,
        )

        # bottom row (always cap on the midline, padded horizontally)
        if stimulus.unknown == "result":
            w_change = width_for(stimulus.change, whole)
            _cap(
                ax,
                LEFT + SUBTRACTION_GAP_X,
                LEFT + width_for(whole, whole) - w_change - SUBTRACTION_GAP_X,
                CAP_Y_BESIDE,  # Beside the box for subtraction diagrams
                stimulus.variable_symbol,
                ACCENT,
            )
            _bar(
                ax,
                LEFT + width_for(whole, whole) - w_change,
                BOT_Y,
                w_change,
                H,
                FILL,
                EDGE,
                stimulus.change,
            )
        elif stimulus.unknown == "change":
            w_result = width_for(stimulus.result, whole)
            _bar(ax, LEFT, BOT_Y, w_result, H, FILL, EDGE, stimulus.result)
            _cap(
                ax,
                LEFT + w_result + SUBTRACTION_GAP_X,
                LEFT + width_for(whole, whole) - CAP_GAP_X,
                CAP_Y_BESIDE,  # Beside the box for subtraction diagrams
                stimulus.variable_symbol,
                ACCENT,
            )
        else:  # unknown start
            w_result = width_for(stimulus.result, whole)
            _bar(ax, LEFT, BOT_Y, w_result, H, FILL, EDGE, stimulus.result)
            _cap(
                ax,
                LEFT + w_result + SUBTRACTION_GAP_X,
                LEFT + w_result + width_for(stimulus.change, whole) - CAP_GAP_X,
                CAP_Y_BESIDE,  # Beside the box for subtraction diagrams
                stimulus.change,
                ACCENT,
            )

    # ---------------- Equal Groups ----------------
    elif isinstance(stimulus, EqualGroupsDiagram):
        if stimulus.unknown == "groups":
            if stimulus.total is None or stimulus.group_size in (None, 0):
                raise ValueError(
                    "When unknown='groups', total and non-zero group_size are required"
                )
            n = int(stimulus.total / stimulus.group_size)
            whole = stimulus.total
            top_label = stimulus.total
        elif stimulus.unknown == "total":
            n = int(stimulus.groups)
            whole = n * stimulus.group_size
            top_label = stimulus.variable_symbol
        else:
            n = int(stimulus.groups)
            whole = stimulus.total
            top_label = stimulus.total

        # top bar
        _bar(ax, LEFT, TOP_Y, width_for(whole, whole), H, FILL, EDGE, top_label)

        # bottom: n equal groups with appropriate cap placement
        if stimulus.unknown == "total":
            seg_w = ensure_precise_equal_groups(int(n), stimulus.group_size, whole)
            for i in range(int(n)):
                _bar(
                    ax,
                    LEFT + i * seg_w,
                    BOT_Y,
                    seg_w,
                    H,
                    FILL,
                    EDGE,
                    stimulus.group_size,
                )
            _cap(
                ax,
                LEFT + CAP_GAP_X,
                LEFT + int(n) * seg_w - CAP_GAP_X,
                CAP_Y_BELOW,  # below the bars to avoid overlap
                stimulus.variable_symbol,
                ACCENT,
            )
        else:
            seg_val = whole / n
            seg_w = ensure_precise_equal_groups(int(n), seg_val, whole)
            for i in range(int(n)):
                _bar(
                    ax,
                    LEFT + i * seg_w,
                    BOT_Y,
                    seg_w,
                    H,
                    FILL,
                    EDGE,
                    stimulus.variable_symbol,
                )
            _cap(
                ax,
                LEFT + CAP_GAP_X,
                LEFT + int(n) * seg_w - CAP_GAP_X,
                CAP_Y_BELOW,  # below the bars to avoid overlap
                "",
                ACCENT,
            )

    # ---------------- Fraction Strip New (Simple) ----------------
    elif isinstance(stimulus, FractionStripNew):
        # Simple layout: equal parts with bottom total line
        # Center the diagram vertically by using a single row at middle position
        CENTER_Y = 0.55  # Center the single row of boxes

        # Calculate box dimensions with gaps
        n = stimulus.total_parts
        gap = (RIGHT - LEFT) * 0.01
        total_width = RIGHT - LEFT
        total_gap_width = (n - 1) * gap
        width_for_boxes = total_width - total_gap_width
        seg_w = width_for_boxes / n

        # Draw the equal parts (boxes)
        for i in range(n):
            _bar(
                ax,
                LEFT + i * (seg_w + gap),
                CENTER_Y,
                seg_w,
                H,
                FILL,
                EDGE,
                stimulus.part_value,
            )

        # Draw the bottom total line with label
        # Make the line slightly wider than the boxes to encapsulate them
        total_boxes_width = n * seg_w + (n - 1) * gap
        line_extension = total_boxes_width * 0.02  # Extend line by 2% on each side
        _cap(
            ax,
            LEFT - line_extension,
            LEFT + total_boxes_width + line_extension,
            CENTER_Y - 0.12,  # Below the boxes
            stimulus.total_value,
            ACCENT,
        )

    # ---------------- Multiplication (Simplified) ----------------
    elif isinstance(stimulus, MultiplicationDiagram):
        # Convert to simple fraction strip format
        if stimulus.unknown == "product":
            # factor × factor2 = product (unknown product)
            n = int(stimulus.factor)  # number of parts
            part_value = str(stimulus.factor2)  # what goes in each part
            total_value = stimulus.variable_symbol  # total is unknown
        elif stimulus.unknown == "factor":
            # Handle both cases: factor unknown or factor2 unknown
            if stimulus.factor is None:
                # factor is unknown, factor2 and product are known
                n = int(
                    stimulus.product / stimulus.factor2
                )  # calculate number of parts
                part_value = str(stimulus.factor2)  # what goes in each part
            else:
                # factor2 is unknown, factor and product are known
                n = int(stimulus.factor)  # number of parts
                part_value = (
                    stimulus.variable_symbol
                )  # what goes in each part (unknown)
            total_value = str(stimulus.product)  # total is known
        else:  # unknown factor2
            n = int(stimulus.factor)  # number of parts
            part_value = (
                stimulus.variable_symbol
            )  # what goes in each part (unknown factor2)
            total_value = str(stimulus.product)  # total is known

        # Simple layout: equal parts with bottom total line
        CENTER_Y = 0.55  # Center the single row of boxes

        # Calculate box dimensions with gaps
        gap = (RIGHT - LEFT) * 0.01
        total_width = RIGHT - LEFT
        total_gap_width = (n - 1) * gap
        width_for_boxes = total_width - total_gap_width
        seg_w = width_for_boxes / n

        # Draw the equal parts (boxes)
        for i in range(n):
            _bar(
                ax,
                LEFT + i * (seg_w + gap),
                CENTER_Y,
                seg_w,
                H,
                FILL,
                EDGE,
                part_value,
            )

        # Draw the bottom total line with label
        # Make the line slightly wider than the boxes to encapsulate them
        total_boxes_width = n * seg_w + (n - 1) * gap
        line_extension = total_boxes_width * 0.02  # Extend line by 2% on each side
        _cap(
            ax,
            LEFT - line_extension,
            LEFT + total_boxes_width + line_extension,
            CENTER_Y - 0.12,  # Below the boxes
            total_value,
            ACCENT,
        )

    # ---------------- Division (Simplified) ----------------
    elif isinstance(stimulus, DivisionDiagram):
        # Convert to simple fraction strip format
        if stimulus.unknown == "quotient":
            # dividend ÷ divisor = quotient (unknown quotient)
            n = int(stimulus.divisor)  # number of parts (divisor)
            part_value = (
                stimulus.variable_symbol
            )  # what goes in each part (unknown quotient)
            total_value = str(stimulus.dividend)  # total is known (dividend)
        elif stimulus.unknown == "divisor":
            # dividend ÷ divisor = quotient (unknown divisor)
            n = int(stimulus.dividend / stimulus.quotient)  # calculate number of parts
            part_value = str(stimulus.quotient)  # what goes in each part
            total_value = str(stimulus.dividend)  # total is known (dividend)
        else:  # unknown dividend
            n = int(stimulus.divisor)  # number of parts
            part_value = str(stimulus.quotient)  # what goes in each part
            total_value = stimulus.variable_symbol  # total is unknown (dividend)

        # Simple layout: equal parts with bottom total line
        CENTER_Y = 0.55  # Center the single row of boxes

        # Calculate box dimensions with gaps
        gap = (RIGHT - LEFT) * 0.01
        total_width = RIGHT - LEFT
        total_gap_width = (n - 1) * gap
        width_for_boxes = total_width - total_gap_width
        seg_w = width_for_boxes / n

        # Draw the equal parts (boxes)
        for i in range(n):
            _bar(
                ax,
                LEFT + i * (seg_w + gap),
                CENTER_Y,
                seg_w,
                H,
                FILL,
                EDGE,
                part_value,
            )

        # Draw the bottom total line with label
        # Make the line slightly wider than the boxes to encapsulate them
        total_boxes_width = n * seg_w + (n - 1) * gap
        line_extension = total_boxes_width * 0.02  # Extend line by 2% on each side
        _cap(
            ax,
            LEFT - line_extension,
            LEFT + total_boxes_width + line_extension,
            CENTER_Y - 0.12,  # Below the boxes
            total_value,
            ACCENT,
        )

    # ---------------- Comparison ----------------
    elif isinstance(stimulus, ComparisonDiagram):
        # Re-render for a vertical (stacked) comparison
        plt.close(fig)  # close the unused initial fig to avoid a leak
        fig = plt.figure(figsize=(8.2, 8.4))  # portrait-ish

        # Top (Diagram A)
        ax1 = fig.add_axes([0.06, 0.55, 0.88, 0.36])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")
        # Bottom (Diagram B)
        ax2 = fig.add_axes([0.06, 0.12, 0.88, 0.36])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        # Always show neutral placeholders on the image so answers don't leak
        top_label = "Diagram A"
        bottom_label = "Diagram B"
        fig.text(
            0.50, 0.94, top_label, ha="center", va="top", fontsize=16, weight="bold"
        )
        fig.text(
            0.50, 0.51, bottom_label, ha="center", va="top", fontsize=16, weight="bold"
        )

        # Render diagrams based on correct_position
        # If correct_position is "A", correct goes on top (ax1), distractor on bottom (ax2)
        # If correct_position is "B", distractor goes on top (ax1), correct on bottom (ax2)
        correct_stimulus = _coerce(stimulus.correct_diagram)
        distractor_stimulus = _coerce(stimulus.distractor_diagram)

        if stimulus.correct_position == "A":
            # Correct diagram on top (Diagram A), distractor on bottom (Diagram B)
            _render_single_diagram(ax1, correct_stimulus, FILL, EDGE, ACCENT)
            _render_single_diagram(ax2, distractor_stimulus, FILL, EDGE, ACCENT)
        else:  # correct_position == "B"
            # Distractor on top (Diagram A), correct diagram on bottom (Diagram B)
            _render_single_diagram(ax1, distractor_stimulus, FILL, EDGE, ACCENT)
            _render_single_diagram(ax2, correct_stimulus, FILL, EDGE, ACCENT)

        # Save the comparison diagram
        out_dir = settings.additional_content_settings.image_destination_folder
        ext = settings.additional_content_settings.stimulus_image_format
        filename = f"equation_tape_comparison_{int(time.time())}.{ext}"
        out_path = os.path.join(out_dir, filename)
        plt.savefig(
            out_path, dpi=800, bbox_inches="tight", transparent=False, format=ext
        )
        plt.close(fig)
        return out_path

    else:
        raise ValueError("Unsupported stimulus description")

    # save like the histogram renderer does
    out_dir = settings.additional_content_settings.image_destination_folder
    ext = settings.additional_content_settings.stimulus_image_format
    filename = f"equation_tape_{int(time.time())}.{ext}"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=800, bbox_inches="tight", transparent=False, format=ext)
    plt.close(fig)
    return out_path


# Belt & suspenders: guarantee the advertised schema model is a Pydantic class
draw_equation_tape_diagram.stimulus_type = EquationTapeDiagramWrapper
