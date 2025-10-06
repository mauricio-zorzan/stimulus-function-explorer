# NOTE: keep this file WITHOUT `from __future__ import annotations`

import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
    HistogramDescription,
    HistogramWithDottedBinDescription,
    MultiHistogramDescription,
)
from content_generators.settings import settings
from matplotlib.ticker import MaxNLocator

# ---- Color pairs for alternating bins ----
COLOR_PAIRS: List[Tuple[str, str]] = [
    ("#81CAD6", "#DC3E26"),  # Green, Purple
    ("#3B53A5", "#E8785D"),  # Blue, Orange
    ("#E26274", "#F9EC7E"),  # Magenta, Yellow
    ("#B2456E", "#FBEAE7"),  # Red, Blue
    ("#1A2A4D", "#FFDD98"),  # Dark Blue, White
    ("#0B922F", "#F5F5DC"),  # Green, Beige
    ("#6A0DAD", "#D4AF37"),  # Purple, Gold
    ("#800020", "#EAE0D5"),  # Burgundy, Beige
    ("#0047AB", "#FFAC1C"),  # Blue, Amber/Orange
]


# ---- Legacy: Pastel, professional palette (alternating) ----
PASTEL_A = "#0047AB"  # blue
PASTEL_B = "#FFAC1C"  # amber/orange


def _alternating_colors(n: int, color_pair: Tuple[str, str] | None = None):
    """Return [A, B, A, B, ...] of length n using the specified color pair."""
    if color_pair is None:
        # Use legacy colors for backward compatibility
        return [PASTEL_A if i % 2 == 0 else PASTEL_B for i in range(n)]

    color_a, color_b = color_pair
    return [color_a if i % 2 == 0 else color_b for i in range(n)]


def _get_random_color_pair() -> Tuple[str, str]:
    """Get a random color pair from the available options."""
    return random.choice(COLOR_PAIRS)


def _render_one(
    ax, desc: HistogramDescription, color_pair: Tuple[str, str] | None = None
):
    # Compute left edges, widths, heights
    lefts = [b.start for b in desc.bins]
    widths = [b.end - b.start + 1 for b in desc.bins]
    heights = [b.frequency for b in desc.bins]

    # ---- UPDATED: alternate colors per bin using selected color pair ----
    colors = _alternating_colors(len(desc.bins), color_pair)

    # Draw bars aligned to the left edge of each bin
    ax.bar(
        lefts,
        heights,
        width=widths,
        align="edge",
        edgecolor="black",
        color=colors,  # <— apply alternating palette
    )

    # X ticks at bin centers with labels - increased font size
    centers = [l + w / 2 for l, w in zip(lefts, widths)]
    ax.set_xticks(centers)
    ax.set_xticklabels(desc.tick_labels(), rotation=0, ha="center", fontsize=14)

    # Axis labels & title - increased font sizes
    ax.set_xlabel(desc.effective_x_label(), fontsize=16)
    ax.set_title(desc.effective_title(), fontsize=18, fontweight="bold")

    # Force integer y-ticks with larger font size
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=14)


@stimulus_function
def draw_histogram(stimulus: HistogramDescription) -> str:
    # Defensive: fail fast if a class slipped through
    if isinstance(stimulus, type):
        raise TypeError(
            "Renderer received a class, not an instance. "
            "Check that HistogramDescription subclasses StimulusDescription and JSON matches the schema."
        )

    # Select a random color pair for this generation
    selected_color_pair = _get_random_color_pair()

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    _render_one(ax, stimulus, selected_color_pair)

    # Shared y-label with increased font size
    ax.set_ylabel(stimulus.y_label, fontsize=16)

    # Harmonize y-limit
    ymax = max(b.frequency for b in stimulus.bins)
    ax.set_ylim(0, max(1, ymax))

    # Save
    file_path = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"histogram_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_path,
        dpi=800,
        bbox_inches="tight",
        transparent=False,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)
    return file_path


@stimulus_function
def draw_histogram_pair(stimulus: MultiHistogramDescription) -> str:
    # Defensive
    if isinstance(stimulus, type):
        raise TypeError(
            "Renderer received a class, not an instance. "
            "Check that MultiHistogramDescription subclasses StimulusDescription and JSON matches the schema."
        )

    # Select a random color pair for this generation
    selected_color_pair = _get_random_color_pair()

    # Two side-by-side panels sharing Y
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5.5), sharey=True, constrained_layout=True
    )

    # Titles with increased font size
    fig.suptitle(stimulus.effective_title(), fontsize=20, fontweight="bold")

    # Always render histograms[0] on left (ax1) and histograms[1] on right (ax2)
    # The correct_histogram_position only determines which one is correct, not positioning
    left_histogram = stimulus.histograms[0]
    right_histogram = stimulus.histograms[1]

    # Render both (each will use the same alternating A/B palette)
    _render_one(ax1, left_histogram, selected_color_pair)
    _render_one(ax2, right_histogram, selected_color_pair)

    # ---- Y-axis labels: ALWAYS label both subplots with increased font size ----
    # (Ignore top-level stimulus.y_label here; use per-panel labels.)
    y1 = (left_histogram.y_label or "Frequency").strip()
    y2 = (right_histogram.y_label or "Frequency").strip()

    ax1.set_ylabel(y1, labelpad=8, fontsize=16)  # left panel label
    ax2.set_ylabel(y2, labelpad=8, fontsize=16)  # right panel label

    # Show y tick values on both panels with larger font size
    ax1.tick_params(axis="y", labelleft=True, labelsize=14)
    ax2.tick_params(axis="y", labelleft=True, labelsize=14)

    # Normalize y-limits across both for fair visual comparison
    ymax = max(
        max(b.frequency for b in left_histogram.bins),
        max(b.frequency for b in right_histogram.bins),
    )
    ax1.set_ylim(0, max(1, ymax))
    ax2.set_ylim(0, max(1, ymax))

    # Save
    file_path = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"histogram_pair_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_path,
        dpi=800,
        bbox_inches="tight",
        transparent=False,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)
    return file_path


@stimulus_function
def draw_histogram_with_dotted_bin(stimulus: HistogramWithDottedBinDescription) -> str:
    """
    Draw a histogram where one bin is shown as a dotted outline that needs to be completed.

    Args:
        stimulus: HistogramWithDottedBinDescription with the data and dotted bin index
    """
    # Defensive: fail fast if a class slipped through
    if isinstance(stimulus, type):
        raise TypeError(
            "Renderer received a class, not an instance. "
            "Check that HistogramWithDottedBinDescription subclasses StimulusDescription and JSON matches the schema."
        )

    # Select a random color pair for this generation
    selected_color_pair = _get_random_color_pair()

    dotted_bin_index = stimulus.dotted_bin_index
    if dotted_bin_index < 0 or dotted_bin_index >= len(stimulus.bins):
        raise ValueError(
            f"dotted_bin_index {dotted_bin_index} is out of range for {len(stimulus.bins)} bins"
        )

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    # Compute left edges, widths, heights
    lefts = [b.start for b in stimulus.bins]
    widths = [b.end - b.start + 1 for b in stimulus.bins]
    heights = [b.frequency for b in stimulus.bins]

    # Get alternating colors using the selected color pair
    colors = _alternating_colors(len(stimulus.bins), selected_color_pair)

    # Draw solid bars for all bins except the dotted one
    for i, (left, width, height, color) in enumerate(
        zip(lefts, widths, heights, colors)
    ):
        if i != dotted_bin_index:
            ax.bar(
                left,
                height,
                width=width,
                align="edge",
                edgecolor="black",
                color=color,
                linewidth=1,
            )

    # Draw the dotted bin as an outline
    dotted_bin = stimulus.bins[dotted_bin_index]
    dotted_left = dotted_bin.start
    dotted_width = dotted_bin.end - dotted_bin.start + 1
    dotted_height = dotted_bin.frequency
    dotted_color = colors[dotted_bin_index]

    # Create a rectangle patch for the dotted outline
    from matplotlib.patches import Rectangle

    dotted_rect = Rectangle(
        (dotted_left, 0),
        dotted_width,
        dotted_height,
        linewidth=3,
        edgecolor=dotted_color,
        facecolor="none",
        linestyle="--",
        alpha=0.8,
    )
    ax.add_patch(dotted_rect)

    # X ticks at bin centers with labels - increased font size
    centers = [l + w / 2 for l, w in zip(lefts, widths)]
    ax.set_xticks(centers)
    ax.set_xticklabels(stimulus.tick_labels(), rotation=0, ha="center", fontsize=14)

    # Axis labels & title - increased font sizes
    ax.set_xlabel(stimulus.effective_x_label(), fontsize=16)
    ax.set_title(stimulus.effective_title(), fontsize=18, fontweight="bold")

    # Force integer y-ticks with larger font size
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=14)

    # Shared y-label with increased font size
    ax.set_ylabel(stimulus.y_label, fontsize=16)

    # Harmonize y-limit
    ymax = max(b.frequency for b in stimulus.bins)
    ax.set_ylim(0, max(1, ymax))

    # Save
    file_path = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"histogram_dotted_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_path,
        dpi=800,
        bbox_inches="tight",
        transparent=False,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)
    return file_path
