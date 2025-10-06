import math
import random
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction import (
    DividedShapeList,
    DivisionModel,
    FractionList,
    FractionNumber,
    FractionPairList,
    FractionPairSetList,
    FractionSet,
    FractionShape,
    FractionStrips,
    MixedFractionList,
    UnequalFractionList,
    WholeFractionalShapes,
)
from content_generators.settings import settings
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


###########################
# Fraction Models - Basic #
###########################
@stimulus_function
def draw_fractional_models_multiplication_units(model_data: FractionSet):
    # Parse the fractions
    frac1 = model_data.fractions[0].split("/")
    frac2 = model_data.fractions[1].split("/")

    num1, den1 = int(frac1[0]), int(frac1[1])
    num2, den2 = int(frac2[0]), int(frac2[1])

    length = den1
    width = den2

    shaded_length = num1
    shaded_width = num2

    fig, ax = plt.subplots()

    ax.add_patch(
        patches.Rectangle(
            (0, 0), length, width, edgecolor="black", facecolor="none", linewidth=4
        )
    )

    ax.add_patch(
        patches.Rectangle(
            (0, width - shaded_width),
            shaded_length,
            shaded_width,
            edgecolor="none",
            facecolor="yellow",
            alpha=0.5,
        )
    )

    for i in range(length + 1):
        ax.axvline(i, color="gray", linewidth=2)
    for j in range(width + 1):
        ax.axhline(j, color="gray", linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.set_aspect("equal", "box")

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/frac_model_multi_units{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def draw_fractional_sets_models(model_data: FractionPairSetList):
    num_models = len(model_data)
    max_shapes_per_model = max(len(model.fractions) for model in model_data)

    fig, axs = plt.subplots(
        num_models, max_shapes_per_model + 1, figsize=(2 * num_models, 6)
    )
    if num_models == 1:
        axs = [axs]  # Ensure axs is a list even with a single model

    for i, fractional_model in enumerate(model_data):
        shape = fractional_model.shape
        figure_label = "Figure " + str(i + 1)
        axs[i][0].axis("off")
        axs[i][0].text(
            0.25,
            0.5,
            figure_label,
            ha="center",
            va="top",
            transform=axs[i][0].transAxes,
            fontsize=16,
            style="italic",
        )
        for z, fraction in enumerate(fractional_model.fractions):
            k = z + 1
            numerator, denominator = map(int, fraction.split("/"))
            shade_color = (
                "blue" if fractional_model.color is None else fractional_model.color
            )

            if shape == FractionShape.RECTANGLE:
                for j in range(denominator):
                    facecolor = shade_color if j < numerator else "none"
                    axs[i][k].add_patch(
                        patches.Rectangle(
                            (j / denominator, 0),
                            1 / denominator,
                            1,
                            edgecolor="black",
                            facecolor=facecolor,
                            linewidth=2,
                        )
                    )
            elif shape == FractionShape.CIRCLE:
                for j in range(denominator):
                    angle_start = 360 * j / denominator
                    angle_end = 360 * (j + 1) / denominator
                    facecolor = shade_color if j < numerator else "none"
                    axs[i][k].add_patch(
                        patches.Wedge(
                            (0.5, 0.5),
                            0.5,
                            angle_start,
                            angle_end,
                            facecolor=facecolor,
                            edgecolor="black",
                            linewidth=2,
                        )
                    )
            # Set the limits, aspect, and label position
            axs[i][k].set_xlim(0, 1)
            axs[i][k].set_ylim(0, 1)
            axs[i][k].set_aspect("equal")
            axs[i][k].axis("off")
        for k in range(len(fractional_model.fractions) + 1, max_shapes_per_model + 1):
            axs[i][k].set_xlim(0, 0.1)
            axs[i][k].set_ylim(0, 0.1)
            axs[i][k].axis("off")

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/frac_model_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    # plt.show()

    return file_name


color_cache = {}


def random_color(id):
    if id not in color_cache:
        color_cache[id] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return color_cache[id]


def calculate_subplot_grid(num_models):
    """
    Calculate the optimal grid layout for the given number of models.
    """
    rows = math.ceil(math.sqrt(num_models))
    cols = math.ceil(num_models / rows)
    return rows, cols


@stimulus_function
def draw_fractional_pair_models(model_data: FractionPairList):
    num_models = len(model_data)
    axs: list[Axes]

    # Determine the grid layout
    rows, cols = calculate_subplot_grid(num_models)

    # Create figure and axis objects
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_models == 1:
        axs = [axs]  # type: ignore
    if rows > 1:
        axs = axs.flatten()  # type: ignore
    # Iterate over each model's data to draw them
    for i, fractional_model in enumerate(model_data):
        shape = fractional_model.shape
        fractions = [
            list(map(int, frac.split("/"))) for frac in fractional_model.fractions
        ]
        denominator = fractions[0][1]

        if shape == FractionShape.RECTANGLE:
            for j in range(denominator):
                facecolor = "none"
                for idx, (numerator, _) in enumerate(fractions):
                    if j < sum(num for num, _ in fractions[: idx + 1]):
                        if idx == 0:
                            facecolor = "blue"
                        elif idx == 1:
                            facecolor = "green"
                        else:
                            facecolor = random_color(idx)
                        break
                axs[i].add_patch(
                    patches.Rectangle(
                        (j / denominator, 0),
                        1 / denominator,
                        1,
                        edgecolor="black",
                        facecolor=facecolor,
                        linewidth=3,
                    )
                )
        elif shape == FractionShape.CIRCLE:
            for j in range(denominator):
                angle_start = 360 * j / denominator
                angle_end = 360 * (j + 1) / denominator
                facecolor = "none"
                for idx, (numerator, _) in enumerate(fractions):
                    if j < sum(num for num, _ in fractions[: idx + 1]):
                        if idx == 0:
                            facecolor = "blue"
                        elif idx == 1:
                            facecolor = "green"
                        else:
                            facecolor = random_color(idx)
                        break
                axs[i].add_patch(
                    patches.Wedge(
                        (0.5, 0.5),
                        0.5,
                        angle_start,
                        angle_end,
                        facecolor=facecolor,
                        edgecolor="black",
                        linewidth=3,
                    )
                )

        # Set the limits, aspect, and label position
        axs[i].set_xlim(-0.05, 1.05)
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_aspect("equal")
        axs[i].axis("off")
        if num_models > 1:
            axs[i].set_title(f"Figure {i + 1}", fontsize=30)
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_frac_model.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=1000,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


def _draw_triangle_with_division(ax, numerator: int, denominator: int, color):
    """
    Draw a triangle divided into halves (2), thirds (3), or sixths (6).
    Uses clean outline approach to eliminate edge artifacts.
    Adapts to different axes sizes by using the current axes limits.
    """
    # Get current axes limits to adapt triangle size
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate triangle bounds within the current axes
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Use 90% of the available space with proper margins
    margin_x = 0.05 * x_range
    margin_y = 0.05 * y_range

    left = xlim[0] + margin_x
    right = xlim[1] - margin_x
    bottom = ylim[0] + margin_y
    top = ylim[1] - margin_y

    # Triangle vertices - equilateral triangle that fits the space
    center_x = (left + right) / 2
    OUTER = [(center_x, top), (left, bottom), (right, bottom)]

    def add_fill_triangle(vertices, filled: bool):
        # NO edge on the fills; the outline is drawn once at the end
        tri = patches.Polygon(
            vertices,
            facecolor=(color if filled else "none"),
            edgecolor="none",
            linewidth=0,
            zorder=1,
        )
        ax.add_patch(tri)

    if denominator == 2:
        for j in range(2):
            filled = j < numerator
            if j == 0:
                vertices = [(center_x, top), (left, bottom), (center_x, bottom)]
            else:
                vertices = [(center_x, top), (center_x, bottom), (right, bottom)]
            add_fill_triangle(vertices, filled)

        # add the dividing median (top -> midpoint of base)
        base_mid = (center_x, bottom)
        line = Line2D(
            [center_x, base_mid[0]],
            [top, base_mid[1]],
            linewidth=1,
            color="black",
            solid_capstyle="butt",
            zorder=5,
        )
        ax.add_line(line)

    elif denominator == 3:
        # Calculate triangle centroid for divisions
        tri_center_x = (center_x + left + right) / 3
        tri_center_y = (top + bottom + bottom) / 3

        # Calculate inset proportional to triangle size
        triangle_height = top - bottom
        inset = 0.05 * triangle_height

        top_vertex = (center_x, top - inset)
        left_vertex = (left + inset, bottom + inset)
        right_vertex = (right - inset, bottom + inset)

        parts = [
            [top_vertex, left_vertex, (tri_center_x, tri_center_y)],
            [top_vertex, (tri_center_x, tri_center_y), right_vertex],
            [
                (left, bottom),
                (right, bottom),
                (tri_center_x, tri_center_y),
            ],  # bottom section uses actual triangle corners
        ]
        for j, vertices in enumerate(parts):
            add_fill_triangle(vertices, j < numerator)

        # draw ALL three spokes from the center so every seam is visible
        line_extension = 0.08 * triangle_height
        spokes = [
            (center_x, top - inset + line_extension),  # top
            (
                left + inset - line_extension * 0.5,
                bottom + inset - line_extension * 0.5,
            ),  # left
            (
                right - inset + line_extension * 0.5,
                bottom + inset - line_extension * 0.5,
            ),  # right
        ]
        for x2, y2 in spokes:
            ax.add_line(
                Line2D(
                    [tri_center_x, x2],
                    [tri_center_y, y2],
                    linewidth=1,
                    color="black",
                    solid_capstyle="butt",
                    zorder=5,
                )
            )

    elif denominator == 6:
        # Calculate triangle centroid for divisions
        tri_center_x = (center_x + left + right) / 3
        tri_center_y = (top + bottom + bottom) / 3

        # Calculate inset proportional to triangle size
        triangle_height = top - bottom
        inset = 0.05 * triangle_height

        top_vertex = (center_x, top - inset)
        left_vertex = (left + inset, bottom + inset)
        right_vertex = (right - inset, bottom + inset)

        mid_top_left = (
            (top_vertex[0] + left_vertex[0]) / 2,
            (top_vertex[1] + left_vertex[1]) / 2,
        )
        mid_top_right = (
            (top_vertex[0] + right_vertex[0]) / 2,
            (top_vertex[1] + right_vertex[1]) / 2,
        )

        parts = [
            [top_vertex, mid_top_left, (tri_center_x, tri_center_y)],
            [top_vertex, (tri_center_x, tri_center_y), mid_top_right],
            [mid_top_left, left_vertex, (tri_center_x, tri_center_y)],
            [
                (left, bottom),
                ((left_vertex[0] + right_vertex[0]) / 2, bottom),
                (tri_center_x, tri_center_y),
            ],  # bottom left - use actual triangle corners
            [
                ((left_vertex[0] + right_vertex[0]) / 2, bottom),
                (right, bottom),
                (tri_center_x, tri_center_y),
            ],  # bottom right - use actual triangle corners
            [(tri_center_x, tri_center_y), right_vertex, mid_top_right],
        ]
        for j, vertices in enumerate(parts):
            add_fill_triangle(vertices, j < numerator)

        # draw the three medians (vertex -> opposite side midpoint)
        mid_left_side = (
            (top_vertex[0] + left_vertex[0]) / 2,
            (top_vertex[1] + left_vertex[1]) / 2,
        )
        mid_right_side = (
            (top_vertex[0] + right_vertex[0]) / 2,
            (top_vertex[1] + right_vertex[1]) / 2,
        )

        line_extension = 0.06 * triangle_height
        medians = [
            (
                (center_x, top - inset + line_extension),
                (
                    (left_vertex[0] + right_vertex[0]) / 2,
                    bottom,
                ),  # extend to actual base
            ),  # top -> base midpoint - extend to triangle edge
            (
                (
                    left + inset - line_extension * 0.5,
                    bottom + inset - line_extension * 0.5,
                ),
                mid_right_side,
            ),  # left -> midpoint of right side - extend left endpoint
            (
                (
                    right - inset + line_extension * 0.5,
                    bottom + inset - line_extension * 0.5,
                ),
                mid_left_side,
            ),  # right -> midpoint of left side - extend right endpoint
        ]
        for (x1, y1), (x2, y2) in medians:
            ax.add_line(
                Line2D(
                    [x1, x2],
                    [y1, y2],
                    linewidth=1,
                    color="black",
                    solid_capstyle="butt",
                    zorder=5,
                )
            )

    # Draw ONE clean outer outline on top
    outline = patches.Polygon(
        OUTER,
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=1,
        joinstyle="miter",
        zorder=10,
    )
    ax.add_patch(outline)

    # Optionally clip internal lines to the outer triangle to guarantee nothing spills
    for artist in ax.lines:
        artist.set_clip_path(outline)


def _determine_rectangle_division_type(denominator: int) -> str:
    """
    Determine how to divide a rectangle based on the denominator.
    Returns 'vertical', 'horizontal', or 'both' based on factors and random selection.
    """
    # Find factors of the denominator to determine possible grid divisions
    factors = []
    for i in range(1, int(denominator**0.5) + 1):
        if denominator % i == 0:
            factors.append((i, denominator // i))

    # Remove 1xN and Nx1 cases for grid division (they're covered by vertical/horizontal)
    grid_factors = [(r, c) for r, c in factors if r > 1 and c > 1]

    # Determine available division types
    available_types = ["vertical", "horizontal"]
    if grid_factors:
        available_types.append("both")

    # Randomly select from available types
    return random.choice(available_types)


def _draw_rectangle_with_division(
    ax, numerator: int, denominator: int, division_type: str, color
):
    """
    Draw a rectangle with the specified division type.
    """
    if division_type == "vertical":
        # Traditional vertical division (current implementation)
        for j in range(denominator):
            facecolor = color if j < numerator else "none"
            ax.add_patch(
                patches.Rectangle(
                    (j / denominator, 0),
                    1 / denominator,
                    1,
                    edgecolor="black",
                    facecolor=facecolor,
                    linewidth=1,
                )
            )

    elif division_type == "horizontal":
        # Horizontal division
        for j in range(denominator):
            facecolor = color if j < numerator else "none"
            ax.add_patch(
                patches.Rectangle(
                    (0, j / denominator),
                    1,
                    1 / denominator,
                    edgecolor="black",
                    facecolor=facecolor,
                    linewidth=1,
                )
            )

    elif division_type == "both":
        # Grid division (both horizontal and vertical)
        # Find suitable factors for creating a grid
        factors = []
        for i in range(1, int(denominator**0.5) + 1):
            if denominator % i == 0:
                factors.append((i, denominator // i))

        # Choose the factor pair that creates the most square-like grid
        grid_factors = [(r, c) for r, c in factors if r > 1 and c > 1]
        if grid_factors:
            # Select the factor pair closest to square
            rows, cols = min(grid_factors, key=lambda x: abs(x[0] - x[1]))
        else:
            # Fallback to vertical if no suitable grid factors
            rows, cols = 1, denominator

        # Draw grid
        cell_width = 1 / cols
        cell_height = 1 / rows

        for j in range(denominator):
            row = j // cols
            col = j % cols
            facecolor = color if j < numerator else "none"
            ax.add_patch(
                patches.Rectangle(
                    (col * cell_width, row * cell_height),
                    cell_width,
                    cell_height,
                    edgecolor="black",
                    facecolor=facecolor,
                    linewidth=1,
                )
            )


@stimulus_function
def draw_fractional_models(model_data: FractionList):
    """
    Function to draw shapes split into segments based on the parsed fractions from dictionaries, each with a thick black border.
    """
    # Calculate the number of models
    num_models = len(model_data)

    # Create figure and axis objects
    fig, axs = plt.subplots(1, num_models, figsize=(0.6 * num_models, 0.6))
    if num_models == 1:
        axs = [axs]  # Ensure axs is a list even with a single model

    # Iterate over each model's data to draw them
    for i, fractional_model in enumerate(model_data):
        shape = fractional_model.shape
        fraction_str = fractional_model.fraction
        numerator, denominator = map(int, fraction_str.split("/"))

        # Generate random color for this model - RGB values between 0.3 and 0.8 to avoid very dark or very light colors
        random_color = (
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
        )

        if shape == FractionShape.RECTANGLE:
            # Determine division type based on denominator and random selection
            division_type = _determine_rectangle_division_type(denominator)
            _draw_rectangle_with_division(
                axs[i], numerator, denominator, division_type, random_color
            )
        elif shape == FractionShape.CIRCLE:
            # Add random rotation to circle (0-360 degrees)
            rotation_offset = random.uniform(0, 360)

            # Fill the fraction with segments, each with a thick border
            for j in range(denominator):
                angle_start = 360 * j / denominator + 90 + rotation_offset
                angle_end = 360 * (j + 1) / denominator + 90 + rotation_offset
                facecolor = random_color if j < numerator else "none"
                axs[i].add_patch(
                    patches.Wedge(
                        (0.5, 0.5),
                        0.5,
                        angle_start,
                        angle_end,
                        facecolor=facecolor,
                        edgecolor="black",
                        linewidth=1,
                    )
                )
        elif shape == FractionShape.TRIANGLE:
            # Draw triangle divided into halves, thirds, or sixths
            _draw_triangle_with_division(axs[i], numerator, denominator, random_color)

        # Set the limits, aspect, and label position
        axs[i].set_xlim(-0.05, 1.05)
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_aspect("equal")
        axs[i].axis("off")
        if num_models > 1:
            figure_label = "Figure " + str(i + 1)
            axs[i].text(
                0.5,
                -0.05,
                figure_label,
                ha="center",
                va="top",
                transform=axs[i].transAxes,
                fontsize=5,
            )
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_frac_model.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def draw_fractional_models_labeled(model_data: FractionList):
    """
    Function to draw shapes split into segments based on the parsed fractions from dictionaries, each with a thick black border.
    Instead of Figure labels, each model is labeled with the fraction in numerator/denominator format.
    """
    # Calculate the number of models
    num_models = len(model_data)

    # Create figure and axis objects
    fig, axs = plt.subplots(1, num_models, figsize=(0.6 * num_models, 0.6))
    if num_models == 1:
        axs = [axs]  # Ensure axs is a list even with a single model

    # Iterate over each model's data to draw them
    for i, fractional_model in enumerate(model_data):
        shape = fractional_model.shape
        fraction_str = fractional_model.fraction
        numerator, denominator = map(int, fraction_str.split("/"))

        # Generate a random color that's not too dark or too light
        # RGB values between 0.3 and 0.8 to avoid very dark or very light colors
        random_color = (
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
        )

        if shape == FractionShape.RECTANGLE:
            # Draw each segment with a thick border
            for j in range(denominator):
                facecolor = random_color if j < numerator else "none"
                axs[i].add_patch(
                    patches.Rectangle(
                        (j / denominator, 0),
                        1 / denominator,
                        1,
                        edgecolor="black",
                        facecolor=facecolor,
                        linewidth=1,
                    )
                )
        elif shape == FractionShape.CIRCLE:
            # Fill the fraction with segments, each with a thick border
            for j in range(denominator):
                angle_start = 360 * j / denominator + 90
                angle_end = 360 * (j + 1) / denominator + 90
                facecolor = random_color if j < numerator else "none"
                axs[i].add_patch(
                    patches.Wedge(
                        (0.5, 0.5),
                        0.5,
                        angle_start,
                        angle_end,
                        facecolor=facecolor,
                        edgecolor="black",
                        linewidth=1,
                    )
                )
        elif shape == FractionShape.TRIANGLE:
            # Draw triangle divided into halves, thirds, or sixths
            _draw_triangle_with_division(axs[i], numerator, denominator, random_color)

        # Set the limits, aspect, and add fraction label
        axs[i].set_xlim(-0.05, 1.05)
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_aspect("equal")
        axs[i].axis("off")

        # Add fraction label using numerator over denominator format
        fraction_label = f"$\\frac{{{numerator}}}{{{denominator}}}$"
        axs[i].text(
            0.5,
            -0.05,
            fraction_label,
            ha="center",
            va="top",
            transform=axs[i].transAxes,
            fontsize=8,
        )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_frac_model_labeled.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def draw_fractional_models_no_shade(model_data: DividedShapeList):
    """
    Function to draw shapes split into segments based on the parsed fractions from dictionaries, each with a thick black border.
    """
    # Calculate the number of models
    num_models = len(model_data)

    # Create figure and axis objects
    fig, axs = plt.subplots(1, num_models, figsize=(5 * num_models, 6))
    if num_models == 1:
        axs = [axs]  # Ensure axs is a list even with a single model

    # Iterate over each model's data to draw them
    for i, model in enumerate(model_data):
        denominator = model.denominator
        numerator = 0

        if model.shape == FractionShape.RECTANGLE:
            # Draw each segment with a thick border
            for j in range(denominator):
                facecolor = "blue" if j < numerator else "none"
                axs[i].add_patch(
                    patches.Rectangle(
                        (j / denominator, 0),
                        1 / denominator,
                        1,
                        edgecolor="black",
                        facecolor=facecolor,
                        linewidth=3,
                    )
                )
        elif model.shape == FractionShape.CIRCLE:
            # Fill the fraction with segments, each with a thick border
            for j in range(denominator):
                angle_start = 360 * j / denominator
                angle_end = 360 * (j + 1) / denominator
                facecolor = "blue" if j < numerator else "none"
                axs[i].add_patch(
                    patches.Wedge(
                        (0.5, 0.5),
                        0.5,
                        angle_start,
                        angle_end,
                        facecolor=facecolor,
                        edgecolor="black",
                        linewidth=3,
                    )
                )

        # Set the limits, aspect, and label position
        axs[i].set_xlim(-0.1, 1.1)
        axs[i].set_ylim(-0.1, 1.1)
        axs[i].set_aspect("equal")
        axs[i].axis("off")
        if num_models > 1:
            figure_label = "Figure " + str(i + 1)
            axs[i].text(
                0.5,
                0,
                figure_label,
                ha="center",
                va="top",
                transform=axs[i].transAxes,
                fontsize=20,
            )
    file_name = f"{settings.additional_content_settings.image_destination_folder}/frac_model_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def draw_fractional_models_full_shade(model_data: DividedShapeList):
    """
    Function to draw shapes split into segments based on the parsed fractions from dictionaries,
    each fully shaded with the color blue and a thick black border.
    """
    # Calculate the number of models
    num_models = len(model_data)

    # Create figure and axis objects
    fig, axs = plt.subplots(1, num_models, figsize=(5 * num_models, 6))
    if num_models == 1:
        axs = [axs]  # Ensure axs is a list even with a single model

    # Iterate over each model's data to draw them
    for i, model in enumerate(model_data):
        shape = model.shape
        denominator = model.denominator

        if shape == FractionShape.RECTANGLE:
            # Draw each segment with a thick border, fully shaded in blue
            for j in range(denominator):
                axs[i].add_patch(
                    patches.Rectangle(
                        (j / denominator, 0),
                        1 / denominator,
                        1,
                        edgecolor="black",
                        facecolor="skyblue",
                        linewidth=3,
                    )
                )
        elif shape == FractionShape.CIRCLE:
            # Fill the fraction with segments, each fully shaded in blue with a thick border
            for j in range(denominator):
                angle_start = 360 * j / denominator
                angle_end = 360 * (j + 1) / denominator
                axs[i].add_patch(
                    patches.Wedge(
                        (0.5, 0.5),
                        0.5,
                        angle_start,
                        angle_end,
                        facecolor="skyblue",
                        edgecolor="black",
                        linewidth=3,
                    )
                )

        # Set the limits, aspect, and label position
        axs[i].set_xlim(-0.1, 1.1)
        axs[i].set_ylim(-0.1, 1.1)
        axs[i].set_aspect("equal")
        axs[i].axis("off")
        if num_models > 1:
            figure_label = "Figure " + str(i + 1)
            axs[i].text(
                0.5,
                0,
                figure_label,
                ha="center",
                va="top",
                transform=axs[i].transAxes,
                fontsize=20,
            )
    file_name = f"{settings.additional_content_settings.image_destination_folder}/frac_model_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def draw_fractional_models_unequal(model_data: UnequalFractionList):
    """
    Function to draw shapes split into segments based on the parsed fractions from dictionaries.
    Some shapes are equally divided while others are unequally divided.
    All shapes are fully shaded with random colors and have thin black borders.
    """
    # Calculate the number of models
    num_models = len(model_data)

    # Create figure and axis objects
    fig, axs = plt.subplots(1, num_models, figsize=(0.6 * num_models, 0.6))
    if num_models == 1:
        axs = [axs]  # Ensure axs is a list even with a single model

    # Iterate over each model's data to draw them
    for i, fractional_model in enumerate(model_data):
        shape = fractional_model.shape
        divided_parts = fractional_model.divided_parts
        equally_divided = fractional_model.equally_divided

        # Generate random color for this model - RGB values between 0.3 and 0.8 to avoid very dark or very light colors
        random_color = (
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
            random.uniform(0.3, 0.8),
        )

        if shape == FractionShape.RECTANGLE:
            if equally_divided:
                # Draw each segment with equal width and thick border, fully shaded
                for j in range(divided_parts):
                    axs[i].add_patch(
                        patches.Rectangle(
                            (j / divided_parts, 0),
                            1 / divided_parts,
                            1,
                            edgecolor="black",
                            facecolor=random_color,
                            linewidth=1,
                        )
                    )
            else:
                # Draw segments with unequal widths, fully shaded
                segment_widths = []
                remaining_width = 1.0
                for j in range(divided_parts - 1):
                    # Ensure we leave enough width for remaining segments
                    max_width = remaining_width - (divided_parts - j - 1) * 0.05
                    min_width = 0.05

                    # Special handling for 2, 3, and 4 parts to ensure obvious inequality
                    if divided_parts == 2:
                        # Ensure the division is noticeably unequal (avoid center area)
                        # Either make it significantly less than 0.5 or significantly more
                        if random.choice([True, False]):
                            # Make first segment smaller (20% to 35%)
                            min_width = 0.2
                            max_width = 0.35
                        else:
                            # Make first segment larger (65% to 80%)
                            min_width = 0.65
                            max_width = 0.8
                    elif divided_parts == 3:
                        # Avoid segments close to 1/3 (33.33%)
                        # Either make it significantly less than 1/3 or significantly more
                        if random.choice([True, False]):
                            # Make first segment smaller (10% to 25%)
                            min_width = 0.1
                            max_width = 0.25
                        else:
                            # Make first segment larger (45% to 60%)
                            min_width = 0.45
                            max_width = 0.6
                    elif divided_parts == 4:
                        # Avoid segments close to 1/4 (25%)
                        # Either make it significantly less than 1/4 or significantly more
                        if random.choice([True, False]):
                            # Make first segment smaller (5% to 15%)
                            min_width = 0.05
                            max_width = 0.15
                        else:
                            # Make first segment larger (35% to 50%)
                            min_width = 0.35
                            max_width = 0.5

                    if max_width <= min_width:
                        width = min_width
                    else:
                        width = random.uniform(min_width, max_width)
                    segment_widths.append(width)
                    remaining_width -= width
                segment_widths.append(remaining_width)

                # Draw the segments, fully shaded
                current_x = 0
                for j in range(divided_parts):
                    axs[i].add_patch(
                        patches.Rectangle(
                            (current_x, 0),
                            segment_widths[j],
                            1,
                            edgecolor="black",
                            facecolor=random_color,
                            linewidth=1,
                        )
                    )
                    current_x += segment_widths[j]

        elif shape == FractionShape.CIRCLE:
            # 1) draw the full circle once
            circle = patches.Circle(
                (0.5, 0.5),  # center
                0.5,  # radius
                facecolor=random_color,
                edgecolor="black",
                linewidth=1,
            )
            axs[i].add_patch(circle)

            # 2) compute the list of segment angles
            if equally_divided:
                segment_angles = [360 / divided_parts] * divided_parts
            else:
                segment_angles = []
                remaining = 360
                for j in range(divided_parts - 1):
                    max_ang = remaining - (divided_parts - j - 1) * 10
                    min_ang = 10

                    # Special handling for 2, 3, and 4 parts to ensure obvious inequality
                    if divided_parts == 2:
                        # Ensure the division is noticeably unequal (avoid center area)
                        # Either make it significantly less than 180° or significantly more
                        if random.choice([True, False]):
                            # Make first segment smaller (72° to 126°, or 20% to 35% of 360°)
                            min_ang = 72
                            max_ang = 126
                        else:
                            # Make first segment larger (234° to 288°, or 65% to 80% of 360°)
                            min_ang = 234
                            max_ang = 288
                    elif divided_parts == 3:
                        # Avoid segments close to 120° (360°/3)
                        # Either make it significantly less than 120° or significantly more
                        if random.choice([True, False]):
                            # Make first segment smaller (36° to 90°, or 10% to 25% of 360°)
                            min_ang = 36
                            max_ang = 90
                        else:
                            # Make first segment larger (162° to 216°, or 45% to 60% of 360°)
                            min_ang = 162
                            max_ang = 216
                    elif divided_parts == 4:
                        # Avoid segments close to 90° (360°/4)
                        # Either make it significantly less than 90° or significantly more
                        if random.choice([True, False]):
                            # Make first segment smaller (18° to 54°, or 5% to 15% of 360°)
                            min_ang = 18
                            max_ang = 54
                        else:
                            # Make first segment larger (126° to 180°, or 35% to 50% of 360°)
                            min_ang = 126
                            max_ang = 180

                    angle = (
                        min_ang
                        if max_ang <= min_ang
                        else random.uniform(min_ang, max_ang)
                    )
                    segment_angles.append(angle)
                    remaining -= angle
                segment_angles.append(remaining)

            # 3) draw the radial lines
            current_angle = 90  # start at "12 o'clock"
            for ang in segment_angles:
                θ = math.radians(current_angle)
                x_end = 0.5 + 0.5 * math.cos(θ)
                y_end = 0.5 + 0.5 * math.sin(θ)

                axs[i].add_line(
                    Line2D([0.5, x_end], [0.5, y_end], linewidth=1, color="black")
                )
                current_angle += ang

        elif shape == FractionShape.TRIANGLE:
            if equally_divided:
                # Draw triangle fully shaded - all sections filled (force numerator > denominator to ensure full shading)
                _draw_triangle_with_division(
                    axs[i], divided_parts + 1, divided_parts, random_color
                )
            else:
                # Draw triangle with unequal vertical divisions
                # First, draw the full triangle filled with color
                triangle_vertices = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
                triangle = patches.Polygon(
                    triangle_vertices,
                    facecolor=random_color,
                    edgecolor="black",
                    linewidth=1,
                )
                axs[i].add_patch(triangle)

                # Generate random x-positions for vertical division lines
                # These lines will divide the triangle into unequal vertical sections
                line_positions = []
                min_x, max_x = 0.1, 0.9  # Triangle base x-coordinates
                available_width = max_x - min_x

                for j in range(divided_parts - 1):
                    # Ensure we leave enough width for remaining divisions
                    remaining_divisions = divided_parts - j - 1
                    min_spacing = 0.05

                    if j == 0:  # First division
                        # Special handling based on number of parts to ensure obvious inequality
                        if divided_parts == 2:
                            # Avoid center area (around 0.5) to prevent appearing equal
                            if random.choice([True, False]):
                                # Position closer to left (20% to 35% from left)
                                min_pos = min_x + 0.2 * available_width
                                max_pos = min_x + 0.35 * available_width
                            else:
                                # Position closer to right (65% to 80% from left)
                                min_pos = min_x + 0.65 * available_width
                                max_pos = min_x + 0.8 * available_width
                        elif divided_parts == 3:
                            # Avoid positions around 1/3 and 2/3
                            if random.choice([True, False]):
                                # First division closer to left (10% to 25% from left)
                                min_pos = min_x + 0.1 * available_width
                                max_pos = min_x + 0.25 * available_width
                            else:
                                # First division closer to center-right (45% to 60% from left)
                                min_pos = min_x + 0.45 * available_width
                                max_pos = min_x + 0.6 * available_width
                        else:
                            # For 4+ divisions, allow more variation
                            min_pos = min_x + min_spacing
                            max_pos = max_x - remaining_divisions * min_spacing
                    else:
                        # Subsequent divisions
                        last_pos = line_positions[-1] if line_positions else min_x
                        min_pos = last_pos + min_spacing
                        max_pos = max_x - remaining_divisions * min_spacing

                    if max_pos <= min_pos:
                        x_pos = min_pos
                    else:
                        x_pos = random.uniform(min_pos, max_pos)

                    line_positions.append(x_pos)

                # Draw the vertical division lines
                for x_pos in line_positions:
                    # Calculate where the vertical line intersects the triangle edges
                    # Triangle has vertices at (0.1, 0.1), (0.9, 0.1), (0.5, 0.9)
                    # Left edge: from (0.1, 0.1) to (0.5, 0.9)
                    # Right edge: from (0.9, 0.1) to (0.5, 0.9)

                    if x_pos <= 0.5:
                        # Line intersects left edge
                        # Left edge equation: y = 0.1 + (0.8 / 0.4) * (x - 0.1) = 0.1 + 2 * (x - 0.1)
                        y_top = 0.1 + 2 * (x_pos - 0.1)
                    else:
                        # Line intersects right edge
                        # Right edge equation: y = 0.1 + (0.8 / -0.4) * (x - 0.9) = 0.1 - 2 * (x - 0.9)
                        y_top = 0.1 - 2 * (x_pos - 0.9)

                    # Draw vertical line from base to intersection point
                    axs[i].add_line(
                        Line2D([x_pos, x_pos], [0.1, y_top], linewidth=1, color="black")
                    )

        # Set the limits, aspect, and label position
        axs[i].set_xlim(-0.05, 1.05)
        axs[i].set_ylim(-0.05, 1.05)
        axs[i].set_aspect("equal")
        axs[i].axis("off")
        if num_models > 1:
            figure_label = "Figure " + str(i + 1)
            axs[i].text(
                0.5,
                -0.05,
                figure_label,
                ha="center",
                va="top",
                transform=axs[i].transAxes,
                fontsize=5,
            )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_frac_model_unequal.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def draw_mixed_fractional_models(model_data: MixedFractionList):
    """
    Function to draw shapes for mixed numbers and improper fractions.
    Shows up to 4 fractional models in a 2×2 grid.
    If fraction > 1, additional shapes are added to show the whole‐number part.
    - If only 1 figure: use a single large subplot, centered, no label.
    - If 2 figures: use a single row with 2 large subplots, both centered and large.
    - If 3 figures: use a single row with 3 subplots.
    - If 4 figures: standard 2x2 grid, show labels.
    """
    num_models = len(model_data)

    # constant sizing
    shape_width = 1.0
    spacing = shape_width * 1.2
    margin = 0.05
    max_shapes = 4
    xlim_max = (max_shapes - 1) * spacing + shape_width

    # create subplots
    if num_models == 1:
        fig, axs = plt.subplots(1, 1, figsize=(8, 8))
        axs = [axs]
    elif num_models == 2:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs = list(axs)
    elif num_models == 3:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs = list(axs)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # remove vertical space between rows and create slight overlap
        fig.subplots_adjust(hspace=-1, wspace=0.2)
        axs = axs.flatten()

    # draw each model
    for i, ax in enumerate(axs[:num_models]):
        numerator, denominator = map(int, model_data.root[i].fraction.split("/"))
        whole = numerator // denominator
        rem = numerator % denominator
        count = whole + (1 if rem > 0 else 0)

        # alignment: mixed flush-left else centered
        if whole > 0 and rem > 0:
            start_x = 0
        else:
            total_w = (count - 1) * spacing + shape_width
            start_x = (xlim_max - total_w) / 2

        for s in range(count):
            x0 = start_x + s * spacing
            shape = model_data.root[i].shape
            if shape == FractionShape.RECTANGLE:
                ax.add_patch(
                    patches.Rectangle(
                        (x0, 0),
                        shape_width,
                        shape_width,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=3,
                    )
                )
                if s < whole:
                    for j in range(denominator):
                        ax.add_patch(
                            patches.Rectangle(
                                (x0 + j / denominator * shape_width, 0),
                                shape_width / denominator,
                                shape_width,
                                edgecolor="black",
                                facecolor="lightblue",
                                linewidth=2,
                            )
                        )
                elif s == whole and rem > 0:
                    for j in range(denominator):
                        fc = "lightblue" if j < rem else "none"
                        ax.add_patch(
                            patches.Rectangle(
                                (x0 + j / denominator * shape_width, 0),
                                shape_width / denominator,
                                shape_width,
                                edgecolor="black",
                                facecolor=fc,
                                linewidth=2,
                            )
                        )
            else:
                center = (x0 + shape_width / 2, shape_width / 2)
                ax.add_patch(
                    patches.Circle(
                        center,
                        shape_width / 2,
                        edgecolor="black",
                        facecolor="none",
                        linewidth=3,
                    )
                )
                if s < whole:
                    for j in range(denominator):
                        a0 = 360 * j / denominator + 90
                        a1 = 360 * (j + 1) / denominator + 90
                        ax.add_patch(
                            patches.Wedge(
                                center,
                                shape_width / 2,
                                a0,
                                a1,
                                facecolor="lightblue",
                                edgecolor="black",
                                linewidth=2,
                            )
                        )
                elif s == whole and rem > 0:
                    for j in range(denominator):
                        a0 = 360 * j / denominator + 90
                        a1 = 360 * (j + 1) / denominator + 90
                        fc = "lightblue" if j < rem else "none"
                        ax.add_patch(
                            patches.Wedge(
                                center,
                                shape_width / 2,
                                a0,
                                a1,
                                facecolor=fc,
                                edgecolor="black",
                                linewidth=2,
                            )
                        )

        ax.set_xlim(-margin, xlim_max + margin)
        ax.set_ylim(-margin, shape_width + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        if num_models > 1:
            ax.text(
                0.5,
                -0.15,
                f"Figure {i+1}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                fontsize=24,
            )

    plt.tight_layout()
    timestamp = int(time.time())
    dest = settings.additional_content_settings
    file_name = f"{dest.image_destination_folder}/{timestamp}_mixed_frac_model.{dest.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=650,
        transparent=False,
        bbox_inches="tight",
        format=dest.stimulus_image_format,
    )
    plt.close(fig)
    return file_name


@stimulus_function
def draw_whole_fractional_models(model_data: WholeFractionalShapes):
    """
    Function to draw 1-5 fully shaded shapes (circles or rectangles) representing whole fractions.
    Each shape is divided into the same number of parts (common denominator) and fully shaded.
    Represents fractions like 3/3, 6/3, 9/3, etc. where count * divisions / divisions = count whole units.
    All shapes are arranged in a single row without labels.
    """
    count = model_data.count
    shape = model_data.shape
    divisions = model_data.divisions

    # Create figure with single row layout
    fig, axs = plt.subplots(1, count, figsize=(5 * count, 6))
    if count == 1:
        axs = [axs]  # Ensure axs is a list even with a single shape
    else:
        axs = list(axs)

    # Draw each shape
    for i in range(count):
        ax = axs[i]

        if shape == FractionShape.RECTANGLE:
            # Draw each segment fully shaded with common denominator divisions
            for j in range(divisions):
                ax.add_patch(
                    patches.Rectangle(
                        (j / divisions, 0),
                        1 / divisions,
                        1,
                        edgecolor="black",
                        facecolor="lightblue",
                        linewidth=3,
                    )
                )
        elif shape == FractionShape.CIRCLE:
            # Draw circle with fully shaded wedges using common denominator
            for j in range(divisions):
                angle_start = 360 * j / divisions + 90  # Start at 12 o'clock
                angle_end = 360 * (j + 1) / divisions + 90
                ax.add_patch(
                    patches.Wedge(
                        (0.5, 0.5),
                        0.5,
                        angle_start,
                        angle_end,
                        facecolor="lightblue",
                        edgecolor="black",
                        linewidth=3,
                    )
                )

        # Set the limits, aspect, and styling
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    # Save the figure
    timestamp = int(time.time())
    dest = settings.additional_content_settings
    file_name = f"{dest.image_destination_folder}/{timestamp}_whole_frac_model.{dest.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=650,
        transparent=False,
        bbox_inches="tight",
        format=dest.stimulus_image_format,
    )
    plt.close(fig)
    return file_name


def _generate_pastel_color():
    """Generate a random bright pastel color with vibrant, appealing tones."""
    # Create brighter pastels by using higher saturation and optimal lightness
    # This creates vibrant but soft colors similar to the purple shown in the reference image

    # Define beautiful color palettes for brighter pastels
    bright_pastel_colors = [
        # Purple/Lavender family (like in the image)
        (0.75, 0.65, 0.85),  # Light purple
        (0.80, 0.70, 0.90),  # Lavender
        (0.85, 0.75, 0.95),  # Pale lavender
        # Pink family
        (0.95, 0.75, 0.85),  # Light pink
        (0.90, 0.70, 0.80),  # Rose pink
        (0.95, 0.80, 0.90),  # Soft pink
        # Blue family
        (0.75, 0.85, 0.95),  # Light blue
        (0.70, 0.80, 0.90),  # Sky blue
        (0.80, 0.90, 0.95),  # Pale blue
        # Green family
        (0.80, 0.90, 0.75),  # Light green
        (0.75, 0.85, 0.70),  # Mint green
        (0.85, 0.95, 0.80),  # Pale green
        # Yellow/Peach family
        (0.95, 0.90, 0.70),  # Light yellow
        (0.95, 0.85, 0.75),  # Peach
        (0.90, 0.80, 0.70),  # Light orange
        # Coral/Salmon family
        (0.95, 0.80, 0.75),  # Coral
        (0.90, 0.75, 0.70),  # Salmon
        (0.95, 0.85, 0.80),  # Light coral
    ]

    # Randomly select from the predefined bright pastel colors
    return random.choice(bright_pastel_colors)


def _darken_color(color, factor=0.3):
    """Create a darker version of the given color for borders."""
    r, g, b = color
    # Reduce each RGB component by the factor to make it darker
    return (r * (1 - factor), g * (1 - factor), b * (1 - factor))


@stimulus_function
def draw_fraction_strips(model_data: FractionStrips):
    """
    Function to draw fraction strips showing fraction decomposition.
    - For whole_number=1: Original behavior with 2-3 stacked rectangles
    - For whole_number>1: Shows n copies of the fraction strips in a 2x2 grid layout
    """
    whole_number = model_data.whole_number
    splits = model_data.splits
    first_division = model_data.first_division
    second_division = model_data.second_division
    target_numerator = model_data.target_numerator
    # Note: target_denominator is used for validation only, not in drawing logic

    # If target_numerator is not provided, color all segments by default
    if target_numerator is None:
        if splits == 2:
            target_numerator = first_division
        else:  # splits == 3
            target_numerator = second_division if second_division else first_division

    # Use light yellow for whole number, generate colors for fraction parts
    whole_color = "lightyellow"
    first_division_color = _generate_pastel_color()
    second_division_color = _generate_pastel_color() if splits == 3 else None

    # Note: Using black borders for all rectangles for consistency

    if whole_number == 1:
        # Original single fraction strip behavior
        return _draw_single_fraction_strip(
            splits,
            first_division,
            second_division,
            target_numerator,
            whole_color,
            first_division_color,
            second_division_color,
        )
    else:
        # Modular 2x2 grid layout with n copies of the fraction strip
        return _draw_grid_fraction_strips(
            whole_number,
            splits,
            first_division,
            second_division,
            target_numerator,
            whole_color,
            first_division_color,
            second_division_color,
        )


def _draw_single_fraction_strip(
    splits,
    first_division,
    second_division,
    target_numerator,
    whole_color,
    first_division_color,
    second_division_color,
):
    """Draw the original single fraction strip (whole_number=1)"""
    # Create figure with very compact layout
    fig_height = 0.6 * splits + 0.2  # Very compact height
    fig, axs = plt.subplots(splits, 1, figsize=(12, fig_height))
    plt.subplots_adjust(
        hspace=0.02, left=0.05, right=0.95, top=0.95, bottom=0.05
    )  # Minimal spacing

    if splits == 2:
        axs = [axs[0], axs[1]]
    else:
        axs = [axs[0], axs[1], axs[2]]

    # First rectangle - whole with "1" in the middle
    ax1 = axs[0]
    ax1.add_patch(
        patches.Rectangle(
            (0, 0),
            3,
            0.3,  # Reduced height for more compact look
            edgecolor="black",
            facecolor=whole_color,
            linewidth=2,  # Thinner lines
        )
    )
    ax1.text(
        1.5, 0.15, "1", ha="center", va="center", fontsize=16, fontweight="bold"
    )  # Adjusted position and size
    ax1.set_xlim(-0.05, 3.05)
    ax1.set_ylim(-0.02, 0.32)
    ax1.set_aspect("equal")
    ax1.axis("off")

    if splits == 2:
        # Second rectangle - split into unit fractions
        ax2 = axs[1]
        for i in range(first_division):
            # Color only the first target_numerator segments
            is_colored = i < target_numerator
            facecolor = first_division_color if is_colored else "white"

            ax2.add_patch(
                patches.Rectangle(
                    (i * 3 / first_division, 0),
                    3 / first_division,
                    0.3,  # Reduced height
                    edgecolor="black",
                    facecolor="lightcoral",
                    linewidth=2,  # Thinner lines
                )
            )
            # Add fraction label
            center_x = (i + 0.5) * 3 / first_division
            ax2.text(
                center_x,
                0.21,  # Adjusted position
                "1",
                ha="center",
                va="center",
                fontsize=12,  # Smaller font
                fontweight="bold",
            )
            # Add fraction line
            line_half_width = 0.04  # Smaller line
            ax2.plot(
                [center_x - line_half_width, center_x + line_half_width],
                [0.15, 0.15],  # Adjusted position
                "k-",
                linewidth=1.5,  # Thinner line
            )
            # Add denominator
            ax2.text(
                center_x,
                0.09,  # Adjusted position
                str(first_division),
                ha="center",
                va="center",
                fontsize=12,  # Smaller font
                fontweight="bold",
            )

        ax2.set_xlim(-0.05, 3.05)  # Tighter bounds
        ax2.set_ylim(-0.02, 0.32)  # Tighter bounds
        ax2.set_aspect("equal")
        ax2.axis("off")

    else:  # splits == 3
        # Second rectangle - single unit fraction
        ax2 = axs[1]
        unit_width = 3 / first_division
        ax2.add_patch(
            patches.Rectangle(
                (0, 0),
                unit_width,
                0.3,  # Reduced height to match
                edgecolor="black",
                facecolor="orange",
                linewidth=2,  # Thinner lines
            )
        )
        center_x = unit_width / 2
        ax2.text(
            center_x,
            0.21,  # Adjusted position to match
            "1",
            ha="center",
            va="center",
            fontsize=12,  # Smaller font
            fontweight="bold",
        )
        line_half_width = 0.04  # Smaller line
        ax2.plot(
            [center_x - line_half_width, center_x + line_half_width],
            [0.15, 0.15],  # Adjusted position to match
            "k-",
            linewidth=1.5,  # Thinner line
        )
        ax2.text(
            center_x,
            0.09,  # Adjusted position to match
            str(first_division),
            ha="center",
            va="center",
            fontsize=12,  # Smaller font
            fontweight="bold",
        )

        ax2.set_xlim(-0.05, 3.05)  # Tighter bounds
        ax2.set_ylim(-0.02, 0.32)  # Tighter bounds
        ax2.set_aspect("equal")
        ax2.axis("off")

        # Third rectangle - division of the unit fraction
        ax3 = axs[2]
        for i in range(second_division):
            # Color only the first target_numerator segments
            is_colored = i < target_numerator
            facecolor = second_division_color if is_colored else "white"

            ax3.add_patch(
                patches.Rectangle(
                    (i * unit_width / second_division, 0),
                    unit_width / second_division,
                    0.3,  # Reduced height to match
                    edgecolor="black",
                    facecolor=facecolor,
                    linewidth=2,  # Thinner lines
                )
            )
            center_x = (i + 0.5) * unit_width / second_division
            ax3.text(
                center_x,
                0.21,  # Adjusted position to match
                "1",
                ha="center",
                va="center",
                fontsize=10,  # Smaller font
                fontweight="bold",
            )
            line_half_width = 0.03  # Smaller line
            ax3.plot(
                [center_x - line_half_width, center_x + line_half_width],
                [0.15, 0.15],  # Adjusted position to match
                "k-",
                linewidth=1.5,  # Thinner line
            )
            ax3.text(
                center_x,
                0.09,  # Adjusted position to match
                str(first_division * second_division),
                ha="center",
                va="center",
                fontsize=10,  # Smaller font
                fontweight="bold",
            )

        ax3.set_xlim(-0.05, 3.05)  # Tighter bounds
        ax3.set_ylim(-0.02, 0.32)  # Tighter bounds
        ax3.set_aspect("equal")
        ax3.axis("off")

    plt.tight_layout()

    # Save the figure
    timestamp = int(time.time())
    dest = settings.additional_content_settings
    file_name = f"{dest.image_destination_folder}/{timestamp}_fraction_strips.{dest.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=650,
        transparent=False,
        bbox_inches="tight",
        format=dest.stimulus_image_format,
    )
    plt.close(fig)
    return file_name


def _draw_grid_fraction_strips(
    whole_number,
    splits,
    first_division,
    second_division=None,
    target_numerator=None,
    whole_color=None,
    first_division_color=None,
    second_division_color=None,
):
    """Draw n identical copies of the same fraction strip in a 2x2 grid layout"""
    # Create figure with very compact size based on whole_number and splits
    if whole_number <= 2:
        # For 1-2 wholes: very compact horizontal layout
        fig_height = 0.8 + (splits * 0.6)  # Very compact vertical spacing
        fig = plt.figure(figsize=(12, fig_height))
    else:
        # For 3-4 wholes: compact 2x2 grid
        fig_height = 1.2 + (splits * 0.8)  # Compact space for 2x2 grid
        fig = plt.figure(figsize=(12, fig_height))

    # Calculate grid positions and subplot layout based on whole_number
    if whole_number <= 2:
        # For 1-2 wholes: arrange horizontally in a single row
        grid_positions = [(0, 0), (0, 1)]
    else:
        # For 3-4 wholes: use full 2x2 grid
        grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for grid_idx in range(whole_number):
        row, col = grid_positions[grid_idx]

        # Create subplot for this grid cell
        # Each cell contains 'splits' number of fraction strips stacked vertically
        for strip_idx in range(splits):
            if whole_number <= 2:
                # For 1-2 wholes: horizontal layout with tighter vertical spacing
                subplot_idx = strip_idx * whole_number + grid_idx + 1
                ax = plt.subplot(splits, whole_number, subplot_idx)
            else:
                # For 3-4 wholes: 2x2 grid layout
                subplot_idx = (row * splits + strip_idx) * 2 + col + 1
                ax = plt.subplot(splits * 2, 2, subplot_idx)

            if strip_idx == 0:
                # First strip - whole rectangle with "1"
                ax.add_patch(
                    patches.Rectangle(
                        (0, 0),
                        3,
                        0.3,  # Reduced height
                        edgecolor="black",
                        facecolor=whole_color,
                        linewidth=2,
                    )
                )
                ax.text(
                    1.5,
                    0.15,  # Adjusted position
                    "1",
                    ha="center",
                    va="center",
                    fontsize=14,  # Smaller font
                    fontweight="bold",
                )

            elif strip_idx == 1:
                # Second strip - divided into fractions
                if splits == 2:
                    # For 2 splits: show all unit fractions
                    for i in range(first_division):
                        # Color only the first target_numerator segments
                        is_colored = i < target_numerator
                        facecolor = first_division_color if is_colored else "white"

                        ax.add_patch(
                            patches.Rectangle(
                                (i * 3 / first_division, 0),
                                3 / first_division,
                                0.3,  # Reduced height to match
                                edgecolor="black",
                                facecolor=facecolor,
                                linewidth=2,
                            )
                        )
                        # Add fraction labels
                        center_x = (i + 0.5) * 3 / first_division
                        ax.text(
                            center_x,
                            0.21,  # Adjusted position to match
                            "1",
                            ha="center",
                            va="center",
                            fontsize=10,  # Smaller font for grid
                            fontweight="bold",
                        )
                        # Add fraction line
                        line_half_width = 0.03  # Smaller for grid
                        ax.plot(
                            [center_x - line_half_width, center_x + line_half_width],
                            [0.15, 0.15],  # Adjusted position to match
                            "k-",
                            linewidth=1.5,
                        )
                        # Add denominator
                        ax.text(
                            center_x,
                            0.09,  # Adjusted position to match
                            str(first_division),
                            ha="center",
                            va="center",
                            fontsize=10,  # Smaller font for grid
                            fontweight="bold",
                        )
                else:  # splits == 3
                    # For 3 splits: show single unit fraction
                    unit_width = 3 / first_division
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0),
                            unit_width,
                            0.3,  # Reduced height to match
                            edgecolor="black",
                            facecolor=first_division_color,
                            linewidth=2,
                        )
                    )
                    center_x = unit_width / 2
                    ax.text(
                        center_x,
                        0.21,  # Adjusted position to match
                        "1",
                        ha="center",
                        va="center",
                        fontsize=10,  # Smaller font for grid
                        fontweight="bold",
                    )
                    line_half_width = 0.03  # Smaller for grid
                    ax.plot(
                        [center_x - line_half_width, center_x + line_half_width],
                        [0.15, 0.15],  # Adjusted position to match
                        "k-",
                        linewidth=1.5,
                    )
                    ax.text(
                        center_x,
                        0.09,  # Adjusted position to match
                        str(first_division),
                        ha="center",
                        va="center",
                        fontsize=10,  # Smaller font for grid
                        fontweight="bold",
                    )

            elif strip_idx == 2:  # Third strip (only for splits == 3)
                # Third strip - division of the unit fraction
                unit_width = 3 / first_division
                for i in range(second_division):
                    # Color only the first target_numerator segments
                    is_colored = i < target_numerator
                    facecolor = second_division_color if is_colored else "white"

                    ax.add_patch(
                        patches.Rectangle(
                            (i * unit_width / second_division, 0),
                            unit_width / second_division,
                            0.3,  # Reduced height to match
                            edgecolor="black",
                            facecolor=facecolor,
                            linewidth=2,
                        )
                    )
                    center_x = (i + 0.5) * unit_width / second_division
                    ax.text(
                        center_x,
                        0.21,  # Adjusted position to match
                        "1",
                        ha="center",
                        va="center",
                        fontsize=8,  # Even smaller for third strip
                        fontweight="bold",
                    )
                    line_half_width = 0.02  # Smaller for third strip
                    ax.plot(
                        [center_x - line_half_width, center_x + line_half_width],
                        [0.15, 0.15],  # Adjusted position to match
                        "k-",
                        linewidth=1.5,
                    )
                    ax.text(
                        center_x,
                        0.09,  # Adjusted position to match
                        str(first_division * second_division),
                        ha="center",
                        va="center",
                        fontsize=8,  # Even smaller for third strip
                        fontweight="bold",
                    )

            ax.set_xlim(-0.05, 3.05)  # Tighter bounds
            ax.set_ylim(-0.02, 0.32)  # Tighter bounds
            ax.set_aspect("equal")
            ax.axis("off")

    # Fill empty grid positions with empty subplots (only for 3-4 wholes)
    if whole_number > 2:
        for grid_idx in range(whole_number, 4):
            row, col = grid_positions[grid_idx]
            for strip_idx in range(splits):
                subplot_idx = (row * splits + strip_idx) * 2 + col + 1
                ax = plt.subplot(splits * 2, 2, subplot_idx)
                ax.axis("off")

    # Apply very compact layout with minimal spacing
    if whole_number <= 2:
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.05,
            hspace=0.02,
            wspace=0.05,  # Minimal spacing for 1-2 wholes
        )
    else:
        plt.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.05,
            hspace=0.05,
            wspace=0.1,  # Compact spacing for 3-4 wholes
        )

    # Save the figure
    timestamp = int(time.time())
    dest = settings.additional_content_settings
    file_name = f"{dest.image_destination_folder}/{timestamp}_fraction_strips.{dest.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=650,
        transparent=False,
        bbox_inches="tight",
        format=dest.stimulus_image_format,
    )
    plt.close(fig)
    return file_name


if __name__ == "__main__":
    # Testing function
    stimulus_description = [
        {"shape": "Rectangle", "fraction": "1/12"},
        {"shape": "Circle", "fraction": "1/10"},
    ]
    print(draw_fractional_models(stimulus_description))


def _draw_bar(ax, x0, y0, w, h, parts, shaded, edge, fill):
    """
    Helper: draw one bar split into `parts` equal sections,
            with the first `shaded` of them coloured.
    """
    part_w = w / parts
    for k in range(parts):
        fc = fill if k < shaded else "white"
        ax.add_patch(
            patches.Rectangle(
                (x0 + k * part_w, y0),
                part_w,
                h,
                linewidth=2,
                edgecolor=edge,
                facecolor=fc,
            )
        )


@stimulus_function
def draw_division_model(stimulus: DivisionModel, *, figsize=None):
    """
    Draws a fraction bar model for a single division exercise.

    Parameters
    ----------
    stimulus : DivisionModel
        Pydantic model containing dividend and divisor information
    figsize  : tuple, optional
        Size of the matplotlib figure. If None, will be calculated dynamically.
    """
    # ------------------------------------------------------------------
    # 1) Work out three things:
    #    a) how many separate bars to draw              (# of wholes)
    #    b) how many equal pieces each bar must show    (denominator)
    #    c) how many of those pieces get shaded         (numerator)
    # ------------------------------------------------------------------

    # Convert Pydantic models to the format expected by the original function
    def convert_operand(operand):
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, FractionNumber):
            return {"numerator": operand.numerator, "denominator": operand.denominator}
        else:
            raise ValueError(f"Unexpected operand type: {type(operand)}")

    dividend = convert_operand(stimulus.dividend)
    divisor = convert_operand(stimulus.divisor)

    # Which operand contains the *proper* fraction that supplies
    # the denominator for the model?
    def is_frac(x):
        return isinstance(x, dict)

    def denom(x):
        return x["denominator"]

    def numer(x):
        return x["numerator"]

    if is_frac(dividend):  #  dividend is the fraction
        n_bars = 1
        pieces = denom(dividend)
        shaded = numer(dividend)
        edge, fill = "#0077ff", "#8ec3ff"  # blue scheme
    else:  #  dividend is a whole number
        n_bars = int(dividend)
        frac_source = divisor if is_frac(divisor) else dividend
        pieces = denom(frac_source)
        shaded = pieces  # all parts filled
        edge, fill = "#2a9d47", "#8dde9a"  # green scheme

    # ------------------------------------------------------------------
    # 2) Draw
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    bar_w, bar_h = pieces, 1
    gap = 0.2  # reduced space between bars

    for b in range(n_bars):
        x_start = b * (bar_w + gap)
        _draw_bar(
            ax,
            x_start,
            0,
            bar_w,
            bar_h,
            parts=pieces,
            shaded=shaded,
            edge=edge,
            fill=fill,
        )

    # Calculate total width and set minimal margins
    total_width = n_bars * bar_w + (n_bars - 1) * gap
    margin_x = 0.02 * total_width  # 2% margin
    margin_y = 0.05  # minimal vertical margin

    # Set tight axis limits
    ax.set_xlim(-margin_x, total_width + margin_x)
    ax.set_ylim(-margin_y, bar_h + margin_y)

    # Calculate optimal figure size based on content
    if figsize is None:
        # Aspect ratio to minimize white space
        content_width = total_width + 2 * margin_x
        content_height = bar_h + 2 * margin_y
        aspect_ratio = content_width / content_height

        # Set reasonable limits to avoid encoding issues
        max_width = 20.0  # Maximum 20 inches wide
        fig_height = 2.0  # Fixed height for consistency
        fig_width = min(fig_height * aspect_ratio, max_width)

        # If we hit the width limit, adjust height to maintain reasonable proportions
        if fig_width == max_width:
            fig_height = max_width / aspect_ratio
            # Ensure minimum height
            fig_height = max(fig_height, 1.0)

        fig.set_size_inches(fig_width, fig_height)

    plt.tight_layout(pad=0.1)  # minimal padding

    # Save the file following the established pattern
    timestamp = int(time.time())
    dest = settings.additional_content_settings
    file_name = f"{dest.image_destination_folder}/{timestamp}_division_model.{dest.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        transparent=False,
        bbox_inches="tight",
        format=dest.stimulus_image_format,
    )
    plt.close(fig)
    return file_name
