import colorsys
import math
import random
import time
from collections import defaultdict
from fractions import Fraction
from math import sqrt
from typing import List

import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.drawing_functions.prism_nets import (
    Label,
    labels_overlap,
    measure_text_dimensions,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.area_stimulus import (
    AreaStimulusParams,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.circle import (
    Circle,
    CircleDiagram,
    CircleElementType,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.circle_arc import (
    CircleWithArcsDescription,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.complex_figure import (
    CompositeRectangularGrid,
    CompositeRectangularTriangularGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.composite_rectangular_prism import (
    CompositeRectangularPrism,
    CompositeRectangularPrism2,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.geometric_shapes import (
    GeometricShapeList,
    GeometricShapeListWithRotation,
    GeometricShapeWithAngleList,
    ParallelQuadrilateral,
    QuadrilateralVennDiagram,
    RegularIrregularPolygonList,
    ShapeWithRightAngles,
    ValidGeometricShape,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_fully_labeled import (
    PolygonFullyLabeled,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    PolygonList,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_perimeter import (
    PolygonPerimeter,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_string_sides import (
    PolygonStringSides,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.quadrilateral_figures import (
    ParallelogramWithHeight,
    QuadrilateralFigures,
    QuadrilateralShapeType,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangle_with_area import (
    RectangleWithHiddenSide,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid import (
    MultipleGrids,
    RectangularGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_grid_list import (
    RectangularGridList,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.shape import (
    Shape,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.similar_triangles import (
    SimilarRightTriangles,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.trapezoid_grid import (
    ETrapezoidType,
    TrapezoidGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.triangular_grid_list import (
    TriangularGrid,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.triangular_grid_opt import (
    TriangularGridOpt,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.unit_squares import (
    UnitSquares,
)
from content_generators.settings import settings
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)


# Color selection for visual variety
def get_random_polygon_color() -> str:
    """
    Randomly select a color from a set of regular, visually distinct colors.

    Returns:
        str: A color name suitable for matplotlib plotting
    """
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "darkgreen",
        "darkblue",
    ]
    return random.choice(colors)


# Fraction handling helper functions for educational dimension display
def decimal_to_mixed_number(decimal_value: float, tolerance: float = 1e-6) -> str:
    """
    Convert decimal numbers to mixed number format for educational display.
    Handles common fractions like 1/3, 2/3, 1/2, 1/4, etc.

    Examples:
        3.333333 -> "3 1/3"
        6.666667 -> "6 2/3"
        4.5 -> "4 1/2"
        2.25 -> "2 1/4"
        5.0 -> "5"
    """
    # Handle exact integers
    if abs(decimal_value - round(decimal_value)) < tolerance:
        return str(int(round(decimal_value)))

    # Convert to fraction with reasonable denominator limit
    frac = Fraction(decimal_value).limit_denominator(100)

    if frac.denominator == 1:
        return str(frac.numerator)

    # Convert improper fraction to mixed number
    whole_part = frac.numerator // frac.denominator
    remainder = frac.numerator % frac.denominator

    if whole_part == 0:
        return f"{remainder}/{frac.denominator}"
    else:
        return f"{whole_part} {remainder}/{frac.denominator}"


def should_use_fraction_display(value: float) -> bool:
    """
    Determine if a decimal value should be displayed as a fraction.
    Returns True for values that have nice fraction representations.
    """
    # Convert to fraction and check if it has a reasonable denominator
    frac = Fraction(value).limit_denominator(100)

    # Use fractions for denominators that are commonly used in education
    common_denominators = {2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20}

    return frac.denominator in common_denominators and abs(value - float(frac)) < 1e-6


def format_dimension_label(value: float, unit: str) -> str:
    """
    Format a dimension value with appropriate fraction display and units.

    Examples:
        format_dimension_label(3.333333, "ft") -> "3 1/3 ft"
        format_dimension_label(4.5, "cm") -> "4 1/2 cm"
        format_dimension_label(5.0, "in") -> "5 in"
    """
    if should_use_fraction_display(value):
        fraction_str = decimal_to_mixed_number(value)
        return f"{fraction_str} {unit}"
    else:
        # For non-fraction values, format to avoid unnecessary decimals
        if value == int(value):
            return f"{int(value)} {unit}"
        else:
            return f"{value:.2f} {unit}"


def _draw_side_dash(
    ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness, frac=0.5
):
    px = x1 + (x2 - x1) * frac
    py = y1 + (y2 - y1) * frac
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    perp_dx = -dy / length
    perp_dy = dx / length
    start_x = px - perp_dx * dash_length / 2
    start_y = py - perp_dy * dash_length / 2
    end_x = px + perp_dx * dash_length / 2
    end_y = py + perp_dy * dash_length / 2
    ax.plot(
        [start_x, end_x], [start_y, end_y], color=dash_color, linewidth=dash_thickness
    )


def _draw_side_double_dash(ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness):
    _draw_side_dash(
        ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness, frac=9 / 18
    )
    _draw_side_dash(
        ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness, frac=9.5 / 18
    )


@stimulus_function
def draw_circle_with_radius(data: Circle):
    fig, ax = plt.subplots()
    draw_circle(ax, "", data.radius, color="black", fill_color="white")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/circle_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    # plt.show()
    return file_name


@stimulus_function
def draw_circle_diagram(data: CircleDiagram):
    """
    Draw a comprehensive circle diagram with support for a radius, a diameter, or a chord element
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set up the plot
        ax.set_xlim(-data.radius * 1.5, data.radius * 1.5)
        ax.set_ylim(-data.radius * 1.5, data.radius * 1.5)

        # Draw the main circle using existing function
        draw_circle(ax, title="", radius=data.radius, color="black", fill_color="white")

        # Draw the single element (radius, diameter, or chord) if specified
        if data.element:
            element = data.element
            # Generate a random angle (0-360 degrees)
            random_angle_degrees = random.uniform(0, 360)
            start_angle_rad = np.radians(random_angle_degrees)
            start_x = data.radius * np.cos(start_angle_rad)
            start_y = data.radius * np.sin(start_angle_rad)

            if element.element_type == CircleElementType.RADIUS:
                # Draw radius from center to circumference
                ax.plot([0, start_x], [0, start_y], "k-", linewidth=2)

                # Draw points at both endpoints
                ax.plot(0, 0, "ko", markersize=6)  # Center point
                ax.plot(start_x, start_y, "ko", markersize=6)  # Circumference point

                # Add endpoint labels
                if len(element.endpoint_labels) >= 2:
                    label_offset = 0.08 * data.radius
                    # Label at center (first endpoint)
                    center_label_x = -label_offset * np.cos(start_angle_rad)
                    center_label_y = -label_offset * np.sin(start_angle_rad)
                    ax.text(
                        center_label_x,
                        center_label_y,
                        element.endpoint_labels[0],
                        fontsize=16,
                        ha="center",
                        va="center",
                    )
                    # Label at circumference (second endpoint)
                    label_x = start_x + label_offset * np.cos(start_angle_rad)
                    label_y = start_y + label_offset * np.sin(start_angle_rad)
                    ax.text(
                        label_x,
                        label_y,
                        element.endpoint_labels[1],
                        fontsize=16,
                        ha="center",
                        va="center",
                    )

            elif element.element_type == CircleElementType.DIAMETER:
                # Draw diameter across the circle through center
                end_x = -start_x  # Opposite point
                end_y = -start_y
                ax.plot([start_x, end_x], [start_y, end_y], "k-", linewidth=2)

                # Draw points at both endpoints and center
                ax.plot(0, 0, "ko", markersize=6)  # Center point
                ax.plot(start_x, start_y, "ko", markersize=6)  # First endpoint
                ax.plot(end_x, end_y, "ko", markersize=6)  # Second endpoint

                # Add endpoint labels
                if len(element.endpoint_labels) >= 2:
                    label_offset = 0.08 * data.radius
                    # First endpoint
                    label1_x = start_x + label_offset * np.cos(start_angle_rad)
                    label1_y = start_y + label_offset * np.sin(start_angle_rad)
                    ax.text(
                        label1_x,
                        label1_y,
                        element.endpoint_labels[0],
                        fontsize=16,
                        ha="center",
                        va="center",
                    )
                    # Second endpoint (opposite side)
                    label2_x = end_x - label_offset * np.cos(start_angle_rad)
                    label2_y = end_y - label_offset * np.sin(start_angle_rad)
                    ax.text(
                        label2_x,
                        label2_y,
                        element.endpoint_labels[1],
                        fontsize=16,
                        ha="center",
                        va="center",
                    )

            elif element.element_type == CircleElementType.CHORD:
                # For chord, use a span of 120 degrees centered on the specified angle
                chord_span = 120  # degrees
                chord_start_angle = start_angle_rad - np.radians(chord_span / 2)
                chord_end_angle = start_angle_rad + np.radians(chord_span / 2)

                chord_start_x = data.radius * np.cos(chord_start_angle)
                chord_start_y = data.radius * np.sin(chord_start_angle)
                chord_end_x = data.radius * np.cos(chord_end_angle)
                chord_end_y = data.radius * np.sin(chord_end_angle)

                ax.plot(
                    [chord_start_x, chord_end_x],
                    [chord_start_y, chord_end_y],
                    "k-",
                    linewidth=2,
                )

                # Draw points at both endpoints
                ax.plot(
                    chord_start_x, chord_start_y, "ko", markersize=6
                )  # First endpoint
                ax.plot(chord_end_x, chord_end_y, "ko", markersize=6)  # Second endpoint

                # Add endpoint labels
                if len(element.endpoint_labels) >= 2:
                    label_offset = 0.08 * data.radius
                    # First endpoint
                    label1_x = chord_start_x + label_offset * np.cos(chord_start_angle)
                    label1_y = chord_start_y + label_offset * np.sin(chord_start_angle)
                    ax.text(
                        label1_x,
                        label1_y,
                        element.endpoint_labels[0],
                        fontsize=16,
                        ha="center",
                        va="center",
                    )
                    # Second endpoint
                    label2_x = chord_end_x + label_offset * np.cos(chord_end_angle)
                    label2_y = chord_end_y + label_offset * np.sin(chord_end_angle)
                    ax.text(
                        label2_x,
                        label2_y,
                        element.endpoint_labels[1],
                        fontsize=16,
                        ha="center",
                        va="center",
                    )

        # Save the figure
        file_name = f"{settings.additional_content_settings.image_destination_folder}/circle_diagram_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
        plt.tight_layout()
        plt.savefig(
            file_name,
            dpi=800,
            transparent=False,
            bbox_inches="tight",
            format=settings.additional_content_settings.stimulus_image_format,
        )
        plt.close()

        return file_name
    except Exception as e:
        print(f"Error drawing circle diagram: {e}")
        raise


############################
# Geometric Shapes - Basic #
############################
def draw_rectangle(ax, title="Rectangle", width=3, height=2, color="blue", rotation=0):
    x = np.array([0, width, width, 0, 0])
    y = np.array([0, 0, height, height, 0])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Add 90-degree angle markers (small squares) at each corner - calculated from rotated vertices
    marker_size = (
        min(width, height) * 0.1
    )  # Size of the angle marker relative to the shape

    # Get the rotated vertices for marker calculation
    corners = [(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])]

    # Draw right angle markers at each corner
    for i in range(4):
        corner_x, corner_y = corners[i]
        next_corner_x, next_corner_y = corners[(i + 1) % 4]
        prev_corner_x, prev_corner_y = corners[(i - 1) % 4]

        # Calculate direction vectors from corner to adjacent corners
        dx1 = next_corner_x - corner_x
        dy1 = next_corner_y - corner_y
        len1 = np.sqrt(dx1**2 + dy1**2)
        if len1 > 0:
            dx1, dy1 = dx1 / len1, dy1 / len1

        dx2 = prev_corner_x - corner_x
        dy2 = prev_corner_y - corner_y
        len2 = np.sqrt(dx2**2 + dy2**2)
        if len2 > 0:
            dx2, dy2 = dx2 / len2, dy2 / len2

        # Create marker square
        p1_x = corner_x + marker_size * dx1
        p1_y = corner_y + marker_size * dy1
        p2_x = corner_x + marker_size * dx2
        p2_y = corner_y + marker_size * dy2
        p3_x = p1_x + marker_size * dx2
        p3_y = p1_y + marker_size * dy2

        ax.plot(
            [corner_x, p1_x, p3_x, p2_x, corner_x],
            [corner_y, p1_y, p3_y, p2_y, corner_y],
            color=color,
            linewidth=1,
        )

    # Dash parameters
    dash_length = min(width, height) * 0.06
    dash_color = "red"
    dash_thickness = 2.5

    # Helper to draw a dash at a given fraction along a side, midpoint on the side
    def draw_side_dash(x1, y1, x2, y2, frac=0.5):
        px = x1 + (x2 - x1) * frac
        py = y1 + (y2 - y1) * frac
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        perp_dx = -dy / length
        perp_dy = dx / length
        # Dash is centered on the side, extends equally inside and outside
        start_x = px - perp_dx * dash_length / 2
        start_y = py - perp_dy * dash_length / 2
        end_x = px + perp_dx * dash_length / 2
        end_y = py + perp_dy * dash_length / 2
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            color=dash_color,
            linewidth=dash_thickness,
        )

    # For double dashes, draw at 9/18 and 9.5/18 along the side (almost touching)
    def draw_side_double_dash(x1, y1, x2, y2):
        draw_side_dash(x1, y1, x2, y2, frac=9 / 18)
        draw_side_dash(x1, y1, x2, y2, frac=9.5 / 18)

    # Draw dashes for equivalent sides using rotated coordinates
    if width == height:  # Square
        # All sides are equal, draw single dash on each side
        for i in range(4):
            draw_side_dash(x[i], y[i], x[i + 1], y[i + 1])
    else:  # Rectangle
        # Draw single dash on shorter sides, double dash on longer sides
        if width < height:
            draw_side_dash(x[0], y[0], x[1], y[1])  # Bottom
            draw_side_double_dash(x[1], y[1], x[2], y[2])  # Right
            draw_side_dash(x[2], y[2], x[3], y[3])  # Top
            draw_side_double_dash(x[3], y[3], x[0], y[0])  # Left
        else:
            draw_side_double_dash(x[0], y[0], x[1], y[1])  # Bottom
            draw_side_dash(x[1], y[1], x[2], y[2])  # Right
            draw_side_double_dash(x[2], y[2], x[3], y[3])  # Top
            draw_side_dash(x[3], y[3], x[0], y[0])  # Left

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_circle(ax, title="Circle", radius: float = 1, color="blue", fill_color=None):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, color=color)
    ax.fill(x, y, color=fill_color if fill_color is not None else color, alpha=0.3)
    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_regular_polygon(
    ax, title="Polygon", sides=3, radius=1, color="blue", rotation=0
):
    theta = np.linspace(0, 2 * np.pi, sides + 1)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Draw dashes for all sides (all equal) - using rotated coordinates
    dash_length = radius * 0.09
    dash_color = "red"
    dash_thickness = 2.5
    for i in range(sides):
        _draw_side_dash(
            ax, x[i], y[i], x[i + 1], y[i + 1], dash_length, dash_color, dash_thickness
        )

    # Add right angle markers for regular quadrilateral (square) - using rotated coordinates
    if sides == 4:
        marker_size = radius * 0.1  # Size relative to the radius

        # For each corner, add a right angle marker
        for i in range(4):
            # Get current vertex and the two adjacent vertices
            curr_x, curr_y = x[i], y[i]
            next_x, next_y = x[i + 1], y[i + 1]
            prev_x, prev_y = x[(i - 1) % 4], y[(i - 1) % 4]

            # Calculate unit vectors along the two edges from current vertex
            # Vector to next vertex
            dx1, dy1 = next_x - curr_x, next_y - curr_y
            len1 = np.sqrt(dx1**2 + dy1**2)
            if len1 > 0:
                dx1, dy1 = dx1 / len1, dy1 / len1

            # Vector to previous vertex
            dx2, dy2 = prev_x - curr_x, prev_y - curr_y
            len2 = np.sqrt(dx2**2 + dy2**2)
            if len2 > 0:
                dx2, dy2 = dx2 / len2, dy2 / len2

            # Create the right angle marker square using the same approach as rectangle
            # Move inward along both edges to create a small square
            p1_x = curr_x + marker_size * dx1
            p1_y = curr_y + marker_size * dy1
            p2_x = curr_x + marker_size * dx2
            p2_y = curr_y + marker_size * dy2
            p3_x = p1_x + marker_size * dx2  # Complete the square
            p3_y = p1_y + marker_size * dy2

            # Draw the right angle marker as a small square (same order as rectangle)
            ax.plot(
                [curr_x, p1_x, p3_x, p2_x, curr_x],
                [curr_y, p1_y, p3_y, p2_y, curr_y],
                color=color,
                linewidth=1,
            )

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_rhombus(
    ax, title="Rhombus", diagonal1=3, diagonal2=4, color="blue", rotation=0
):
    x = np.array([0, diagonal1 / 2, 0, -diagonal1 / 2, 0])
    y = np.array([diagonal2 / 2, 0, -diagonal2 / 2, 0, diagonal2 / 2])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Add dashes for all sides (all equal in a rhombus) - using rotated coordinates
    dash_length = min(diagonal1, diagonal2) * 0.09
    dash_color = "red"
    dash_thickness = 2.5
    for i in range(4):
        _draw_side_dash(
            ax, x[i], y[i], x[i + 1], y[i + 1], dash_length, dash_color, dash_thickness
        )

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_parallelogram(
    ax, title="Parallelogram", base=3, side=2, angle_deg=30, color="blue", rotation=0
):
    angle_rad = np.radians(angle_deg)
    offset_x = np.cos(angle_rad) * side
    offset_y = np.sin(angle_rad) * side
    x = np.array([0, base, base + offset_x, offset_x, 0])
    y = np.array([0, 0, offset_y, offset_y, 0])

    # Apply rotation
    x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Add dashes for equal sides in parallelogram
    dash_length = min(base, side) * 0.09
    dash_color = "red"
    dash_thickness = 2.5

    # Single dashes for base sides (opposite sides are equal)
    _draw_side_dash(
        ax, x[0], y[0], x[1], y[1], dash_length, dash_color, dash_thickness
    )  # Bottom
    _draw_side_dash(
        ax, x[2], y[2], x[3], y[3], dash_length, dash_color, dash_thickness
    )  # Top

    # Double dashes for side edges (opposite sides are equal)
    _draw_side_double_dash(
        ax, x[1], y[1], x[2], y[2], dash_length, dash_color, dash_thickness
    )  # Right
    _draw_side_double_dash(
        ax, x[3], y[3], x[0], y[0], dash_length, dash_color, dash_thickness
    )  # Left

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def _draw_height_line(ax, base_point, top_point, color="black", linestyle="--"):
    """Draw a dashed perpendicular height line with right angle marker."""
    # Draw dashed line
    ax.plot(
        [base_point[0], top_point[0]],
        [base_point[1], top_point[1]],
        color=color,
        linestyle=linestyle,
        linewidth=1,
        zorder=3,  # Ensure height line appears above fill
    )

    # Add right angle marker at base - INSIDE the parallelogram
    marker_size = 0.15  # Slightly smaller for better proportion
    # Draw vertical line first
    ax.plot(
        [base_point[0] - marker_size, base_point[0] - marker_size],
        [base_point[1], base_point[1] + marker_size],
        color=color,
        linestyle="-",  # Solid line for marker
        linewidth=1,
        zorder=3,
    )
    # Draw horizontal line at the TOP of vertical line
    ax.plot(
        [base_point[0], base_point[0] - marker_size],
        [
            base_point[1] + marker_size,
            base_point[1] + marker_size,
        ],  # Changed to be at top
        color=color,
        linestyle="-",  # Solid line for marker
        linewidth=1,
        zorder=3,
    )


def _add_label(
    ax,
    label_text,
    start_point,
    end_point,
    fontsize=14,
    fontstyle="normal",
    label_type=None,
):
    """Add a label to a line segment."""
    # Calculate midpoint for label placement
    mid_x = (start_point[0] + end_point[0]) / 2
    mid_y = (start_point[1] + end_point[1]) / 2

    # Calculate direction vector
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    length = np.sqrt(dx**2 + dy**2)

    if length > 0:
        # Normalize direction vector
        dx, dy = dx / length, dy / length
        # Add perpendicular offset (rotate 90 degrees)
        perp_dx, perp_dy = -dy, dx

        # Position labels based on type (for parallelogram)
        if label_type == "base":
            # Place base label below the base line
            offset = -0.3  # Negative to go below
            label_x = mid_x
            label_y = mid_y + offset
            ha = "center"  # Horizontal alignment
        elif label_type == "height":
            # For height label, check which side has more space
            offset = 0.15  # Reduced from 0.5 to 0.35 to keep label closer to line
            # Use mid_x to determine which side to place label
            if mid_x < 2.5:  # Left half of the figure
                label_x = mid_x + offset
                ha = "left"  # Align left when label is on right side
            else:  # Right half of the figure
                label_x = mid_x - offset
                ha = "right"  # Align right when label is on left side
            label_y = mid_y
        elif label_type == "slant":
            # Place slant label outside the parallelogram
            offset = 0.3
            label_x = mid_x + offset * perp_dx
            label_y = mid_y + offset * perp_dy
            ha = "center"
        else:
            # Default positioning
            offset = 0.3
            label_x = mid_x + offset * perp_dx
            label_y = mid_y + offset * perp_dy
            ha = "center"

        ax.text(
            label_x,
            label_y,
            label_text,
            fontsize=fontsize,
            fontstyle=fontstyle,
            ha=ha,  # Use calculated horizontal alignment
            va="center",
        )


def extract_measurement(label: str) -> float | None:
    """Extract numeric measurement from a label."""
    if label in ["h", "b"]:
        return None
    try:
        return float(label.replace("cm", "").strip())
    except (ValueError, AttributeError):
        return None


def generate_proportional_parallelogram_coordinates(
    base: float, height: float, slant: float
):
    """Generate coordinates for a parallelogram with correct proportions."""
    # Calculate the horizontal shift using actual measurements
    # Using Pythagorean theorem: slant² = height² + horizontal_shift²
    horizontal_shift = math.sqrt(slant**2 - height**2)

    # Create coordinates array with actual proportions
    coords = np.array(
        [
            [0, 0],  # Bottom left
            [base, 0],  # Bottom right
            [base + horizontal_shift, height],  # Top right
            [horizontal_shift, height],  # Top left
            [0, 0],  # Close the shape
        ]
    )

    # Scale to fit view while preserving proportions
    max_dim = max(base, height, slant)  # Compare all dimensions
    scale_factor = 5.0 / max_dim

    # Scale coordinates
    scaled_coords = coords * scale_factor

    # Verify the scaled lengths match the original proportions
    scaled_base = np.linalg.norm(scaled_coords[1] - scaled_coords[0])  # Base length
    scaled_slant = np.linalg.norm(scaled_coords[2] - scaled_coords[1])  # Slant length
    scaled_height = scaled_coords[3][1]  # Height

    # Print for debugging
    print(f"Original - Base: {base}, Height: {height}, Slant: {slant}")
    print(
        f"Scaled - Base: {scaled_base}, Height: {scaled_height}, Slant: {scaled_slant}"
    )
    print(
        f"Ratios - Base/Slant: {base/slant}, Scaled Base/Slant: {scaled_base/scaled_slant}"
    )

    return scaled_coords


def draw_single_parallelogram_with_height(
    ax, data: ParallelogramWithHeight, color="blue"
):
    """Draw a single parallelogram with height line and labels."""
    # Extract measurements
    base = extract_measurement(data.base_label)
    height = extract_measurement(data.height_label)
    slant = extract_measurement(data.slant_side_label)

    if slant is None:
        raise ValueError("Slant side measurement is required")

    # If base or height is unknown ('b' or 'h'), use reasonable values for drawing
    if base is None and height is not None:
        # Calculate minimum base needed for the given height and slant
        min_base = math.sqrt(slant**2 - height**2)
        base = min_base * 1.2  # Add 20% for visual clarity
    elif height is None and base is not None:
        # Calculate maximum possible height for the given base and slant
        max_height = math.sqrt(slant**2 - (base / 2) ** 2)
        height = max_height * 0.8  # Use 80% of max for visual clarity
    elif base is None and height is None:
        raise ValueError("Cannot have both base and height unknown")

    # Generate coordinates with correct proportions
    coords = generate_proportional_parallelogram_coordinates(base, height, slant)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Draw the parallelogram
    ax.plot(x_coords, y_coords, color="black", linewidth=2)
    ax.fill(x_coords, y_coords, color=color, alpha=0.3)

    # Calculate perpendicular foot point
    top_point = [x_coords[3], y_coords[3]]  # Top left vertex

    # Get base line vector and normalize it
    base_vector = [x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]]
    base_length = np.sqrt(base_vector[0] ** 2 + base_vector[1] ** 2)
    base_unit = [base_vector[0] / base_length, base_vector[1] / base_length]

    # Project top point onto base line
    top_to_start = [top_point[0] - x_coords[0], top_point[1] - y_coords[0]]
    projection = top_to_start[0] * base_unit[0] + top_to_start[1] * base_unit[1]

    # Clamp projection to stay within base line segment
    projection = max(0, min(base_length, projection))

    # Calculate foot point
    foot_point = [
        x_coords[0] + projection * base_unit[0],
        y_coords[0] + projection * base_unit[1],
    ]
    # Draw height line from top left to foot point
    _draw_height_line(
        ax,
        foot_point,  # base point where perpendicular meets base
        top_point,  # top left vertex
    )

    # Add labels
    # Base label
    _add_label(
        ax,
        data.base_label,
        [x_coords[0], y_coords[0]],
        [x_coords[1], y_coords[1]],
        label_type="base",  # Add type for specific positioning
    )

    # Height label
    _add_label(
        ax,
        data.height_label,
        foot_point,
        top_point,
        label_type="height",  # Add type for specific positioning
    )

    # Slant side label
    _add_label(
        ax,
        data.slant_side_label,
        [x_coords[1], y_coords[1]],
        [x_coords[2], y_coords[2]],
        label_type="slant",  # Add type for specific positioning
    )
    ax.set_aspect("equal")
    ax.axis("off")


@stimulus_function
def draw_parallelogram_with_height(data: ParallelogramWithHeight):
    """Draw a parallelogram with height line and specific labeling."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_single_parallelogram_with_height(ax, data)
    plt.tight_layout(pad=0.2)

    # Save the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/parallelogram_{int(time.time() * 1000)}.{settings.additional_content_settings.stimulus_image_format}"
    fig.savefig(file_name, transparent=False, bbox_inches="tight", dpi=500)
    return file_name


def draw_scalene_triangle(
    ax, title="Scalene Triangle", sides=[3, 4, 5], color="blue", rotation=0
):
    # Sort sides for consistent labeling
    a, b, c = sorted(sides)
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
    angle_C = np.arccos(cos_C)
    x = np.array([0, c, a * np.cos(angle_C)])
    y = np.array([0, 0, a * np.sin(angle_C)])
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)
    # Dash logic: compare side lengths - using rotated coordinates
    pts = [(x[0], y[0]), (x[1], y[1]), (x[2], y[2])]
    side_lengths = [np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(3)]
    tol = 1e-6

    # Group sides by length (with tolerance)
    groups = []
    used = [False, False, False]
    for i in range(3):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, 3):
            if not used[j] and abs(side_lengths[i] - side_lengths[j]) < tol:
                group.append(j)
        for idx in group:
            used[idx] = True
        groups.append(group)

    # Check for right triangle and add right angle marker using rotated coordinates
    if abs(a**2 + b**2 - c**2) < 1e-6:
        # Find the right angle vertex (opposite the longest side)
        # For right triangle with sides a, b, c where c is hypotenuse,
        # the right angle is at the vertex opposite to c
        right_angle_vertex_idx = 0  # This is the vertex between sides a and b

        # Get the right angle vertex and adjacent vertices
        curr_x, curr_y = x[right_angle_vertex_idx], y[right_angle_vertex_idx]
        next_x, next_y = x[right_angle_vertex_idx + 1], y[right_angle_vertex_idx + 1]
        prev_x, prev_y = (
            x[(right_angle_vertex_idx - 1) % 3],
            y[(right_angle_vertex_idx - 1) % 3],
        )

        # Calculate direction vectors from corner to adjacent corners
        dx1 = next_x - curr_x
        dy1 = next_y - curr_y
        len1 = np.sqrt(dx1**2 + dy1**2)
        if len1 > 0:
            dx1, dy1 = dx1 / len1, dy1 / len1

        dx2 = prev_x - curr_x
        dy2 = prev_y - curr_y
        len2 = np.sqrt(dx2**2 + dy2**2)
        if len2 > 0:
            dx2, dy2 = dx2 / len2, dy2 / len2

        # Create marker square
        marker_size = min(a, b) * 0.1
        p1_x = curr_x + marker_size * dx1
        p1_y = curr_y + marker_size * dy1
        p2_x = curr_x + marker_size * dx2
        p2_y = curr_y + marker_size * dy2
        p3_x = p1_x + marker_size * dx2
        p3_y = p1_y + marker_size * dy2

        ax.plot(
            [curr_x, p1_x, p3_x, p2_x, curr_x],
            [curr_y, p1_y, p3_y, p2_y, curr_y],
            color=color,
            linewidth=1,
        )

    dash_color = "red"
    dash_thickness = 2.5
    dash_length = min(side_lengths) * 0.1

    for group in groups:
        if len(group) > 1:
            # Only draw dashes for equal sides
            for idx in group:
                x1, y1 = pts[idx]
                x2, y2 = pts[(idx + 1) % 3]
                if len(group) == 2:
                    _draw_side_dash(
                        ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness
                    )

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_isosceles_trapezoid(
    ax, title="Isosceles Trapezoid", top_length=2, base_length=3, height=2, color="blue"
):
    # Calculate the coordinates for the isosceles trapezoid
    offset = (base_length - top_length) / 2
    x = np.array([0, base_length, base_length - offset, offset, 0])
    y = np.array([0, 0, height, height, 0])

    # Draw the basic trapezoid shape
    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Add dashes for equal sides in isosceles trapezoid
    dash_length = min(base_length, top_length, height) * 0.09
    dash_color = "red"
    dash_thickness = 2.5

    # Single dashes for the two equal legs (non-parallel sides)
    _draw_side_dash(
        ax, x[1], y[1], x[2], y[2], dash_length, dash_color, dash_thickness
    )  # Right leg
    _draw_side_dash(
        ax, x[3], y[3], x[0], y[0], dash_length, dash_color, dash_thickness
    )  # Left leg

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_right_trapezoid(ax, title="Right Trapezoid", color="blue", rotation=0):
    """Draw a right trapezoid with right angle markers and parallel sides indicators."""
    # Create a right trapezoid with one right angle
    x = np.array([0, 4, 4, 1, 0])
    y = np.array([0, 0, 2, 2, 0])

    # Apply horizontal flip for right trapezoid to make it look different
    # Flip the x-coordinates horizontally around x=2 to keep within bounds
    x_flipped = 4 - x

    # Apply rotation if specified
    if rotation != 0:
        x_flipped, y = apply_rotation(x_flipped, y, rotation)

    # Draw the basic trapezoid shape
    ax.plot(x_flipped, y, color=color)
    ax.fill(x_flipped, y, color=color, alpha=0.3)

    # Add right angle markers using the same approach as rectangle
    marker_size = 0.2

    # Get the vertices for marker calculation
    corners = [
        (x_flipped[0], y[0]),
        (x_flipped[1], y[1]),
        (x_flipped[2], y[2]),
        (x_flipped[3], y[3]),
    ]

    # Right trapezoid has right angles at vertices 1 and 2 (where vertical side meets horizontal sides)
    right_angle_vertices = [1, 2]

    for i in right_angle_vertices:
        corner_x, corner_y = corners[i]
        next_corner_x, next_corner_y = corners[(i + 1) % 4]
        prev_corner_x, prev_corner_y = corners[(i - 1) % 4]

        # Calculate direction vectors from corner to adjacent corners
        dx1 = next_corner_x - corner_x
        dy1 = next_corner_y - corner_y
        len1 = np.sqrt(dx1**2 + dy1**2)
        if len1 > 0:
            dx1, dy1 = dx1 / len1, dy1 / len1

        dx2 = prev_corner_x - corner_x
        dy2 = prev_corner_y - corner_y
        len2 = np.sqrt(dx2**2 + dy2**2)
        if len2 > 0:
            dx2, dy2 = dx2 / len2, dy2 / len2

        # Create marker square
        p1_x = corner_x + marker_size * dx1
        p1_y = corner_y + marker_size * dy1
        p2_x = corner_x + marker_size * dx2
        p2_y = corner_y + marker_size * dy2
        p3_x = p1_x + marker_size * dx2
        p3_y = p1_y + marker_size * dy2

        ax.plot(
            [corner_x, p1_x, p3_x, p2_x, corner_x],
            [corner_y, p1_y, p3_y, p2_y, corner_y],
            color=color,
            linewidth=1,
        )

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


@stimulus_function
def draw_geometric_shapes(shape_list: GeometricShapeList):
    num_shapes = len(shape_list)
    # Always use a single row layout
    num_cols = num_shapes
    num_rows = 1

    # Adjust size per shape based on number of shapes
    size_per_shape = 3 if num_shapes > 6 else 4
    # Calculate total width but cap it at a reasonable maximum
    total_width = min(size_per_shape * num_cols, 30)
    # Adjust height to maintain aspect ratio
    height = size_per_shape

    _, axs = plt.subplots(num_rows, num_cols, figsize=(total_width, height))
    axs = np.array(axs).flatten()  # Flatten to easily index

    shape_map = {
        ValidGeometricShape.RECTANGLE: lambda i, item: draw_rectangle(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.RHOMBUS: lambda i, item: draw_rhombus(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.PARALLELOGRAM: lambda i, item: draw_parallelogram(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=random.randint(0, 360),
        ),
        ValidGeometricShape.CIRCLE: lambda i, item: draw_circle(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.REGULAR_QUADRILATERAL: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=4, color=item.color
        ),
        ValidGeometricShape.SCALENE_TRIANGLE: lambda i, item: draw_scalene_triangle(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[5, 7, 10],
            color=item.color,
        ),
        ValidGeometricShape.TRAPEZOID: lambda i, item: draw_trapezoid_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.ISOSCELES_TRAPEZOID: lambda i,
        item: draw_isosceles_trapezoid(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.RIGHT_TRAPEZOID: lambda i, item: draw_right_trapezoid(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.QUADRILATERAL: lambda i,
        item: draw_quadrilateral_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.KITE: lambda i, item: draw_kite_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.TRIANGLE: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=3, color=item.color
        ),
        ValidGeometricShape.SQUARE: lambda i, item: draw_rectangle(
            axs[i],
            "" if num_shapes == 1 else item.label,
            width=2,
            height=2,
            color=item.color,
        ),
        ValidGeometricShape.PENTAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=5, color=item.color
        ),
        ValidGeometricShape.REGULAR_PENTAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=5, color=item.color
        ),
        ValidGeometricShape.HEXAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=6, color=item.color
        ),
        ValidGeometricShape.REGULAR_HEXAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=6, color=item.color
        ),
        ValidGeometricShape.HEPTAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=7, color=item.color
        ),
        ValidGeometricShape.REGULAR_HEPTAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=7, color=item.color
        ),
        ValidGeometricShape.OCTAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=8, color=item.color
        ),
        ValidGeometricShape.REGULAR_OCTAGON: lambda i, item: draw_regular_polygon(
            axs[i], "" if num_shapes == 1 else item.label, sides=8, color=item.color
        ),
        ValidGeometricShape.RIGHT_TRIANGLE: lambda i, item: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 5],
            color=item.color,
        ),
        ValidGeometricShape.ISOSCELES_TRIANGLE: lambda i, item: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[2, 4, 4],
            color=item.color,
        ),
        ValidGeometricShape.REGULAR_TRIANGLE: lambda i, item: draw_regular_polygon(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
        ),
        ValidGeometricShape.EQUILATERAL_TRIANGLE: lambda i, item: draw_regular_polygon(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=3,
            radius=2,
            color=item.color,
        ),
        ValidGeometricShape.OBTUSE_TRIANGLE: lambda i, item: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 6],
            color=item.color,
        ),
        ValidGeometricShape.ACUTE_TRIANGLE: lambda i, item: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 4],
            color=item.color,
        ),
    }

    for ax in axs[num_shapes:]:
        ax.remove()

    for i, item in enumerate(shape_list):
        if item.shape in shape_map:
            shape_map[item.shape](i, item)
        else:
            raise ValueError(f"Shape Type {item.shape.value} is not supported")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_basic_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    # plt.show()
    return file_name


###################################################
# Geometric Shapes - Rectangles with Side Lengths #
###################################################
@stimulus_function
def generate_rect_with_side_len(data: RectangularGridList):
    target_area = 25  # All rectangles will be scaled to this area (e.g., 5x5 units)
    scaled_lengths = []
    scaled_widths = []
    for rectangle in data:
        l = float(rectangle.length)
        w = float(rectangle.width)
        if l == 0 or w == 0:
            scale = 1  # Avoid division by zero
        else:
            scale = sqrt(target_area / (l * w))
        scaled_l = l * scale
        scaled_w = w * scale
        scaled_lengths.append(scaled_l)
        scaled_widths.append(scaled_w)

    # Increase spacing between rectangles
    spacing = 2
    total_width = sum(scaled_widths) + (len(data) - 1) * spacing
    max_height = max(scaled_lengths)
    fig_width = min(total_width + 2, 30)  # Cap the width at 30 inches
    fig_height = min(max_height + 2, 15)  # Cap the height at 15 inches

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_offset = 0

    def format_unit(value, base_unit, labeling_style="abbreviated"):
        """Format unit with proper singular/plural and full name vs abbreviation"""
        # Handle special case for "Units"
        if base_unit == "Units":
            return "Unit" if value == 1 else "Units"

        # Handle other units based on labeling style
        if labeling_style == "full_names":
            # Convert abbreviations to full names with proper singular/plural
            unit_mapping = {
                "cm": "centimeter" if value == 1 else "centimeters",
                "mm": "millimeter" if value == 1 else "millimeters",
                "m": "meter" if value == 1 else "meters",
                "km": "kilometer" if value == 1 else "kilometers",
                "in": "inch" if value == 1 else "inches",
                "ft": "foot" if value == 1 else "feet",
                "yd": "yard" if value == 1 else "yards",
            }
            return unit_mapping.get(base_unit, base_unit)

        # Default: return abbreviation as-is (current behavior)
        return base_unit

    for i, rectangle in enumerate(data):
        length = scaled_lengths[i]
        width = scaled_widths[i]
        unit = rectangle.unit.value

        # Use consistent large font size for both width and length labels
        label_fontsize = 50  # Fixed large font size for all labels

        rect = patches.Rectangle(
            (x_offset, 0),
            width,
            length,
            linewidth=3,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Format the width label with LaTeX fraction if it's a Fraction
        width_value = rectangle.width
        width_unit = format_unit(width_value, unit, rectangle.labeling_style.value)
        width_label = (
            f"$\\frac{{{width_value.numerator}}}{{{width_value.denominator}}}$ {width_unit}"
            if isinstance(width_value, Fraction) and width_value.denominator != 1
            else f"{width_value} {width_unit}"
        )
        width_padding = 0.1 if isinstance(rectangle.width, Fraction) else 0
        ax.text(
            x_offset + width / 2,
            length + width_padding,
            width_label,
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
        )

        # Format the length label with LaTeX fraction if it's a Fraction
        length_value = rectangle.length
        length_unit = format_unit(length_value, unit, rectangle.labeling_style.value)
        length_label = (
            f"$\\frac{{{length_value.numerator}}}{{{length_value.denominator}}}$ {length_unit}"
            if isinstance(length_value, Fraction) and length_value.denominator != 1
            else f"{length_value} {length_unit}"
        )
        length_padding = 0.1 if isinstance(rectangle.length, Fraction) else 0
        # Always rotate length labels for consistent behavior
        ax.text(
            x_offset - length_padding,
            length / 2,
            length_label,
            va="center",
            ha="right",
            rotation="vertical",
            fontsize=label_fontsize,
        )

        if len(data) > 1:
            ax.text(
                x_offset + width / 2,
                -0.2 * max_height,
                f"Figure {i+1}",
                ha="center",
                fontsize=32,
            )
        x_offset += width + spacing

    # Set axis limits with some padding
    ax.set_xlim(-spacing / 2, total_width + spacing / 2)
    ax.set_ylim(-spacing / 2, max_height + spacing / 2)
    ax.axis("off")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_geo_rect.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_single_rect_with_side_len_stimulus(data: RectangularGridList):
    """
    Generate a single rectangle with side length labels optimized for small, consistent image size.
    """
    if len(data) != 1:
        raise ValueError(
            "generate_single_rect_with_side_len_stimulus requires exactly one rectangle"
        )

    rectangle = data[0]

    # Convert to float for calculations
    length = float(rectangle.length)
    width = float(rectangle.width)
    unit = rectangle.unit.value

    # Always use square figure size for consistency
    fig_size = 0.8  # inches - all images will be square
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Calculate rectangle aspect ratio with a maximum cap of 5:1
    aspect_ratio = width / length if length > 0 else 1

    # Cap the aspect ratio to prevent extreme distortions
    max_ratio = 5.0
    if aspect_ratio > max_ratio:
        aspect_ratio = max_ratio
    elif aspect_ratio < 1 / max_ratio:
        aspect_ratio = 1 / max_ratio

    # Scale rectangle to fit within a fixed area while maintaining capped aspect ratio
    max_dimension = 0.9

    if aspect_ratio > 1:  # Wide rectangle
        scaled_width = max_dimension
        scaled_length = max_dimension / aspect_ratio
    else:  # Tall or square rectangle
        scaled_length = max_dimension
        scaled_width = max_dimension * aspect_ratio

    # Center the rectangle in the figure
    x_center = 0.5
    y_center = 0.5
    x_pos = x_center - scaled_width / 2
    y_pos = y_center - scaled_length / 2

    # Draw rectangle
    rect = patches.Rectangle(
        (x_pos, y_pos),
        scaled_width,
        scaled_length,
        linewidth=2,
        edgecolor="blue",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Format unit labels
    def format_unit(value, base_unit, labeling_style="abbreviated"):
        """Format unit with proper singular/plural and full name vs abbreviation"""
        if base_unit == "Units":
            return "Unit" if value == 1 else "Units"

        if labeling_style == "full_names":
            unit_mapping = {
                "cm": "centimeter" if value == 1 else "centimeters",
                "mm": "millimeter" if value == 1 else "millimeters",
                "m": "meter" if value == 1 else "meters",
                "km": "kilometer" if value == 1 else "kilometers",
                "in": "inch" if value == 1 else "inches",
                "ft": "foot" if value == 1 else "feet",
                "yd": "yard" if value == 1 else "yards",
            }
            return unit_mapping.get(base_unit, base_unit)

        return base_unit

    # Add width label (top)
    width_value = rectangle.width
    width_unit = format_unit(width_value, unit, rectangle.labeling_style.value)
    width_label = (
        f"$\\frac{{{width_value.numerator}}}{{{width_value.denominator}}}$ {width_unit}"
        if isinstance(width_value, Fraction) and width_value.denominator != 1
        else f"{width_value} {width_unit}"
    )

    ax.text(
        x_center,
        y_pos + scaled_length + 0.08,
        width_label,
        ha="center",
        va="bottom",
        fontsize=5,
    )

    # Add length label (left side, rotated)
    length_value = rectangle.length
    length_unit = format_unit(length_value, unit, rectangle.labeling_style.value)
    length_label = (
        f"$\\frac{{{length_value.numerator}}}{{{length_value.denominator}}}$ {length_unit}"
        if isinstance(length_value, Fraction) and length_value.denominator != 1
        else f"{length_value} {length_unit}"
    )

    ax.text(
        x_pos - 0.08,
        y_center,
        length_label,
        ha="right",
        va="center",
        rotation="vertical",
        fontsize=5,
    )

    # Set axis limits and remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Save with high DPI for sharp rendering
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_single_rect.{settings.additional_content_settings.stimulus_image_format}"

    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=300,
    )
    plt.close()

    return file_name


###################################################
# Geometric Shapes - Triangles with Side Lengths #
###################################################
@stimulus_function
def generate_triangle_with_side_len(data: TriangularGrid):
    """
    Draws one triangle with:
      - side lengths labeled outside edges
      - interior angles labeled inside, except the right angle
      - a right-angle marker at the right-angle vertex, correctly aligned
    """
    # preserve input order: side1=left, side2=slanted, side3=base
    b, a, c = data.side1, data.side2, data.side3

    # scale longest side to ≤ 10 units
    scale = min(1.0, 10.0 / max(a, b, c))

    # law of cosines for apex
    x2 = (b**2 + c**2 - a**2) / (2 * c)
    y2 = math.sqrt(max(0.0, b**2 - x2**2))

    # vertices: P0, P1, P2
    verts = np.array([[0.0, 0.0], [c, 0.0], [x2, y2]]) * scale

    fig, ax = plt.subplots(figsize=(6, 6))
    # draw triangle
    xs = np.append(verts[:, 0], verts[0, 0])
    ys = np.append(verts[:, 1], verts[0, 1])
    ax.plot(xs, ys, color="blue", linewidth=3)

    # centroid for inside/outside tests
    cx, cy = verts.mean(axis=0)
    pad_side = 0.10 * max(a, b, c) * scale
    pad_angle = 0.20 * max(a, b, c) * scale
    # Make top angle padding adaptive based on both height and base
    pad_angle_top = (
        0.25 * min(y2, c) * scale
    )  # Use the smaller of height or base to ensure balanced padding

    # side labels outside
    def label_side(i, j, length):
        x0, y0 = verts[i]
        x1, y1 = verts[j]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        norm = math.hypot(nx, ny) or 1
        # flip if inward
        if (nx * (cx - mx) + ny * (cy - my)) > 0:
            nx, ny = -nx, -ny
        ox, oy = (nx / norm) * pad_side, (ny / norm) * pad_side
        ax.text(
            mx + ox,
            my + oy,
            f"{length} {data.unit.value}",
            ha="center",
            va="center",
            fontsize=18,
            clip_on=False,
        )

    label_side(0, 1, c)
    label_side(1, 2, a)
    label_side(2, 0, b)

    # Add tick marks for congruent sides
    side_lengths = [c, a, b]  # corresponding to sides 0-1, 1-2, 2-0
    side_coords = [(0, 1), (1, 2), (2, 0)]  # vertex pairs for each side

    # Group sides by length to find congruent sides
    length_groups = {}
    for i, length in enumerate(side_lengths):
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(i)

    # Filter to only groups with multiple sides (congruent sides)
    congruent_groups = [group for group in length_groups.values() if len(group) > 1]

    # Assign tick mark types (single for first group, double for second group, etc.)
    tick_types = {}
    for group_idx, group in enumerate(congruent_groups):
        tick_type = "single" if group_idx == 0 else "double"
        for side_idx in group:
            tick_types[side_idx] = tick_type

    # Draw tick marks
    dash_length = 0.05 * max(a, b, c) * scale  # Scale with triangle size
    dash_color = "red"
    dash_thickness = 5

    for side_idx, tick_type in tick_types.items():
        i, j = side_coords[side_idx]
        x1, y1 = verts[i]
        x2, y2 = verts[j]

        if tick_type == "single":
            _draw_side_dash(ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness)
        elif tick_type == "double":
            _draw_side_double_dash(
                ax, x1, y1, x2, y2, dash_length, dash_color, dash_thickness
            )

    # interior angles
    ang0 = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
    ang1 = math.acos((a**2 + c**2 - b**2) / (2 * a * c))
    ang2 = math.pi - ang0 - ang1

    # Convert to degrees and round to integers
    ang0_deg = round(ang0 * 180 / math.pi)
    ang1_deg = round(ang1 * 180 / math.pi)
    ang2_deg = round(ang2 * 180 / math.pi)

    # Ensure the sum is exactly 180 degrees
    angle_sum = ang0_deg + ang1_deg + ang2_deg
    if angle_sum != 180:
        # Adjust the largest angle to make the sum exactly 180
        angles_deg = [ang0_deg, ang1_deg, ang2_deg]
        max_idx = angles_deg.index(max(angles_deg))
        angles_deg[max_idx] += 180 - angle_sum
        ang0_deg, ang1_deg, ang2_deg = angles_deg

    # Convert back to radians for right-angle detection
    angles = [
        ang0_deg * math.pi / 180,
        ang1_deg * math.pi / 180,
        ang2_deg * math.pi / 180,
    ]
    angles_deg = [ang0_deg, ang1_deg, ang2_deg]

    # find right-angle index
    right_i = None
    for idx, ang in enumerate(angles):
        if abs(ang - math.pi / 2) < 1e-2:
            right_i = idx
            break

    # angle labels inside (skip right angle)
    for i, ang_deg in enumerate(angles_deg):
        if i == right_i:
            continue
        xi, yi = verts[i]
        dx, dy = cx - xi, cy - yi
        norm = math.hypot(dx, dy) or 1
        # Use smaller padding for top vertex (index 2)
        current_pad = pad_angle_top if i == 2 else pad_angle
        ox, oy = (dx / norm) * current_pad, (dy / norm) * current_pad
        ax.text(
            xi + ox,
            yi + oy,
            f"{ang_deg}\u00b0",
            ha="center",
            va="center",
            fontsize=18,
            clip_on=False,
        )

    # complete black right-angle marker
    if right_i is not None:
        P = verts[right_i]
        i0 = (right_i + 1) % 3
        i1 = (right_i + 2) % 3
        v0 = verts[i0] - P
        v1 = verts[i1] - P
        u0 = v0 / np.linalg.norm(v0)
        u1 = v1 / np.linalg.norm(v1)
        ms = pad_angle * 0.5
        A = P + u0 * ms
        C = P + u1 * ms
        B = A + u1 * ms
        # draw four sides of square in black
        ax.plot([P[0], A[0]], [P[1], A[1]], color="black", linewidth=2)
        ax.plot([A[0], B[0]], [A[1], B[1]], color="black", linewidth=2)
        ax.plot([B[0], C[0]], [B[1], C[1]], color="black", linewidth=2)
        ax.plot([C[0], P[0]], [C[1], P[1]], color="black", linewidth=2)

    ax.set_aspect("equal")
    ax.axis("off")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_triangle{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


###################################################
# Geometric Shapes - Decomposition - Unit Square #
###################################################
@stimulus_function
def generate_stimulus_with_grid(rectangle: RectangularGrid):
    length = rectangle.length
    width = rectangle.width
    unit = rectangle.unit
    padding = 0.1

    fig, ax = plt.subplots(figsize=(width + 1, length + 2))

    # Determine irregular/skewing behavior from input
    irregular = getattr(rectangle, "irregular", False)

    # Choose whether to skew rows or columns when irregular
    if irregular:
        skew_type = random.choice(["columns", "rows"])  # choose axis for skew
        if skew_type == "columns":
            # Build smooth offsets via random walk to keep the shape connected
            column_offsets = [0] * width
            if width > 0:
                current = random.randint(-1, 1)
                column_offsets[0] = current
                for i in range(1, width):
                    step = random.choice([-1, 0, 1])
                    current += step
                    column_offsets[i] = current
            row_offsets = [0] * length
        else:
            # Smooth row offsets via random walk
            row_offsets = [0] * length
            if length > 0:
                current = random.randint(-1, 1)
                row_offsets[0] = current
                for i in range(1, length):
                    step = random.choice([-1, 0, 1])
                    current += step
                    row_offsets[i] = current
            column_offsets = [0] * width
    else:
        skew_type = None
        column_offsets = [0] * width
        row_offsets = [0] * length

    # Prepare extras distribution (only if irregular)
    extras_total = getattr(rectangle, "extra_unit_squares", None)
    if not irregular:
        extras_total = None
    # extras must be between 1 and max(length,width); schema enforces but guard anyway
    if extras_total is not None:
        extras_total = int(extras_total)
        max_allowed = max(length, width)
        if extras_total < 1 or extras_total > max_allowed:
            extras_total = None

    attach_mode = None
    extras_per_index = None
    if extras_total:
        # Choose to append by rows or by columns
        attach_mode = random.choice(["rows", "columns"])  # attach across chosen indices
        if attach_mode == "rows":
            indices = list(range(length))
        else:
            indices = list(range(width))

        random.shuffle(indices)
        extras_per_index = [0] * (length if attach_mode == "rows" else width)
        remaining = extras_total
        i = 0
        # Distribute 1 or 2 per selected index until exhausted
        while remaining > 0 and indices:
            idx = indices[i % len(indices)]
            add = 2 if remaining >= 2 and random.randint(1, 2) == 2 else 1
            extras_per_index[idx] += add
            remaining -= add
            i += 1

    # Calculate bounds for positioning (no outer rectangle, just for layout)
    min_x = 0
    max_x = width
    min_y = 0
    max_y = length

    # Adjust bounds if skewing is applied
    if irregular:
        for x in range(width):
            for y in range(length):
                if skew_type == "columns":
                    actual_x = x
                    actual_y = y + column_offsets[x]
                elif skew_type == "rows":
                    actual_x = x + row_offsets[y]
                    actual_y = y
                else:
                    actual_x = x
                    actual_y = y
                min_x = min(min_x, actual_x)
                max_x = max(max_x, actual_x + 1)
                min_y = min(min_y, actual_y)
                max_y = max(max_y, actual_y + 1)

    # Choose one random fill color for all unit squares
    fill_color = get_random_polygon_color()

    # Add the unit square grid with skewing and optional extras
    for x in range(width):
        for y in range(length):
            # Base position with skew
            if not irregular:
                actual_x = x
                actual_y = y
            elif skew_type == "columns":
                actual_x = x
                actual_y = y + column_offsets[x]
            else:  # rows
                actual_x = x + row_offsets[y]
                actual_y = y

            square = patches.Rectangle(
                (actual_x, actual_y),
                1,
                1,
                linewidth=2,
                edgecolor="black",
                facecolor=fill_color,
            )
            ax.add_artist(square)
            min_x = min(min_x, actual_x)
            max_x = max(max_x, actual_x + 1)
            min_y = min(min_y, actual_y)
            max_y = max(max_y, actual_y + 1)

        # If attaching by columns, draw extras for this column after normal rows
        if extras_per_index is not None and attach_mode == "columns":
            extra_count = extras_per_index[x]
            for k in range(extra_count):
                y_extra = length + k
                if not irregular:
                    actual_x = x
                    actual_y = y_extra
                elif skew_type == "columns":
                    actual_x = x
                    actual_y = y_extra + column_offsets[x]
                else:  # rows skew has horizontal shifts
                    actual_x = x + (row_offsets[length - 1] if length > 0 else 0)
                    actual_y = y_extra
                square = patches.Rectangle(
                    (actual_x, actual_y),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor=fill_color,
                )
                ax.add_artist(square)
                min_x = min(min_x, actual_x)
                max_x = max(max_x, actual_x + 1)
                min_y = min(min_y, actual_y)
                max_y = max(max_y, actual_y + 1)

    # If attaching by rows, draw extras per row beyond the last column
    if extras_per_index is not None and attach_mode == "rows":
        for y in range(length):
            extra_count = extras_per_index[y]
            for k in range(extra_count):
                x_extra = width + k
                if not irregular:
                    actual_x = x_extra
                    actual_y = y
                elif skew_type == "columns":
                    actual_x = x_extra
                    # Use nearest column offset to keep append contiguous vertically
                    nearest_col = min(max(0, width - 1), width - 1) if width > 0 else 0
                    actual_y = y + (column_offsets[nearest_col] if width > 0 else 0)
                else:  # rows skew
                    actual_x = x_extra + row_offsets[y]
                    actual_y = y
                square = patches.Rectangle(
                    (actual_x, actual_y),
                    1,
                    1,
                    linewidth=2,
                    edgecolor="black",
                    facecolor=fill_color,
                )
                ax.add_artist(square)
                min_x = min(min_x, actual_x)
                max_x = max(max_x, actual_x + 1)
                min_y = min(min_y, actual_y)
                max_y = max(max_y, actual_y + 1)

    # Add a single square below the main rectangle at a reasonable distance
    single_square_y = min_y - 1.5  # Position below the main rectangle
    single_square_x = min_x  # Left-aligned with the main drawing
    single_square = patches.Rectangle(
        (single_square_x, single_square_y),
        1,
        1,
        linewidth=2,
        edgecolor="black",  # Already black borders
        facecolor=fill_color,  # Same color as the unit squares
        alpha=1,
    )
    ax.add_patch(single_square)

    # Set xlim and ylim according to rectangle dimensions and the additional square
    ax.set_xlim(min_x - 0.45 - padding, max_x + padding)
    ax.set_ylim(single_square_y - 0.5 - padding, max_y + padding)

    ax.axis("off")  # disable x-axis and y-axis
    ax.set_aspect("equal", "box")

    # Fixed font size - constant for all images
    font_size = 16

    # Label the single square next to it with "1 square [unit]"
    # Handle special case for UNITS to avoid awkward "Units" display
    if unit.value == "Units":
        square_label = "1 square unit"
    else:
        square_label = f"1 square {str(unit.value)}"
    ax.text(
        single_square_x + 1.1,
        single_square_y + 0.5,
        square_label,
        ha="left",
        va="center",
        fontsize=font_size,
    )

    # Fixed border dimensions - same for all images to ensure consistency
    legend_padding = 0.15
    legend_left = single_square_x - legend_padding
    # Fixed width that accommodates longest possible text (e.g., "1 square centimeters")
    fixed_legend_width = 3.0  # Wide enough for any unit text
    legend_right = single_square_x + 1.1 + fixed_legend_width
    legend_bottom = single_square_y - legend_padding
    legend_top = single_square_y + 1 + legend_padding

    # Draw the legend border rectangle
    legend_border = patches.Rectangle(
        (legend_left, legend_bottom),
        legend_right - legend_left,
        legend_top - legend_bottom,
        linewidth=1.5,
        edgecolor="gray",
        facecolor="none",  # Transparent fill
        linestyle="-",  # Solid line
    )
    ax.add_patch(legend_border)

    # Update plot limits to ensure legend border is not cropped
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    # Extend limits to include the full legend border with some extra padding
    ax.set_xlim(
        min(current_xlim[0], legend_left - 0.1),
        max(current_xlim[1], legend_right + 0.1),
    )
    ax.set_ylim(
        min(current_ylim[0], legend_bottom - 0.1),
        max(current_ylim[1], legend_top + 0.1),
    )

    fig.canvas.draw()  # Redraw the canvas

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_decomp_unit_square{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


###################################################
# Geometric Shape with Base and Height #
###################################################
@stimulus_function
def generate_shape_with_base_and_height(params: Shape):
    shape_type = params.shape_type
    height = params.height
    base = params.base
    unit = params.unit

    fig, ax = plt.subplots(figsize=(6, 4))
    y_start = 1  # Starting y-position for better visibility

    # Calculate scaling factor - slightly reduce to give more room for mixed fraction labels
    max_dimension = max(height, base)
    scale_factor = 4.5 / max_dimension  # Reduced from 5 to 4.5 for better label spacing

    scaled_height = height * scale_factor
    scaled_base = base * scale_factor

    right_edge = 1 + scaled_base
    if shape_type.value == "parallelogram":
        right_edge += 1
    elif shape_type.value == "rhombus":
        right_edge += 0.5  # Rhombus needs some extra space for the slanted sides

    # Increase margins to ensure mixed fraction labels are always visible
    ax.set_xlim(-0.2, right_edge + 0.7)  # More right margin for mixed fractions
    ax.set_ylim(
        -0.2, y_start + scaled_height + 0.7
    )  # More top margin for mixed fractions
    ax.axis("off")

    if shape_type.value == "rectangle":
        shape = patches.Rectangle(
            (1, y_start),
            scaled_base,
            scaled_height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(shape)
    elif shape_type.value == "parallelogram":
        shape = patches.Polygon(
            [
                [1, y_start],
                [1 + scaled_base, y_start],
                [1 + scaled_base + 1, y_start + scaled_height],
                [1 + 1, y_start + scaled_height],
            ],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(shape)
    elif shape_type.value == "rhombus":
        # For rhombus, base and height represent the diagonals
        # Center the rhombus horizontally
        center_x = 1 + scaled_base / 2
        center_y = y_start + scaled_height / 2

        # Calculate the four vertices of the rhombus
        # The diagonals intersect at the center
        vertices = [
            [center_x, center_y + scaled_height / 2],  # top
            [center_x + scaled_base / 2, center_y],  # right
            [center_x, center_y - scaled_height / 2],  # bottom
            [center_x - scaled_base / 2, center_y],  # left
        ]

        shape = patches.Polygon(
            vertices,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(shape)
    elif shape_type.value == "triangle":
        shape = patches.Polygon(
            [
                [1, y_start],
                [1 + scaled_base, y_start],
                [1 + 0.5 * scaled_base, y_start + scaled_height],
            ],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(shape)

    # Base dimension line - positioned lower to avoid mixed fraction overlap
    ax.annotate(
        "",
        xy=(1, y_start - 0.3),
        xytext=(1 + scaled_base, y_start - 0.3),
        arrowprops=dict(arrowstyle="<->", color="blue", linewidth=2),
    )

    # Height dimension line - positioned further left to avoid mixed fraction overlap
    ax.annotate(
        "",
        xy=(0.6, y_start),  # Moved further left from 0.7
        xytext=(0.6, y_start + scaled_height),
        arrowprops=dict(arrowstyle="<->", color="blue", linewidth=2),
    )

    # Base label with fraction formatting - positioned lower for visibility
    ax.annotate(
        format_dimension_label(base, unit),
        (1 + scaled_base / 2, y_start - 0.9),  # Moved lower from -0.8
        ha="center",
        fontsize=20,
    )

    # Height label with fraction formatting - positioned further left for visibility
    ax.annotate(
        format_dimension_label(height, unit),
        (0.1, y_start + scaled_height / 2),  # Moved further left from 0.2
        va="center",
        rotation=90,
        fontsize=20,
    )

    ax.set_aspect("equal", adjustable="box")

    plt.subplots_adjust(
        left=0.2, right=0.98, top=0.9, bottom=0.15
    )  # Increased bottom margin

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_shape.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_composite_rect_prism(stimulus_description: CompositeRectangularPrism):
    prisms = stimulus_description.figures
    unit = stimulus_description.units
    hide_measurements = stimulus_description.hide_measurements

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    heights, widths, lengths = zip(*prisms)
    axis_lim = max(sum(heights), max(widths), max(lengths))
    ax.set_xlim(-1.5, axis_lim)
    ax.set_ylim(-1.5, axis_lim)
    ax.set_zlim(-0.5, axis_lim)

    height_offsets = []
    offset = 0
    for h, w, l in prisms:
        height_offsets.append(offset)
        offset += h

    elev = np.deg2rad(ax.elev)
    azim = np.deg2rad(ax.azim)
    view_vec = np.array(
        [np.cos(elev) * np.sin(azim), -np.cos(elev) * np.cos(azim), np.sin(elev)]
    )

    centroids = [
        np.array([lengths[i] / 2, widths[i] / 2, height_offsets[i] + heights[i] / 2])
        for i in range(len(prisms))
    ]
    depths = [c.dot(view_vec) for c in centroids]
    draw_order = sorted(range(len(prisms)), key=lambda i: depths[i])

    # Phase 1: draw the prism bodies
    for idx in draw_order:
        h, w, l = prisms[idx]
        ho = height_offsets[idx]
        bh = ho + h
        dp = depths[idx]
        verts = [
            [(0, 0, ho), (l, 0, ho), (l, w, ho), (0, w, ho)],  # bottom
            [(0, 0, bh), (l, 0, bh), (l, w, bh), (0, w, bh)],  # top
            [(0, 0, ho), (l, 0, ho), (l, 0, bh), (0, 0, bh)],  # front
            [(0, w, ho), (l, w, ho), (l, w, bh), (0, w, bh)],  # back (width face)
            [(0, 0, ho), (0, w, ho), (0, w, bh), (0, 0, bh)],  # left
            [(l, 0, ho), (l, w, ho), (l, w, bh), (l, 0, bh)],  # right
        ]
        face_colors = [
            "#b3d1ff",  # bottom
            "#b3d1ff",  # top
            "#b3d1ff",  # front
            "#7fb2e5",  # back (width face, shadow)
            "#b3d1ff",  # left
            "#b3d1ff",  # right
        ]
        poly = Poly3DCollection(
            verts, facecolors=face_colors, edgecolor="black", linewidth=1
        )
        poly._sort_zpos = dp
        ax.add_collection3d(poly)

    # Phase 2: overlay measurement lines and labels (high zorder)
    for idx in draw_order:
        # Skip measurements if this prism index is in the hide_measurements list
        if idx in hide_measurements:
            continue

        h, w, l = prisms[idx]
        ho = height_offsets[idx]
        bh = ho + h
        # length (green)
        ax.plot([0, l], [-0.5, -0.5], [ho, ho], "--", color="green", zorder=1000)
        ax.plot([0, 0], [0, -1], [ho, ho], "--", color="green", zorder=1000)
        ax.plot([l, l], [0, -1], [ho, ho], "--", color="green", zorder=1000)
        ax.text(
            l / 2,
            -0.3 * max(l, w, h),  # Increased relative padding
            ho,
            f"{l} {unit}",
            color="green",
            zorder=1000,
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.7, pad=4
            ),  # Increased padding
        )
        # width (blue)
        ax.plot([l + 0.5, l + 0.5], [0, w], [ho, ho], "--", color="blue", zorder=1000)
        ax.plot([l, l + 1], [0, 0], [ho, ho], "--", color="blue", zorder=1000)
        ax.plot([l, l + 1], [w, w], [ho, ho], "--", color="blue", zorder=1000)
        ax.text(
            l + 0.3 * max(l, w, h),  # Increased relative padding
            w / 2,
            ho,
            f"{w} {unit}",
            color="blue",
            zorder=1000,
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.7, pad=4
            ),  # Increased padding
        )
        # height (red)
        ax.plot([-0.5, -0.5], [0, 0], [ho, bh], "--", color="red", zorder=1000)
        ax.plot([0, -1], [0, 0], [ho, ho], "--", color="red", zorder=1000)
        ax.plot([0, -1], [0, 0], [bh, bh], "--", color="red", zorder=1000)
        ax.text(
            -0.3 * max(l, w, h),  # Increased relative padding
            0,
            ho,
            f"{h} {unit}",
            color="red",
            ha="center",
            zorder=1000,
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.7, pad=4
            ),  # Increased padding
        )

    for ax_call in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        try:
            ax_call([])
        except Exception:
            pass
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/composite_rect_prism_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


@stimulus_function
def generate_geometric_shapes_transformations(polygons: PolygonList):
    def apply_rotation(point, angle, direction, center):
        """Apply rotation to a point around a center point."""
        # Translate to origin relative to center
        x = point.x - center.x
        y = point.y - center.y

        # Adjust angle for direction
        if direction == "clockwise":
            angle = -angle

        # Apply rotation using exact values to avoid floating point errors
        if angle == 90:
            new_x = -y
            new_y = x
        elif angle == 180:
            new_x = -x
            new_y = -y
        elif angle == 270:
            new_x = y
            new_y = -x
        elif angle == -90:  # clockwise 90
            new_x = y
            new_y = -x
        elif angle == -180:  # clockwise 180 (same as counterclockwise 180)
            new_x = -x
            new_y = -y
        elif angle == -270:  # clockwise 270
            new_x = -y
            new_y = x
        else:
            # Fallback to no rotation
            new_x = x
            new_y = y

        # Translate back from center
        return new_x + center.x, new_y + center.y

    fig, ax = plt.subplots()

    ax.set_aspect("equal", "box")

    plt.axis("on")

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    max_coord = 0
    for polygon in polygons:
        for vertex in polygon.points:
            # Check both original and potentially rotated coordinates
            max_coord = max(max_coord, abs(vertex.x), abs(vertex.y))

            # If rotation is specified, also check rotated coordinates
            if (
                polygon.rotation_angle is not None
                and polygon.rotation_direction is not None
            ):
                center = polygon.rotation_center or type(vertex)(label="O", x=0, y=0)
                rotated_x, rotated_y = apply_rotation(
                    vertex, polygon.rotation_angle, polygon.rotation_direction, center
                )
                max_coord = max(max_coord, abs(rotated_x), abs(rotated_y))

    if max_coord % 2 == 0:
        max_coord += 4
    else:
        max_coord += 3

    ax.set_xlim(-max_coord, max_coord)
    ax.set_ylim(-max_coord, max_coord)

    ax.set_xticks(range(-int(max_coord), int(max_coord) + 1, 1))
    ax.set_yticks(range(-int(max_coord), int(max_coord) + 1, 1))

    colors = ["red", "blue"]
    legend_handles = []
    for idx, polygon in enumerate(polygons):
        vertices = polygon.points

        # Apply rotation if specified
        if (
            polygon.rotation_angle is not None
            and polygon.rotation_direction is not None
        ):
            center = polygon.rotation_center or type(vertices[0])(label="O", x=0, y=0)
            rotated_vertices = []
            for vertex in vertices:
                rotated_x, rotated_y = apply_rotation(
                    vertex, polygon.rotation_angle, polygon.rotation_direction, center
                )
                rotated_vertices.append((rotated_x, rotated_y, vertex.label))

            x_coords = [v[0] for v in rotated_vertices]
            y_coords = [v[1] for v in rotated_vertices]
            labels = [v[2] for v in rotated_vertices]
        else:
            x_coords = [vertex.x for vertex in vertices]
            y_coords = [vertex.y for vertex in vertices]
            labels = [vertex.label for vertex in vertices]

        color = colors[idx % 2]

        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])

        ax.plot(x_coords, y_coords, marker="o", color=color)

        for label, x, y in zip(labels, x_coords[:-1], y_coords[:-1]):
            ax.text(
                x,
                y,
                " " + label,
                verticalalignment="bottom",
                horizontalalignment="right" if x < 0 else "left",
                fontsize=13,
            )

        (line,) = ax.plot(
            x_coords,
            y_coords,
            marker="o",
            color=color,
        )
        legend_handles.append((line, polygon.label))

    if any(p.label != "No Label" for p in polygons):
        ax.legend(
            handles=[h[0] for h in legend_handles],
            labels=[h[1] for h in legend_handles],
            loc="upper right",
        )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_transformations_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def _projected_depth(ax, verts):
    """Depth of a polygon's centroid in projected coords (screen z)."""
    x = sum(v[0] for v in verts) / 4.0
    y = sum(v[1] for v in verts) / 4.0
    z = sum(v[2] for v in verts) / 4.0
    _, _, zproj = proj3d.proj_transform(x, y, z, ax.get_proj())
    return zproj


def _nice_faces(color_base=None):
    """Return top/front/side face colors + line color, with muted random base if not given."""
    if color_base is None:
        # Generate muted pastel color in HLS space (low saturation, medium-lightness)
        h = random.random()
        l = random.uniform(0.55, 0.7)  # lightness
        s = random.uniform(0.3, 0.5)  # saturation (lower = more muted)
        base = colorsys.hls_to_rgb(h, l, s)
    else:
        base = mc.to_rgb(color_base)

    def tint(f):
        return tuple(min(1.0, c + f * (1 - c)) for c in base)

    def shade(f):
        return tuple(c * (1 - f) for c in base)

    return {
        "top": tint(0.45),
        "front": tint(0.25),
        "side": shade(0.10),
        "line": (0, 0, 0),
    }


def _box_faces(x, y, z, l, w, h, faces):
    """Return the three visible faces (top, front, side) as (verts, facecolor)."""
    v000, v100 = (x, y, z), (x + l, y, z)
    v110 = (x + l, y + w, z)
    v001, v101 = (x, y, z + h), (x + l, y, z + h)
    v111, v011 = (x + l, y + w, z + h), (x, y + w, z + h)
    return [
        ([v001, v101, v111, v011], faces["top"]),  # top
        ([v000, v100, v101, v001], faces["front"]),  # front
        ([v100, v110, v111, v101], faces["side"]),  # side
    ]


def _view_vec(ax):
    elev = np.deg2rad(ax.elev)
    azim = np.deg2rad(ax.azim)
    return np.array(
        [np.cos(elev) * np.sin(azim), -np.cos(elev) * np.cos(azim), np.sin(elev)]
    )


def _nudge_point_toward_view(ax, pt, frac):
    """Nudge a 3D point a tiny fraction of scene diagonal toward the camera."""
    v = _view_vec(ax)
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    diag = np.linalg.norm([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    d = frac * diag
    return (pt[0] - v[0] * d, pt[1] - v[1] * d, pt[2] - v[2] * d)


def _label_dim(
    ax,
    kind,
    x,
    y,
    z,
    l,
    w,
    h,
    txt,
    pad=0.18,
    halo=True,
    *,
    layout=None,
    global_max_h=None,
):
    """Draw a black label for L/W/H.
    - side: W on TOP, aligned with y (rot=90), near +x edge.
            H goes on -x edge for the tallest prism; on +x edge for shorter ones.
    - L   : L on TOP/front edge; W on TOP, aligned with y near +x edge.
    """

    kw = dict(ha="center", va="center")
    halo_fx = [pe.withStroke(linewidth=3, foreground="white")] if halo else []

    if kind == "L":
        if layout == "L":
            # length on TOP, near front (-y) edge
            text_xyz = (x + l / 2.0, y - pad, z + h)
            rot = 0
        else:
            # default: bottom, outside front edge
            text_xyz = (x + l / 2.0, y - pad, z)
            rot = 0

    elif kind == "W":
        if layout in ("side", "L"):
            # WIDTH on TOP, aligned with y (vertical text), near +x edge
            inset = 0.06 * max(l, w, h)
            text_xyz = (x + l - inset, y + w / 2.0, z + h)
            rot = 90
        else:
            # default: right side, vertical text
            text_xyz = (x + l + pad, y + w / 2.0, z)
            rot = 90

    else:  # "H"
        # Default: left (-x) edge. For side-by-side, shorter prisms go to opposite (+x) edge.
        if layout == "side" and (global_max_h is not None) and (h < global_max_h):
            text_xyz = (x + l + pad, y, z + h / 2.0)  # opposite edge (+x)
        else:
            text_xyz = (x - pad, y, z + h / 2.0)  # default edge (-x)
        rot = 90

    # Nudge toward the camera so faces can't occlude labels
    text_xyz = _nudge_point_toward_view(ax, text_xyz, 0.003)

    ax.text(
        *text_xyz,
        txt,
        color="black",
        rotation=rot,
        path_effects=halo_fx,
        zorder=10001,
        **kw,
    )


def _auto_layout(figs, gap, seed=None):
    """
    Deterministic: prefer 'stack' when dims are monotone non-increasing
    (common textbook composite), else choose by proportions.
    """
    hs = [h for h, _, _ in figs]
    ws = [w for _, w, _ in figs]
    ls = [l for *_, l in figs]
    mono = (
        all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))
        and all(ws[i] >= ws[i + 1] for i in range(len(ws) - 1))
        and all(ls[i] >= ls[i + 1] for i in range(len(ls) - 1))
    )
    if mono:
        return "stack"

    Htot = sum(hs)
    Lmax = max(ls)
    Wmax = max(ws)
    if Htot > 1.5 * max(Lmax, Wmax):
        return "step"
    if Lmax > 1.25 * Wmax:
        return "side"

    # Only return L if all heights are the same
    if len(set(hs)) == 1:
        return "L"
    else:
        return "side"  # Fallback to side layout if heights differ


def _layout_positions(figs, layout, gap):
    """Return (x,y,z) for each prism (figs use [h,w,l])."""
    n = len(figs)
    if layout == "stack":
        z, pos = 0.0, []
        for h, w, l in figs:
            pos.append((0.0, 0.0, z))
            z += h + gap
        return pos
    if layout == "side":
        x, pos = 0.0, []
        for h, w, l in figs:
            pos.append((x, 0.0, 0.0))
            x += l + gap
        return pos
    if layout == "L":
        # Optional: require equal heights for a flush L top
        heights = [h for h, w, l in figs]
        if len(set(heights)) > 1:
            raise ValueError("L layout requires all prisms to have the same height")

        n = len(figs)
        pos = [(0.0, 0.0, 0.0) for _ in range(n)]  # prism 0 always at front/left/base

        # Start cursors from prism 0 footprint
        base_h, base_w, base_l = figs[0]
        y_cursor = base_w + gap  # extend back (+y) from prism 0
        x_cursor = base_l + gap  # extend right (+x) from prism 0

        # Always place prism 1 behind (+y); remaining go to the right (+x)
        if n >= 2:
            h1, w1, l1 = figs[1]
            pos[1] = (0.0, y_cursor, 0.0)
            y_cursor += w1 + gap

        for i in range(2, n):
            hi, wi, li = figs[i]
            pos[i] = (x_cursor, 0.0, 0.0)
            x_cursor += li + gap

        return pos
    return [(0.0, 0.0, 0.0) for _ in range(n)]


# ------------------------------------------------------------
# Main (layout-aware painter)
# ------------------------------------------------------------


@stimulus_function
def generate_composite_rect_prism_v2(stimulus_description: CompositeRectangularPrism2):
    sd = stimulus_description
    prisms = sd.figures  # each = [h, w, l]
    unit = sd.units
    hide = set(sd.hide_measurements)

    # Process show_labels: extend with True if shorter than prisms list
    show_labels = sd.show_labels[:]  # copy to avoid modifying the original
    while len(show_labels) < len(prisms):
        show_labels.append(True)  # default to True for missing entries

    # Decide positions
    if sd.positions is not None:
        if len(sd.positions) != len(prisms):
            raise ValueError("positions length must match figures length")
        # Convert list[list[float]] to list[tuple[float, float, float]]
        positions = [tuple(pos) for pos in sd.positions]
        layout = "manual"
    else:
        layout = (
            sd.layout if sd.layout != "auto" else _auto_layout(prisms, sd.gap, sd.seed)
        )
        positions = _layout_positions(prisms, layout, sd.gap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # reproducible randomness if sd.seed is provided
    rng = random.Random(sd.seed)

    # choose a pleasant isometric-ish view (kept in a readable range)
    elev = rng.uniform(16.0, 26.0)  # slight tilt
    azim = rng.uniform(-72.0, -42.0)  # from front-right-ish

    ax.view_init(elev=elev, azim=azim)

    # Scene limits
    xs = [x for (x, _, _) in positions] + [
        positions[i][0] + prisms[i][2] for i in range(len(prisms))
    ]
    ys = [y for (_, y, _) in positions] + [
        positions[i][1] + prisms[i][1] for i in range(len(prisms))
    ]
    zs = [z for (_, _, z) in positions] + [
        positions[i][2] + prisms[i][0] for i in range(len(prisms))
    ]
    pad = max(0.75, 0.08 * max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_zlim(max(0.0, min(zs) - pad * 0.4), max(zs) + pad)

    faces = _nice_faces(None)

    # ---------- Layout-aware draw order ----------
    all_faces = []

    if layout == "stack":
        # Use the same approach as version 1 - calculate box centroids and let matplotlib sort
        elev = np.deg2rad(ax.elev)
        azim = np.deg2rad(ax.azim)
        view_vec = np.array(
            [np.cos(elev) * np.sin(azim), -np.cos(elev) * np.cos(azim), np.sin(elev)]
        )

        # Calculate centroid and depth for each box
        box_data = []
        for i, ((x, y, z), (h, w, l)) in enumerate(zip(positions, prisms)):
            centroid = np.array([x + l / 2, y + w / 2, z + h / 2])
            depth = centroid.dot(view_vec)
            box_data.append((i, x, y, z, h, w, l, depth))

        # Sort by depth (farthest first)
        box_data.sort(key=lambda item: item[7])

        # Draw each box as a single Poly3DCollection with depth info
        for i, x, y, z, h, w, l, depth in box_data:
            faces_data = _box_faces(x, y, z, l, w, h, faces)
            verts = [face_verts for face_verts, _ in faces_data]
            face_colors = [face_color for _, face_color in faces_data]

            poly = Poly3DCollection(
                verts, facecolors=face_colors, edgecolor=faces["line"], linewidth=1.1
            )
            poly._sort_zpos = depth  # This is the key - let matplotlib handle sorting
            if hasattr(poly, "set_depthshade"):
                poly.set_depthshade(False)
            ax.add_collection3d(poly)

        # Skip the normal face collection since we handled it above
        all_faces = []

    elif layout == "side":
        # Order by view-vector depth (far -> near)
        v = _view_vec(ax)

        def depth(i):
            x, y, z = positions[i]
            h, w, l = prisms[i]
            c = np.array([x + l / 2, y + w / 2, z + h / 2])
            return float(c.dot(v))

        idxs = sorted(range(len(prisms)), key=depth)
        for i in idxs:
            x, y, z = positions[i]
            h, w, l = prisms[i]
            all_faces.extend(_box_faces(x, y, z, l, w, h, faces))

    else:
        # Mixed geometries: face-level projected sort (robust)
        for (x, y, z), (h, w, l) in zip(positions, prisms):
            all_faces.extend(_box_faces(x, y, z, l, w, h, faces))
        zmin, zmax = ax.get_zlim3d()
        _, _, znear = proj3d.proj_transform(0, 0, zmin, ax.get_proj())
        _, _, zfar = proj3d.proj_transform(0, 0, zmax, ax.get_proj())
        reverse = zfar > znear
        all_faces.sort(key=lambda item: _projected_depth(ax, item[0]), reverse=reverse)

    # Draw faces
    for verts, fc in all_faces:
        p = Poly3DCollection(
            [verts], facecolors=[fc], edgecolor=faces["line"], linewidth=1.1
        )
        if hasattr(p, "set_depthshade"):
            p.set_depthshade(False)
        if hasattr(p, "set_zsort"):
            p.set_zsort("min")
        ax.add_collection3d(p)

    pad_lbl = 0.18  # same default used by _label_dim
    # For side-by-side only, make shorter prism's H go to the opposite edge
    global_max_h = max(h for h, _, _ in prisms) if layout == "side" else None
    if layout == "L":
        for i, ((x, y, z), (h, w, l)) in enumerate(zip(positions, prisms)):
            if i in hide or not show_labels[i]:
                continue

            # L: first prism length at front; second prism length at back
            if i == 1:
                # push y forward so y - pad == (original y + w + pad) => back edge
                _label_dim(
                    ax,
                    "L",
                    x,
                    y + w + 2 * pad_lbl,
                    z,
                    l,
                    w,
                    h,
                    f"{l} {unit}",
                    layout=layout,
                    pad=pad_lbl,
                )
            else:
                _label_dim(
                    ax, "L", x, y, z, l, w, h, f"{l} {unit}", layout=layout, pad=pad_lbl
                )

            # W on top (your existing L behavior)
            _label_dim(
                ax, "W", x, y, z, l, w, h, f"{w} {unit}", layout=layout, pad=pad_lbl
            )

            # show H only once (first prism)
            if i == 0:
                _label_dim(
                    ax, "H", x, y, z, l, w, h, f"{h} {unit}", layout=layout, pad=pad_lbl
                )
    else:
        # non-L layouts unchanged
        for i, ((x, y, z), (h, w, l)) in enumerate(zip(positions, prisms)):
            if i in hide or not show_labels[i]:
                continue
            _label_dim(
                ax,
                "L",
                x,
                y,
                z,
                l,
                w,
                h,
                f"{l} {unit}",
                layout=layout,
                global_max_h=global_max_h,
            )
            _label_dim(
                ax,
                "W",
                x,
                y,
                z,
                l,
                w,
                h,
                f"{w} {unit}",
                layout=layout,
                global_max_h=global_max_h,
            )
            _label_dim(
                ax,
                "H",
                x,
                y,
                z,
                l,
                w,
                h,
                f"{h} {unit}",
                layout=layout,
                global_max_h=global_max_h,
            )

    # Cleanup
    for setter in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        try:
            setter([])
        except:  # noqa: E722
            pass
    plt.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)

    # Save
    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/composite_rect_prism_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        file_name,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


###################################################
# Geometric Shapes - Decomposition - Unit Square #
###################################################
@stimulus_function
def generate_unit_squares_unitless(data: UnitSquares):
    max_width = max(d.width for d in data)
    max_length = max(d.length for d in data)
    scale_factor = 0.5
    fig, axs = plt.subplots(
        1,
        len(data),
        figsize=(5 * max_width * scale_factor, 7 * max_length * scale_factor),
    )

    for i, ax in enumerate(axs):
        rectangle = data[i]
        length = rectangle.length
        width = rectangle.width

        # Add the scaled rectangle
        rect = patches.Rectangle(
            (0, 0),
            width * scale_factor,
            length * scale_factor,
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Now add the scaled unit square grid within the rectangle
        for x in range(width):
            for y in range(length):
                square = patches.Rectangle(
                    (x * scale_factor, y * scale_factor),
                    scale_factor,
                    scale_factor,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor="none",
                )
                ax.add_artist(square)
        # Set xlim and ylim according to scaled dimensions
        ax.set_xlim(
            [-0.5 * scale_factor, max_width * scale_factor + 0.5 * scale_factor]
        )
        ax.set_ylim(
            [-0.5 * scale_factor, max_length * scale_factor + 0.5 * scale_factor]
        )

        ax.axis("off")  # disable x-axis and y-axis
        ax.set_aspect("equal", "box")

        # Place the title text centered under the figure
        if len(data) > 1:
            ax.text(
                width * scale_factor / 2,
                -0.5 * scale_factor,
                f"Figure {i+1}",
                ha="center",
                fontsize=16,
            )

    plt.subplots_adjust(bottom=0.15)
    fig.canvas.draw()  # Redraw the canvas
    plt.tight_layout(pad=0.5)  # Set padding to 0.5

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_unit_square_unitless{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_area_stimulus(params: AreaStimulusParams):
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )  # Keep the figure square for better proportions

    def add_right_angle_marker(ax, x, y, size, position="bottom-left"):
        """Add a small right-angle marker at the specified (x, y) position."""
        # Calculate the aspect ratio of the plot
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        aspect_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])

        # Adjust the size to maintain a square shape
        adjusted_size_x = size
        adjusted_size_y = size / aspect_ratio

        if position == "bottom-left":
            right_angle_marker = np.array(
                [
                    [x, y],
                    [x + adjusted_size_x, y],
                    [x + adjusted_size_x, y + adjusted_size_y],
                    [x, y + adjusted_size_y],
                ]
            )
        elif position == "bottom-right":
            right_angle_marker = np.array(
                [
                    [x - adjusted_size_x, y],
                    [x, y],
                    [x, y + adjusted_size_y],
                    [x - adjusted_size_x, y + adjusted_size_y],
                ]
            )
        elif position == "top-right":
            right_angle_marker = np.array(
                [
                    [x - adjusted_size_x, y - adjusted_size_y],
                    [x, y - adjusted_size_y],
                    [x, y],
                    [x - adjusted_size_x, y],
                ]
            )
        elif position == "top-left":
            right_angle_marker = np.array(
                [
                    [x, y - adjusted_size_y],
                    [x + adjusted_size_x, y - adjusted_size_y],
                    [x + adjusted_size_x, y],
                    [x, y],
                ]
            )

        ax.add_patch(
            patches.Polygon(
                right_angle_marker,
                closed=True,
                fill=None,
                edgecolor="black",
                linewidth=linewidth,
            )
        )

    base = params.base
    height = params.height
    shape = params.shape
    not_to_scale_note = params.not_to_scale_note

    # Set the largest dimension to a fixed size (e.g., 5) and scale the smaller dimension proportionally
    max_size = 3

    # Normalize dimensions to a fixed range
    if base > height:
        normalized_base = 1.5
        normalized_height = 1
    elif height > base:
        normalized_base = 1
        normalized_height = 1.5
    else:
        normalized_base = 1
        normalized_height = 1

    # Scale the normalized dimensions to a fixed size
    scaled_base = normalized_base * max_size
    scaled_height = normalized_height * max_size

    # Set constant padding and offsets
    base_offset = -0.4  # Constant offset for base label
    height_offset = -0.4  # Constant offset for height label
    note_offset = -0.8  # Constant padding for the "not to scale" note

    linewidth = 2  # Set the line thickness
    font_size = 18  # Increase the font size for labels
    right_angle_marker_size = 0.3  # Constant size for right-angle marker

    if shape == "right_triangle":
        # Define the coordinates for the right triangle's vertices
        triangle_points = np.array([[0, 0], [scaled_base, 0], [0, scaled_height]])

        # Plot the right triangle
        triangle = patches.Polygon(
            triangle_points,
            closed=True,
            fill=None,
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_patch(triangle)

        # Set limits for x and y axis to accommodate the scaled shape with some padding
        ax.set_xlim(-0.5, scaled_base + 0.5)
        ax.set_ylim(-0.5, scaled_height + 0.2)

        # Add a right-angle marker at the right-angle corner (0, 0) with constant size
        add_right_angle_marker(
            ax, 0, 0, size=right_angle_marker_size, position="bottom-left"
        )

        # Label the base and height with constant offsets
        ax.text(
            scaled_base / 2, base_offset, f"{base}", fontsize=font_size, ha="center"
        )  # Label for the base
        ax.text(
            height_offset,
            scaled_height / 2,
            f"{height}",
            fontsize=font_size,
            va="center",
        )  # Label for the height

        # Add the note that the figure is not drawn to scale
        plt.text(
            scaled_base / 2,
            note_offset,
            not_to_scale_note,
            fontsize=font_size,
            ha="center",
        )

    elif shape == "rectangle":
        # Define the coordinates for the rectangle's vertices
        rectangle_points = np.array(
            [[0, 0], [0, scaled_height], [scaled_base, scaled_height], [scaled_base, 0]]
        )

        # Plot the rectangle
        rectangle = patches.Polygon(
            rectangle_points,
            closed=True,
            fill=None,
            edgecolor="black",
            linewidth=linewidth,
        )
        ax.add_patch(rectangle)

        # Adjust axis limits to reduce padding around the shape
        ax.set_xlim(-0.5, scaled_base + 0.5)
        ax.set_ylim(-0.5, scaled_height + 0.5)

        # Add right-angle markers at each corner of the rectangle with constant size
        add_right_angle_marker(
            ax, 0, 0, size=right_angle_marker_size, position="bottom-left"
        )  # Bottom-left corner
        add_right_angle_marker(
            ax, scaled_base, 0, size=right_angle_marker_size, position="bottom-right"
        )  # Bottom-right corner
        add_right_angle_marker(
            ax,
            scaled_base,
            scaled_height,
            size=right_angle_marker_size,
            position="top-right",
        )  # Top-right corner
        add_right_angle_marker(
            ax, 0, scaled_height, size=right_angle_marker_size, position="top-left"
        )  # Top-left corner

        # Label the base and height with constant offsets
        ax.text(
            scaled_base / 2, base_offset, f"{base}", fontsize=font_size, ha="center"
        )
        ax.text(
            height_offset,
            scaled_height / 2,
            f"{height}",
            fontsize=font_size,
            va="center",
        )

        # Add the note that the figure is not drawn to scale
        plt.text(
            scaled_base / 2,
            note_offset,
            not_to_scale_note,
            fontsize=font_size,
            ha="center",
        )

    # Hide the axes for a cleaner appearance
    ax.axis("off")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/area_stimulus_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=600,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_similar_right_triangles(params: SimilarRightTriangles):
    fig, ax = plt.subplots(figsize=(8, 4))

    # Define the coordinates for the larger triangle
    large_triangle_points = np.array([[0, 0], [4, 0], [0, 3]])
    small_triangle_points = np.array([[5, 0], [7, 0], [5, 1.5]])

    # Plot the larger triangle
    large_triangle = patches.Polygon(
        large_triangle_points,
        closed=True,
        fill=None,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(large_triangle)

    # Plot the smaller triangle
    small_triangle = patches.Polygon(
        small_triangle_points,
        closed=True,
        fill=None,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(small_triangle)

    # Add right-angle markers
    ax.add_patch(
        patches.Rectangle((0, 0), 0.3, 0.3, fill=None, edgecolor="black", linewidth=1)
    )
    ax.add_patch(
        patches.Rectangle((5, 0), 0.15, 0.15, fill=None, edgecolor="black", linewidth=1)
    )

    # Access angle labels from params
    angle_labels = params.angle_labels

    # Label the angles closer to the triangles
    ax.text(-0.1, -0.1, angle_labels[2], fontsize=12, ha="right", va="top")  # Adjusted
    ax.text(4.1, -0.1, angle_labels[0], fontsize=12, ha="left", va="top")  # Adjusted
    ax.text(
        -0.1, 3.1, angle_labels[1], fontsize=12, ha="right", va="bottom"
    )  # Adjusted

    ax.text(5.0, -0.05, angle_labels[5], fontsize=12, ha="right", va="top")  # Adjusted
    ax.text(7.0, -0.05, angle_labels[3], fontsize=12, ha="left", va="top")  # Adjusted
    ax.text(
        5.0, 1.55, angle_labels[4], fontsize=12, ha="right", va="bottom"
    )  # Adjusted

    # Add the note
    ax.text(3.5, -1, "Note: Figures not drawn to scale.", fontsize=12, ha="center")

    # Set limits and hide axes
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 4)
    ax.axis("off")

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/similar_right_triangles_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=600,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_circle_with_arcs(params: CircleWithArcsDescription):
    fig, ax = plt.subplots()

    # Draw the circle
    circle = patches.Circle((0, 0), 1, edgecolor="black", facecolor="none", linewidth=2)
    ax.add_artist(circle)

    # Calculate the arc endpoints based on the arc_size
    arc_rad = np.radians(params.arc_size)
    arc_x = np.cos(arc_rad)
    arc_y = np.sin(arc_rad)

    # Draw intersecting lines
    ax.plot([-arc_x, arc_x], [-arc_y, arc_y], color="black", linewidth=2)
    ax.plot([-arc_x, arc_x], [arc_y, -arc_y], color="black", linewidth=2)

    # Offset for labels
    offset = 0.1

    # Add labels for the points at the arc endpoints with slight offset
    ax.text(
        -arc_x - offset,
        -arc_y - offset,
        params.point_labels[0],
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.text(
        arc_x + offset,
        -arc_y - offset,
        params.point_labels[1],
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.text(
        -arc_x - offset,
        arc_y + offset,
        params.point_labels[3],
        fontsize=12,
        ha="center",
        va="center",
    )
    ax.text(
        arc_x + offset,
        arc_y + offset,
        params.point_labels[2],
        fontsize=12,
        ha="center",
        va="center",
    )

    # Dynamic position for 'O'
    if params.arc_size <= 45:
        ax.text(0, offset, "O", fontsize=12, ha="center", va="center")  # Move up
    else:
        ax.text(-offset, 0, "O", fontsize=12, ha="center", va="center")  # Move left

    # Set limits and aspect
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add note
    plt.figtext(0.5, 0, "Note: Figure not drawn to scale.", ha="center", fontsize=10)

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/circle_with_intersecting_lines_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=600,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_composite_rectangular_grid(data: CompositeRectangularGrid):
    return draw_composite_rectangular_grid_internal(data)


@stimulus_function
def draw_composite_rectangular_grid_with_dashed_lines(data: CompositeRectangularGrid):
    return draw_composite_rectangular_grid_internal(data, dashed_lines=True)


def draw_composite_rectangular_grid_internal(
    data: CompositeRectangularGrid, dashed_lines: bool = False
):
    """
    Draw the union of two rectangles, fill it, and label every outer edge
    (including concave cuts) with a horizontal length label just outside the shape.
    Segments sharing the same x (or y) get merged into one continuous label.
    """
    # 1) Build & union
    rects = data.rectangles
    unit = rects[0].unit.value
    polys = [box(r.x, r.y, r.x + r.width, r.y + r.length) for r in rects]
    union_poly = polys[0].union(polys[1])

    def format_unit(value, base_unit):
        """Format unit to be singular (Unit) or plural (Units) based on value"""
        if base_unit == "Units":
            return "Unit" if value == 1 else "Units"
        return base_unit

    # 2) Extract exterior ring coords
    if isinstance(union_poly, Polygon):
        exterior = list(union_poly.exterior.coords)
    elif isinstance(union_poly, MultiPolygon):
        exterior = list(max(union_poly.geoms, key=lambda p: p.area).exterior.coords)
    else:
        raise ValueError("Unexpected geometry type for union.")
    xs, ys = zip(*exterior)

    # 3) Draw shape
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill(xs, ys, color="#b3f5a3", alpha=0.6, edgecolor="green", linewidth=2)
    ax.plot(xs, ys, color="green", linewidth=2)

    if dashed_lines:
        # 3.5) Draw dashed lines to show individual rectangles
        # This helps students identify each rectangle when they overlap or are adjacent

        # Check if rectangles overlap or are adjacent
        rect1_intersects_rect2 = polys[0].intersects(polys[1])

        if rect1_intersects_rect2:
            # Draw the boundary of each rectangle with dashed lines
            for i, poly in enumerate(polys):
                boundary = poly.boundary
                if isinstance(boundary, LineString):
                    coords = list(boundary.coords)
                    if len(coords) >= 2:
                        xs_dash, ys_dash = zip(*coords)
                        # Use different colors or patterns for each rectangle
                        color = "black" if i == 0 else "darkgray"
                        ax.plot(
                            xs_dash,
                            ys_dash,
                            color=color,
                            linewidth=1.5,
                            linestyle="--",
                            alpha=0.7,
                            label=f"Rectangle {i+1}" if len(polys) > 1 else None,
                        )
                elif isinstance(boundary, (MultiLineString, GeometryCollection)):
                    # Handle MultiLineString and GeometryCollection cases
                    for geom in boundary.geoms:
                        if isinstance(geom, LineString):
                            coords = list(geom.coords)
                            if len(coords) >= 2:
                                xs_dash, ys_dash = zip(*coords)
                                color = "black" if i == 0 else "darkgray"
                                ax.plot(
                                    xs_dash,
                                    ys_dash,
                                    color=color,
                                    linewidth=1.5,
                                    linestyle="--",
                                    alpha=0.7,
                                )
        else:
            # For non-overlapping rectangles, just draw the common edge if it exists
            rect1_boundary = polys[0].boundary
            rect2_boundary = polys[1].boundary
            common_edge = rect1_boundary.intersection(rect2_boundary)

            if not common_edge.is_empty:
                if isinstance(common_edge, LineString):
                    coords = list(common_edge.coords)
                    if len(coords) >= 2:
                        xs_dash, ys_dash = zip(*coords)
                        ax.plot(
                            xs_dash,
                            ys_dash,
                            color="black",
                            linewidth=2,
                            linestyle="--",
                            alpha=0.8,
                        )
                elif isinstance(common_edge, (MultiLineString, GeometryCollection)):
                    for geom in common_edge.geoms:
                        if isinstance(geom, LineString):
                            coords = list(geom.coords)
                            if len(coords) >= 2:
                                xs_dash, ys_dash = zip(*coords)
                                ax.plot(
                                    xs_dash,
                                    ys_dash,
                                    color="black",
                                    linewidth=2,
                                    linestyle="--",
                                    alpha=0.8,
                                )

    # 4) Compute base offset (10% of avg side)
    eps = 1e-8
    lengths = []
    for i in range(len(exterior) - 1):
        x0, y0 = exterior[i]
        x1, y1 = exterior[i + 1]
        if abs(x0 - x1) < eps or abs(y0 - y1) < eps:
            lengths.append(np.hypot(x1 - x0, y1 - y0))
    avg_len = np.mean(lengths) if lengths else 1.0
    offset = 0.1 * avg_len
    h_offset = 2.5 * offset  # extra horizontal padding for vertical labels

    # 5) Group axis‐aligned segments
    verticals = defaultdict(list)
    horizontals = defaultdict(list)
    for i in range(len(exterior) - 1):
        x0, y0 = exterior[i]
        x1, y1 = exterior[i + 1]
        if abs(x0 - x1) < eps:
            verticals[round(x0, 8)].append((min(y0, y1), max(y0, y1)))
        elif abs(y0 - y1) < eps:
            horizontals[round(y0, 8)].append((min(x0, x1), max(x0, x1)))

    # 6) Compute bounds
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 7) Create Label objects for all edges with collision detection
    all_labels = []

    # Create vertical edge labels
    for x, segs in verticals.items():
        y0_all = min(s[0] for s in segs)
        y1_all = max(s[1] for s in segs)
        mid_y = 0.5 * (y0_all + y1_all)
        length = y1_all - y0_all

        # test a point half‐offset right; reverse if inside
        test_pt = Point(x + offset / 2, mid_y)
        direction = 1 if not union_poly.contains(test_pt) else -1
        lx = x + direction * h_offset

        val = int(length) if abs(length - int(length)) < 1e-6 else round(length, 2)
        formatted_unit = format_unit(val, unit)
        text = f"{val} {formatted_unit}"

        # Create label with measured dimensions
        text_width, text_height = measure_text_dimensions(text, 14, ax)
        label = Label(
            x=lx,
            y=mid_y,
            text=text,
            fontsize=14,
            ha="center",
            va="center",
            width=text_width,
            height=text_height,
        )

        # Store positioning info for potential repositioning
        label.positioning_params = {
            "edge_x": x,
            "edge_y_range": (y0_all, y1_all),
            "is_vertical": True,
            "base_offset": h_offset,
            "direction": direction,
            "union_poly": union_poly,
        }

        all_labels.append(label)

    # Create horizontal edge labels
    for y, segs in horizontals.items():
        x0_all = min(s[0] for s in segs)
        x1_all = max(s[1] for s in segs)
        mid_x = 0.5 * (x0_all + x1_all)
        length = x1_all - x0_all

        test_pt = Point(mid_x, y + offset / 2)
        direction = 1 if not union_poly.contains(test_pt) else -1
        ly = y + direction * offset

        val = int(length) if abs(length - int(length)) < 1e-6 else round(length, 2)
        formatted_unit = format_unit(val, unit)
        text = f"{val} {formatted_unit}"

        # Create label with measured dimensions
        text_width, text_height = measure_text_dimensions(text, 14, ax)
        label = Label(
            x=mid_x,
            y=ly,
            text=text,
            fontsize=14,
            ha="center",
            va="center",
            width=text_width,
            height=text_height,
        )

        # Store positioning info for potential repositioning
        label.positioning_params = {
            "edge_y": y,
            "edge_x_range": (x0_all, x1_all),
            "is_vertical": False,
            "base_offset": offset,
            "direction": direction,
            "union_poly": union_poly,
        }

        all_labels.append(label)

    # 8) Apply collision detection and repositioning
    adjusted_labels = _resolve_label_collisions(all_labels, offset, h_offset)

    # 9) Draw all labels
    for label in adjusted_labels:
        ax.text(
            label.x,
            label.y,
            label.text,
            fontsize=label.fontsize,
            ha=label.ha,
            va=label.va,
            rotation=0,
        )

    # 9) Finalize & save
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(min_x - h_offset, max_x + h_offset)
    ax.set_ylim(min_y - offset, max_y + offset)
    plt.tight_layout()

    fname = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/composite_rectangular_grid_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        fname,
        dpi=600,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return fname


def _resolve_label_collisions(labels, offset, h_offset):
    """
    Resolve label collisions by repositioning overlapping labels.
    For small edges, prefer inside placement to avoid cluttering.
    Returns adjusted list of labels with collision-free positioning.
    """
    if len(labels) <= 1:
        return labels

    # Calculate edge lengths to identify small edges
    edge_lengths = []
    for label in labels:
        params = label.positioning_params
        if params and params["is_vertical"]:
            y0, y1 = params["edge_y_range"]
            edge_lengths.append(y1 - y0)
        elif params and not params["is_vertical"]:
            x0, x1 = params["edge_x_range"]
            edge_lengths.append(x1 - x0)
        else:
            edge_lengths.append(float("inf"))  # Unknown length

    # Determine threshold for "small" edges (bottom 40% of edge lengths)
    valid_lengths = [length for length in edge_lengths if length != float("inf")]
    if valid_lengths:
        small_edge_threshold = sorted(valid_lengths)[int(len(valid_lengths) * 0.4)]
    else:
        small_edge_threshold = 0

    adjusted_labels = []

    for i, label in enumerate(labels):
        # Check for collisions with already placed labels
        current_position = label
        collision_found = False

        for placed_label in adjusted_labels:
            if labels_overlap(current_position, placed_label, buffer=0.2):
                collision_found = True
                break

        if not collision_found:
            adjusted_labels.append(current_position)
            continue

        # Try to find alternative position
        params = label.positioning_params
        if params is None:
            # If no positioning params, keep original position
            adjusted_labels.append(current_position)
            continue

        # Check if this is a small edge that should be placed inside
        is_small_edge = edge_lengths[i] <= small_edge_threshold and edge_lengths[
            i
        ] != float("inf")

        alternative_label = _find_alternative_position(
            label,
            adjusted_labels,
            params,
            offset,
            h_offset,
            prefer_inside=is_small_edge,
        )
        adjusted_labels.append(alternative_label)

    return adjusted_labels


def _find_alternative_position(
    label, existing_labels, params, offset, h_offset, prefer_inside=False
):
    """
    Find an alternative position for a label to avoid collisions.
    If prefer_inside=True, try inside placement first for small edges.
    """
    if params["is_vertical"]:
        return _find_alternative_vertical_position(
            label, existing_labels, params, h_offset, prefer_inside
        )
    else:
        return _find_alternative_horizontal_position(
            label, existing_labels, params, offset, prefer_inside
        )


def _find_alternative_vertical_position(
    label, existing_labels, params, h_offset, prefer_inside=False
):
    """
    Find alternative position for vertical edge labels.
    Try inside placement first if prefer_inside=True, otherwise try different offsets.
    """
    edge_x = params["edge_x"]
    y0, y1 = params["edge_y_range"]
    mid_y = 0.5 * (y0 + y1)
    direction = params["direction"]

    # If prefer_inside, try inside placement first
    if prefer_inside:
        inside_label = _try_inside_vertical_placement(label, existing_labels, params)
        if inside_label:
            return inside_label

    # Try different horizontal distances (outside placement)
    alternative_offsets = [
        h_offset * 1.5,  # Further out
        h_offset * 2.0,  # Even further
        h_offset * 0.7,  # Closer in
    ]

    # Try different vertical positions along the edge
    vertical_positions = [
        mid_y,  # Original center
        mid_y + (y1 - y0) * 0.2,  # Slightly up
        mid_y - (y1 - y0) * 0.2,  # Slightly down
        mid_y + (y1 - y0) * 0.1,  # Slightly up (smaller)
        mid_y - (y1 - y0) * 0.1,  # Slightly down (smaller)
    ]

    for h_off in alternative_offsets:
        for v_pos in vertical_positions:
            # Constrain vertical position to edge bounds with small margin
            v_pos = max(y0 + 0.1, min(y1 - 0.1, v_pos))

            test_label = Label(
                x=edge_x + direction * h_off,
                y=v_pos,
                text=label.text,
                fontsize=label.fontsize,
                ha=label.ha,
                va=label.va,
                width=label.width,
                height=label.height,
            )

            # Check for collisions
            has_collision = any(
                labels_overlap(test_label, existing, buffer=0.15)
                for existing in existing_labels
            )

            if not has_collision:
                test_label.positioning_params = params
                return test_label

    # If still no good position and didn't try inside yet, try inside now
    if not prefer_inside:
        inside_label = _try_inside_vertical_placement(label, existing_labels, params)
        if inside_label:
            return inside_label

    # If no good position found, use original with increased offset
    fallback_label = Label(
        x=edge_x + direction * h_offset * 2.5,
        y=mid_y,
        text=label.text,
        fontsize=label.fontsize,
        ha=label.ha,
        va=label.va,
        width=label.width,
        height=label.height,
    )
    fallback_label.positioning_params = params
    return fallback_label


def _find_alternative_horizontal_position(
    label, existing_labels, params, offset, prefer_inside=False
):
    """
    Find alternative position for horizontal edge labels.
    Try inside placement first if prefer_inside=True, otherwise try different offsets.
    """
    edge_y = params["edge_y"]
    x0, x1 = params["edge_x_range"]
    mid_x = 0.5 * (x0 + x1)
    direction = params["direction"]

    # If prefer_inside, try inside placement first
    if prefer_inside:
        inside_label = _try_inside_horizontal_placement(label, existing_labels, params)
        if inside_label:
            return inside_label

    # Try different vertical distances (outside placement)
    alternative_offsets = [
        offset * 1.5,  # Further out
        offset * 2.0,  # Even further
        offset * 0.7,  # Closer in
    ]

    # Try different horizontal positions along the edge
    horizontal_positions = [
        mid_x,  # Original center
        mid_x + (x1 - x0) * 0.2,  # Slightly right
        mid_x - (x1 - x0) * 0.2,  # Slightly left
        mid_x + (x1 - x0) * 0.1,  # Slightly right (smaller)
        mid_x - (x1 - x0) * 0.1,  # Slightly left (smaller)
    ]

    for v_off in alternative_offsets:
        for h_pos in horizontal_positions:
            # Constrain horizontal position to edge bounds with small margin
            h_pos = max(x0 + 0.1, min(x1 - 0.1, h_pos))

            test_label = Label(
                x=h_pos,
                y=edge_y + direction * v_off,
                text=label.text,
                fontsize=label.fontsize,
                ha=label.ha,
                va=label.va,
                width=label.width,
                height=label.height,
            )

            # Check for collisions
            has_collision = any(
                labels_overlap(test_label, existing, buffer=0.15)
                for existing in existing_labels
            )

            if not has_collision:
                test_label.positioning_params = params
                return test_label

    # If still no good position and didn't try inside yet, try inside now
    if not prefer_inside:
        inside_label = _try_inside_horizontal_placement(label, existing_labels, params)
        if inside_label:
            return inside_label

    # If no good position found, use original with increased offset
    fallback_label = Label(
        x=mid_x,
        y=edge_y + direction * offset * 2.5,
        text=label.text,
        fontsize=label.fontsize,
        ha=label.ha,
        va=label.va,
        width=label.width,
        height=label.height,
    )
    fallback_label.positioning_params = params
    return fallback_label


def _try_inside_vertical_placement(label, existing_labels, params):
    """
    Try placing a vertical edge label inside the shape.
    Returns Label if successful, None if no good inside position found.
    """
    edge_x = params["edge_x"]
    y0, y1 = params["edge_y_range"]
    mid_y = 0.5 * (y0 + y1)
    union_poly = params["union_poly"]

    # For vertical edges, place label inside with some margin from the edge
    inside_margin = 0.3

    # Try different positions inside the shape
    inside_positions = [
        (
            edge_x + inside_margin,
            mid_y,
            "left",
            "center",
        ),  # Right of edge, left-aligned
        (
            edge_x - inside_margin,
            mid_y,
            "right",
            "center",
        ),  # Left of edge, right-aligned
        (edge_x + inside_margin * 1.5, mid_y, "left", "center"),  # Further inside
        (
            edge_x - inside_margin * 1.5,
            mid_y,
            "right",
            "center",
        ),  # Further inside (other side)
    ]

    for x_pos, y_pos, ha, va in inside_positions:
        # Check if position is inside the union polygon
        test_point = Point(x_pos, y_pos)
        if not union_poly.contains(test_point):
            continue

        # Constrain to edge bounds
        y_pos = max(y0 + 0.1, min(y1 - 0.1, y_pos))

        test_label = Label(
            x=x_pos,
            y=y_pos,
            text=label.text,
            fontsize=label.fontsize,
            ha=ha,
            va=va,
            width=label.width,
            height=label.height,
        )

        # Check for collisions with existing labels
        has_collision = any(
            labels_overlap(test_label, existing, buffer=0.1)
            for existing in existing_labels
        )

        if not has_collision:
            test_label.positioning_params = params.copy()
            test_label.positioning_params["inside_placement"] = True
            return test_label

    return None


def _try_inside_horizontal_placement(label, existing_labels, params):
    """
    Try placing a horizontal edge label inside the shape.
    Returns Label if successful, None if no good inside position found.
    """
    edge_y = params["edge_y"]
    x0, x1 = params["edge_x_range"]
    mid_x = 0.5 * (x0 + x1)
    union_poly = params["union_poly"]

    # For horizontal edges, place label inside with some margin from the edge
    inside_margin = 0.3

    # Try different positions inside the shape
    inside_positions = [
        (mid_x, edge_y + inside_margin, "center", "bottom"),  # Above edge
        (mid_x, edge_y - inside_margin, "center", "top"),  # Below edge
        (mid_x, edge_y + inside_margin * 1.5, "center", "bottom"),  # Further inside
        (
            mid_x,
            edge_y - inside_margin * 1.5,
            "center",
            "top",
        ),  # Further inside (other side)
    ]

    for x_pos, y_pos, ha, va in inside_positions:
        # Check if position is inside the union polygon
        test_point = Point(x_pos, y_pos)
        if not union_poly.contains(test_point):
            continue

        # Constrain to edge bounds
        x_pos = max(x0 + 0.1, min(x1 - 0.1, x_pos))

        test_label = Label(
            x=x_pos,
            y=y_pos,
            text=label.text,
            fontsize=label.fontsize,
            ha=ha,
            va=va,
            width=label.width,
            height=label.height,
        )

        # Check for collisions with existing labels
        has_collision = any(
            labels_overlap(test_label, existing, buffer=0.1)
            for existing in existing_labels
        )

        if not has_collision:
            test_label.positioning_params = params.copy()
            test_label.positioning_params["inside_placement"] = True
            return test_label

    return None


########################################################
# Geometric Shapes - Trapezoids with Side Lengths    #
########################################################
@stimulus_function
def generate_trapezoid_with_side_len(data: TrapezoidGrid):
    """
    Draws a trapezoid with labeled dimensions for area calculation problems.

    The trapezoid includes:
    - Base length labeled at bottom
    - Top length labeled at top
    - Height labeled on the side
    - All dimensions include units
    - Different trapezoid types (regular, isosceles, right, left)
    """
    # Convert dimensions to float for calculations
    base = float(data.base)
    top_length = float(data.top_length)
    height = float(data.height)
    unit = data.unit.value

    # Scale dimensions to fit well in figure (target total area ~25 square units)
    max_dim = max(base, top_length, height)
    if max_dim > 0:
        scale_factor = min(8.0 / max_dim, 1.0)  # Scale down if too large
    else:
        scale_factor = 1.0

    scaled_base = base * scale_factor
    scaled_top = top_length * scale_factor
    scaled_height = height * scale_factor

    # Use smaller, more efficient figure size
    fig_width = min(scaled_base + 2, 8)
    fig_height = min(scaled_height + 2, 6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate trapezoid vertices based on type
    if data.trapezoid_type == ETrapezoidType.RIGHT_TRAPEZOID:
        # Right trapezoid - right side is vertical
        vertices = [
            [0, 0],
            [scaled_base, 0],
            [scaled_base, scaled_height],
            [scaled_top, scaled_height],
        ]
    elif data.trapezoid_type == ETrapezoidType.LEFT_TRAPEZOID:
        # Left trapezoid - left side is vertical
        vertices = [
            [0, 0],
            [scaled_base, 0],
            [scaled_base - scaled_top, scaled_height],
            [0, scaled_height],
        ]
    else:
        # Regular or isosceles trapezoid - centered top
        offset = (scaled_base - scaled_top) / 2
        vertices = [
            [0, 0],
            [scaled_base, 0],
            [scaled_base - offset, scaled_height],
            [offset, scaled_height],
        ]

    # Draw the trapezoid more efficiently using patches
    trap_patch = patches.Polygon(
        vertices, closed=True, fill=None, edgecolor="blue", linewidth=3
    )
    ax.add_patch(trap_patch)

    # Calculate label positions and add dimensional labels
    label_offset = 0.3
    font_size = 14

    # Format dimension values efficiently
    def format_dimension(original_value):
        if isinstance(original_value, Fraction):
            if original_value.denominator == 1:
                return str(original_value.numerator)
            else:
                return f"{original_value.numerator}/{original_value.denominator}"
        else:
            # Standard numeric format
            if original_value == int(original_value):
                return str(int(original_value))
            else:
                return f"{original_value:.1f}"

    base_str = format_dimension(data.base)
    top_str = format_dimension(data.top_length)
    height_str = format_dimension(data.height) if not data.show_variable_height else "h"

    # Base label (bottom)
    base_mid_x = scaled_base / 2
    ax.text(
        base_mid_x,
        -label_offset,
        f"{base_str} {unit}",
        fontsize=font_size,
        ha="center",
        va="top",
        fontweight="bold",
        clip_on=False,
    )

    # Top label
    if data.trapezoid_type == ETrapezoidType.RIGHT_TRAPEZOID:
        top_mid_x = scaled_base - (scaled_top / 2)
    elif data.trapezoid_type == ETrapezoidType.LEFT_TRAPEZOID:
        top_mid_x = (scaled_base - scaled_top) + (scaled_top / 2)
    else:
        top_mid_x = scaled_base / 2  # Centered

    ax.text(
        top_mid_x,
        scaled_height + label_offset,
        f"{top_str} {unit}",
        fontsize=font_size,
        ha="center",
        va="bottom",
        fontweight="bold",
        clip_on=False,
    )

    # Height label (left side for most types, right side for left trapezoid)
    if data.trapezoid_type == ETrapezoidType.LEFT_TRAPEZOID:
        # Label on right side for left trapezoid
        height_x = scaled_base + label_offset
        height_rotation = 90
        ha = "center"
    else:
        # Label on left side for other types
        height_x = -label_offset
        height_rotation = 90
        ha = "center"

    height_mid_y = scaled_height / 2
    ax.text(
        height_x,
        height_mid_y,
        f"{height_str} {unit}",
        fontsize=font_size,
        ha=ha,
        va="center",
        rotation=height_rotation,
        fontweight="bold",
        clip_on=False,
    )

    # Add title if label is provided
    if data.label:
        ax.text(
            scaled_base / 2,
            scaled_height + 2 * label_offset,
            data.label,
            fontsize=16,
            ha="center",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits with padding
    padding = 1.0
    ax.set_xlim(-padding, scaled_base + padding)
    ax.set_ylim(-padding, scaled_height + padding * 2)

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/trapezoid_with_dimensions_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


def draw_polygon_with_string_side_lengths(
    ax, side_lengths=None, side_labels=None, color="blue", unit=""
):
    """
    Draw a polygon with side lengths specified as strings and optional labels.

    Args:
        ax: matplotlib axis
        side_lengths: list of side lengths (can be strings or numbers)
        side_labels: list of labels for each side (optional)
        color: color of the polygon
        unit: unit string to append to side lengths
    """
    if side_lengths is None:
        side_lengths = ["3", "4", "5"]

    if side_labels is None:
        side_labels = side_lengths

    # Convert string lengths to numbers for geometric construction
    # Apply proportional scaling for better visual representation
    numeric_lengths = []
    for length in side_lengths:
        try:
            # Try to convert to float
            numeric_lengths.append(float(length))
        except (ValueError, TypeError):
            # If it's a string that can't be converted, use a default value
            numeric_lengths.append(3.0)

    # Apply proportional scaling to make longer sides visually larger
    max_length = max(numeric_lengths)
    min_display_size = 2.0
    max_display_size = 6.0
    scale_factor = (
        (max_display_size - min_display_size) / max_length if max_length > 0 else 1.0
    )

    def get_scaled_length(length):
        return min_display_size + (length * scale_factor)

    scaled_lengths = [get_scaled_length(length) for length in numeric_lengths]

    n_sides = len(side_lengths)

    if n_sides == 3:
        # Triangle - use the scalene triangle approach with scaled lengths
        a, b, c = scaled_lengths[0], scaled_lengths[1], scaled_lengths[2]
        # Ensure triangle inequality
        if a + b > c and a + c > b and b + c > a:
            cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
            cos_C = max(-1, min(1, cos_C))  # Clamp to valid range
            angle_C = np.arccos(cos_C)
            x = np.array([0, c, a * np.cos(angle_C)])
            y = np.array([0, 0, a * np.sin(angle_C)])
        else:
            # Fallback to equilateral triangle if triangle inequality fails
            avg_scale = sum(scaled_lengths) / 3
            radius = avg_scale * 0.8
            theta = np.linspace(0, 2 * np.pi, n_sides + 1)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
    elif n_sides == 4:
        # Quadrilateral - simple rectangle-like shape with scaled dimensions
        width = scaled_lengths[0]
        height = scaled_lengths[1] if len(scaled_lengths) > 1 else width
        x = np.array([0, width, width, 0, 0])
        y = np.array([0, 0, height, height, 0])
    else:
        # Regular polygon for other cases - use average scaled length for radius
        avg_scale = sum(scaled_lengths) / len(scaled_lengths)
        radius = avg_scale * 0.8
        theta = np.linspace(0, 2 * np.pi, n_sides + 1)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

    # Close the polygon
    if n_sides != 4:  # Quadrilateral already closed
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Draw the polygon
    ax.plot(x, y, color=color, linewidth=3)
    ax.fill(x, y, color=color, alpha=0.3)

    # Calculate centroid for label positioning
    cx, cy = np.mean(x[:-1]), np.mean(y[:-1])

    # Add side length labels with improved positioning and scaling
    font_size = 24  # Increased for better legibility

    for i in range(n_sides):
        # Get the two points of this side
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]

        # Calculate midpoint of the side
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate normal vector (perpendicular to the side)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx, ny = -dy / length, dx / length

            # Flip normal if it points inward
            if (nx * (cx - mx) + ny * (cy - my)) > 0:
                nx, ny = -nx, -ny

            # Position label on the side line for consistent placement
            label_x = mx
            label_y = my

            # Create the label text
            label_text = side_labels[i]
            if label_text:  # Only add label if it's not empty
                if unit:
                    label_text += f" {unit}"

                # Add the label
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    ax.axis("equal")
    ax.axis("off")


@stimulus_function
def draw_polygon_with_string_sides(side_data: PolygonStringSides):
    """
    Stimulus function to draw a polygon with string side lengths.

    Expected input format:
    {
        "side_lengths": ["3", "4", "5"],
        "side_labels": ["a", "b", "c"],  # optional
        "unit": "cm"  # optional
    }
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    side_lengths = side_data.side_lengths
    side_labels = (
        side_data.side_labels if side_data.side_labels is not None else side_lengths
    )
    unit = side_data.unit if side_data.unit is not None else ""

    # Select random color for visual variety
    polygon_color = get_random_polygon_color()

    draw_polygon_with_string_side_lengths(
        ax,
        side_lengths=side_lengths,
        side_labels=side_labels,
        color=polygon_color,
        unit=unit,
    )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/polygon_string_sides_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_polygon_perimeter_with_all_sides_labeled(side_data: PolygonPerimeter):
    """
    Draw a polygon for perimeter calculation problems.

    Shows all side lengths with units
    Used for educational problems where students need to find the perimeter.

    Features:
    - Supports regular polygons with 3-10 sides
    - Supports L-shaped polygons (6 sides)
    - Supports T-shaped polygons (8 sides)
    - All sides labeled with their length and unit
    - Proper geometric construction for educational clarity
    """
    side_data.unknown_side_indices = []
    return draw_polygon_perimeter_internal(side_data)


@stimulus_function
def draw_polygon_perimeter(side_data: PolygonPerimeter):
    """
    Draw a polygon for perimeter calculation problems.

    Shows all side lengths with units except one side which is marked with "?".
    Used for educational problems where students need to find the missing side.

    Features:
    - Supports regular polygons with 3-10 sides
    - Supports L-shaped polygons (6 sides)
    - Supports T-shaped polygons (8 sides)
    - One side labeled with "?" for the unknown measurement
    - All other sides labeled with their length and unit
    - Proper geometric construction for educational clarity
    """
    return draw_polygon_perimeter_internal(side_data)


def draw_polygon_perimeter_internal(side_data: PolygonPerimeter):
    # Validate that all side lengths are positive
    if not all(length > 0 for length in side_data.side_lengths):
        raise ValueError("All side lengths must be positive")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert integers to strings for geometric construction
    side_lengths_str = [str(length) for length in side_data.side_lengths]

    # Create labels: show "length unit" for known sides, no label for unknown sides
    side_labels = []
    for i, length in enumerate(side_data.side_lengths):
        if i in side_data.unknown_side_indices:
            if side_data.shape_type == "regular":
                side_labels.append("")  # No label for unknown sides
            else:
                side_labels.append("?")  # No label for unknown sides
        else:
            side_labels.append(f"{length}")

    # Select random color for visual variety
    polygon_color = get_random_polygon_color()

    # Draw polygon based on shape type
    shape_type = getattr(side_data, "shape_type", "regular")

    if shape_type == "L-shape":
        draw_l_shaped_polygon(
            ax,
            side_lengths=side_lengths_str,
            side_labels=side_labels,
            unit=side_data.unit,
            color=polygon_color,
        )
    elif shape_type == "T-shape":
        draw_t_shaped_polygon(
            ax,
            side_lengths=side_lengths_str,
            side_labels=side_labels,
            unit=side_data.unit,
            color=polygon_color,
        )
    elif shape_type == "irregular":
        # Irregular polygon that scales to actual side lengths
        draw_irregular_polygon_with_labels(
            ax,
            side_lengths=side_data.side_lengths,  # Use actual numeric values
            side_labels=side_labels,
            color=polygon_color,
            unit=side_data.unit,
        )
    else:
        # Regular polygon (existing functionality)
        draw_polygon_with_string_side_lengths(
            ax,
            side_lengths=side_lengths_str,
            side_labels=side_labels,
            color=polygon_color,
            unit=side_data.unit,
        )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/polygon_perimeter_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_composite_rectangular_triangular_grid(
    data: CompositeRectangularTriangularGrid,
):
    """
    Draw the union of two rectangles and one triangle, fill it, and label every outer edge
    (including concave cuts) with length labels just outside the shape.
    Segments sharing the same x (or y) get merged into one continuous label.
    """
    # 1) Build geometries and union
    rects = data.rectangles
    triangle = data.triangle
    unit = rects[0].unit.value

    # Create rectangle polygons
    rect_polys = [box(r.x, r.y, r.x + r.width, r.y + r.length) for r in rects]

    # Create triangle polygon from right-angle vertex, base, and height
    triangle_vertices = [
        (triangle.right_angle_x, triangle.right_angle_y),  # Right-angle vertex
        (
            triangle.right_angle_x + triangle.base,
            triangle.right_angle_y,
        ),  # Base endpoint
        (
            triangle.right_angle_x,
            triangle.right_angle_y + triangle.height,
        ),  # Height endpoint
    ]
    triangle_poly = Polygon(triangle_vertices)

    # Union all shapes
    union_poly = rect_polys[0].union(rect_polys[1]).union(triangle_poly)

    # 2) Extract exterior ring coords and draw all components
    fig, ax = plt.subplots(figsize=(8, 6))

    if isinstance(union_poly, Polygon):
        # Single polygon - draw as before
        exterior = list(union_poly.exterior.coords)
        xs, ys = zip(*exterior)
        ax.fill(xs, ys, color="#b3f5a3", alpha=0.6, edgecolor="green", linewidth=2)
        ax.plot(xs, ys, color="green", linewidth=2)
    elif isinstance(union_poly, MultiPolygon):
        # Multiple polygons - draw all components
        all_exteriors = []
        for geom in union_poly.geoms:
            exterior = list(geom.exterior.coords)
            all_exteriors.append(exterior)
            xs, ys = zip(*exterior)
            ax.fill(xs, ys, color="#b3f5a3", alpha=0.6, edgecolor="green", linewidth=2)
            ax.plot(xs, ys, color="green", linewidth=2)
        # Use the largest component for bounds and labeling
        exterior = list(max(union_poly.geoms, key=lambda p: p.area).exterior.coords)
    else:
        raise ValueError("Unexpected geometry type for union.")

    # Get coordinates for bounds calculation
    xs, ys = zip(*exterior)

    # 4) Compute base offset (same approach as draw_composite_rectangular_grid)
    eps = 1e-8
    lengths = []
    for i in range(len(exterior) - 1):
        x0, y0 = exterior[i]
        x1, y1 = exterior[i + 1]
        if abs(x0 - x1) < eps or abs(y0 - y1) < eps:
            lengths.append(np.hypot(x1 - x0, y1 - y0))
    avg_len = np.mean(lengths) if lengths else 1.0
    offset = 0.1 * avg_len  # 10% of average edge length
    h_offset = 2.5 * offset  # 2.5 times the base offset

    # 5) Group axis-aligned segments and handle non-axis-aligned edges
    # Label ALL horizontal and vertical edges regardless of length
    # Only filter diagonal edges to remove union operation artifacts
    min_edge_length = 0.5  # Only applies to diagonal edges

    verticals = defaultdict(list)
    horizontals = defaultdict(list)
    diagonal_edges = []

    # Collect edges from all components (if MultiPolygon) or single polygon
    exteriors_to_process = []
    if isinstance(union_poly, Polygon):
        exteriors_to_process = [list(union_poly.exterior.coords)]
    elif isinstance(union_poly, MultiPolygon):
        exteriors_to_process = [list(geom.exterior.coords) for geom in union_poly.geoms]

    for exterior_coords in exteriors_to_process:
        for i in range(len(exterior_coords) - 1):
            x0, y0 = exterior_coords[i]
            x1, y1 = exterior_coords[i + 1]
            edge_length = np.hypot(x1 - x0, y1 - y0)

            if abs(x0 - x1) < eps:  # Vertical edge
                verticals[round(x0, 8)].append((min(y0, y1), max(y0, y1)))
            elif abs(y0 - y1) < eps:  # Horizontal edge
                horizontals[round(y0, 8)].append((min(x0, x1), max(x0, x1)))
            else:  # Diagonal edge (from triangle)
                # Only filter diagonal edges - skip very short ones that are artifacts
                if edge_length >= min_edge_length:
                    diagonal_edges.append(((x0, y0), (x1, y1)))

    # 6) Compute bounds
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 7) Label vertical edges - merge connected segments, label others separately
    for x, segs in verticals.items():
        # Sort segments by y-coordinate
        segs.sort()

        # Merge connected segments
        merged_segs = []
        current_start, current_end = segs[0]

        for y0, y1 in segs[1:]:
            if abs(y0 - current_end) < eps:  # Connected to current segment
                current_end = y1  # Extend current segment
            else:  # Gap found, start new segment
                merged_segs.append((current_start, current_end))
                current_start, current_end = y0, y1

        # Add the final segment
        merged_segs.append((current_start, current_end))

        # Label each merged segment
        for y0, y1 in merged_segs:
            mid_y = 0.5 * (y0 + y1)
            length = y1 - y0

            # Test a point half-offset right; reverse if inside (same as rectangular grid)
            test_pt = Point(x + h_offset / 2, mid_y)
            direction = 1 if not union_poly.contains(test_pt) else -1
            lx = x + direction * h_offset

            val = int(length) if abs(length - int(length)) < 1e-6 else round(length, 2)
            ax.text(
                lx,
                mid_y,
                f"{val} {unit}",
                fontsize=14,
                ha="center",
                va="center",
                rotation=0,
            )

    # 8) Label horizontal edges - merge connected segments, label others separately
    for y, segs in horizontals.items():
        # Sort segments by x-coordinate
        segs.sort()

        # Merge connected segments
        merged_segs = []
        current_start, current_end = segs[0]

        for x0, x1 in segs[1:]:
            if abs(x0 - current_end) < eps:  # Connected to current segment
                current_end = x1  # Extend current segment
            else:  # Gap found, start new segment
                merged_segs.append((current_start, current_end))
                current_start, current_end = x0, x1

        # Add the final segment
        merged_segs.append((current_start, current_end))

        # Label each merged segment
        for x0, x1 in merged_segs:
            mid_x = 0.5 * (x0 + x1)
            length = x1 - x0

            # Test a point half-offset up; reverse if inside (same as rectangular grid)
            test_pt = Point(mid_x, y + offset / 2)
            direction = 1 if not union_poly.contains(test_pt) else -1
            ly = y + direction * offset

            val = int(length) if abs(length - int(length)) < 1e-6 else round(length, 2)
            ax.text(
                mid_x,
                ly,
                f"{val} {unit}",
                fontsize=14,
                ha="center",
                va="center",
                rotation=0,
            )

    # 9) Skip diagonal edges (triangle hypotenuse) - only label base and height
    # The triangle's base and height are already captured as horizontal/vertical edges
    # so we don't need to label the diagonal hypotenuse

    # 10) Finalize & save
    ax.set_aspect("equal")
    ax.axis("off")
    # Expand axis limits to ensure all labels are visible
    # Add extra padding beyond the offset to account for label text size
    label_padding = 0.5  # Extra padding for label text
    ax.set_xlim(min_x - h_offset - label_padding, max_x + h_offset + label_padding)
    ax.set_ylim(min_y - offset - label_padding, max_y + offset + label_padding)
    plt.tight_layout()

    fname = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/composite_rectangular_triangular_grid_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        fname,
        dpi=600,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return fname


###################################################################
# Geometric Shapes - Triangles with Optional Side Lengths       #
# (For Pythagorean Theorem Problems)                             #
###################################################################
@stimulus_function
def generate_triangle_with_opt_side_len(data: TriangularGridOpt):
    """
    Draws a triangle with optional side lengths for Pythagorean theorem problems.

    Features:
    - Calculates missing sides using Pythagorean theorem when only 2 sides provided
    - Supports three exercise types: find_hypotenuse, find_leg, verify_right_triangle
    - Selective labeling based on exercise type and label_sides parameter
    - Conditional right-angle marker display based on show_right_angle_symbol
    - Rotation support (0, 90, 180, 270 degrees)
    - Optionally displays Pythagorean theorem calculation
    - Supports real-world problem solving in two dimensions
    """
    # Get all side values (calculated values filled in by model validation)
    side1, side2, side3 = data.get_side_values()
    original_sides = [data.side1, data.side2, data.side3]

    # Determine which sides were originally provided vs calculated
    provided_sides = [i for i, s in enumerate(original_sides) if s is not None]
    calculated_sides = [i for i, s in enumerate(original_sides) if s is None]

    # Use the sides for triangle construction (same approach as existing function)
    b, a, c = side1, side2, side3  # side1=left, side2=slanted, side3=base

    # Scale longest side to ≤ 10 units for proper display
    scale = min(1.0, 10.0 / max(a, b, c))

    # Law of cosines for triangle construction
    x2 = (b**2 + c**2 - a**2) / (2 * c)
    y2 = math.sqrt(max(0.0, b**2 - x2**2))

    # Triangle vertices: P0(0,0), P1(c,0), P2(x2,y2)
    verts = np.array([[0.0, 0.0], [c, 0.0], [x2, y2]]) * scale

    # Apply rotation if specified
    if data.rotation_angle != 0:
        # Calculate triangle centroid for rotation center
        centroid = verts.mean(axis=0)

        # Apply rotation around centroid
        angle_rad = np.radians(data.rotation_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        # Rotation matrix
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        # Apply rotation to each vertex relative to centroid
        rotated_verts = []
        for vertex in verts:
            # Translate to origin (relative to centroid)
            translated = vertex - centroid
            # Apply rotation
            rotated = np.dot(rotation_matrix, translated)
            # Translate back
            rotated_verts.append(rotated + centroid)

        verts = np.array(rotated_verts)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw triangle
    xs = np.append(verts[:, 0], verts[0, 0])
    ys = np.append(verts[:, 1], verts[0, 1])
    ax.plot(xs, ys, color="blue", linewidth=3)

    # Fill triangle lightly to make it more visible
    ax.fill(xs[:-1], ys[:-1], color="lightblue", alpha=0.2)

    # Calculate padding for labels
    max_side = max(a, b, c) * scale
    pad_side = 0.12 * max_side
    pad_angle = 0.25 * max_side

    # Centroid for inside/outside label positioning
    cx, cy = verts.mean(axis=0)

    # Function to label sides
    def label_side(vertex_i, vertex_j, side_index, original_length):
        x0, y0 = verts[vertex_i]
        x1, y1 = verts[vertex_j]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2

        # Calculate outward normal vector
        dx, dy = x1 - x0, y1 - y0
        nx, ny = -dy, dx
        norm = math.hypot(nx, ny) or 1

        # Ensure normal points outward
        if (nx * (cx - mx) + ny * (cy - my)) > 0:
            nx, ny = -nx, -ny

        ox, oy = (nx / norm) * pad_side, (ny / norm) * pad_side

        # Determine label text based on whether side was provided or calculated
        if side_index in provided_sides:
            # Known side - show actual value with unit
            if should_use_fraction_display(original_length):
                label_text = (
                    f"{decimal_to_mixed_number(original_length)} {data.unit.value}"
                )
            else:
                if original_length == int(original_length):
                    label_text = f"{int(original_length)} {data.unit.value}"
                else:
                    label_text = f"{original_length:.1f} {data.unit.value}"
            font_color = "black"
            font_weight = "bold"
        else:
            # Unknown side - show variable or calculated value based on settings
            if data.label_unknown:
                # Show variable name
                variable_names = ["a", "b", "c"]
                label_text = variable_names[side_index]
                font_color = "red"
                font_weight = "bold"
            else:
                # Show calculated value
                calculated_value = [side1, side2, side3][side_index]
                if should_use_fraction_display(calculated_value):
                    label_text = (
                        f"{decimal_to_mixed_number(calculated_value)} {data.unit.value}"
                    )
                else:
                    if calculated_value == int(calculated_value):
                        label_text = f"{int(calculated_value)} {data.unit.value}"
                    else:
                        label_text = f"{calculated_value:.1f} {data.unit.value}"
                font_color = "darkgreen"
                font_weight = "bold"

        ax.text(
            mx + ox,
            my + oy,
            label_text,
            ha="center",
            va="center",
            fontsize=16,
            color=font_color,
            weight=font_weight,
            clip_on=False,
        )

    # Label sides based on label_sides parameter
    if data.label_sides and len(data.label_sides) == 3:
        if data.label_sides[2]:  # side3 (base)
            label_side(0, 1, 2, c)
        if data.label_sides[1]:  # side2 (slanted)
            label_side(1, 2, 1, a)
        if data.label_sides[0]:  # side1 (left)
            label_side(2, 0, 0, b)
    else:
        # Default: label all sides
        label_side(0, 1, 2, c)  # Base (side3)
        label_side(1, 2, 1, a)  # Slanted side (side2)
        label_side(2, 0, 0, b)  # Left side (side1)

    # Check if this is a right triangle and find the right angle
    angles_deg = []
    # Map side lengths to opposite vertices correctly:
    # side_lengths[i] = length of side opposite vertex i
    side_lengths = [a, b, c]  # [opposite_v0, opposite_v1, opposite_v2]

    # Calculate angles using law of cosines
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        # For angle at vertex i, we need:
        # - The two sides that meet at vertex i: edge(i,j) and edge(i,k)
        # - The side opposite vertex i: edge(j,k)
        # Edge (i,j) has length equal to side opposite vertex k
        # Edge (i,k) has length equal to side opposite vertex j
        # Edge (j,k) has length equal to side opposite vertex i
        edge_ij = side_lengths[k]  # length of edge from vertex i to vertex j
        edge_ik = side_lengths[j]  # length of edge from vertex i to vertex k
        edge_jk = side_lengths[
            i
        ]  # length of edge from vertex j to vertex k (opposite vertex i)

        cos_angle = (edge_ij**2 + edge_ik**2 - edge_jk**2) / (2 * edge_ij * edge_ik)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
        angle_rad = math.acos(cos_angle)
        angles_deg.append(angle_rad * 180 / math.pi)

    # Find right angle (approximately 90 degrees)
    right_angle_index = None
    for i, angle in enumerate(angles_deg):
        if abs(angle - 90) < 1.0:  # Allow small tolerance
            right_angle_index = i
            break

    # Draw right angle marker if it's a right triangle and show_right_angle_symbol is True
    if right_angle_index is not None and data.show_right_angle_symbol:
        vertex_index = right_angle_index
        P = verts[vertex_index]

        # Get the two adjacent vertices
        i0 = (vertex_index + 1) % 3
        i1 = (vertex_index + 2) % 3
        v0 = verts[i0] - P
        v1 = verts[i1] - P

        # Create unit vectors
        u0 = v0 / np.linalg.norm(v0)
        u1 = v1 / np.linalg.norm(v1)

        # Size of right angle marker
        marker_size = pad_angle * 0.4

        # Create right angle marker square
        A = P + u0 * marker_size
        C = P + u1 * marker_size
        B = A + u1 * marker_size

        # Draw the right angle marker
        ax.plot([P[0], A[0]], [P[1], A[1]], color="black", linewidth=2)
        ax.plot([A[0], B[0]], [A[1], B[1]], color="black", linewidth=2)
        ax.plot([B[0], C[0]], [B[1], C[1]], color="black", linewidth=2)
        ax.plot([C[0], P[0]], [C[1], P[1]], color="black", linewidth=2)

    # Add calculation display if requested
    if data.show_calculation and len(calculated_sides) > 0:
        # Find the provided sides for calculation display
        provided_values = [original_sides[i] for i in provided_sides]
        calculated_side_index = calculated_sides[0]
        calculated_value = [side1, side2, side3][calculated_side_index]

        if len(provided_sides) == 2:
            # Show Pythagorean theorem calculation
            provided_values = [original_sides[i] for i in provided_sides]
            a_val, b_val = sorted([v for v in provided_values if v is not None])

            # Determine if we calculated hypotenuse or leg
            if abs(calculated_value - math.sqrt(a_val**2 + b_val**2)) < 0.1:
                # Calculated hypotenuse
                calc_text = f"c² = a² + b²\nc² = {a_val}² + {b_val}²\nc² = {a_val**2} + {b_val**2}\nc² = {a_val**2 + b_val**2}\nc = √{a_val**2 + b_val**2}\nc = {calculated_value:.1f}"
            else:
                # Calculated leg
                non_none_values = [v for v in provided_values if v is not None]
                hyp = max(non_none_values)
                leg = min(non_none_values)
                calc_text = f"a² + b² = c²\na² = c² - b²\na² = {hyp}² - {leg}²\na² = {hyp**2} - {leg**2}\na² = {hyp**2 - leg**2}\na = √{hyp**2 - leg**2}\na = {calculated_value:.1f}"

            # Position calculation text
            text_x = max(verts[:, 0]) + pad_side * 2
            text_y = max(verts[:, 1]) * 0.8

            ax.text(
                text_x,
                text_y,
                calc_text,
                ha="left",
                va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                clip_on=False,
            )

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits with extra padding for labels and calculations
    padding = pad_side * 3
    if data.show_calculation:
        padding_right = max(verts[:, 0]) + pad_side * 8  # Extra space for calculation
    else:
        padding_right = max(verts[:, 0]) + padding

    ax.set_xlim(min(verts[:, 0]) - padding, padding_right)
    ax.set_ylim(min(verts[:, 1]) - padding, max(verts[:, 1]) + padding)

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/triangle_opt_side_len_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def generate_rect_with_side_len_and_area(data: RectangleWithHiddenSide):
    """
    Generate a rectangle with one side shown, the other labeled with "?", and area displayed inside.

    Features:
    - Shows one dimension based on show_length/show_width flags
    - Labels hidden dimension with "?"
    - Displays area in the center of the rectangle
    - Uses proper units: unit for sides, square unit for area
    """
    from fractions import Fraction

    length = float(data.length)
    width = float(data.width)
    area = length * width  # Calculate area from dimensions
    unit = data.unit.value

    # Scale rectangle to fit well in figure (target max dimension ~8 units)
    max_dim = max(length, width)
    if max_dim > 0:
        scale_factor = min(8.0 / max_dim, 1.0)
    else:
        scale_factor = 1.0

    scaled_length = length * scale_factor
    scaled_width = width * scale_factor

    # Create figure with appropriate size
    fig_width = min(scaled_width + 3, 10)
    fig_height = min(scaled_length + 3, 8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Draw the rectangle
    rect = patches.Rectangle(
        (0, 0),
        scaled_width,
        scaled_length,
        linewidth=3,
        edgecolor="blue",
        facecolor="lightblue",
        alpha=0.3,
    )
    ax.add_patch(rect)

    # Calculate label positions
    label_offset = 0.3
    font_size = 16

    # Format dimension values efficiently
    def format_dimension(original_value):
        if isinstance(original_value, Fraction):
            if original_value.denominator == 1:
                return str(original_value.numerator)
            else:
                return f"$\\frac{{{original_value.numerator}}}{{{original_value.denominator}}}$"
        else:
            # Standard numeric format
            if original_value == int(original_value):
                return str(int(original_value))
            else:
                return f"{original_value:.1f}"

    def format_area(original_area):
        if isinstance(original_area, Fraction):
            if original_area.denominator == 1:
                return str(original_area.numerator)
            else:
                return f"$\\frac{{{original_area.numerator}}}{{{original_area.denominator}}}$"
        else:
            if original_area == int(original_area):
                return str(int(original_area))
            else:
                return f"{original_area:.1f}"

    def format_unit(value, base_unit):
        """Format unit to be singular (unit) or plural (units) based on value"""
        if base_unit == "Units":
            return "unit" if value == 1 else "units"
        return base_unit

    # Width label (bottom)
    width_mid_x = scaled_width / 2
    if data.show_width:
        width_str = format_dimension(data.width)
        formatted_unit = format_unit(float(data.width), unit)
        width_label = f"{width_str} {formatted_unit}"
    else:
        width_label = "?"

    ax.text(
        width_mid_x,
        -label_offset,
        width_label,
        fontsize=font_size,
        ha="center",
        va="top",
        fontweight="bold",
        clip_on=False,
    )

    # Length label (left side)
    length_mid_y = scaled_length / 2
    if data.show_length:
        length_str = format_dimension(data.length)
        formatted_unit = format_unit(float(data.length), unit)
        length_label = f"{length_str} {formatted_unit}"
    else:
        length_label = "?"

    ax.text(
        -label_offset,
        length_mid_y,
        length_label,
        fontsize=font_size,
        ha="right",
        va="center",
        rotation=90,
        fontweight="bold",
        clip_on=False,
    )

    # Area label (center of rectangle)
    area_str = format_area(area)
    formatted_unit = format_unit(area, unit)
    area_label = f"{area_str} square {formatted_unit}"

    ax.text(
        scaled_width / 2,
        scaled_length / 2,
        area_label,
        fontsize=font_size + 2,
        ha="center",
        va="center",
        fontweight="bold",
        clip_on=False,
    )

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")

    # Set limits with padding
    padding = 1.0
    ax.set_xlim(-padding, scaled_width + padding)
    ax.set_ylim(-padding, scaled_length + padding)

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/rectangle_with_area_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=500,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return file_name


@stimulus_function
def draw_polygon_fully_labeled(side_data: PolygonFullyLabeled):
    """
    Draw a polygon with all sides labeled with their measurements.

    Shows all side lengths with units for educational problems where students
    need to see complete measurements, such as calculating perimeter when all
    dimensions are provided.

    Features:
    - Supports squares (4 equal sides)
    - Supports rectangles (4 sides, opposite sides equal)
    - Supports regular polygons with 3-10 sides
    - Supports L-shaped polygons (6 sides)
    - Supports T-shaped polygons (8 sides)
    - All sides labeled with their length and unit
    - Proper geometric construction for educational clarity
    """
    # Validation is now handled by the Pydantic model

    # Validate that all side lengths are positive
    if not all(length > 0 for length in side_data.side_lengths):
        raise ValueError("All side lengths must be positive")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.grid(False)
    # Remove title as requested

    # Create side labels (all sides are labeled, no unknowns)
    # For helper functions that add unit themselves, don't include unit here
    side_labels = [str(length) for length in side_data.side_lengths]
    shape_type = side_data.shape_type
    unit = side_data.unit

    # Get random color for all shapes
    polygon_color = get_random_polygon_color()

    # Draw the appropriate polygon based on shape type
    if shape_type == "square":
        # Draw as a square with equal sides
        side_length = side_data.side_lengths[0]  # All sides are equal for a square

        # Clear and redraw as a proper square
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(False)
        # Remove title as requested

        # Define square corners
        corners = [
            (0, 0),
            (side_length, 0),
            (side_length, side_length),
            (0, side_length),
            (0, 0),
        ]

        # Draw square with color fill
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        ax.plot(x_coords, y_coords, color=polygon_color, linewidth=2)
        ax.fill(x_coords, y_coords, color=polygon_color, alpha=0.3)

        # Add side labels for all four sides with consistent formatting
        label_offset = 0.4
        font_size = 14

        # Bottom side
        ax.text(
            side_length / 2,
            -label_offset,
            f"{side_length} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Right side
        ax.text(
            side_length + label_offset,
            side_length / 2,
            f"{side_length} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Top side
        ax.text(
            side_length / 2,
            side_length + label_offset,
            f"{side_length} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Left side
        ax.text(
            -label_offset,
            side_length / 2,
            f"{side_length} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Set appropriate limits
        margin = side_length * 0.2
        ax.set_xlim(-margin, side_length + margin)
        ax.set_ylim(-margin, side_length + margin)
    elif shape_type == "rectangle":
        # For rectangles, we'll draw a simple rectangular shape
        length = side_data.side_lengths[0]  # width
        width = side_data.side_lengths[1]  # height

        # Clear the plot and redraw as a proper rectangle
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(False)
        # Remove title as requested

        # Define rectangle corners
        corners = [(0, 0), (length, 0), (length, width), (0, width), (0, 0)]

        # Draw rectangle with color fill
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        ax.plot(x_coords, y_coords, color=polygon_color, linewidth=2)
        ax.fill(x_coords, y_coords, color=polygon_color, alpha=0.3)

        # Add side labels with consistent formatting
        label_offset = 0.4
        font_size = 14

        # Bottom side
        ax.text(
            length / 2,
            -label_offset,
            f"{length} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Right side
        ax.text(
            length + label_offset,
            width / 2,
            f"{width} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Top side
        ax.text(
            length / 2,
            width + label_offset,
            f"{length} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Left side
        ax.text(
            -label_offset,
            width / 2,
            f"{width} {unit}",
            ha="center",
            va="center",
            fontsize=font_size,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        # Set appropriate limits
        margin = max(length, width) * 0.2
        ax.set_xlim(-margin, length + margin)
        ax.set_ylim(-margin, width + margin)

    elif shape_type == "regular":
        # Draw as a regular polygon using the existing helper function
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(False)
        # Remove title as requested

        draw_polygon_with_string_side_lengths(
            ax,
            side_lengths=side_data.side_lengths,
            side_labels=side_labels,
            color=polygon_color,
            unit=unit,
        )
    elif shape_type == "L-shape":
        # Draw L-shaped polygon
        draw_l_shaped_polygon(
            ax,
            side_lengths=side_data.side_lengths,
            side_labels=side_labels,
            unit=unit,
            color=polygon_color,
        )
    elif shape_type == "T-shape":
        # Draw T-shaped polygon
        draw_t_shaped_polygon(
            ax,
            side_lengths=side_data.side_lengths,
            side_labels=side_labels,
            unit=unit,
            color=polygon_color,
        )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/polygon_fully_labeled_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


def draw_l_shaped_polygon(ax, side_lengths, side_labels, unit="", color="blue"):
    # Convert string lengths to numbers for construction
    numeric_lengths = []
    for length in side_lengths:
        try:
            numeric_lengths.append(float(length))
        except (ValueError, TypeError):
            numeric_lengths.append(3.0)  # Default value

    # L-shape coordinates - scale proportionally to maintain true ratios
    # Apply pure proportional scaling to preserve side length relationships
    max_length = max(numeric_lengths)
    target_max_size = 6.0  # Maximum size for the longest side
    scale_factor = target_max_size / max_length if max_length > 0 else 1.0

    def get_scaled_length(length):
        # Pure proportional scaling - maintains exact ratios between sides
        return length * scale_factor

    l0, l1, l2, l3, l4, l5 = [get_scaled_length(length) for length in numeric_lengths]

    # Construct L-shape coordinates
    # Start at origin, go clockwise
    vertices = [
        [0, 0],  # Start (bottom-left)
        [l0, 0],  # After side 0 (bottom-right corner)
        [l0, l1],  # After side 1 (inner corner)
        [l0 - l2, l1],  # After side 2 (inner top)
        [l0 - l2, l1 + l3],  # After side 3 (top-right corner)
        [0, l1 + l3],  # After side 4 (top-left corner)
        [0, 0],  # Close polygon (back to start)
    ]

    # Draw the L-shaped polygon
    xs, ys = zip(*vertices)
    ax.plot(xs, ys, color=color, linewidth=3)
    ax.fill(xs[:-1], ys[:-1], color=color, alpha=0.3)

    # Calculate centroid for label positioning
    cx = sum(x for x in xs[:-1]) / len(xs[:-1])
    cy = sum(y for y in ys[:-1]) / len(ys[:-1])

    # Add side length labels with improved positioning
    font_size = 24  # Increased for better legibility

    # Define side information for L-shape: (start_vertex_index, end_vertex_index)
    sides = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]

    for i, (start_idx, end_idx) in enumerate(sides):
        # Get the two points of this side
        x1, y1 = vertices[start_idx]
        x2, y2 = vertices[end_idx]

        # Calculate midpoint of the side
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate normal vector (perpendicular to the side)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx, ny = -dy / length, dx / length

            # Flip normal if it points inward (rough check)
            if (nx * (cx - mx) + ny * (cy - my)) > 0:
                nx, ny = -nx, -ny

            # Position label on the side line for consistent placement
            label_x = mx
            label_y = my

            # Create the label text
            label_text = side_labels[i]
            if label_text:  # Only add label if it's not empty
                if unit:
                    label_text += f" {unit}"

                # Add the label
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    ax.set_aspect("equal")
    ax.axis("off")


def draw_t_shaped_polygon(ax, side_lengths, side_labels, unit="", color="blue"):
    # Convert string lengths to numbers for construction
    numeric_lengths = []
    for length in side_lengths:
        try:
            numeric_lengths.append(float(length))
        except (ValueError, TypeError):
            numeric_lengths.append(3.0)  # Default value

    # T-shape coordinates - scale proportionally to maintain true ratios
    # Apply pure proportional scaling to preserve side length relationships
    max_length = max(numeric_lengths)
    target_max_size = 6.0  # Maximum size for the longest side
    scale_factor = target_max_size / max_length if max_length > 0 else 1.0

    def get_scaled_length(length):
        # Pure proportional scaling - maintains exact ratios between sides
        return length * scale_factor

    l0, l1, l2, l3, l4, l5, l6, l7 = [
        get_scaled_length(length) for length in numeric_lengths
    ]

    # Construct T-shape coordinates
    # Start at bottom-left, go clockwise
    vertices = [
        [0, 0],  # Start (bottom-left)
        [l0, 0],  # After side 0 (bottom-right)
        [l0, l1],  # After side 1 (inner bottom-right)
        [l0 + l2, l1],  # After side 2 (outer bottom-right)
        [l0 + l2, l1 + l3],  # After side 3 (outer top-right)
        [0 - l6, l1 + l3],  # After side 4 (outer top-left)
        [0 - l6, l1],  # After side 5 (inner top-left)
        [0, l1],  # After side 6 (inner bottom-left)
        [0, 0],  # Close polygon (back to start)
    ]

    # Adjust coordinates to center the T-shape
    min_x = min(x for x, y in vertices[:-1])
    offset_x = -min_x if min_x < 0 else 0
    vertices = [[x + offset_x, y] for x, y in vertices]

    # Draw the T-shaped polygon
    xs, ys = zip(*vertices)
    ax.plot(xs, ys, color=color, linewidth=3)
    ax.fill(xs[:-1], ys[:-1], color=color, alpha=0.3)

    # Calculate centroid for label positioning
    cx = sum(x for x in xs[:-1]) / len(xs[:-1])
    cy = sum(y for y in ys[:-1]) / len(ys[:-1])

    # Add side length labels with improved positioning
    font_size = 24  # Increased for better legibility

    # Define side information for T-shape: (start_vertex_index, end_vertex_index)
    sides = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]

    for i, (start_idx, end_idx) in enumerate(sides):
        # Get the two points of this side
        x1, y1 = vertices[start_idx]
        x2, y2 = vertices[end_idx]

        # Calculate midpoint of the side
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate normal vector (perpendicular to the side)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx, ny = -dy / length, dx / length

            # Flip normal if it points inward (rough check)
            if (nx * (cx - mx) + ny * (cy - my)) > 0:
                nx, ny = -nx, -ny

            # Position label on the side line for consistent placement
            label_x = mx
            label_y = my

            # Create the label text
            label_text = side_labels[i]
            if label_text:  # Only add label if it's not empty
                if unit:
                    label_text += f" {unit}"

                # Add the label
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    ax.set_aspect("equal")
    ax.axis("off")


def construct_irregular_polygon(side_lengths):
    """
    Construct an irregular polygon from given side lengths.

    This function creates a polygon that approximates the given side lengths
    while ensuring the polygon closes properly. The approach varies based on
    the number of sides:

    - 3 sides: Uses law of cosines for exact triangle construction
    - 4 sides: Uses a modified rectangle approach respecting all sides
    - 5+ sides: Uses a heuristic approach that tries to approximate the side lengths

    Args:
        side_lengths: List of numeric side lengths

    Returns:
        tuple: (x_coordinates, y_coordinates) arrays for the polygon vertices
    """
    n_sides = len(side_lengths)

    if n_sides == 3:
        # For triangles, use geometric intersection to ensure exact side lengths
        # side_lengths[0] = AB, side_lengths[1] = BC, side_lengths[2] = CA
        side_AB, side_BC, side_CA = side_lengths[0], side_lengths[1], side_lengths[2]

        # Place A at origin, B at (side_AB, 0)
        A = np.array([0.0, 0.0])
        B = np.array([side_AB, 0.0])

        # Find C as intersection of two circles:
        # Circle 1: centered at A with radius side_CA
        # Circle 2: centered at B with radius side_BC

        # Distance between circle centers
        d = side_AB
        r1 = side_CA  # radius from A
        r2 = side_BC  # radius from B

        # Check if intersection is possible (triangle inequality)
        if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
            # Fallback to a valid triangle if intersection impossible
            C = np.array([r1 * 0.6, r1 * 0.8])
        else:
            # Calculate intersection point using circle intersection formula
            a = (r1**2 - r2**2 + d**2) / (2 * d)
            h = np.sqrt(r1**2 - a**2)

            # Point on line AB where perpendicular to C intersects
            P = A + a * (B - A) / d

            # C is offset perpendicular to AB
            C = P + h * np.array([0, 1])  # Choose positive y for consistent orientation

        x = np.array([A[0], B[0], C[0]])
        y = np.array([A[1], B[1], C[1]])

        return x, y

    elif n_sides == 4:
        # For quadrilaterals, create a shape that better respects all four sides
        s1, s2, s3, s4 = side_lengths

        # Use the side lengths directly (scaling is handled in the calling function)

        # Create a quadrilateral by adjusting a rectangle
        # Start with a rectangle using average dimensions
        height = (s2 + s4) / 2  # Average of opposite sides

        # Adjust corners to better approximate the given sides
        corner_adjust = 0.3

        x = np.array(
            [
                0,
                s1,  # Use actual first side length
                s1 + corner_adjust * (s2 - height),  # Adjust right side
                corner_adjust * (s4 - height),  # Adjust left side
            ]
        )

        y = np.array(
            [
                0,
                0,
                s2,  # Use actual second side length
                s4,  # Use actual fourth side length
            ]
        )

        return x, y

    else:
        # For polygons with 5+ sides, use a heuristic approach
        # This creates an approximately correct polygon

        # Use the side lengths directly (scaling is handled in the calling function)
        scaled_lengths = side_lengths

        # Start at origin
        x = [0]
        y = [0]

        # Current position and direction
        current_x, current_y = 0, 0
        current_angle = 0

        # Amount to turn each step for approximating polygon shape
        angle_adjustment = 2 * np.pi / n_sides

        for i in range(n_sides - 1):  # n-1 because we'll close the polygon
            # Use the specified side length
            length = scaled_lengths[i]

            # Calculate next vertex
            next_x = current_x + length * np.cos(current_angle)
            next_y = current_y + length * np.sin(current_angle)

            x.append(next_x)
            y.append(next_y)

            # Update position and angle
            current_x, current_y = next_x, next_y
            current_angle += angle_adjustment

            # Add some variation based on side length difference
            # This helps make the polygon look more irregular when side lengths vary
            if i < len(scaled_lengths) - 1:
                length_ratio = scaled_lengths[i] / scaled_lengths[i + 1]
                if length_ratio > 1.2 or length_ratio < 0.8:
                    # Adjust angle if there's significant difference in consecutive sides
                    current_angle += (length_ratio - 1) * 0.2

        # The polygon should automatically close (approximately)
        # due to our angle calculations

        return np.array(x), np.array(y)


def draw_irregular_polygon_with_labels(
    ax, side_lengths, side_labels, color="blue", unit=""
):
    """
    Draw an irregular polygon that scales according to the provided side lengths.

    Args:
        ax: matplotlib axis
        side_lengths: list of numeric side lengths
        side_labels: list of labels for each side
        color: color of the polygon
        unit: unit string to append to side lengths
    """
    # Convert to numeric values
    numeric_lengths = []
    for length in side_lengths:
        try:
            numeric_lengths.append(float(length))
        except (ValueError, TypeError):
            numeric_lengths.append(3.0)

    # Apply proportional scaling to make longer sides visually larger
    max_length = max(numeric_lengths)
    min_display_size = 2.0
    max_display_size = 6.0
    scale_factor = (
        (max_display_size - min_display_size) / max_length if max_length > 0 else 1.0
    )

    def get_scaled_length(length):
        return min_display_size + (length * scale_factor)

    scaled_lengths = [get_scaled_length(length) for length in numeric_lengths]

    # Construct the irregular polygon using scaled lengths
    x, y = construct_irregular_polygon(scaled_lengths)

    # Close the polygon for drawing
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])

    # Draw the polygon
    ax.plot(x_closed, y_closed, color=color, linewidth=3)
    ax.fill(x, y, color=color, alpha=0.3)

    # Calculate centroid for label positioning
    cx, cy = np.mean(x), np.mean(y)

    # Add side length labels for irregular polygon with improved positioning
    font_size = 24  # Increased for better legibility
    n_sides = len(x)

    for i in range(n_sides):
        # Get the two points of this side
        x1, y1 = x[i], y[i]
        x2, y2 = x[(i + 1) % n_sides], y[(i + 1) % n_sides]

        # Calculate midpoint of the side
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate normal vector (perpendicular to the side)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            nx, ny = -dy / length, dx / length

            # Flip normal if it points inward
            if (nx * (cx - mx) + ny * (cy - my)) > 0:
                nx, ny = -nx, -ny

            # Position label on the side line for consistent placement
            label_x = mx
            label_y = my

            # Create the label text
            label_text = side_labels[i]
            if label_text:  # Only add label if it's not empty
                if unit:
                    label_text += f" {unit}"

                # Add the label
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    ax.set_aspect("equal")
    ax.axis("off")


def apply_rotation(x, y, angle_degrees):
    """Apply rotation to x, y coordinates around the centroid."""
    if angle_degrees == 0:
        return x, y

    # Calculate centroid
    cx = np.mean(x[:-1])  # Exclude last point since it's duplicate of first
    cy = np.mean(y[:-1])

    # Translate to origin
    x_centered = x - cx
    y_centered = y - cy

    # Apply rotation
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    x_rotated = x_centered * cos_a - y_centered * sin_a
    y_rotated = x_centered * sin_a + y_centered * cos_a

    # Translate back
    x_final = x_rotated + cx
    y_final = y_rotated + cy

    return x_final, y_final


def draw_rectangle_no_indicators(
    ax, title="Rectangle", width=3, height=2, color="blue", rotation=0
):
    """Draw a rectangle without side length indicators."""
    x = np.array([0, width, width, 0, 0])
    y = np.array([0, 0, height, height, 0])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Add 90-degree angle markers (small squares) at each corner - calculated from rotated vertices
    marker_size = (
        min(width, height) * 0.1
    )  # Size of the angle marker relative to the shape

    # Get the rotated vertices for marker calculation
    corners = [(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])]

    # Draw right angle markers at each corner
    for i in range(4):
        corner_x, corner_y = corners[i]
        next_corner_x, next_corner_y = corners[(i + 1) % 4]
        prev_corner_x, prev_corner_y = corners[(i - 1) % 4]

        # Calculate direction vectors from corner to adjacent corners
        dx1 = next_corner_x - corner_x
        dy1 = next_corner_y - corner_y
        len1 = np.sqrt(dx1**2 + dy1**2)
        if len1 > 0:
            dx1, dy1 = dx1 / len1, dy1 / len1

        dx2 = prev_corner_x - corner_x
        dy2 = prev_corner_y - corner_y
        len2 = np.sqrt(dx2**2 + dy2**2)
        if len2 > 0:
            dx2, dy2 = dx2 / len2, dy2 / len2

        # Create marker square
        p1_x = corner_x + marker_size * dx1
        p1_y = corner_y + marker_size * dy1
        p2_x = corner_x + marker_size * dx2
        p2_y = corner_y + marker_size * dy2
        p3_x = p1_x + marker_size * dx2
        p3_y = p1_y + marker_size * dy2

        ax.plot(
            [corner_x, p1_x, p3_x, p2_x, corner_x],
            [corner_y, p1_y, p3_y, p2_y, corner_y],
            color=color,
            linewidth=1,
        )

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_regular_polygon_no_indicators(
    ax, title="Polygon", sides=3, radius=1, color="blue", rotation=0
):
    """Draw a regular polygon without side length indicators."""
    theta = np.linspace(0, 2 * np.pi, sides + 1)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    # Add right angle markers for regular quadrilateral (square)
    if sides == 4:
        marker_size = radius * 0.1  # Size relative to the radius

        # For each corner, add a right angle marker
        for i in range(4):
            # Get current vertex and the two adjacent vertices
            curr_x, curr_y = x[i], y[i]
            next_x, next_y = x[i + 1], y[i + 1]
            prev_x, prev_y = x[(i - 1) % 4], y[(i - 1) % 4]

            # Calculate unit vectors along the two edges from current vertex
            # Vector to next vertex
            dx1, dy1 = next_x - curr_x, next_y - curr_y
            len1 = np.sqrt(dx1**2 + dy1**2)
            if len1 > 0:
                dx1, dy1 = dx1 / len1, dy1 / len1

            # Vector to previous vertex
            dx2, dy2 = prev_x - curr_x, prev_y - curr_y
            len2 = np.sqrt(dx2**2 + dy2**2)
            if len2 > 0:
                dx2, dy2 = dx2 / len2, dy2 / len2

            # Create the right angle marker square using the same approach as rectangle
            # Move inward along both edges to create a small square
            p1_x = curr_x + marker_size * dx1
            p1_y = curr_y + marker_size * dy1
            p2_x = curr_x + marker_size * dx2
            p2_y = curr_y + marker_size * dy2
            p3_x = p1_x + marker_size * dx2  # Complete the square
            p3_y = p1_y + marker_size * dy2

            # Draw the right angle marker as a small square (same order as rectangle)
            ax.plot(
                [curr_x, p1_x, p3_x, p2_x, curr_x],
                [curr_y, p1_y, p3_y, p2_y, curr_y],
                color=color,
                linewidth=1,
            )

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_rhombus_no_indicators(
    ax, title="Rhombus", diagonal1=3, diagonal2=4, color="blue", rotation=0
):
    """Draw a rhombus without side length indicators."""
    # Calculate the vertices of the rhombus
    x = np.array([0, diagonal1 / 2, 0, -diagonal1 / 2, 0])
    y = np.array([diagonal2 / 2, 0, -diagonal2 / 2, 0, diagonal2 / 2])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_parallelogram_no_indicators(
    ax, title="Parallelogram", base=3, side=2, angle_deg=30, color="blue", rotation=0
):
    """Draw a parallelogram without side length indicators."""
    angle_rad = np.radians(angle_deg)

    # Calculate the vertices
    x = np.array(
        [0, base, base + side * np.cos(angle_rad), side * np.cos(angle_rad), 0]
    )
    y = np.array([0, 0, side * np.sin(angle_rad), side * np.sin(angle_rad), 0])

    # Apply rotation
    x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_scalene_triangle_no_indicators(
    ax, title="Scalene Triangle", sides=[3, 4, 5], color="blue", rotation=0
):
    """Draw a scalene triangle without side length indicators."""
    # Sort sides for consistent labeling
    sides = sorted(sides)
    a, b, c = sides

    # Calculate angles using cosine law
    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)

    # Ensure angles are valid
    cos_A = np.clip(cos_A, -1, 1)

    angle_A = np.arccos(cos_A)

    # Calculate vertices
    x = np.array([0, c, b * np.cos(angle_A), 0])
    y = np.array([0, 0, b * np.sin(angle_A), 0])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_isosceles_trapezoid_no_indicators(
    ax,
    title="Isosceles Trapezoid",
    top_length=2,
    base_length=3,
    height=2,
    color="blue",
    rotation=0,
):
    """Draw an isosceles trapezoid without side length indicators."""
    # Calculate the vertices
    x = np.array(
        [
            0,
            base_length,
            base_length - (base_length - top_length) / 2,
            (base_length - top_length) / 2,
            0,
        ]
    )
    y = np.array([0, 0, height, height, 0])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)

    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_kite_no_indicators(ax, title="Kite", color="blue", rotation=0):
    """Draw a kite shape without side length indicators."""
    # A kite has two pairs of adjacent sides that are equal
    # Create kite vertices positioned along symmetry axes
    center_x, center_y = 2, 1.5

    # Create kite vertices with proper symmetry
    # A kite has one axis of symmetry (vertical in this case)
    vertices = np.array(
        [
            [center_x, center_y + 1.5],  # Top vertex
            [center_x + 1.2, center_y + 0.3],  # Right vertex
            [center_x, center_y - 1.5],  # Bottom vertex
            [center_x - 1.2, center_y + 0.3],  # Left vertex (symmetric to right)
        ]
    )

    # Close the kite shape
    x = np.append(vertices[:, 0], vertices[0, 0])
    y = np.append(vertices[:, 1], vertices[0, 1])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_quadrilateral_no_indicators(
    ax, title="Quadrilateral", color="blue", rotation=0
):
    """Draw a quadrilateral without side length indicators."""
    import random

    # Define 4 different quadrilateral variations
    quadrilateral_variations = [
        # Variation 1: Original quadrilateral
        (np.array([0, 4, 3, 1, 0]), np.array([0, 0, 2, 3, 0])),
        # Variation 2: Different asymmetric quadrilateral
        (np.array([0, 3.5, 4, 0.5, 0]), np.array([0, 0, 2.5, 2, 0])),
        # Variation 3: More irregular quadrilateral
        (np.array([0, 4, 2.5, 1.5, 0]), np.array([0, 0.5, 2, 2.5, 0])),
        # Variation 4: Another different shape
        (np.array([0, 3, 4, 0.8, 0]), np.array([0, 0, 1.8, 2.8, 0])),
    ]

    # Randomly select one of the 4 variations
    x, y = random.choice(quadrilateral_variations)

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_trapezoid_no_indicators(ax, title="Trapezoid", color="blue", rotation=0):
    """Draw a trapezoid without side length indicators."""
    # Create a general trapezoid with one pair of parallel sides
    # Use the original coordinates that look like a proper trapezoid
    x = np.array([0, 4, 3, 1, 0])
    y = np.array([0, 0, 2, 2, 0])

    # Apply rotation if specified
    if rotation != 0:
        x, y = apply_rotation(x, y, rotation)

    ax.plot(x, y, color=color)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


def draw_right_trapezoid_no_indicators(
    ax, title="Right Trapezoid", color="blue", rotation=0
):
    """Draw a right trapezoid without side length indicators."""
    # Create a right trapezoid with one right angle
    # Make it more distinct with a clear right angle
    x = np.array([0, 4, 4, 1, 0])
    y = np.array([0, 0, 2, 2, 0])

    # Apply horizontal flip for right trapezoid to make it look different
    # Flip the x-coordinates horizontally around x=2 to keep within bounds
    x_flipped = 4 - x

    # Apply rotation if specified
    if rotation != 0:
        x_flipped, y = apply_rotation(x_flipped, y, rotation)

    ax.plot(x_flipped, y, color=color)
    ax.fill(x_flipped, y, color=color, alpha=0.3)
    ax.axis("equal")
    ax.title.set_text(title)
    ax.title.set_fontsize(30)
    ax.axis("off")


@stimulus_function
def draw_geometric_shapes_no_indicators(shape_list: GeometricShapeList):
    """
    Draw geometric shapes without side length indicators (dashes).
    This function is similar to draw_geometric_shapes but uses functions
    that don't include side length indicators.
    """
    num_shapes = len(shape_list)
    # Always use a single row layout
    num_cols = num_shapes
    num_rows = 1

    # Adjust size per shape based on number of shapes
    size_per_shape = 3 if num_shapes > 6 else 4
    # Calculate total width but cap it at a reasonable maximum
    total_width = min(size_per_shape * num_cols, 30)
    # Adjust height to maintain aspect ratio
    height = size_per_shape

    _, axs = plt.subplots(num_rows, num_cols, figsize=(total_width, height))
    axs = np.array(axs).flatten()  # Flatten to easily index

    shape_map = {
        ValidGeometricShape.RECTANGLE: lambda i, item: draw_rectangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
        ),
        ValidGeometricShape.RHOMBUS: lambda i, item: draw_rhombus_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
        ),
        ValidGeometricShape.PARALLELOGRAM: lambda i,
        item: draw_parallelogram_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=random.randint(0, 360),  # Only parallelograms get random rotation
        ),
        ValidGeometricShape.CIRCLE: lambda i, item: draw_circle(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.REGULAR_QUADRILATERAL: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=4,
            color=item.color,
        ),
        ValidGeometricShape.SCALENE_TRIANGLE: lambda i,
        item: draw_scalene_triangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[5, 7, 10],
            color=item.color,
        ),
        ValidGeometricShape.TRAPEZOID: lambda i, item: draw_trapezoid_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.ISOSCELES_TRAPEZOID: lambda i,
        item: draw_isosceles_trapezoid_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
        ),
        ValidGeometricShape.RIGHT_TRAPEZOID: lambda i,
        item: draw_right_trapezoid_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.QUADRILATERAL: lambda i,
        item: draw_quadrilateral_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.KITE: lambda i, item: draw_kite_no_indicators(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.TRIANGLE: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
        ),
        ValidGeometricShape.SQUARE: lambda i, item: draw_rectangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            width=2,
            height=2,
            color=item.color,
        ),
        ValidGeometricShape.PENTAGON: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=5,
            color=item.color,
        ),
        ValidGeometricShape.REGULAR_PENTAGON: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=5,
            color=item.color,
        ),
        ValidGeometricShape.HEXAGON: lambda i, item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=6,
            color=item.color,
        ),
        ValidGeometricShape.REGULAR_HEXAGON: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=6,
            color=item.color,
        ),
        ValidGeometricShape.HEPTAGON: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=7,
            color=item.color,
        ),
        ValidGeometricShape.REGULAR_HEPTAGON: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=7,
            color=item.color,
        ),
        ValidGeometricShape.OCTAGON: lambda i, item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=8,
            color=item.color,
        ),
        ValidGeometricShape.REGULAR_OCTAGON: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=8,
            color=item.color,
        ),
        ValidGeometricShape.RIGHT_TRIANGLE: lambda i,
        item: draw_scalene_triangle_no_indicators(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 5],
            color=item.color,
        ),
        ValidGeometricShape.ISOSCELES_TRIANGLE: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
        ),
        ValidGeometricShape.EQUILATERAL_TRIANGLE: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
        ),
        ValidGeometricShape.REGULAR_TRIANGLE: lambda i,
        item: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
        ),
        ValidGeometricShape.OBTUSE_TRIANGLE: lambda i,
        item: draw_scalene_triangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[2, 3, 4],
            color=item.color,
        ),
        ValidGeometricShape.ACUTE_TRIANGLE: lambda i,
        item: draw_scalene_triangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[3, 4, 5],
            color=item.color,
        ),
    }

    # Draw each shape
    for i, item in enumerate(shape_list):
        if item.shape in shape_map:
            shape_map[item.shape](i, item)
        else:
            # Default to regular polygon for unknown shapes
            draw_regular_polygon_no_indicators(
                axs[i], "" if num_shapes == 1 else item.label, sides=3, color=item.color
            )

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_basic_no_indicators_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_geometric_shapes_with_rotation(shape_list: GeometricShapeListWithRotation):
    """
    Draw geometric shapes with rotation support.
    This function provides rotation-enabled versions of geometric shapes.
    Note: Uses the no-indicators drawing functions with rotation support.
    """
    num_shapes = len(shape_list.shapes)
    # Always use a single row layout
    num_cols = num_shapes
    num_rows = 1

    # Adjust size per shape based on number of shapes
    size_per_shape = 3 if num_shapes > 6 else 4
    # Calculate total width but cap it at a reasonable maximum
    total_width = min(size_per_shape * num_cols, 30)
    # Adjust height to maintain aspect ratio
    height = size_per_shape

    _, axs = plt.subplots(num_rows, num_cols, figsize=(total_width, height))
    axs = np.array(axs).flatten()  # Flatten to easily index

    shape_map = {
        ValidGeometricShape.RECTANGLE: lambda i, item, rotation: draw_rectangle(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.RHOMBUS: lambda i, item, rotation: draw_rhombus(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.PARALLELOGRAM: lambda i, item, rotation: draw_parallelogram(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.CIRCLE: lambda i, item, rotation: draw_circle(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.REGULAR_QUADRILATERAL: lambda i,
        item,
        rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=4,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.SCALENE_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[5, 7, 10],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.TRAPEZOID: lambda i,
        item,
        rotation: draw_trapezoid_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.ISOSCELES_TRAPEZOID: lambda i,
        item,
        rotation: draw_isosceles_trapezoid(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.RIGHT_TRAPEZOID: lambda i,
        item,
        rotation: draw_right_trapezoid(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.QUADRILATERAL: lambda i,
        item,
        rotation: draw_quadrilateral_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.KITE: lambda i, item, rotation: draw_kite_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.TRIANGLE: lambda i, item, rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.SQUARE: lambda i, item, rotation: draw_rectangle(
            axs[i],
            "" if num_shapes == 1 else item.label,
            width=2,
            height=2,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.PENTAGON: lambda i, item, rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=5,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_PENTAGON: lambda i,
        item,
        rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=5,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.HEXAGON: lambda i, item, rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=6,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_HEXAGON: lambda i,
        item,
        rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=6,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.HEPTAGON: lambda i, item, rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=7,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_HEPTAGON: lambda i,
        item,
        rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=7,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.OCTAGON: lambda i, item, rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=8,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_OCTAGON: lambda i,
        item,
        rotation: draw_regular_polygon(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=8,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.RIGHT_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 5],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.ISOSCELES_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[2, 4, 4],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_TRIANGLE: lambda i,
        item,
        rotation: draw_regular_polygon(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.EQUILATERAL_TRIANGLE: lambda i,
        item,
        rotation: draw_regular_polygon(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=3,
            radius=2,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.OBTUSE_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 6],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.ACUTE_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 4],
            color=item.color,
            rotation=rotation,
        ),
    }

    for ax in axs[num_shapes:]:
        ax.remove()

    for i, item in enumerate(shape_list.shapes):
        # Determine rotation based on shape_list settings
        if shape_list.rotate:
            # Generate random rotation (0 to 360 degrees)
            shape_rotation = random.randint(0, 360)
        else:
            # No rotation
            shape_rotation = 0

        if item.shape in shape_map:
            shape_map[item.shape](i, item, shape_rotation)
        else:
            raise ValueError(f"Shape Type {item.shape.value} is not supported")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_basic_with_rotation_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_geometric_shapes_no_indicators_with_rotation(
    shape_list: GeometricShapeListWithRotation,
):
    """
    Draw geometric shapes without side length indicators (dashes) with rotation support.
    This function extends draw_geometric_shapes_no_indicators with the ability to apply
    random rotation to all shapes when specified in the shape list.
    """
    num_shapes = len(shape_list.shapes)
    # Always use a single row layout
    num_cols = num_shapes
    num_rows = 1

    # Adjust size per shape based on number of shapes
    size_per_shape = 3 if num_shapes > 6 else 4
    # Calculate total width but cap it at a reasonable maximum
    total_width = min(size_per_shape * num_cols, 30)
    # Adjust height to maintain aspect ratio
    height = size_per_shape

    _, axs = plt.subplots(num_rows, num_cols, figsize=(total_width, height))
    axs = np.array(axs).flatten()  # Flatten to easily index

    shape_map = {
        ValidGeometricShape.RECTANGLE: lambda i,
        item,
        rotation: draw_rectangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.RHOMBUS: lambda i,
        item,
        rotation: draw_rhombus_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.PARALLELOGRAM: lambda i,
        item,
        rotation: draw_parallelogram_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.CIRCLE: lambda i, item, rotation: draw_circle(
            axs[i], "" if num_shapes == 1 else item.label, color=item.color
        ),
        ValidGeometricShape.REGULAR_QUADRILATERAL: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=4,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.SCALENE_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[5, 7, 10],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.TRAPEZOID: lambda i,
        item,
        rotation: draw_trapezoid_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.ISOSCELES_TRAPEZOID: lambda i,
        item,
        rotation: draw_isosceles_trapezoid_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.RIGHT_TRAPEZOID: lambda i,
        item,
        rotation: draw_right_trapezoid_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.QUADRILATERAL: lambda i,
        item,
        rotation: draw_quadrilateral_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.KITE: lambda i, item, rotation: draw_kite_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.TRIANGLE: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.SQUARE: lambda i,
        item,
        rotation: draw_rectangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            width=2,
            height=2,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.PENTAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=5,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_PENTAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=5,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.HEXAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=6,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_HEXAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=6,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.HEPTAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=7,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_HEPTAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=7,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.OCTAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=8,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_OCTAGON: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=8,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.RIGHT_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle_no_indicators(
            ax=axs[i],
            title="" if num_shapes == 1 else item.label,
            sides=[3, 4, 5],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.ISOSCELES_TRIANGLE: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.EQUILATERAL_TRIANGLE: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.REGULAR_TRIANGLE: lambda i,
        item,
        rotation: draw_regular_polygon_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=3,
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.OBTUSE_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[2, 3, 4],
            color=item.color,
            rotation=rotation,
        ),
        ValidGeometricShape.ACUTE_TRIANGLE: lambda i,
        item,
        rotation: draw_scalene_triangle_no_indicators(
            axs[i],
            "" if num_shapes == 1 else item.label,
            sides=[3, 4, 5],
            color=item.color,
            rotation=rotation,
        ),
    }

    # Draw each shape
    for i, item in enumerate(shape_list.shapes):
        # Determine rotation based on shape_list settings
        if shape_list.rotate:
            # Generate random rotation (0 to 360 degrees)
            shape_rotation = random.randint(0, 360)
        else:
            # No rotation
            shape_rotation = 0

        if item.shape in shape_map:
            shape_map[item.shape](i, item, shape_rotation)
        else:
            # Default to regular polygon for unknown shapes
            draw_regular_polygon_no_indicators(
                axs[i],
                "" if num_shapes == 1 else item.label,
                sides=3,
                color=item.color,
                rotation=shape_rotation,
            )

    plt.tight_layout()

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_basic_no_indicators_with_rotation_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


###################################################
# Multiple Grids for Comparison                   #
###################################################
@stimulus_function
def generate_multiple_grids(data: MultipleGrids):
    """
    Generate multiple unit square grids in the same picture for comparison.
    Supports up to 5 grids, each with different dimensions.
    Useful for teaching area comparison, perimeter comparison, and unit understanding.
    """
    grids = data.grids
    num_grids = len(grids)

    if num_grids == 0:
        raise ValueError("At least one grid must be provided")

    # Calculate figure size based on number of grids
    fig_width = max(4, num_grids * 3)  # Minimum 4, then 3 per grid
    fig_height = 6  # Fixed height for consistent layout

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate spacing and positioning
    spacing = 1.0  # Space between grids
    start_x = 0

    # Draw each grid
    for i, grid in enumerate(grids):
        length = grid.length
        width = grid.width

        # Calculate grid position
        grid_x = start_x
        grid_y = 0

        # Draw unit squares for this grid
        # Determine irregularity based on the irregularity setting
        if data.irregularity == "all_regular":
            make_irregular = False
        elif data.irregularity == "all_irregular":
            make_irregular = True
        elif data.irregularity == "mixed":
            make_irregular = random.choice([True, False])
        else:
            make_irregular = True  # Default fallback

        if make_irregular:
            # Create truly irregular grid by removing squares while maintaining connectivity
            print(f"Grid {i+1} is IRREGULAR - creating irregular shape")

            # Ensure minimum 2 rows and 2 columns for irregularity
            if width < 2 or length < 2:
                print(f"  Grid too small ({width}×{length}), skipping irregularity")
                make_irregular = False
                removed_positions = set()
            else:
                # Check if we have target units for precise control
                if (
                    hasattr(data, "target_units")
                    and data.target_units is not None
                    and i < len(data.target_units)
                ):
                    target = data.target_units[i]
                    # Calculate exact squares to remove to reach target
                    squares_to_remove = (width * length) - target
                    print(
                        f"  TARGET: {target} units, removing exactly {squares_to_remove} squares from {width}×{length} grid"
                    )
                else:
                    # Fall back to random removal (10-25%) to maintain connectivity
                    squares_to_remove = random.randint(
                        max(1, (width * length) // 10), (width * length) // 4
                    )
                    print(
                        f"  No target specified, removing {squares_to_remove} squares from {width}×{length} grid"
                    )

                removed_positions = set()

                # For small grids, ensure we remove enough squares to be visible
                if width * length <= 10:
                    squares_to_remove = max(
                        squares_to_remove, 1
                    )  # At least 1 square removed

                # Safety check: don't remove more squares than possible while maintaining connectivity
                max_removable = (width * length) - max(
                    width, length
                )  # Keep at least 1 square per row/column
                squares_to_remove = min(squares_to_remove, max_removable)

                if squares_to_remove <= 0:
                    print(
                        "  Cannot remove squares while maintaining connectivity, keeping grid regular"
                    )
                    make_irregular = False
                    removed_positions = set()
                else:
                    print(
                        f"  Will remove {squares_to_remove} squares (max safe: {max_removable})"
                    )

                    # Helper function to check if remaining squares are connected using DFS
                    def is_connected(removed_set):
                        # Find all remaining squares
                        remaining_squares = []
                        for x in range(width):
                            for y in range(length):
                                if (x, y) not in removed_set:
                                    remaining_squares.append((x, y))

                        if not remaining_squares:
                            return True  # Empty set is considered connected

                        # Start DFS from the first remaining square
                        start = remaining_squares[0]
                        visited = set()
                        stack = [start]

                        while stack:
                            current = stack.pop()
                            if current in visited:
                                continue
                            visited.add(current)

                            # Check all 4 adjacent positions (up, down, left, right)
                            x, y = current
                            neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

                            for nx, ny in neighbors:
                                # Check if neighbor is within bounds and not removed
                                if (
                                    0 <= nx < width
                                    and 0 <= ny < length
                                    and (nx, ny) not in removed_set
                                    and (nx, ny) not in visited
                                ):
                                    stack.append((nx, ny))

                        # Check if all remaining squares were visited
                        return len(visited) == len(remaining_squares)

                    # Remove squares while ensuring connectivity
                    for _ in range(squares_to_remove):
                        # Find all positions that could be removed while maintaining connectivity
                        valid_positions = []

                        for x in range(width):
                            for y in range(length):
                                if (x, y) not in removed_positions:
                                    # Try removing this square temporarily
                                    test_removed = removed_positions | {(x, y)}

                                    # Check if the remaining squares would still be connected
                                    if is_connected(test_removed):
                                        valid_positions.append((x, y))

                        if valid_positions:
                            # Choose a random valid position that maintains connectivity
                            pos = random.choice(valid_positions)
                            removed_positions.add(pos)

                            print(
                                f"    Removed square at {pos}, remaining squares still connected"
                            )
                        else:
                            print(
                                "    No more valid positions for removal while maintaining connectivity, stopping early"
                            )
                            break

                    print(
                        f"  Actually removed {len(removed_positions)} squares: {sorted(removed_positions)}"
                    )

                    # Final connectivity verification
                    if not is_connected(removed_positions):
                        print(
                            "  WARNING: Final shape is not connected! This should not happen."
                        )
                    else:
                        print("  ✓ Final shape connectivity verified")
        else:
            # Regular grid - no squares removed
            removed_positions = set()
            print(f"Grid {i+1} is REGULAR (complete grid)")
        base_color = (
            random.uniform(0.4, 0.9),
            random.uniform(0.4, 0.9),
            random.uniform(0.4, 0.9),
        )

        # Create darker edge color by reducing brightness by about 40%
        edge_color = (base_color[0] * 0.6, base_color[1] * 0.6, base_color[2] * 0.6)

        for x in range(width):
            for y in range(length):
                # Skip removed squares for irregular grids
                if (x, y) not in removed_positions:
                    square = patches.Rectangle(
                        (grid_x + x, grid_y + y),
                        1,
                        1,
                        linewidth=3,  # Bolder lines for better visibility
                        edgecolor=edge_color,  # Darker shade of the same color
                        facecolor=base_color,  # Lighter base color for all squares in this grid
                    )
                    ax.add_patch(square)
                else:
                    print(f"    Skipping square at ({x}, {y}) - it was removed")

        # Add per-grid label (LLM-controlled; avoid revealing counts)
        # Only add labels when there are multiple grids
        if num_grids > 1:
            # Default to letters if label missing: Figure A, Figure B, ...
            default_letter = chr(ord("A") + i)
            label_text = getattr(grid, "label", None) or f"Figure {default_letter}"
            ax.text(
                grid_x + width / 2,
                grid_y - 0.5,
                label_text,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
            )

        # Update starting position for next grid
        start_x += width + spacing

    # Set plot limits and properties
    ax.set_xlim(-0.5, start_x - spacing + 0.5)
    ax.set_ylim(-1.5, max(grid.length for grid in grids) + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/multiple_grids_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


def generate_rhombus_coordinates(side_labels=None):
    """Generate rhombus coordinates with proportional scaling based on side labels."""
    # Extract numeric side length from labels if provided
    if side_labels and len(side_labels) > 0:
        try:
            # Parse the first label to get side length (all sides equal in rhombus)
            side_length_str = side_labels[0].strip()
            if (
                side_length_str
                and side_length_str.replace(".", "").replace("-", "").isdigit()
            ):
                base_side_length = float(side_length_str)
                # Scale to a reasonable display size (target max dimension around 4-5 units)
                scale_factor = min(4.0 / base_side_length, 1.0)
                side_length = base_side_length * scale_factor
            else:
                # Fallback for non-numeric labels like 'x'
                side_length = random.uniform(2.5, 4.0)
        except (ValueError, IndexError):
            side_length = random.uniform(2.5, 4.0)
    else:
        side_length = random.uniform(2.5, 4.0)

    # Decide if this should look like a square (50% chance)
    is_square_like = random.random() < 0.5

    if is_square_like:
        # Make it a square or nearly square
        return np.array(
            [
                [0, 0],
                [side_length, 0],
                [side_length, side_length],
                [0, side_length],
                [0, 0],  # Close the shape
            ]
        )
    else:
        # Generate a proper rhombus with random angles
        # Angle between 30 and 150 degrees (avoiding very flat rhombuses)
        angle = random.uniform(30, 150)
        angle_rad = np.radians(angle)

        # Calculate the second vertex position
        dx = side_length * np.cos(angle_rad)
        dy = side_length * np.sin(angle_rad)

        return np.array(
            [
                [0, 0],
                [side_length, 0],
                [side_length + dx, dy],
                [dx, dy],
                [0, 0],  # Close the shape
            ]
        )


def generate_parallelogram_with_height_coordinates():
    """Generate coordinates for a parallelogram with horizontal base and slanted sides."""
    # Base points (vary the proportions)
    base_width = random.uniform(3, 5)  # More variety in width
    base_height = 0
    x_left = 0
    x_right = base_width

    # Calculate top points with slant
    height = random.uniform(1.5, 2.5)  # More variety in height
    slant_angle = random.uniform(20, 45)  # Increased angle range for more variety
    horizontal_shift = height * math.tan(math.radians(slant_angle))

    # Randomly choose direction (left or right slant)
    if random.choice([True, False]):
        horizontal_shift = -horizontal_shift  # Left slant

    # Create coordinates array
    coords = np.array(
        [
            [x_left, base_height],  # Bottom left
            [x_right, base_height],  # Bottom right
            [x_right + horizontal_shift, height],  # Top right
            [x_left + horizontal_shift, height],  # Top left
            [x_left, base_height],  # Close the shape
        ]
    )
    return coords


def generate_parallelogram_coordinates(side_labels=None):
    """Generate parallelogram coordinates with proportional scaling based on side labels."""
    # Extract side lengths from labels if provided
    if side_labels and len(side_labels) >= 4:
        try:
            # Parse side labels - in parallelogram, opposite sides should be equal
            # side_labels[0] and side_labels[2] should be equal (base sides)
            # side_labels[1] and side_labels[3] should be equal (slant sides)
            base_str = side_labels[0].strip()
            slant_str = side_labels[1].strip()

            if (
                base_str
                and base_str.replace(".", "").replace("-", "").isdigit()
                and slant_str
                and slant_str.replace(".", "").replace("-", "").isdigit()
            ):
                base_length = float(base_str)
                slant_length = float(slant_str)

                # Scale to reasonable display size
                max_dim = max(base_length, slant_length)
                scale_factor = min(4.0 / max_dim, 1.0)
                base = base_length * scale_factor
                target_slant = slant_length * scale_factor

                # Calculate height from target slant length
                # We'll use a random angle but ensure the slant matches
                angle = random.uniform(30, 150)
                angle_rad = np.radians(angle)
                height = target_slant * np.sin(angle_rad)

            else:
                # Fallback for non-numeric labels
                base = random.uniform(3.0, 4.5)
                height = random.uniform(2.0, 3.5)
        except (ValueError, IndexError):
            base = random.uniform(3.0, 4.5)
            height = random.uniform(2.0, 3.5)
    else:
        # Default random dimensions
        base = random.uniform(3.0, 4.5)
        height = random.uniform(2.0, 3.5)

    # Random skew angle (30 to 150 degrees)
    skew_angle = random.uniform(30, 150)
    skew_rad = np.radians(skew_angle)

    # Calculate skew offset to ensure opposite sides are equal
    skew_x = height / np.tan(skew_rad)

    # Construct parallelogram with guaranteed equal opposite sides
    # Bottom side: [0,0] to [base, 0] (length = base)
    # Right side: [base,0] to [base + skew_x, height] (length = sqrt(skew_x^2 + height^2))
    # Top side: [base + skew_x, height] to [skew_x, height] (length = base, same as bottom)
    # Left side: [skew_x, height] to [0, 0] (length = sqrt(skew_x^2 + height^2), same as right)

    return np.array(
        [
            [0, 0],
            [base, 0],
            [base + skew_x, height],
            [skew_x, height],
            [0, 0],  # Close the shape
        ]
    )


def generate_irregular_quadrilateral_coordinates(side_labels=None):
    """Generate irregular quadrilateral coordinates with proportional scaling based on side labels."""
    # Extract side lengths from labels if provided
    if side_labels and len(side_labels) >= 4:
        try:
            # Parse side labels to get actual numeric values
            numeric_labels = []
            for label in side_labels:
                label_str = label.strip()
                if label_str and label_str.replace(".", "").replace("-", "").isdigit():
                    numeric_labels.append(float(label_str))
                else:
                    # For non-numeric labels, use a default value
                    numeric_labels.append(random.uniform(3.0, 8.0))

            if len(numeric_labels) >= 4:
                # Use the actual side lengths but scale them to reasonable display size
                max_dim = max(numeric_labels)
                scale_factor = min(4.5 / max_dim, 1.0)
                scaled_side_lengths = [
                    length * scale_factor for length in numeric_labels
                ]

                # Use the proper geometric construction that respects side lengths
                x_coords, y_coords = construct_irregular_polygon(scaled_side_lengths)

                # Convert to the format expected by the rest of the code
                points = []
                for i in range(len(x_coords)):
                    points.append([x_coords[i], y_coords[i]])

                # Close the shape
                points.append(points[0])

                return np.array(points)
            else:
                # Fallback for insufficient numeric labels
                return _generate_default_irregular_quadrilateral()
        except (ValueError, IndexError, Exception):
            return _generate_default_irregular_quadrilateral()
    else:
        # Default random dimensions when no labels provided
        return _generate_default_irregular_quadrilateral()


def _generate_default_irregular_quadrilateral():
    """Generate a default irregular quadrilateral with random dimensions."""
    # Generate random side lengths
    side_lengths = [
        random.uniform(3.0, 5.0),
        random.uniform(2.5, 4.5),
        random.uniform(3.5, 5.5),
        random.uniform(2.0, 4.0),
    ]

    # Use the proper geometric construction
    x_coords, y_coords = construct_irregular_polygon(side_lengths)

    # Convert to the format expected by the rest of the code
    points = []
    for i in range(len(x_coords)):
        points.append([x_coords[i], y_coords[i]])

    # Close the shape
    points.append(points[0])

    return np.array(points)


def draw_single_quadrilateral(
    ax,
    shape_type,
    side_labels,
    show_ticks,
    figure_label=None,
    rotation=False,
    label_fontsize=8,
):
    """Draw a single quadrilateral with specified properties.

    Args:
        rotation: If True, applies random rotation. If False, uses standard orientation.
    """

    # Select random color for the shape
    import random

    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray"]
    color = random.choice(colors)

    # Generate coordinates based on shape type and side labels for proportional scaling
    if shape_type == QuadrilateralShapeType.RHOMBUS:
        coords = generate_rhombus_coordinates(side_labels)
    elif shape_type == QuadrilateralShapeType.PARALLELOGRAM:
        coords = generate_parallelogram_coordinates(side_labels)
    else:  # IRREGULAR_QUADRILATERAL
        coords = generate_irregular_quadrilateral_coordinates(side_labels)

    # Apply random rotation if enabled
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    if rotation:
        rotation_angle = random.uniform(0, 360)
        x_rotated, y_rotated = apply_rotation(x_coords, y_coords, rotation_angle)
    else:
        x_rotated, y_rotated = x_coords, y_coords

    # Draw the shape
    ax.plot(x_rotated, y_rotated, color=color, linewidth=2)
    ax.fill(x_rotated, y_rotated, color=color, alpha=0.3)

    # Add figure label if provided (positioned below the shape)
    if figure_label:
        # Calculate horizontal center and bottom position for label placement
        cx = np.mean(x_rotated[:-1])  # Exclude last point since it's duplicate of first
        min_y = np.min(y_rotated[:-1])  # Find bottom of the shape
        label_y = min_y - 0.5  # Position label below the shape with some margin
        ax.text(
            cx,
            label_y,
            figure_label,
            ha="center",
            va="top",  # Top of text aligns with the y position (below shape)
            fontsize=8,
            fontweight="bold",
        )

    # Add side labels or tick marks
    for i in range(4):  # 4 sides
        x1, y1 = x_rotated[i], y_rotated[i]
        x2, y2 = x_rotated[i + 1], y_rotated[i + 1]

        if show_ticks:
            # Draw red tick marks using the existing function
            _draw_side_dash(
                ax, x1, y1, x2, y2, dash_length=0.3, dash_color="red", dash_thickness=2
            )
        else:
            # Add side labels
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2

            # Calculate normal vector for label positioning
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Normal vector pointing outward
                nx, ny = -dy / length, dx / length

                # Check if normal points inward and flip if needed
                cx = np.mean(x_rotated[:-1])
                cy = np.mean(y_rotated[:-1])
                if (nx * (cx - mx) + ny * (cy - my)) > 0:
                    nx, ny = -nx, -ny

                # Offset for label positioning
                offset = 0.9
                label_x = mx + nx * offset
                label_y = my + ny * offset

                ax.text(
                    label_x,
                    label_y,
                    side_labels[i],
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                )


@stimulus_function
def draw_single_quadrilateral_stimulus(data: QuadrilateralFigures):
    """
    Draw a single quadrilateral figure with small, optimized dimensions.

    This function is specifically for single quadrilateral figures with:
    - Small file size (240×240px at 300 DPI)
    - Optimized spacing for web display
    - No figure labels (since it's just one figure)

    Args:
        data: QuadrilateralFigures with exactly one shape
    """
    if len(data.shape_types) != 1:
        raise ValueError(
            "draw_single_quadrilateral_stimulus requires exactly one shape"
        )

    # Create small figure optimized for single quadrilateral
    fig, ax = plt.subplots(1, 1, figsize=(0.8, 0.8))  # 0.8" × 300 DPI = 240px

    # Draw the single quadrilateral (color will be randomly selected internally)
    draw_single_quadrilateral(
        ax,
        data.shape_types[0],
        data.side_labels[0],
        data.show_ticks,
        figure_label=None,  # No figure label needed for single figure
        rotation=data.rotation,
        label_fontsize=6,
    )

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.1)  # Standard padding

    # Tight layout for clean appearance
    plt.tight_layout(pad=0.1)

    # Save with optimized settings for small size
    file_name = f"{settings.additional_content_settings.image_destination_folder}/single_quadrilateral_{int(time.time() * 1000)}.{settings.additional_content_settings.stimulus_image_format}"
    fig.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=300,  # Good quality for small images
    )
    plt.close()

    return file_name


@stimulus_function
def draw_quadrilateral_figures(data: QuadrilateralFigures):
    """
    Draw 2-4 quadrilateral figures with configurable properties.

    Note: For single quadrilaterals, use draw_single_quadrilateral_stimulus instead.

    Features:
    - Support for rhombus, parallelogram, and irregular quadrilateral shapes
    - Optional side labels or tick marks
    - Configurable rotation: random rotation when enabled, standard orientation when disabled
    - Random colors for each shape
    - Automatic figure labeling for multiple shapes
    - Rhombus shapes sometimes appear square-like for variety
    """

    num_shapes = len(data.shape_types)

    # Require multiple shapes for this function
    if num_shapes == 1:
        raise ValueError(
            "draw_quadrilateral_figures requires 2-4 shapes. Use draw_single_quadrilateral_stimulus for single shapes."
        )

    # Set up the figure layout for multiple shapes - use larger dimensions
    # Create 2-row grid: top row for labels, bottom row for shapes
    if num_shapes == 2:
        fig, axes_grid = plt.subplots(
            2, 2, figsize=(3, 2.5)
        )  # Larger for better visibility
    elif num_shapes == 3:
        fig, axes_grid = plt.subplots(2, 3, figsize=(4.5, 2.5))
    else:  # 4 shapes
        fig, axes_grid = plt.subplots(2, 4, figsize=(6, 2.5))

    # Extract bottom row for shapes (axes_grid[1, :])
    axes = axes_grid[1, :]
    # Extract top row for labels (axes_grid[0, :])
    label_axes = axes_grid[0, :]

    # Draw each shape and add labels to top row
    for i, (shape_type, side_labels) in enumerate(
        zip(data.shape_types, data.side_labels)
    ):
        # Draw the shape in the bottom row (color will be randomly selected internally)
        draw_single_quadrilateral(
            axes[i],
            shape_type,
            side_labels,
            data.show_ticks,
            label_fontsize=12,
            figure_label=None,
            rotation=data.rotation,
        )

        # Set axis properties for shape axes
        axes[i].set_aspect("equal")
        axes[i].axis("off")

        # Add normal padding around the shape (no extra bottom margin needed)
        axes[i].margins(0.1)

        # Add figure labels to top row (always multiple shapes in this function)
        figure_label = f"Figure {i + 1}"

        # Configure label axes
        label_axes[i].set_xlim(0, 1)
        label_axes[i].set_ylim(0, 1)
        label_axes[i].axis("off")

        # Add label text centered in the top row
        label_axes[i].text(
            0.5,
            0.5,  # Center of the label axes
            figure_label,
            fontsize=12,  # Larger font size for better visibility in larger figures
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
            fontweight="bold",
            transform=label_axes[i].transAxes,
        )

    # Adjust layout with proper spacing between subplots and room for labels
    plt.tight_layout(pad=0.1, w_pad=0.2)  # More horizontal spacing between figures
    plt.subplots_adjust(bottom=0.2)  # Extra space at bottom for figure labels

    # Save the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/quadrilateral_figures_{int(time.time() * 1000)}.{settings.additional_content_settings.stimulus_image_format}"
    fig.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=500,
    )
    plt.close()

    return file_name


def generate_random_shape_with_right_angles(num_right_angles):
    """
    Generate a polygon with the specified number of interior right angles.
    Returns vertices (x, y arrays) and indices of vertices with right angles.

    Supported shapes (interior right angles only):
    - 0: Circle, equilateral triangle, pentagon, hexagon, or quadrilateral (no right angles)
    - 1: Right triangle or right angle kite shape
    - 2: Pentagon with 2 right angles at base or right trapezoid
    - 3: File icon shape (triangle + rectangle)
    - 4: Rectangle or square
    """
    import math
    import random

    if num_right_angles == 0:
        # Shapes with no right angles
        shapes = [
            "circle",
            "equilateral_triangle",
            "pentagon",
            "hexagon",
            "quadrilateral",
        ]
        shape_type = random.choice(shapes)

        if shape_type == "circle":
            # Circle - return special marker for circle
            return ("circle", None, [])
        elif shape_type == "equilateral_triangle":
            # Equilateral triangle
            side = 4
            height = side * math.sqrt(3) / 2
            x = np.array([0, side, side / 2, 0])
            y = np.array([0, 0, height, 0])
            right_angle_indices = []
        elif shape_type == "pentagon":
            # Regular pentagon
            angles = np.linspace(0, 2 * math.pi, 6)  # 6 points to close the shape
            radius = 2
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            right_angle_indices = []
        elif shape_type == "hexagon":
            # Regular hexagon
            angles = np.linspace(0, 2 * math.pi, 7)  # 7 points to close the shape
            radius = 2
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            right_angle_indices = []
        else:  # quadrilateral
            # Use existing quadrilateral variations (no right angles)
            quadrilateral_variations = [
                # Variation 1: Original quadrilateral
                (np.array([0, 4, 3, 1, 0]), np.array([0, 0, 2, 3, 0])),
                # Variation 2: Different asymmetric quadrilateral
                (np.array([0, 3.5, 4, 0.5, 0]), np.array([0, 0, 2.5, 2, 0])),
                # Variation 3: More irregular quadrilateral
                (np.array([0, 4, 2.5, 1.5, 0]), np.array([0, 0.5, 2, 2.5, 0])),
                # Variation 4: Another different shape
                (np.array([0, 3, 4, 0.8, 0]), np.array([0, 0, 1.8, 2.8, 0])),
            ]
            # Randomly select one of the 4 variations
            x, y = random.choice(quadrilateral_variations)
            right_angle_indices = []

    elif num_right_angles == 1:
        # Random choice between right triangle and right angle kite
        shapes = ["right_triangle", "right_kite"]
        shape_type = random.choice(shapes)

        if shape_type == "right_triangle":
            # Right triangle (3-4-5 triangle)
            x = np.array([0, 4, 0, 0])
            y = np.array([0, 0, 3, 0])
            right_angle_indices = [0]  # Right angle at origin
        else:  # right_kite
            # Kite-like shape with one clear right angle
            x = np.array([0, 3, 4, 0, 0])
            y = np.array([0, 0, 2, 2, 0])
            right_angle_indices = [
                0
            ]  # Right angle at bottom-left vertex (horizontal to vertical)

    elif num_right_angles == 2:
        # Random choice between pentagon and right trapezoid
        shapes = ["pentagon", "right_trapezoid"]
        shape_type = random.choice(shapes)

        if shape_type == "pentagon":
            # Pentagon with 2 right angles at base
            x = np.array([0, 4, 4, 2, 0, 0])
            y = np.array([0, 0, 2, 3, 2, 0])
            right_angle_indices = [0, 1]  # Right angles at base corners
        else:  # right_trapezoid
            # Right trapezoid with 2 right angles
            x = np.array([0, 4, 3, 0, 0])
            y = np.array([0, 0, 3, 3, 0])
            right_angle_indices = [0, 3]  # Right angles at left corners

    elif num_right_angles == 3:
        # File icon shape: triangle at top-left corner of rectangle
        # Rectangle: (0,0) to (4,3), triangle: (0,2) to (1,3) to (0,3)
        x = np.array([0, 4, 4, 1, 0, 0])
        y = np.array([0, 0, 2, 3, 3, 0])
        right_angle_indices = [
            0,
            1,
            4,
        ]  # Right angles at rectangle corners + triangle corner

    elif num_right_angles == 4:
        # Rectangle or square (randomly choose)
        if random.choice([True, False]):
            # Rectangle
            x = np.array([0, 5, 5, 0, 0])
            y = np.array([0, 0, 3, 3, 0])
        else:
            # Square
            x = np.array([0, 3, 3, 0, 0])
            y = np.array([0, 0, 3, 3, 0])
        right_angle_indices = [0, 1, 2, 3]  # All four corners

    else:
        # Default to right triangle for unsupported counts
        x = np.array([0, 4, 0, 0])
        y = np.array([0, 0, 3, 0])
        right_angle_indices = [0]

    return x, y, right_angle_indices


def draw_right_angle_marker_at_vertex(
    ax, x, y, vertex_idx, marker_size=0.2, color="red"
):
    """
    Draw a right angle marker (small square) at the specified vertex.
    """
    if vertex_idx >= len(x) - 1:  # Skip if vertex index is out of bounds
        return

    # Get current vertex and adjacent vertices (handle wrapping for closed polygons)
    curr_x, curr_y = x[vertex_idx], y[vertex_idx]
    next_idx = (vertex_idx + 1) % (len(x) - 1)  # Skip the duplicate closing vertex
    prev_idx = (vertex_idx - 1) % (len(x) - 1)
    next_x, next_y = x[next_idx], y[next_idx]
    prev_x, prev_y = x[prev_idx], y[prev_idx]

    # Calculate direction vectors from current vertex to adjacent vertices
    dx1 = next_x - curr_x
    dy1 = next_y - curr_y
    len1 = np.sqrt(dx1**2 + dy1**2)
    if len1 > 0:
        dx1, dy1 = dx1 / len1, dy1 / len1

    dx2 = prev_x - curr_x
    dy2 = prev_y - curr_y
    len2 = np.sqrt(dx2**2 + dy2**2)
    if len2 > 0:
        dx2, dy2 = dx2 / len2, dy2 / len2

    # Create right angle marker square with better positioning
    p1_x = curr_x + marker_size * dx1
    p1_y = curr_y + marker_size * dy1
    p2_x = curr_x + marker_size * dx2
    p2_y = curr_y + marker_size * dy2
    p3_x = p1_x + marker_size * dx2
    p3_y = p1_y + marker_size * dy2

    # Draw the right angle marker (outline only, no fill)
    ax.plot(
        [curr_x, p1_x, p3_x, p2_x, curr_x],
        [curr_y, p1_y, p3_y, p2_y, curr_y],
        color=color,
        linewidth=2,
    )


@stimulus_function
def draw_shape_with_right_angles(data: ShapeWithRightAngles):
    """
    Draw a shape with the specified number of interior right angles.
    Right angles are marked with small squares matching the border color.
    Each shape is randomly rotated by 0°, 90°, 180°, or 270° for educational variety.

    Supported shapes (interior angles only):
    - 0: Circle, equilateral triangle, pentagon, hexagon, or quadrilateral (no right angles)
    - 1: Right triangle or right angle kite shape
    - 2: Pentagon with 2 right angles at base or right trapezoid
    - 3: File icon shape (triangle + rectangle)
    - 4: Rectangle or square
    """
    import random

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get random color for the shape
    color = get_random_polygon_color()

    # Generate shape with right angles
    result = generate_random_shape_with_right_angles(data.num_right_angles)

    # Handle special case for circle
    if isinstance(result[0], str) and result[0] == "circle":
        # Draw a circle
        circle = plt.Circle((0, 0), 2, color=color, fill=False, linewidth=2)
        ax.add_patch(circle)
        # Also add a filled version with transparency
        circle_fill = plt.Circle((0, 0), 2, color=color, alpha=0.3)
        ax.add_patch(circle_fill)
        # No right angle markers for circle

        # Set formatting and limits for circle
        ax.set_aspect("equal")
        ax.axis("off")
        margin = 0.5
        ax.set_xlim(-2 - margin, 2 + margin)
        ax.set_ylim(-2 - margin, 2 + margin)
    else:
        # Regular polygon
        x, y, right_angle_indices = result

        # Apply random rotation (0, 90, 180, or 270 degrees)
        rotation_angles = [0, 90, 180, 270]
        random_rotation = random.choice(rotation_angles)

        if random_rotation != 0:
            x, y = apply_rotation(x, y, random_rotation)

        # Draw the main shape
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill(x, y, color=color, alpha=0.3)

        # Draw right angle markers using the same approach as draw_rectangle
        # Markers are drawn based on the already rotated coordinates
        marker_size = 0.15
        for vertex_idx in right_angle_indices:
            draw_right_angle_marker_at_vertex(
                ax, x, y, vertex_idx, marker_size, color=color
            )

        # Set formatting and limits for polygon
        ax.set_aspect("equal")
        ax.axis("off")
        margin = 0.5
        ax.set_xlim(min(x) - margin, max(x) + margin)
        ax.set_ylim(min(y) - margin, max(y) + margin)

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/random_shape_right_angles_{data.num_right_angles}_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


# New angle marking functions to be appended to geometric_shapes.py


@stimulus_function
def draw_geometric_shapes_with_angles(shape_list: GeometricShapeWithAngleList):
    """
    Draw geometric shapes with specified angle markings.

    This function is similar to draw_geometric_shapes_no_indicators but adds
    angle markers (arcs for acute/obtuse, squares for right angles) at specified vertices.
    """
    num_shapes = len(shape_list)
    # Always use a single row layout
    num_cols = num_shapes
    num_rows = 1

    # Adjust size per shape based on number of shapes
    size_per_shape = 3 if num_shapes > 6 else 4
    # Calculate total width but cap it at a reasonable maximum
    total_width = min(size_per_shape * num_cols, 30)
    # Adjust height to maintain aspect ratio
    height = size_per_shape

    _, axs = plt.subplots(num_rows, num_cols, figsize=(total_width, height))
    axs = np.array(axs).flatten()  # Flatten to easily index

    # Helper function to draw shape and get coordinates
    def draw_shape_and_get_coordinates(shape_type, ax, label, color):
        """Draw a shape without indicators and return its vertex coordinates."""
        if shape_type == ValidGeometricShape.RECTANGLE:
            coords = _draw_rectangle_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.SQUARE:
            coords = _draw_square_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.RIGHT_TRIANGLE:
            coords = _draw_right_triangle_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.OBTUSE_TRIANGLE:
            coords = _draw_obtuse_triangle_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.ACUTE_TRIANGLE:
            coords = _draw_acute_triangle_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.ISOSCELES_TRIANGLE:
            coords = _draw_isosceles_triangle_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.TRAPEZOID:
            coords = _draw_trapezoid_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.RIGHT_TRAPEZOID:
            coords = _draw_right_trapezoid_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.KITE:
            coords = _draw_kite_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.PARALLELOGRAM:
            coords = _draw_parallelogram_clean(ax, label, color)
        elif shape_type == ValidGeometricShape.RHOMBUS:
            coords = _draw_rhombus_clean(ax, label, color)
        else:
            raise ValueError(f"Shape type {shape_type.value} not supported yet")

        return coords

    # Remove unused subplots
    for ax in axs[num_shapes:]:
        ax.remove()

    # Draw each shape with angle markers
    for i, shape_item in enumerate(shape_list):
        ax = axs[i]
        label = "" if num_shapes == 1 else shape_item.label

        # Draw the shape and get coordinates
        coordinates = draw_shape_and_get_coordinates(
            shape_item.shape, ax, label, shape_item.color
        )

        # Add angle marker - automatically find the right vertex
        _add_angle_marker_smart(ax, coordinates, shape_item.angle_type)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_shapes_with_angles_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def _draw_rectangle_clean(ax, title, color):
    """Draw rectangle without any indicators and return coordinates."""
    width, height = 3, 2
    x = np.array([0, width, width, 0, 0])
    y = np.array([0, 0, height, height, 0])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return [(0, 0), (width, 0), (width, height), (0, height)]


def _draw_square_clean(ax, title, color):
    """Draw square without any indicators and return coordinates."""
    size = 2
    x = np.array([0, size, size, 0, 0])
    y = np.array([0, 0, size, size, 0])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return [(0, 0), (size, 0), (size, size), (0, size)]


def _draw_right_triangle_clean(ax, title, color):
    """Draw right triangle without any indicators and return coordinates."""
    coords = [(0, 0), (3, 0), (0, 2)]
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_obtuse_triangle_clean(ax, title, color):
    """Draw obtuse triangle without any indicators and return coordinates."""
    coords = [(0, 0), (2, 0), (3, 1)]  # obtuse angle at vertex 1
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_trapezoid_clean(ax, title, color):
    """Draw trapezoid without any indicators and return coordinates.

    Vertex angles:
    - Vertex 0 (bottom-left): acute (~45°)
    - Vertex 1 (bottom-right): obtuse (~135°)
    - Vertex 2 (top-right): obtuse (~135°)
    - Vertex 3 (top-left): acute (~45°)
    """
    coords = [
        (0, 0),
        (5, 0),
        (3.5, 2),
        (1.5, 2),
    ]  # Trapezoid with more extreme acute and obtuse angles
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_right_trapezoid_clean(ax, title, color):
    """Draw right trapezoid without any indicators and return coordinates.

    Vertex angles:
    - Vertex 0 (bottom-left): right (90°)
    - Vertex 1 (bottom-right): acute (~70°)
    - Vertex 2 (top-right): obtuse (~110°)
    - Vertex 3 (top-left): right (90°)
    """
    coords = [(0, 0), (4, 0), (3, 2), (0, 2)]  # Right trapezoid
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_kite_clean(ax, title, color):
    """Draw kite without any indicators and return coordinates.

    Vertex angles:
    - Vertex 0 (top): obtuse (~125°)
    - Vertex 1 (right): obtuse (~115°)
    - Vertex 2 (bottom): acute (~25°) - very sharp point
    - Vertex 3 (left): obtuse (~115°)
    """
    coords = [(2, 5), (3.2, 2.5), (2, 0), (0.8, 2.5)]  # Very sharp bottom angle (~25°)
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_parallelogram_clean(ax, title, color):
    """Draw parallelogram without any indicators and return coordinates.

    Vertex angles:
    - Vertex 0 (bottom-left): acute (~50°)
    - Vertex 1 (bottom-right): obtuse (~130°)
    - Vertex 2 (top-right): acute (~50°)
    - Vertex 3 (top-left): obtuse (~130°)
    """
    coords = [
        (0, 0),
        (4, 0),
        (5.5, 2),
        (1.5, 2),
    ]  # Parallelogram with moderate acute and obtuse angles
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_rhombus_clean(ax, title, color):
    """Draw rhombus without any indicators and return coordinates.

    Vertex angles:
    - Vertex 0 (bottom): acute (~30°)
    - Vertex 1 (right): obtuse (~150°)
    - Vertex 2 (top): acute (~30°)
    - Vertex 3 (left): obtuse (~150°)
    """
    coords = [
        (2, 0),
        (4.5, 1.5),
        (2, 3),
        (-0.5, 1.5),
    ]  # Very flat rhombus with 30°/150° angles
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_acute_triangle_clean(ax, title, color):
    """Draw acute triangle without any indicators and return coordinates."""
    coords = [(0, 0), (2, 0), (1, 1.5)]  # All angles < 90°
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _draw_isosceles_triangle_clean(ax, title, color):
    """Draw isosceles triangle without any indicators and return coordinates.

    Vertex angles:
    - Vertex 0 (bottom-left): acute (~35°)
    - Vertex 1 (bottom-right): acute (~35°)
    - Vertex 2 (top): obtuse (~110°)
    """
    coords = [
        (0, 0),
        (5, 0),
        (2.5, 1.4),
    ]  # Flat isosceles triangle with 35°-35°-110° angles
    x = np.array([coord[0] for coord in coords] + [coords[0][0]])
    y = np.array([coord[1] for coord in coords] + [coords[0][1]])

    ax.plot(x, y, color=color, linewidth=2)
    ax.fill(x, y, color=color, alpha=0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.title.set_text(title)
        ax.title.set_fontsize(20)

    return coords


def _add_angle_marker_smart(ax, coordinates, angle_type: str):
    """Add angle marker by automatically finding a vertex with the requested angle type.

    Args:
        ax: matplotlib axis
        coordinates: list of (x, y) vertex coordinates
        angle_type: "acute", "obtuse", or "right"
    """

    def calculate_angle_at_vertex(coords, vertex_idx):
        """Calculate the interior angle at a vertex in degrees."""
        num_vertices = len(coords)
        vertex = np.array(coords[vertex_idx])
        prev_vertex = np.array(coords[(vertex_idx - 1) % num_vertices])
        next_vertex = np.array(coords[(vertex_idx + 1) % num_vertices])

        # Calculate vectors from vertex to adjacent vertices
        v1 = prev_vertex - vertex
        v2 = next_vertex - vertex

        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2

        # Calculate angle
        dot_product = np.dot(v1_norm, v2_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product)
        return np.degrees(angle_radians), v1_norm, v2_norm, vertex

    # Find all angles and categorize them
    vertex_angles = []
    for i in range(len(coordinates)):
        angle_deg, v1_norm, v2_norm, vertex = calculate_angle_at_vertex(coordinates, i)
        vertex_angles.append(
            {
                "index": i,
                "angle": angle_deg,
                "v1_norm": v1_norm,
                "v2_norm": v2_norm,
                "vertex": vertex,
                "type": "right"
                if abs(angle_deg - 90) < 5.0
                else ("acute" if angle_deg < 90 else "obtuse"),
            }
        )

    # Find a vertex matching the requested angle type
    matching_vertices = [va for va in vertex_angles if va["type"] == angle_type]

    if not matching_vertices:
        print(
            f"Warning: No {angle_type} angle found in shape with angles: {[va['angle'] for va in vertex_angles]}"
        )
        return

    # Use the first matching vertex (or could add logic to pick the "best" one)
    chosen_vertex = matching_vertices[0]

    # Draw the appropriate marker
    if chosen_vertex["type"] == "right":
        # Draw square marker for right angle
        marker_size = 0.2
        vertex = chosen_vertex["vertex"]
        v1_norm = chosen_vertex["v1_norm"]
        v2_norm = chosen_vertex["v2_norm"]

        # Create square marker
        p1 = vertex + v1_norm * marker_size
        p2 = vertex + v2_norm * marker_size
        p3 = p1 + v2_norm * marker_size

        # Draw square marker
        square_x = [vertex[0], p1[0], p3[0], p2[0], vertex[0]]
        square_y = [vertex[1], p1[1], p3[1], p2[1], vertex[1]]
        ax.plot(square_x, square_y, color="black", linewidth=2)

    else:
        # Draw arc marker for acute/obtuse angles
        vertex = chosen_vertex["vertex"]
        v1 = coordinates[(chosen_vertex["index"] - 1) % len(coordinates)] - vertex
        v2 = coordinates[(chosen_vertex["index"] + 1) % len(coordinates)] - vertex

        # Calculate angles for the arc
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])

        # Ensure we draw the interior angle (the smaller arc)
        angle_diff = angle2 - angle1
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # Swap if needed to ensure angle2 > angle1 for arc drawing
        if angle_diff < 0:
            angle1, angle2 = angle2, angle1

        arc_radius = 0.3 if chosen_vertex["type"] == "obtuse" else 0.45
        arc = patches.Arc(
            (vertex[0], vertex[1]),
            2 * arc_radius,
            2 * arc_radius,
            angle=0,
            theta1=np.degrees(angle1),
            theta2=np.degrees(angle2),
            color="black",
            linewidth=2,
        )
        ax.add_patch(arc)


def draw_right_angle_markers(ax, points, color="black", marker_size=0.2):
    """
    Draw right angle markers at all corners of a quadrilateral (for squares and rectangles).

    Args:
        ax: matplotlib axes
        points: array of quadrilateral vertices (including closing point)
        color: color for the markers
        marker_size: size of the square markers
    """
    # Remove closing point for processing
    vertices = points[:-1]

    for i in range(4):
        # Get current vertex and its neighbors
        vertex = vertices[i]
        prev_vertex = vertices[(i - 1) % 4]
        next_vertex = vertices[(i + 1) % 4]

        # Calculate normalized direction vectors
        v1 = prev_vertex - vertex
        v2 = next_vertex - vertex

        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2

        # Create square marker points
        p1 = vertex + v1_norm * marker_size
        p2 = vertex + v2_norm * marker_size
        p3 = p1 + v2_norm * marker_size

        # Draw square marker
        square_x = [vertex[0], p1[0], p3[0], p2[0], vertex[0]]
        square_y = [vertex[1], p1[1], p3[1], p2[1], vertex[1]]
        ax.plot(square_x, square_y, color=color, linewidth=1.5)


def check_segments_intersect(p1, p2, p3, p4):
    """Check if line segments p1-p2 and p3-p4 intersect."""

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def is_simple_quadrilateral(points):
    """Check if a quadrilateral is simple (no self-intersections)."""
    # Check if any non-adjacent sides intersect
    for i in range(4):
        for j in range(i + 2, 4):
            if j == i + 3 and i == 0:  # Skip adjacent sides (side 0 and side 3)
                continue

            p1, p2 = points[i], points[(i + 1) % 4]
            p3, p4 = points[j], points[(j + 1) % 4]

            if check_segments_intersect(p1, p2, p3, p4):
                return False
    return True


def generate_irregular_quadrilateral_no_parallel_sides():
    """Generate a truly irregular quadrilateral with no parallel sides and no self-intersections."""
    max_attempts = 100

    for attempt in range(max_attempts):
        # Choose from different generation methods for more variety
        method = random.choice(
            ["angular_variation", "distorted_rectangle", "random_convex", "kite_shape"]
        )

        if method == "angular_variation":
            # Generate points with highly varied angular spacing
            center_x, center_y = 0, 0
            base_radius = random.uniform(1.0, 4.0)

            # Much more varied angular distribution
            angles = []
            angle_variations = [
                random.uniform(15, 75),  # First quadrant
                random.uniform(90, 150),  # Second quadrant
                random.uniform(180, 240),  # Third quadrant
                random.uniform(270, 345),  # Fourth quadrant
            ]
            angles = [np.radians(a) for a in angle_variations]

            points = []
            for angle in angles:
                # Much more radius variation
                r = base_radius * random.uniform(0.3, 2.0)
                x = center_x + r * np.cos(angle)
                y = center_y + r * np.sin(angle)
                points.append([x, y])

        elif method == "distorted_rectangle":
            # Start with rectangle and heavily distort it
            width = random.uniform(2.0, 5.0)
            height = random.uniform(1.5, 4.0)

            # Base rectangle points
            base_points = [
                [-width / 2, -height / 2],
                [width / 2, -height / 2],
                [width / 2, height / 2],
                [-width / 2, height / 2],
            ]

            # Apply heavy distortion to each point
            points = []
            for px, py in base_points:
                # Add large random offset
                dx = random.uniform(-width * 0.4, width * 0.4)
                dy = random.uniform(-height * 0.4, height * 0.4)
                points.append([px + dx, py + dy])

        elif method == "random_convex":
            # Generate random convex quadrilateral
            points = []
            for i in range(4):
                # More spread out points
                x = random.uniform(-3, 3)
                y = random.uniform(-3, 3)
                points.append([x, y])

            # Sort points to make convex hull
            center_x = sum(p[0] for p in points) / 4
            center_y = sum(p[1] for p in points) / 4

            def angle_from_center(p):
                return np.arctan2(p[1] - center_y, p[0] - center_x)

            points.sort(key=angle_from_center)

        elif method == "kite_shape":
            # Generate a kite shape with two pairs of adjacent equal sides
            # and no parallel sides

            # Create kite along a main diagonal
            main_axis_length = random.uniform(3.0, 5.0)

            # Position of the "waist" along the main axis (where diagonals cross)
            waist_position = random.uniform(0.3, 0.7)  # Asymmetric for variety

            # Length from center to top and bottom points
            top_length = main_axis_length * waist_position
            bottom_length = main_axis_length * (1 - waist_position)

            # Width of the kite (perpendicular to main axis)
            max_width = random.uniform(2.0, 4.0)

            # Asymmetric width positioning for more variety
            left_width = max_width * random.uniform(0.4, 0.6)
            right_width = max_width * random.uniform(0.4, 0.6)

            # Create the four kite vertices
            points = [
                [0, top_length],  # Top vertex
                [right_width, 0],  # Right vertex
                [0, -bottom_length],  # Bottom vertex
                [-left_width, 0],  # Left vertex
            ]

            # Add small random perturbations to prevent exact symmetries
            # (which could create parallel sides)
            for i, (px, py) in enumerate(points):
                if i > 0:  # Don't perturb the top vertex to maintain kite-like shape
                    perturbation_x = random.uniform(-0.3, 0.3)
                    perturbation_y = random.uniform(-0.2, 0.2)
                    points[i] = [px + perturbation_x, py + perturbation_y]

        # Check if it's a simple quadrilateral
        if not is_simple_quadrilateral(points):
            continue

        # Check that no two sides are parallel
        sides = []
        valid_sides = True
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            if np.linalg.norm(direction) > 0.1:  # Avoid zero-length sides
                sides.append(direction / np.linalg.norm(direction))
            else:
                valid_sides = False
                break

        if not valid_sides:
            continue

        # Check if any two sides are parallel (more strict threshold for irregularity)
        parallel_found = False
        for i in range(4):
            for j in range(i + 1, 4):
                dot_product = abs(np.dot(sides[i], sides[j]))
                if dot_product > 0.9:  # Stricter threshold
                    parallel_found = True
                    break
            if parallel_found:
                break

        if not parallel_found:
            # Close the shape
            points.append(points[0])
            return np.array(points)

    # Fallback: create a simple irregular quadrilateral
    points = [[-2, -1], [3, -0.5], [2, 2.5], [-1.5, 1.8]]
    points.append(points[0])
    return np.array(points)


def generate_trapezoid_one_parallel_side():
    """Generate a trapezoid with exactly one pair of parallel sides - includes scalene, right, and isosceles types."""
    max_attempts = 50  # Prevent infinite loops

    for attempt in range(max_attempts):
        trapezoid_type = random.choice(["scalene", "right", "isosceles"])

        # Much more varied dimensions
        base_width = random.uniform(2.5, 6.0)
        height = random.uniform(1.5, 4.5)

        if trapezoid_type == "isosceles":
            # Isosceles trapezoid - symmetric with equal legs
            top_width = random.uniform(base_width * 0.3, base_width * 0.8)

            # Center the top side
            base_left = -base_width / 2
            base_right = base_width / 2
            top_left = -top_width / 2
            top_right = top_width / 2

            points = [
                [base_left, 0],
                [base_right, 0],
                [top_right, height],
                [top_left, height],
                [base_left, 0],
            ]

        elif trapezoid_type == "right":
            # Right trapezoid - has right angles (vertical sides)
            top_width = random.uniform(base_width * 0.4, base_width * 0.9)

            # One side is vertical, creating right angles
            if random.choice([True, False]):  # Left side vertical
                points = [
                    [-base_width / 2, 0],  # Bottom left
                    [base_width / 2, 0],  # Bottom right
                    [base_width / 2 - (base_width - top_width), height],  # Top right
                    [-base_width / 2, height],  # Top left (vertical line)
                    [-base_width / 2, 0],
                ]
            else:  # Right side vertical
                points = [
                    [-base_width / 2, 0],  # Bottom left
                    [base_width / 2, 0],  # Bottom right (vertical line)
                    [base_width / 2, height],  # Top right
                    [-base_width / 2 + (base_width - top_width), height],  # Top left
                    [-base_width / 2, 0],
                ]

        else:  # scalene trapezoid - FIXED: Keep same height for both top points
            # Scalene trapezoid - no equal sides, but keep proper quadrilateral shape
            top_width = random.uniform(
                base_width * 0.3, base_width * 0.85
            )  # Safer range

            # Asymmetric positioning - top base can be shifted
            base_left = -base_width / 2
            base_right = base_width / 2

            # Shift top base for scalene effect (reduced range to prevent degeneracy)
            top_shift = random.uniform(-base_width * 0.2, base_width * 0.2)
            top_left = -top_width / 2 + top_shift
            top_right = top_width / 2 + top_shift

            # CRITICAL FIX: Use the same height for both top points to maintain parallel sides
            # Vary the shape through the top_shift and different side angles instead
            points = [
                [base_left, 0],
                [base_right, 0],
                [top_right, height],  # Same height
                [top_left, height],  # Same height
                [base_left, 0],
            ]

        # Validate that we have a proper quadrilateral
        unique_points = []
        for p in points[:-1]:  # Exclude closing point
            # Check if this point is too close to any existing point
            is_unique = True
            for existing_p in unique_points:
                distance = np.sqrt(
                    (p[0] - existing_p[0]) ** 2 + (p[1] - existing_p[1]) ** 2
                )
                if distance < 0.1:  # Too close, not a valid quadrilateral
                    is_unique = False
                    break
            if is_unique:
                unique_points.append(p)

        # Must have exactly 4 unique points
        if len(unique_points) != 4:
            continue

        # Check that we actually have exactly one pair of parallel sides
        points_no_close = points[:-1]  # Remove closing point for validation
        if not validate_trapezoid(points_no_close):
            continue

        # If we reach here, we have a valid trapezoid
        # Randomly flip orientation (vertical vs horizontal parallel sides)
        if random.choice([True, False]):
            # Rotate 90 degrees to make vertical parallel sides
            rotated_points = []
            for px, py in points[:-1]:  # Exclude closing point
                rotated_points.append([-py, px])
            rotated_points.append(rotated_points[0])  # Close the shape
            points = rotated_points

        return np.array(points)

    # Fallback: generate a simple, guaranteed-valid trapezoid
    base_width = 4.0
    top_width = 2.5
    height = 3.0

    points = [
        [-base_width / 2, 0],  # Bottom left
        [base_width / 2, 0],  # Bottom right
        [top_width / 2, height],  # Top right
        [-top_width / 2, height],  # Top left
        [-base_width / 2, 0],  # Close
    ]

    return np.array(points)


def validate_trapezoid(points):
    """Validate that a quadrilateral has exactly one pair of parallel sides."""
    if len(points) != 4:
        return False

    # Get all four sides
    sides = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        length = np.linalg.norm(direction)
        if length < 0.1:  # Degenerate side
            return False
        sides.append(direction / length)  # Normalized direction vector

    # Count parallel pairs (opposite sides only for trapezoids)
    parallel_pairs = 0

    # Check side 0 vs side 2 (opposite sides)
    dot_02 = abs(np.dot(sides[0], sides[2]))
    if dot_02 > 0.95:  # Parallel
        parallel_pairs += 1

    # Check side 1 vs side 3 (opposite sides)
    dot_13 = abs(np.dot(sides[1], sides[3]))
    if dot_13 > 0.95:  # Parallel
        parallel_pairs += 1

    # For a trapezoid, we want exactly 1 pair of parallel sides
    return parallel_pairs == 1


def generate_parallelogram_two_parallel_sides():
    """Generate a parallelogram with much more variety - squares, rectangles, rhombuses, and generic parallelograms.

    Returns:
        tuple: (points_array, shape_type) where shape_type is one of: 'square', 'rectangle', 'rhombus', 'generic_parallelogram'
    """
    shape_type = random.choice(
        ["rectangle", "square", "rhombus", "generic_parallelogram"]
    )

    if shape_type == "square":
        # Much more size variation
        side_length = random.uniform(1.5, 5.0)

        # Sometimes create a slightly imperfect square for more variety
        if random.random() < 0.3:  # 30% chance of slight imperfection
            width = side_length * random.uniform(0.95, 1.05)
            height = side_length * random.uniform(0.95, 1.05)
        else:
            width = height = side_length

        points = [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
            [-width / 2, -height / 2],
        ]

    elif shape_type == "rectangle":
        # Much more varied aspect ratios
        width = random.uniform(2.0, 6.0)
        height = random.uniform(1.0, 4.5)

        # Ensure significant difference between width and height for rectangles
        if abs(width - height) < 0.5:
            if width > height:
                height = width * random.uniform(0.4, 0.7)
            else:
                width = height * random.uniform(0.4, 0.7)

        points = [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
            [-width / 2, -height / 2],
        ]

    elif shape_type == "rhombus":
        # More varied rhombuses
        side_length = random.uniform(2.0, 5.0)

        # Much wider range of angles for more variety
        angle = random.uniform(30, 150)  # Wider angle range
        angle_rad = np.radians(angle)

        # Create rhombus with varied orientation
        if random.random() < 0.5:
            # Diamond orientation
            points = [
                [0, -side_length * np.sin(angle_rad / 2)],
                [side_length * np.cos(angle_rad / 2), 0],
                [0, side_length * np.sin(angle_rad / 2)],
                [-side_length * np.cos(angle_rad / 2), 0],
                [0, -side_length * np.sin(angle_rad / 2)],
            ]
        else:
            # Slanted orientation
            dx = side_length * np.cos(angle_rad)
            dy = side_length * np.sin(angle_rad)
            points = [
                [0, 0],
                [side_length, 0],
                [side_length + dx, dy],
                [dx, dy],
                [0, 0],
            ]

    else:  # generic_parallelogram
        # Much more varied parallelograms
        base = random.uniform(2.5, 6.0)
        height = random.uniform(1.5, 4.5)

        # Much more varied skew angles
        skew_angle = random.uniform(15, 75)  # Angle in degrees
        skew = height / np.tan(np.radians(skew_angle))

        # Random orientation
        if random.choice([True, False]):
            points = [[0, 0], [base, 0], [base + skew, height], [skew, height], [0, 0]]
        else:
            # Flipped orientation
            points = [[0, 0], [base, 0], [base - skew, height], [-skew, height], [0, 0]]

    # Randomly apply additional transformations for more variety
    if random.random() < 0.3:  # 30% chance
        # Apply slight random rotation (small angle to maintain general shape recognition)
        small_rotation = random.uniform(-15, 15)
        angle_rad = np.radians(small_rotation)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_points = []
        for px, py in points[:-1]:  # Exclude closing point
            x_rot = px * cos_a - py * sin_a
            y_rot = px * sin_a + py * cos_a
            rotated_points.append([x_rot, y_rot])
        rotated_points.append(rotated_points[0])  # Close the shape
        points = rotated_points

    return np.array(points), shape_type


@stimulus_function
def create_parallel_quadrilateral(data: ParallelQuadrilateral):
    """
    Create a quadrilateral with a specified number of parallel sides.

    Args:
        data: ParallelQuadrilateral object containing:
            - num_parallel_sides: 0, 1, or 2 pairs of parallel sides
            - rotate: Whether to apply random rotation
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate the appropriate quadrilateral based on parallel sides count
    shape_type = None
    if data.num_parallel_sides == 0:
        points = generate_irregular_quadrilateral_no_parallel_sides()
    elif data.num_parallel_sides == 1:
        points = generate_trapezoid_one_parallel_side()
    else:  # num_parallel_sides == 2
        points, shape_type = generate_parallelogram_two_parallel_sides()

    # Apply rotation if requested
    if data.rotate:
        rotation_angle = random.uniform(0, 360)
        rotation_rad = np.radians(rotation_angle)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_rad), -np.sin(rotation_rad)],
                [np.sin(rotation_rad), np.cos(rotation_rad)],
            ]
        )

        # Apply rotation to all points except the last one (which is the closing point)
        for i in range(len(points) - 1):
            points[i] = np.dot(rotation_matrix, points[i])

        # Update the closing point
        points[-1] = points[0]

    # Draw the quadrilateral
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Get random color for visual variety
    color = get_random_polygon_color()

    # Draw outline and fill with the same color (different transparency)
    ax.plot(x_coords, y_coords, color=color, linewidth=2)
    ax.fill(x_coords, y_coords, color=color, alpha=0.3)

    # Add right angle markers for squares and rectangles
    if shape_type in ["square", "rectangle"]:
        draw_right_angle_markers(ax, points, color=color, marker_size=0.15)

    # Set equal aspect ratio and adjust limits
    ax.set_aspect("equal")

    # Calculate appropriate limits based on the points
    margin = 1.0
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Remove axes for cleaner appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/parallel_quadrilateral_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


# =============================================================================
# Regular/Irregular Polygon Functions
# =============================================================================

TEXT_GRAY = "#666666"
MARK_RED = "red"
OUTLINE_W = 3.0
MARK_W = 2.6  # a hair thicker for visibility


# ───────────────────────── geometry helpers ─────────────────────────


def _regular_vertices(
    n: int, R: float = 2.2, center=(0.0, 0.0), start_deg: float = -90.0
):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.deg2rad(start_deg)
    x = center[0] + R * np.cos(ang)
    y = center[1] + R * np.sin(ang)
    pts = list(zip(x, y))
    pts.append(pts[0])
    return pts


def _walk_sides_angles(
    side_lengths: List[float], interior_angles_deg: List[float], start_deg=0.0
):
    n = len(side_lengths)
    ext = [180.0 - a for a in interior_angles_deg]
    x, y = [0.0], [0.0]
    head = math.radians(start_deg)
    cx = cy = 0.0
    for i in range(n - 1):
        L = float(side_lengths[i])
        cx += L * math.cos(head)
        cy += L * math.sin(head)
        x.append(cx)
        y.append(cy)
        head += math.radians(ext[i])
    x.append(x[0])
    y.append(y[0])
    return list(zip(x, y))


def _from_sides_only(n: int, side_lengths: List[float]):
    exterior = 360.0 / n
    return _walk_sides_angles(side_lengths, [180.0 - exterior] * n, start_deg=0.0)


def _scale_center(verts, target=2.3):
    xs = np.array([p[0] for p in verts[:-1]])
    ys = np.array([p[1] for p in verts[:-1]])
    cx, cy = xs.mean(), ys.mean()
    xs -= cx
    ys -= cy
    m = max(abs(xs).max(), abs(ys).max(), 1e-6)
    s = target / m
    xs *= s
    ys *= s
    out = list(zip(xs, ys))
    out.append(out[0])
    return out


def _mid(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _fmt_num(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


# orientation-aware bisector that always points toward polygon interior
def _inward_bisector(v, p_prev, p_next, centroid):
    u1 = np.array(p_prev, dtype=float) - np.array(v, dtype=float)
    u2 = np.array(p_next, dtype=float) - np.array(v, dtype=float)
    l1 = np.linalg.norm(u1) or 1.0
    l2 = np.linalg.norm(u2) or 1.0
    u1 /= l1
    u2 /= l2
    bis = u1 + u2
    if np.linalg.norm(bis) == 0:
        bis = np.array([0.3, 0.0])
    bis = bis / (np.linalg.norm(bis) or 1.0)
    to_center = np.array(centroid) - np.array(v)
    if np.dot(bis, to_center) < 0:
        bis = -bis  # flip outward → inward
    return bis, u1, u2, l1, l2


def _arc_span(u_prev, u_next):
    a1 = math.degrees(math.atan2(u_prev[1], u_prev[0]))
    a2 = math.degrees(math.atan2(u_next[1], u_next[0]))
    while a2 <= a1:  # note: <= to avoid 0-sweep
        a2 += 360.0
    sweep = a2 - a1
    if sweep > 180.0:
        a1, a2 = a2, a1 + 360.0
        sweep = a2 - a1
    # epsilon so Arc always has a visible sweep
    if sweep < 0.5:
        a2 = a1 + 0.5
    return a1, a2


# ───────────────────────── indicators ─────────────────────────


def _draw_side_labels(ax, verts, side_lengths):
    if not side_lengths:
        return
    xs = [p[0] for p in verts[:-1]]
    ys = [p[1] for p in verts[:-1]]
    cx, cy = np.mean(xs), np.mean(ys)
    n = len(verts) - 1
    base_off = 0.55 if n <= 6 else (0.60 if n <= 8 else 0.65)
    for i in range(n):
        p1, p2 = verts[i], verts[(i + 1) % n]
        mx, my = _mid(p1, p2)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        if (cx - mx) * nx + (cy - my) * ny > 0:
            nx, ny = -nx, -ny
        ax.text(
            mx + base_off * nx,
            my + base_off * ny,
            _fmt_num(float(side_lengths[i])),
            fontsize=14,
            ha="center",
            va="center",
            color="black",
            weight="bold",
        )


def _tick(ax, p1, p2, count=1):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy) or 1.0
    px, py = -dy / L, dx / L
    tick_len = 0.18
    # slightly larger spacing (per your request)
    if count == 1:
        ts = [0.50]
    elif count == 2:
        ts = [0.40, 0.45]
    else:
        ts = [0.44, 0.48, 0.52]
    for t in ts:
        cx, cy = p1[0] + t * dx, p1[1] + t * dy
        ax.plot(
            [cx - tick_len * px, cx + tick_len * px],
            [cy - tick_len * py, cy + tick_len * py],
            color=MARK_RED,
            linewidth=MARK_W,
            solid_capstyle="round",
        )


def _apply_tick_groups(ax, verts, n, side_equal_groups, side_lengths, show_tick_marks):
    if side_equal_groups:
        for g in side_equal_groups:
            cnt = {"single": 1, "double": 2, "triple": 3}.get(g.tick_style, 1)
            for s in g.sides:
                i = int(s) % n
                _tick(ax, verts[i], verts[(i + 1) % n], count=cnt)
    elif show_tick_marks:
        uniq = []
        group_index = {}
        for i, v in enumerate(side_lengths or []):
            if v not in group_index:
                group_index[v] = len(uniq)
                uniq.append(v)
            gi = group_index[v]
            cnt = [1, 2, 3][gi % 3]
            _tick(ax, verts[i], verts[(i + 1) % n], count=cnt)


def _draw_angle_set(
    ax, verts, angles_deg, angle_style, equal_groups=None, show_numbers=True
):
    n = len(verts) - 1
    xs = [p[0] for p in verts[:-1]]
    ys = [p[1] for p in verts[:-1]]
    centroid = (float(np.mean(xs)), float(np.mean(ys)))

    arc_count = {i: 1 for i in range(n)}
    for g in equal_groups or []:
        for v in g.vertices:
            arc_count[int(v) % n] = max(1, min(3, int(getattr(g, "arc_count", 1))))

    for i in range(n):
        v = verts[i]
        p_prev = verts[(i - 1) % n]
        p_next = verts[(i + 1) % n]
        bis, u1, u2, l1, l2 = _inward_bisector(v, p_prev, p_next, centroid)
        ang = float(angles_deg[i]) if angles_deg else None

        # compute radius based on local edge lengths
        base = 0.22 * min(float(l1), float(l2))
        if ang is not None and ang > 110:
            base *= 1.15
        r = max(0.20, min(base, 0.55))

        # numbers-only style
        if angle_style == "number":
            if show_numbers and ang is not None:
                tx, ty = v[0] + bis[0] * (r + 0.12), v[1] + bis[1] * (r + 0.12)
                ax.text(
                    tx,
                    ty,
                    f"{int(round(ang))}°",
                    fontsize=14,
                    ha="center",
                    va="center",
                    color=TEXT_GRAY,
                    weight="bold",
                )
            continue

        # right-angle square style
        if angle_style == "square":
            # draw a proper open square (3 sides) inside the corner
            s = r * 0.8
            # basis along the two incident edges
            q1 = v + u1 * s  # along edge 1
            q4 = v + u2 * s  # along edge 2
            # the far corners of the little square
            q2 = q1 + u2 * s
            q3 = q4 + u1 * s

            # three sides of the square, open toward the vertex v
            ax.plot(
                [q1[0], q2[0]],
                [q1[1], q2[1]],
                color=MARK_RED,
                linewidth=MARK_W,
                zorder=5,
            )
            ax.plot(
                [q2[0], q3[0]],
                [q2[1], q3[1]],
                color=MARK_RED,
                linewidth=MARK_W,
                zorder=5,
            )
            ax.plot(
                [q3[0], q4[0]],
                [q3[1], q4[1]],
                color=MARK_RED,
                linewidth=MARK_W,
                zorder=5,
            )

            # (optional numbers next to the square if you enabled them)
            if ang is not None and show_numbers:
                tx, ty = v[0] + bis[0] * (s + 0.22), v[1] + bis[1] * (s + 0.22)
                ax.text(
                    tx,
                    ty,
                    f"{int(round(ang))}°",
                    fontsize=14,
                    ha="center",
                    va="center",
                    color=TEXT_GRAY,
                    weight="bold",
                    zorder=6,
                )
            continue

        # "arc" style
        # For reflex angles (>= 175°), show numbers only to keep everything inside neatly.
        if ang is not None and ang >= 175:
            if show_numbers:
                tx, ty = v[0] + bis[0] * (r + 0.14), v[1] + bis[1] * (r + 0.14)
                ax.text(
                    tx,
                    ty,
                    f"{int(round(ang))}°",
                    fontsize=14,
                    ha="center",
                    va="center",
                    color=TEXT_GRAY,
                    weight="bold",
                )
            continue

        th1, th2 = _arc_span(u1, u2)
        for k in range(arc_count.get(i, 1)):
            rk = r + 0.08 * k
            arc = patches.Arc(
                (v[0], v[1]),
                2 * rk,
                2 * rk,
                angle=0,
                theta1=th1,
                theta2=th2,
                color=MARK_RED,
                linewidth=MARK_W,
                zorder=5,
            )
            ax.add_patch(arc)
        if show_numbers and ang is not None:
            tx, ty = (
                v[0] + bis[0] * (r + 0.08 * arc_count.get(i, 1) + 0.10),
                v[1] + bis[1] * (r + 0.08 * arc_count.get(i, 1) + 0.10),
            )
            ax.text(
                tx,
                ty,
                f"{int(round(ang))}°",
                fontsize=14,
                ha="center",
                va="center",
                color=TEXT_GRAY,
                weight="bold",
            )


# ───────────────────────── drawers ─────────────────────────


@stimulus_function
def draw_regular_irregular_polygons(payload: RegularIrregularPolygonList):
    items = payload.root if hasattr(payload, "root") else payload
    m = len(items)

    cols = max(1, m)
    slot_w = 4.8 if cols <= 6 else 3.8
    fig_w = min(cols * slot_w, 36)
    fig_h = 5.2
    fig, axs = plt.subplots(1, cols, figsize=(fig_w, fig_h), dpi=200)
    axs = [axs] if cols == 1 else np.asarray(axs).flatten()
    fig.patch.set_facecolor("white")

    for i, poly in enumerate(items):
        ax = axs[i]
        ax.set_facecolor("white")

        # geometry
        if poly.side_lengths and poly.angles:
            verts = _walk_sides_angles(
                [float(s) for s in poly.side_lengths],
                [float(a) for a in poly.angles],
                start_deg=0.0,
            )
        elif poly.side_lengths:
            verts = _from_sides_only(
                poly.num_sides, [float(s) for s in poly.side_lengths]
            )
        else:
            verts = _regular_vertices(poly.num_sides)
        verts = _scale_center(verts)
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        n = len(verts) - 1

        # resolve side mode when angles requested (exactly one of labels/ticks)
        if poly.show_side_lengths and poly.show_tick_marks:
            show_labels, show_ticks = True, False  # prefer labels
        else:
            show_labels, show_ticks = (
                bool(poly.show_side_lengths),
                bool(poly.show_tick_marks),
            )
        if poly.show_angle_markings and not (show_labels ^ show_ticks):
            show_labels, show_ticks = False, True  # default to ticks

        # outline
        ax.plot(xs, ys, color=poly.color, linewidth=OUTLINE_W, solid_capstyle="round")

        # sides
        side_disp = None
        if show_labels or show_ticks:
            side_disp = (
                [float(s) for s in poly.side_lengths]
                if poly.side_lengths
                else [2] * poly.num_sides
            )
        if show_labels and side_disp:
            _draw_side_labels(ax, verts, side_disp)
        elif show_ticks and side_disp:
            _apply_tick_groups(ax, verts, n, poly.side_equal_groups, side_disp, True)

        # angles
        if poly.show_angle_markings:
            if poly.angles:
                ang_disp = [float(a) for a in poly.angles]
            else:
                reg = (poly.num_sides - 2) * 180.0 / poly.num_sides
                ang_disp = [reg] * poly.num_sides
            _draw_angle_set(
                ax,
                verts,
                ang_disp,
                poly.angle_style,
                poly.angle_equal_groups,
                show_numbers=(poly.angle_style == "number"),
            )

        # label above (multi-panel) if provided
        if poly.label and m > 1:
            ax.text(
                np.mean(xs[:-1]),
                max(ys) + 0.9,
                str(poly.label),
                ha="center",
                va="bottom",
                fontsize=14,
                weight="bold",
                color="black",
            )

        # viewport
        pad = 0.9
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    out = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/regular_irregular_polygons_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.06)
    plt.savefig(
        out,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)
    return out


@stimulus_function
def draw_polygons_bare(payload: RegularIrregularPolygonList):
    items = payload.root if hasattr(payload, "root") else payload
    cols = len(items) or 1
    fig, axs = plt.subplots(1, cols, figsize=(max(4, 3.6 * cols), 4.6), dpi=200)
    axs = [axs] if cols == 1 else np.asarray(axs).flatten()
    for i, poly in enumerate(items):
        ax = axs[i]
        if poly.side_lengths and poly.angles:
            verts = _walk_sides_angles(
                [float(s) for s in poly.side_lengths],
                [float(a) for a in poly.angles],
                start_deg=0.0,
            )
        elif poly.side_lengths:
            verts = _from_sides_only(
                poly.num_sides, [float(s) for s in poly.side_lengths]
            )
        else:
            verts = _regular_vertices(poly.num_sides)
        verts = _scale_center(verts)
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        ax.plot(xs, ys, color=poly.color, linewidth=OUTLINE_W, solid_capstyle="round")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    out = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/polygons_bare_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(
        out,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    return out


@stimulus_function
def draw_quadrilateral_venn_diagram(data: QuadrilateralVennDiagram) -> str:
    """
    Draw the quadrilateral classification diagram (no overlaps), matching the reference.
    """

    # Canvas: 1400 × 1000, white (increased width for proper spacing)
    fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()  # Match image coordinate system (top-left origin)

    # Outer frame (rounded square): increased size for proper spacing
    frame = patches.FancyBboxPatch(
        (40, 40),
        1320,
        920,
        boxstyle="round,pad=0,rounding_size=16",
        fill=False,
        edgecolor="#A3A6A8",
        linewidth=2.5,
        zorder=1,
    )
    ax.add_patch(frame)

    # Title: "quadrilaterals" always, positioned below top edge with 2rem gap
    if data.show_labels:
        ax.text(
            700,  # Center of wider canvas
            100,  # 2rem (32px) gap from top edge (40 + 32 + 28 for text height)
            "quadrilaterals",
            fontsize=40,
            weight="bold",
            ha="center",
            va="center",
            color="#333",
            zorder=5,
        )

    # Main enclosing circle: positioned with 2rem gap from top, bottom aligned with small circle
    # Bottom of big circle should be at same level as bottom of small circle
    big_circle_bottom = 900  # Bottom level for both circles
    big_radius = 350
    big_center_y = big_circle_bottom - big_radius  # 900 - 350 = 550
    big_center_x = 500  # Centered in left portion

    main_circle = patches.Circle(
        (big_center_x, big_center_y),
        big_radius,
        fill=False,
        edgecolor="#A3A6A8",
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(main_circle)

    # Small trapezoids circle: positioned with 3rem gap from big circle, same bottom level
    small_radius = 130
    small_center_y = big_circle_bottom - small_radius  # Same bottom level
    gap_between_circles = 96  # 3rem = 48px, so gap between edges = 96px
    small_center_x = (
        big_center_x + big_radius + gap_between_circles + small_radius
    )  # 500 + 350 + 96 + 120 = 1066

    trap_circle = patches.Circle(
        (small_center_x, small_center_y),
        small_radius,
        fill=False,
        edgecolor="#A3A6A8",
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(trap_circle)

    # Venn diagram - two equal circles inside the big circle
    venn_radius = 170
    venn_y = big_center_y + 70  # Position lower in the big circle
    left_venn_x = big_center_x - 100  # Left of center
    right_venn_x = big_center_x + 100  # Right of center

    left_circle = patches.Circle(
        (left_venn_x, venn_y),
        venn_radius,
        fill=False,
        edgecolor="#A3A6A8",
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(left_circle)

    right_circle = patches.Circle(
        (right_venn_x, venn_y),
        venn_radius,
        fill=False,
        edgecolor="#A3A6A8",
        linewidth=2.5,
        zorder=2,
    )
    ax.add_patch(right_circle)

    # Labels positioned based on new circle coordinates
    if data.show_labels:
        ax.text(
            big_center_x,
            big_center_y - 270,
            "parallelograms",
            fontsize=26,
            ha="center",
            va="center",
            color="#333",
            zorder=5,
        )

        # "rectangles" label: in left circle, avoiding intersection
        ax.text(
            left_venn_x - 60,
            venn_y - 50,
            "rectangles",
            fontsize=16,
            ha="center",
            va="center",
            color="#333",
            zorder=5,
        )

        # "rhombuses" label: in right circle, avoiding intersection
        ax.text(
            right_venn_x + 60,
            venn_y - 50,
            "rhombuses",
            fontsize=16,
            ha="center",
            va="center",
            color="#333",
            zorder=5,
        )

        # "squares" label: in intersection area
        ax.text(
            big_center_x,
            venn_y + 80,  # Below the intersection
            "squares",
            fontsize=16,
            ha="center",
            va="center",
            color="#333",
            zorder=5,
        )

        # "trapezoids" label: in small circle
        ax.text(
            small_center_x,
            small_center_y - 40,
            "trapezoids",
            fontsize=20,
            ha="center",
            va="center",
            color="#333",
            zorder=5,
        )

    # Example shapes positioned based on new circle coordinates
    if data.show_shape_examples:
        teal_para = patches.Polygon(
            [
                (big_center_x - 50, big_center_y - 200),  # Top side parallel to bottom
                (big_center_x + 70, big_center_y - 200),  # Top right
                (big_center_x + 50, big_center_y - 150),  # Bottom side parallel to top
                (big_center_x - 70, big_center_y - 150),  # Bottom left - tilted left
            ],
            fill=False,
            edgecolor="#10A7A0",
            linewidth=6,
            zorder=6,
        )
        ax.add_patch(teal_para)

        blue_rect = patches.Rectangle(
            (left_venn_x - 120, venn_y - 20),
            80,
            120,
            fill=False,
            edgecolor="#1DA8F5",
            linewidth=6,
            zorder=6,
        )
        ax.add_patch(blue_rect)

        # Red square: in intersection of both circles
        red_square = patches.Rectangle(
            (big_center_x - 25, venn_y - 25),
            70,
            70,
            fill=False,
            edgecolor="#E85451",
            linewidth=6,
            zorder=6,
        )
        ax.add_patch(red_square)

        # Green rhombus: in right circle only (rhombuses region)
        # Elongated rhombus with different diagonal lengths to avoid square appearance
        rhombus_center_x = right_venn_x + 60
        rhombus_center_y = venn_y + 40
        rhombus_width = 60  # Half the horizontal diagonal (moderately wider)
        rhombus_height = 45  # Half the vertical diagonal (narrower)
        green_rhombus = patches.Polygon(
            [
                (
                    rhombus_center_x,
                    rhombus_center_y - rhombus_height - 20,
                ),  # Top vertex
                (
                    rhombus_center_x - 20 + rhombus_width,
                    rhombus_center_y,
                ),  # Right vertex
                (
                    rhombus_center_x,
                    rhombus_center_y + rhombus_height + 20,
                ),  # Bottom vertex
                (
                    rhombus_center_x - rhombus_width + 20,
                    rhombus_center_y,
                ),  # Left vertex
            ],
            fill=False,
            edgecolor="#19A74A",
            linewidth=6,
            zorder=6,
        )
        ax.add_patch(green_rhombus)

        # Blue trapezoid: in small circle
        trap_shape_x = small_center_x
        trap_shape_y = small_center_y + 20
        blue_trapezoid = patches.Polygon(
            [
                (trap_shape_x - 30, trap_shape_y - 30),  # Top left (shorter top edge)
                (trap_shape_x + 30, trap_shape_y - 30),  # Top right (shorter top edge)
                (
                    trap_shape_x + 80,
                    trap_shape_y + 30,
                ),  # Bottom right (longer bottom edge)
                (
                    trap_shape_x - 80,
                    trap_shape_y + 30,
                ),  # Bottom left (longer bottom edge)
            ],
            fill=False,
            edgecolor="#1DA8F5",
            linewidth=6,
            zorder=6,
        )
        ax.add_patch(blue_trapezoid)

        # Purple irregular quad: angular, asymmetric shape like the reference
        purple_x = small_center_x
        purple_y = small_center_y - 300  # Positioned near small circle
        purple_quad = patches.Polygon(
            [
                (purple_x - 40, purple_y + 75),  # Top left (vertical flip)
                (purple_x + 120, purple_y + 75),  # Top right (vertical flip)
                (purple_x + 40, purple_y - 20),  # Bottom right (vertical flip)
                (purple_x - 40, purple_y - 40),  # Bottom left (vertical flip)
            ],
            fill=False,
            edgecolor="#8D7CF6",
            linewidth=6,
            zorder=6,
        )
        ax.add_patch(purple_quad)

    # ---- save ----
    out = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/quadrilateral_venn_diagram_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.tight_layout()
    plt.savefig(
        out,
        dpi=200,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close(fig)
    return out
