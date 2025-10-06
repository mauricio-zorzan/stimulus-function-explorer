import math
import random
import time
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_scale import (
    PolygonScale,
)
from content_generators.settings import settings


def generate_polygon_coordinates(
    polygon_type: str, vertex_labels: List[str], measurements: List[Union[int, float]]
) -> List[Tuple[float, float]]:
    """
    Generate polygon coordinates based on the polygon type and all side measurements.
    This creates a polygon that will have the exact measurements specified.
    """
    num_vertices = len(vertex_labels)

    # For simplicity, we'll generate coordinates for common polygon types
    # More complex polygons can be added as needed

    if polygon_type == "triangle" and num_vertices == 3:
        # Create a triangle with specified side measurements
        # Place first vertex at origin, second on x-axis
        coords = [(0.0, 0.0)]

        # Use first measurement for AB
        ab_length = float(measurements[0]) if len(measurements) > 0 else 10.0
        coords.append((ab_length, 0.0))

        # For third vertex, use second measurement for BC if available
        if len(measurements) >= 2:
            bc_length = float(measurements[1])
            # Create a triangle with these measurements
            # Place C at a position that creates the desired BC length
            angle = math.pi / 3  # 60 degrees for a reasonable triangle
            c_x = ab_length - bc_length * math.cos(angle)
            c_y = bc_length * math.sin(angle)
            coords.append((c_x, c_y))
        else:
            coords.append((ab_length / 2, 8.0))  # Default third vertex

        return coords

    elif polygon_type == "quadrilateral" and num_vertices == 4:
        # Create a quadrilateral (rectangle-like for simplicity)
        if len(measurements) >= 2:
            width = float(measurements[0])  # AB
            height = float(measurements[1])  # BC
        else:
            width = 10.0
            height = 8.0

        return [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]

    elif polygon_type == "pentagon" and num_vertices == 5:
        # Create a regular pentagon and scale to match measurements
        coords = []
        radius = 10.0  # Base radius
        if measurements:
            # Scale based on first measurement
            radius = float(measurements[0]) / (2 * math.sin(math.pi / 5))

        for i in range(5):
            angle = 2 * math.pi * i / 5 - math.pi / 2  # Start from top
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            coords.append((x, y))
        return coords

    elif polygon_type == "hexagon" and num_vertices == 6:
        # Create a regular hexagon
        coords = []
        radius = 8.0
        if measurements:
            radius = float(measurements[0])

        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            coords.append((x, y))
        return coords

    else:
        # For irregular or unknown types, create a truly irregular polygon
        # Use measurements and vertex labels as seed for consistent randomness
        seed_value = sum(hash(str(label)) for label in vertex_labels) + sum(
            int(m * 1000) for m in measurements
        )
        random.seed(seed_value)

        coords = []
        base_radius = 10.0

        # Create irregular angles (not evenly spaced)
        angles = []
        remaining_angle = 2 * math.pi
        for i in range(num_vertices):
            if i == num_vertices - 1:
                # Last angle gets whatever remains
                angles.append(remaining_angle)
            else:
                # Random variation around equal division
                base_angle = 2 * math.pi / num_vertices
                variation = base_angle * 0.4  # Allow up to 40% variation
                angle = base_angle + random.uniform(-variation, variation)
                # Ensure we don't use up all remaining angle
                min_remaining = base_angle * 0.6 * (num_vertices - i - 1)
                if angle > remaining_angle - min_remaining:
                    angle = remaining_angle - min_remaining
                angles.append(max(angle, base_angle * 0.6))  # Minimum angle
                remaining_angle -= angle

        # Create irregular radii (different distances from center)
        radii = []
        for i in range(num_vertices):
            radius_variation = random.uniform(0.6, 1.4)  # 60% to 140% of base radius
            radii.append(base_radius * radius_variation)

        # Generate coordinates with irregular angles and radii
        current_angle = 0
        for i in range(num_vertices):
            x = radii[i] * math.cos(current_angle)
            y = radii[i] * math.sin(current_angle)
            coords.append((x, y))
            current_angle += angles[i]

        return coords


def get_side_index(side_name: str, vertex_labels: List[str]) -> int:
    """Get the index of a side based on its name and vertex labels."""
    for i, vertex in enumerate(vertex_labels):
        next_vertex = vertex_labels[(i + 1) % len(vertex_labels)]
        if side_name == f"{vertex}{next_vertex}":
            return i
    return -1


def get_measurements_for_visible_sides(
    visible_sides: List[str],
    all_measurements: List[Union[int, float]],
    vertex_labels: List[str],
) -> List[Union[int, float]]:
    """
    Extract measurements for only the visible sides from the full measurements list.
    This allows showing only some measurements while having all measurements for coordinate generation.
    """
    measurements_for_display = []
    for side in visible_sides:
        side_index = get_side_index(side, vertex_labels)
        if side_index != -1 and side_index < len(all_measurements):
            measurements_for_display.append(all_measurements[side_index])
        # If side not found, we could add error handling here
    return measurements_for_display


def draw_vertex_labels(
    ax, vertex_labels: List[str], positions: List[Tuple[float, float]]
) -> None:
    """Draw vertex labels at the polygon corners, positioned outside the polygon."""
    # Calculate polygon center for outward positioning
    center_x = sum(pos[0] for pos in positions) / len(positions)
    center_y = sum(pos[1] for pos in positions) / len(positions)

    for label, position in zip(vertex_labels, positions):
        # Calculate direction from center to vertex
        dx = position[0] - center_x
        dy = position[1] - center_y

        # Normalize the direction vector
        distance = math.sqrt(dx**2 + dy**2)
        if distance > 0:
            dx_norm = dx / distance
            dy_norm = dy / distance
        else:
            # Fallback if vertex is at center (shouldn't happen normally)
            dx_norm, dy_norm = 0, 1

        # Place label outside the polygon by extending the direction
        label_offset = 1.2  # Much more distance from vertices
        label_x = position[0] + dx_norm * label_offset
        label_y = position[1] + dy_norm * label_offset

        ax.text(
            label_x,
            label_y,
            label,
            fontsize=16,
            fontweight="bold",  # Vertex label font size
            ha="center",
            va="center",
            color="black",
        )  # Removed bbox to eliminate border


def draw_polygon_with_measurements(
    ax,
    polygon_label: str,
    vertex_labels: List[str],
    positions: List[Tuple[float, float]],
    color: str,
    visible_sides: List[str],
    measurements: List[Union[int, float]],
    measurement_unit: str,
) -> None:
    """Draw a polygon with specified side measurements and vertex labels using provided positions."""
    num_points = len(positions)

    # Extract coordinates from positions
    x_coords = [pos[0] for pos in positions] + [positions[0][0]]  # Close the polygon
    y_coords = [pos[1] for pos in positions] + [positions[0][1]]

    # Draw the polygon
    ax.plot(x_coords, y_coords, color=color, linewidth=2)
    ax.fill(x_coords[:-1], y_coords[:-1], color=color, alpha=0.1)

    # Add vertex labels
    draw_vertex_labels(ax, vertex_labels, positions)

    # Add polygon label below the polygon
    center_x = sum(pos[0] for pos in positions) / num_points
    min_y = min(pos[1] for pos in positions)
    ax.text(
        center_x,
        min_y - 4.5,
        polygon_label,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="top",
    )

    # Add side measurements - only for the visible sides
    # Calculate polygon center for outward positioning
    center_x = sum(pos[0] for pos in positions) / num_points
    center_y = sum(pos[1] for pos in positions) / num_points

    for side_name, measurement in zip(visible_sides, measurements):
        # Find the side index
        side_index = get_side_index(side_name, vertex_labels)
        if side_index == -1:
            continue

        pos1 = positions[side_index]
        pos2 = positions[(side_index + 1) % num_points]

        # Find midpoint of the side using display positions
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2

        # Calculate perpendicular offset for label placement
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        side_length = math.sqrt(dx**2 + dy**2)

        if side_length > 0:
            # Calculate two possible perpendicular vectors (rotated 90 degrees)
            perp_x1 = -dy / side_length
            perp_y1 = dx / side_length
            perp_x2 = dy / side_length
            perp_y2 = -dx / side_length

            # Choose the perpendicular vector that points away from the polygon center
            # Test which direction moves away from center
            test_x1 = mid_x + perp_x1 * 0.1
            test_y1 = mid_y + perp_y1 * 0.1
            test_x2 = mid_x + perp_x2 * 0.1
            test_y2 = mid_y + perp_y2 * 0.1

            # Calculate distances from center
            dist1_sq = (test_x1 - center_x) ** 2 + (test_y1 - center_y) ** 2
            dist2_sq = (test_x2 - center_x) ** 2 + (test_y2 - center_y) ** 2

            # Choose the direction that increases distance from center (outward)
            if dist1_sq > dist2_sq:
                perp_x, perp_y = perp_x1, perp_y1
            else:
                perp_x, perp_y = perp_x2, perp_y2

            # Offset the label away from the side (outward)
            label_offset = 1.5  # Reduced offset for closer positioning
            label_x = mid_x + perp_x * label_offset
            label_y = mid_y + perp_y * label_offset

            # Format the measurement to remove .0 for whole numbers
            if isinstance(measurement, float) and measurement.is_integer():
                measurement_text = f"{int(measurement)} {measurement_unit}"
            else:
                measurement_text = f"{measurement} {measurement_unit}"

            # Add measurement label
            ax.text(
                label_x,
                label_y,
                measurement_text,
                fontsize=16,
                ha="center",
                va="center",  # Same font size as vertex labels
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )


def calculate_polygon_positions(
    original_coords: List[Tuple[float, float]], scaled_coords: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Calculate display positions for the two polygons side by side.
    Returns lists of (x, y) coordinates for drawing.
    """
    # Calculate bounds of original polygon
    orig_x_coords = [pos[0] for pos in original_coords]
    orig_y_coords = [pos[1] for pos in original_coords]
    orig_min_x, orig_max_x = min(orig_x_coords), max(orig_x_coords)
    orig_min_y = min(orig_y_coords)
    orig_width = orig_max_x - orig_min_x

    # Calculate bounds of scaled polygon
    scaled_x_coords = [pos[0] for pos in scaled_coords]
    scaled_y_coords = [pos[1] for pos in scaled_coords]
    scaled_min_x = min(scaled_x_coords)
    scaled_min_y = min(scaled_y_coords)

    # Position original polygon on the left
    left_margin = 2
    bottom_margin = 2

    # Translate original polygon to the left
    orig_offset_x = left_margin - orig_min_x
    orig_offset_y = bottom_margin - orig_min_y

    original_positions = []
    for pos in original_coords:
        original_positions.append((pos[0] + orig_offset_x, pos[1] + orig_offset_y))

    # Position scaled polygon to the right with some spacing
    spacing = 7  # Further increased spacing between polygons for measurement labels
    scaled_start_x = left_margin + orig_width + spacing

    # Translate scaled polygon
    scaled_offset_x = scaled_start_x - scaled_min_x
    scaled_offset_y = bottom_margin - scaled_min_y

    scaled_positions = []
    for pos in scaled_coords:
        scaled_positions.append((pos[0] + scaled_offset_x, pos[1] + scaled_offset_y))

    return original_positions, scaled_positions


@stimulus_function
def draw_polygon_scale(data: PolygonScale):
    """
    Draw two scaled polygons side by side with measurements and vertex labels.
    Original and scaled polygons can show different numbers of side measurements.
    """
    fig, ax = plt.subplots(
        figsize=(16, 10)
    )  # Increased figure size to prevent compression

    # Generate coordinates for both polygons using all available measurements
    original_coords = generate_polygon_coordinates(
        data.polygon_type, data.original_vertex_labels, data.original_measurements
    )

    # Scale the coordinates for the scaled polygon
    scaled_coords = [
        (x * data.scale_factor, y * data.scale_factor) for x, y in original_coords
    ]

    # Calculate display positions for optimal layout
    original_positions, scaled_positions = calculate_polygon_positions(
        original_coords, scaled_coords
    )

    # Extract measurements for only the visible sides
    original_display_measurements = get_measurements_for_visible_sides(
        data.original_visible_sides,
        data.original_measurements,
        data.original_vertex_labels,
    )

    scaled_display_measurements = get_measurements_for_visible_sides(
        data.scaled_visible_sides, data.scaled_measurements, data.scaled_vertex_labels
    )

    # Draw both polygons with their respective visible measurements
    draw_polygon_with_measurements(
        ax,
        data.original_polygon_label,
        data.original_vertex_labels,
        original_positions,
        "green",
        data.original_visible_sides,
        original_display_measurements,
        data.measurement_unit,
    )

    draw_polygon_with_measurements(
        ax,
        data.scaled_polygon_label,
        data.scaled_vertex_labels,
        scaled_positions,
        "blue",
        data.scaled_visible_sides,
        scaled_display_measurements,
        data.measurement_unit,
    )

    # Calculate plot bounds
    all_x = []
    all_y = []
    for pos in original_positions + scaled_positions:
        all_x.append(pos[0])
        all_y.append(pos[1])

    margin = 3.0  # Adjusted margin for measurement labels at 1.5 offset
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(
        min(all_y) - margin - 5.0, max(all_y) + margin
    )  # Extra space for polygon labels at -4.5

    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")  # Remove axes and numberings

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/polygon_scale_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

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
