import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.drawing_functions.common.text_helper import (
    TextHelper,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.angle_diagram import (
    AngleDiagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.angles import (
    AngleList,
    SingleAngle,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.fractional_angle import (
    FractionalAngle,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.parallel_lines_transversal import (
    ParallelLinesTransversal,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.points_and_lines import (
    PointsAndLines,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.transversal import (
    TransversalAngleParams,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.triangle import (
    RightTriangleWithRay,
    TriangleStimulusDescription,
)
from content_generators.settings import settings
from matplotlib.patches import FancyArrowPatch


@stimulus_function
def generate_angles(data: TriangleStimulusDescription):
    points = data.triangle.points
    angles = {angle.vertex: angle.measure for angle in data.triangle.angles}

    sorted_points = sorted(
        points, key=lambda p: len(str(angles[p.label])), reverse=True
    )
    point_labels = [point.label for point in sorted_points]
    coordinates = {
        point_labels[0]: (0, 0),
        point_labels[1]: (1, 0),
        point_labels[2]: (0.5, 0.5),
    }

    fig, ax = plt.subplots()

    x_coords, y_coords = zip(*coordinates.values())

    ax.plot(x_coords + (x_coords[0],), y_coords + (y_coords[0],), "k-")

    for label, (x, y) in coordinates.items():
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="black",
            fontsize=16,
        )

    y_text_bottom_padding = 0.02
    y_text_top_padding = 0.1
    x_padding = 0.05

    for label, (x, y) in coordinates.items():
        angle_measure = (
            f"{angles[label]}°"
            if not str(angles[label]).endswith("°")
            else str(angles[label])
        )

        if x == 0 and y == 0:
            angle_label_x = x + x_padding
            angle_label_y = y + y_text_bottom_padding
        elif x == 1 and y == 0:
            angle_label_x = (
                x
                - x_padding
                - len(angle_measure) * (0.023 if len(angle_measure) < 5 else 0.02)
            )
            angle_label_y = y + y_text_bottom_padding
        else:
            angle_label_x = 0.47 if len(angle_measure) < 4 else 0.45
            angle_label_y = y - y_text_top_padding

        ax.annotate(
            angle_measure, (angle_label_x, angle_label_y), color="black", fontsize=16
        )

    padding_buffer = 0.05
    min_x, max_x = min(x_coords) - padding_buffer, max(x_coords) + padding_buffer
    min_y, max_y = min(y_coords) - padding_buffer, max(y_coords) + padding_buffer

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_axis_off()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout(pad=1.2)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/angle_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    # plt.show()

    return file_name


def line_intersection(p1, p2, p3, p4):
    """Finds the intersection of two lines (p1p2 and p3p4) if it exists"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # Lines are parallel

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if 0 <= ua <= 1:
        intersection_x = x1 + ua * (x2 - x1)
        intersection_y = y1 + ua * (y2 - y1)
        return np.array([intersection_x, intersection_y])
    return None


@stimulus_function
def draw_triangle_with_ray(data: RightTriangleWithRay):
    triangle = data.triangle
    # Extract points and angles
    points = {point.label: None for point in triangle.points}
    angles = {angle.vertex: angle.measure for angle in triangle.angles}
    rays = triangle.rays

    # Identify the right angle vertex
    right_angle_vertex = [
        vertex for vertex, measure in angles.items() if int(measure) == 90
    ][0]
    other_vertices = [
        vertex for vertex in points.keys() if vertex != right_angle_vertex
    ]

    points[right_angle_vertex] = np.array([0, 0])  # type: ignore
    points[other_vertices[0]] = np.array([1, 0])  # type: ignore
    points[other_vertices[1]] = np.array([0, 1])  # type: ignore

    # Create the triangle
    triangle_points = np.array(
        [points[vertex] for vertex in points] + [points[list(points.keys())[0]]]
    )

    plt.figure()
    plt.plot(triangle_points[:, 0], triangle_points[:, 1], "b-")

    # Label the points
    for label, coord in points.items():
        if coord is not None and coord[1] == 0:
            plt.text(coord[0], coord[1] - 0.05, f" {label}", fontsize=12, ha="right")
        elif coord is not None:
            plt.text(coord[0], coord[1] + 0.03, f" {label}", fontsize=12, ha="right")

    # Plot the ray if present
    for ray in rays:
        start_label = ray.start_label
        measure = int(ray.measures[0])

        start_point = points[start_label]

        if start_point is None:
            continue

        # Calculate the direction of the ray
        angle_rad = np.deg2rad(measure)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])

        # Extend the ray from the starting point
        ray_end = (
            start_point + direction * 10
        )  # Arbitrarily large value to ensure intersection

        # Check intersections with the triangle sides
        intersections = []
        for i in range(3):
            p1, p2 = triangle_points[i], triangle_points[i + 1]
            intersection = line_intersection(start_point, ray_end, p1, p2)
            if intersection is not None and intersection[0] > 0 and intersection[1] > 0:
                intersections.append(intersection)

        if intersections:
            closest_intersection = min(
                intersections,
                key=lambda point: np.linalg.norm(point - start_point),  # type: ignore
            )
            plt.plot(
                [start_point[0], closest_intersection[0]],
                [start_point[1], closest_intersection[1]],
                "b-",
            )

            # Label the ray with the angle measurement near the start point
            plt.text(
                start_point[0] + 0.02,
                start_point[1] + 0.15,
                ray.measures[1],  # type: ignore
                fontsize=12,
                color="blue",
                ha="left",
            )
            if measure < 45:
                plt.text(
                    start_point[0] + 0.12,
                    start_point[1] + 0.02,
                    f"{measure}°",
                    fontsize=12,
                    color="blue",
                    ha="left",
                )
            else:
                plt.text(
                    start_point[0] + 0.04,
                    start_point[1] + 0.02,
                    f"{measure}°",
                    fontsize=12,
                    color="blue",
                    ha="left",
                )

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.box(False)
    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/angle_rays_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def generate_angle_types(angles: AngleList):
    fig, ax = plt.subplots()

    line_length = 40
    offset = 50
    spacing = 200
    width = 0.004

    for index, angle_info in enumerate(angles):
        vertex_x = offset + index * spacing
        vertex_y = 0

        end_x1 = vertex_x + line_length
        end_y1 = vertex_y

        end_x2 = vertex_x + line_length * np.cos(np.radians(angle_info.measure))
        end_y2 = vertex_y + line_length * np.sin(np.radians(angle_info.measure))

        ax.quiver(
            vertex_x,
            vertex_y,
            end_x1 - vertex_x,
            end_y1 - vertex_y,
            angles="xy",
            scale_units="xy",
            scale=0.4,
            color="black",
            headwidth=3,
            headlength=5,
            linewidth=2,
            width=width,
        )
        ax.quiver(
            vertex_x,
            vertex_y,
            end_x2 - vertex_x,
            end_y2 - vertex_y,
            angles="xy",
            scale_units="xy",
            scale=0.4,
            color="black",
            headwidth=3,
            headlength=5,
            linewidth=2,
            width=width,
        )

        ax.text(
            vertex_x,
            -25,
            angle_info.label,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=20,
        )

        if angle_info.measure == 90:
            side_length = 10
            square = patches.Rectangle(
                (vertex_x, vertex_y),
                side_length,
                side_length,
                color="black",
                fill=False,
            )
            ax.add_patch(square)
        else:
            angle_radius = 20
            angle_arc = patches.Arc(
                (vertex_x, vertex_y),
                angle_radius,
                angle_radius,
                theta1=0,
                theta2=angle_info.measure,
                color="black",
                linewidth=2,
            )
            ax.add_patch(angle_arc)

    ax.set_xlim(-50, offset + len(angles) * spacing - 100)
    ax.set_ylim(-30, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_angle.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=600,
    )
    plt.close()
    # plt.show()

    return file_name


@stimulus_function
def generate_single_angle_type(single_angle: SingleAngle):
    """Generate a visual representation of a single angle using rays."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Parameters for drawing the angle
    ray_length = 2.5  # Reduced length for tighter bounds
    vertex_x = 0  # Keep vertex at origin
    vertex_y = 0
    line_thickness = 3  # Increased from 2

    angle_measure = single_angle.measure

    # Calculate endpoints for the two rays
    # First ray along positive x-axis
    end_x1 = vertex_x + ray_length
    end_y1 = vertex_y

    # Second ray at the specified angle
    end_x2 = vertex_x + ray_length * np.cos(np.radians(angle_measure))
    end_y2 = vertex_y + ray_length * np.sin(np.radians(angle_measure))

    # Draw the two rays with arrows at the ends
    # First ray (along positive x-axis)
    ray1 = FancyArrowPatch(
        (vertex_x, vertex_y),
        (end_x1, end_y1),
        arrowstyle="->",
        mutation_scale=25,  # Slightly larger arrow heads
        linewidth=line_thickness,
        color="black",
    )
    ax.add_patch(ray1)

    # Second ray (at the specified angle)
    ray2 = FancyArrowPatch(
        (vertex_x, vertex_y),
        (end_x2, end_y2),
        arrowstyle="->",
        mutation_scale=25,
        linewidth=line_thickness,
        color="black",
    )
    ax.add_patch(ray2)

    # Add vertex point
    ax.plot(vertex_x, vertex_y, "ko", markersize=8)  # Slightly larger point

    # Add angle marking (arc or square for right angle)
    if angle_measure == 90:
        # Draw a small square for right angles
        side_length = 0.25
        square = patches.Rectangle(
            (vertex_x, vertex_y),
            side_length,
            side_length,
            color="black",
            fill=False,
            linewidth=line_thickness,
        )
        ax.add_patch(square)
    else:
        # Draw an arc for other angles
        arc_radius = 0.4
        angle_arc = patches.Arc(
            (vertex_x, vertex_y),
            2 * arc_radius,
            2 * arc_radius,
            theta1=0,
            theta2=angle_measure,
            color="black",
            linewidth=line_thickness,
        )
        ax.add_patch(angle_arc)

    # Calculate dynamic bounds based on angle geometry
    all_x_coords = [vertex_x, end_x1, end_x2]
    all_y_coords = [vertex_y, end_y1, end_y2]

    # Include arc extent in bounds calculation
    if angle_measure != 90:
        arc_extent = arc_radius
        # Add arc boundary points
        for angle_deg in range(
            0, int(angle_measure) + 1, max(1, int(angle_measure) // 8)
        ):
            arc_x = vertex_x + arc_extent * np.cos(np.radians(angle_deg))
            arc_y = vertex_y + arc_extent * np.sin(np.radians(angle_deg))
            all_x_coords.append(arc_x)
            all_y_coords.append(arc_y)

    # Calculate tight bounds
    min_x, max_x = min(all_x_coords), max(all_x_coords)
    min_y, max_y = min(all_y_coords), max(all_y_coords)

    # Add minimal padding - just enough for arrow heads and clarity
    padding = 0.3

    # Ensure minimum size for very small angles
    width = max_x - min_x
    height = max_y - min_y
    min_size = 1.0

    if width < min_size:
        center_x = (min_x + max_x) / 2
        min_x = center_x - min_size / 2
        max_x = center_x + min_size / 2

    if height < min_size:
        center_y = (min_y + max_y) / 2
        min_y = center_y - min_size / 2
        max_y = center_y + min_size / 2

    # Set dynamic axis limits
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    # Save the file
    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_single_angle.{settings.additional_content_settings.stimulus_image_format}"
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
def generate_angle_diagram(stimulus_description: AngleDiagram):
    return generate_angle_diagram_internal(stimulus_description)


@stimulus_function
def generate_angle_diagram_360(stimulus_description: AngleDiagram):
    return generate_angle_diagram_internal(stimulus_description, True)


def generate_angle_diagram_internal(
    stimulus_description: AngleDiagram, use_right_angle_marker: bool = False
):
    angles_info = stimulus_description.diagram.angles
    adjust_points = stimulus_description.adjust_points

    # Don't sort - use angles in the order they appear
    # angles_info.sort(key=lambda x: x.get_numeric_measure(), reverse=True)

    # Validate that all angles share a common central point
    adjusted_points = [adjust_points(angle.points) for angle in angles_info]
    center_points = [points[1] for points in adjusted_points]
    if len(set(center_points)) != 1:
        raise ValueError("Angles must share a common central point")

    # Simple sequential approach: place angles in order they appear
    coords = {}
    center_point = adjust_points(angles_info[0].points)[1]  # Get center point
    coords[center_point] = np.array([0, 0])  # Center point at origin

    current_angle = 0  # Start at 0 degrees (positive x-axis)
    radius = 1.0

    # Place angles sequentially in the order they appear
    for angle in angles_info:
        p1, p2, p3 = adjust_points(angle.points)
        measure = angle.get_numeric_measure()

        # Place p1 at current_angle if not already placed
        if p1 not in coords:
            coords[p1] = np.array(
                [
                    radius * np.cos(np.deg2rad(current_angle)),
                    radius * np.sin(np.deg2rad(current_angle)),
                ]
            )

        # Place p3 at current_angle + measure
        p3_angle = current_angle + measure
        coords[p3] = np.array(
            [
                radius * np.cos(np.deg2rad(p3_angle)),
                radius * np.sin(np.deg2rad(p3_angle)),
            ]
        )

        # Next angle starts where this one ended
        current_angle = p3_angle

    # Debug: Print final coordinates for verification
    # print(f"Final coordinates: {coords}")
    # for angle in angles_info:
    #     p1, p2, p3 = adjust_points(angle.points)
    #     if all(key in coords for key in [p1, p2, p3]):
    #         v1 = coords[p1] - coords[p2]
    #         v2 = coords[p3] - coords[p2]
    #         angle_calc = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
    #         print(f"Angle {p1}-{p2}-{p3}: calculated={np.degrees(angle_calc):.1f}°, expected={angle.get_numeric_measure()}°")

    fig, ax = plt.subplots()

    # Draw lines (regular lines instead of arrows)
    line_width = 3
    for angle in angles_info:
        p1, p2, p3 = adjust_points(angle.points)
        if all(key in coords for key in [p1, p2, p3]):
            # Draw line from center to p1
            ax.plot(
                [coords[p2][0], coords[p1][0]],
                [coords[p2][1], coords[p1][1]],
                color="black",
                linewidth=line_width,
            )
            # Draw line from center to p3
            ax.plot(
                [coords[p2][0], coords[p3][0]],
                [coords[p2][1], coords[p3][1]],
                color="black",
                linewidth=line_width,
            )

    # Calculate smart label positions to avoid line and arc overlaps
    def get_smart_label_position(point, coord, coords, angles_info, adjust_points):
        """Calculate a label position that avoids overlapping with lines and arcs from this point."""
        # Find all rays emanating from this point
        ray_directions = []
        for other_point, other_coord in coords.items():
            if other_point != point:
                direction = other_coord - coord
                if np.linalg.norm(direction) > 0:
                    ray_directions.append(direction / np.linalg.norm(direction))

        # If no rays (shouldn't happen), place label above
        if not ray_directions:
            return coord[0], coord[1] + 0.15

        # Check if this is the center point (appears in all angles)
        is_center_point = True
        arc_mid_angles = []

        for angle in angles_info:
            p1, p2, p3 = adjust_points(angle.points)
            if p2 != point:  # This point is not the center of this angle
                is_center_point = False
                break

        # If this is the center point, calculate arc positions to avoid
        if is_center_point:
            for angle in angles_info:
                p1, p2, p3 = adjust_points(angle.points)
                if all(key in coords for key in [p1, p2, p3]):
                    # Calculate the mid-angle of this arc
                    v1 = coords[p1] - coords[p2]
                    v2 = coords[p3] - coords[p2]
                    angle1 = np.arctan2(v1[1], v1[0])
                    angle2 = np.arctan2(v2[1], v2[0])

                    # Ensure angle2 > angle1 for consistent arc direction
                    if angle2 < angle1:
                        angle2 += 2 * np.pi

                    # Calculate mid-angle of the arc
                    mid_angle = (angle1 + angle2) / 2
                    arc_mid_angles.append(mid_angle)

        # Try different label positions and pick the one farthest from all rays and arcs
        label_distance = 0.15  # Closer to points (50% reduction from 0.2)
        candidate_angles = np.linspace(
            0, 2 * np.pi, 24
        )  # More positions for better avoidance
        best_angle = 0
        best_min_separation = -1

        for candidate_angle in candidate_angles:
            label_direction = np.array(
                [np.cos(candidate_angle), np.sin(candidate_angle)]
            )
            min_separation = float("inf")

            # Calculate separation from all rays
            for ray_dir in ray_directions:
                dot_product = np.clip(np.dot(label_direction, ray_dir), -1, 1)
                separation = np.arccos(
                    abs(dot_product)
                )  # Use abs to consider both directions
                min_separation = min(min_separation, separation)

            # For center points, also avoid arc positions
            if is_center_point:
                for arc_angle in arc_mid_angles:
                    # Calculate angular distance to arc center
                    arc_separation = abs(candidate_angle - arc_angle)
                    # Handle wraparound (e.g., angle close to 0 and 2π)
                    if arc_separation > np.pi:
                        arc_separation = 2 * np.pi - arc_separation
                    # Avoid being too close to arc centers (within 45 degrees)
                    if arc_separation < np.deg2rad(45):
                        min_separation = min(min_separation, arc_separation)

            # Pick the position with the largest minimum separation
            if min_separation > best_min_separation:
                best_min_separation = min_separation
                best_angle = candidate_angle

        # Position label in the best direction
        label_x = coord[0] + label_distance * np.cos(best_angle)
        label_y = coord[1] + label_distance * np.sin(best_angle)
        return label_x, label_y

    def scale_rectangle(p1, p2, p3, scale_factor):
        """
        Given 3 corner points of a rectangle and a scale factor,
        returns the 4 corners of the scaled rectangle starting from p1.

        Parameters:
            p1, p2, p3 (tuple): Coordinates of 3 corners (x, y)
            scale_factor (float): Scale factor for the new rectangle

        Returns:
            List[tuple]: List of 4 points representing the smaller rectangle
        """
        # Vectors from p1 to other corners
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p1[0], p3[1] - p1[1])

        # Scale the vectors
        v1_scaled = (v1[0] * scale_factor, v1[1] * scale_factor)
        v2_scaled = (v2[0] * scale_factor, v2[1] * scale_factor)

        # Compute new rectangle points
        p1_new = p1
        p2_new = (p1_new[0] + v1_scaled[0], p1_new[1] + v1_scaled[1])
        p3_new = (p1_new[0] + v2_scaled[0], p1_new[1] + v2_scaled[1])
        p4_new = (
            p1_new[0] + v1_scaled[0] + v2_scaled[0],
            p1_new[1] + v1_scaled[1] + v2_scaled[1],
        )

        return np.array(
            [p1_new, p2_new, p4_new, p3_new]
        )  # Returning in rectangle order

    # Draw points and smart-positioned labels
    for point, coord in coords.items():
        ax.plot(coord[0], coord[1], "o", color="black", markersize=7)

        # Calculate smart label position
        label_x, label_y = get_smart_label_position(
            point, coord, coords, angles_info, adjust_points
        )

        ax.text(
            label_x,
            label_y,
            f"{point}",
            color="black",
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=24,
        )

    # Draw angle arcs and angle measure labels
    arc_radius = 0.28
    base_arc_label_radius = 0.38
    for idx, angle in enumerate(angles_info):
        p1, p2, p3 = adjust_points(angle.points)
        if all(key in coords for key in [p1, p2, p3]):
            v1 = coords[p1] - coords[p2]
            v2 = coords[p3] - coords[p2]
            angle1 = np.arctan2(v1[1], v1[0])
            angle2 = np.arctan2(v2[1], v2[0])
            # Ensure angle2 > angle1 for arc drawing
            if angle2 < angle1:
                angle2 += 2 * np.pi
            if use_right_angle_marker and angle.get_numeric_measure() == 90:
                scale = 0.14
                right_angle_marker = scale_rectangle(
                    coords[p2], coords[p1], coords[p3], scale
                )

                ax.add_patch(
                    patches.Polygon(
                        right_angle_marker,
                        closed=True,
                        fill=None,
                        edgecolor="blue",
                        linewidth=3,
                    )
                )
            elif not use_right_angle_marker:
                arc = patches.Arc(
                    coords[p2],
                    arc_radius,
                    arc_radius,
                    angle=0,
                    theta1=np.degrees(angle1),
                    theta2=np.degrees(angle2),
                    color="black",
                    lw=3,
                )
                ax.add_patch(arc)

            # Adjust label radius for small angles to avoid crowding
            angle_measure = angle.get_numeric_measure()
            if angle_measure < 40:
                arc_label_radius = (
                    base_arc_label_radius + 0.25
                )  # Move further out for small angles
            else:
                arc_label_radius = base_arc_label_radius

            # Place angle label at the middle of the arc
            mid_angle = (angle1 + angle2) / 2
            label_x = coords[p2][0] + arc_label_radius * np.cos(mid_angle)
            label_y = coords[p2][1] + arc_label_radius * np.sin(mid_angle)
            ax.text(
                label_x,
                label_y,
                angle.get_display_text(),
                color="black",
                fontsize=20,
                fontweight="bold",
                ha="center",
                va="center",
            )

    ax.axis("off")
    ax.axis("equal")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_angle.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=1200,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    return file_name


def validate_lines_and_angles(input_data):
    # Extract lines and angles from input data
    lines = input_data["lines"]
    angles = input_data["angles"]

    # Extract points from lines
    line1, line2 = lines
    middle_point = set(line1).intersection(set(line2))

    # Validation 1: Check for a common middle point and 4 distinct edge points
    if len(middle_point) != 1:
        raise ValueError("No Single common middle point")

    middle_point = middle_point.pop()
    edge_points = set(line1 + line2) - {middle_point}

    if len(edge_points) != 4:
        raise ValueError("No Distinct 4 edge points.")

    # Validation 2: Check if the sum of all angles is 360 degrees
    total_angle = sum(angle["measure"] for angle in angles)

    if total_angle != 360:
        raise ValueError("Sum of angles is not 360")

    def check_supplementary_angles(start_point, middle_point, end_point, end_point2):
        angle1 = None
        angle2 = None
        for angle in angles:
            points = set(angle["points"])
            if {start_point, middle_point, end_point} == points:
                angle1 = angle["measure"]
            elif {start_point, middle_point, end_point2} == points:
                angle2 = angle["measure"]

        if angle1 is None or angle2 is None:
            raise ValueError("Required angles for new validation not found")

        if angle1 + angle2 != 180:
            raise ValueError(f"Angles {angle1} and {angle2} do not sum to 180 degrees")

    check_supplementary_angles(line1[0], middle_point, line2[0], line2[2])
    check_supplementary_angles(line1[2], middle_point, line2[0], line2[2])

    check_supplementary_angles(line2[0], middle_point, line1[0], line1[2])
    check_supplementary_angles(line2[2], middle_point, line1[0], line1[2])


def generate_angle_supplementary(data):
    validate_lines_and_angles(data)
    cleaned_data = data  # process_angle_data(data)
    lines = cleaned_data["lines"]
    angle_info = cleaned_data["angles"][0]

    center_point = angle_info["points"][1]
    angle_degrees = angle_info["measure"]
    angle_radians = np.radians(angle_degrees)

    points = {center_point: np.array([0, 0])}
    point_distances = 2

    p1, p2 = angle_info["points"][0], angle_info["points"][2]

    first_line = next(line for line in lines if center_point in line)
    left_point = first_line[0] if first_line[1] == center_point else first_line[2]
    right_point = first_line[2] if first_line[1] == center_point else first_line[0]
    points[left_point] = np.array([-point_distances, 0])
    points[right_point] = np.array([point_distances, 0])

    second_line = next(line for line in lines if line != first_line)
    other_left = second_line[0] if second_line[1] == center_point else second_line[2]
    other_right = second_line[2] if second_line[1] == center_point else second_line[0]

    if p1 == left_point or p2 == right_point:
        angle = angle_radians
    else:
        angle = -angle_radians

    if other_left == second_line[0]:
        points[other_left] = points[center_point] + np.array(
            [point_distances * np.cos(angle), point_distances * np.sin(angle)]
        )
        points[other_right] = points[center_point] - np.array(
            [point_distances * np.cos(angle), point_distances * np.sin(angle)]
        )
    else:
        points[other_right] = points[center_point] + np.array(
            [point_distances * np.cos(angle), point_distances * np.sin(angle)]
        )
        points[other_left] = points[center_point] - np.array(
            [point_distances * np.cos(angle), point_distances * np.sin(angle)]
        )

    fig, ax = plt.subplots()
    for line in lines:
        x_vals = [points[pt][0] for pt in line]
        y_vals = [points[pt][1] for pt in line]
        ax.plot(x_vals, y_vals, "o-", color="black")
        for pt in line:
            label_offset = (
                np.array([0.2, -0.2]) if pt == center_point else np.array([0.2, 0.2])
            )
            ax.text(
                points[pt][0] + label_offset[0],
                points[pt][1] + label_offset[1],
                pt,
                fontsize=12,
                color="black",
                horizontalalignment="center",
                verticalalignment="center",
            )

    ax.set_aspect("equal")
    ax.axis("off")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/angle_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()
    # plt.show()

    return file_name


def is_collinear(p1, p2, p3):
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) == (p2[1] - p1[1]) * (p3[0] - p1[0])


def on_segment(p1, p2, p):
    return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[
        1
    ] <= max(p1[1], p2[1])


def validate_input(data: PointsAndLines):
    points = {point.label: point.coordinates for point in data.points}
    lines = data.lines

    all_rays = []
    all_segments = []
    for line in lines:
        if line.type == "ray":
            all_rays.append((line.start_label, line.end_label))
        elif line.type == "line":
            all_rays.append((line.start_label, line.end_label))
            all_rays.append((line.end_label, line.start_label))
        elif line.type == "segment":
            all_segments.append((line.start_label, line.end_label))

    for start1, end1 in all_segments:
        p1, p2 = points[start1], points[end1]

        for start2, end2 in all_rays:
            if (start1 == start2 and end1 == end2) or (
                start1 == end2 and end1 == start2
            ):
                continue
            p3, p4 = points[start2], points[end2]

            if is_collinear(p1, p2, p3) and is_collinear(p1, p2, p4):
                # Check if p3 lies on the segment p1p2 (including endpoints)
                if on_segment(p1, p2, p3):
                    raise ValueError(
                        f"Invalid input: Line segment {start1}{end1} and ray {start2}{end2} intersect at collinear points {start1}, {end1}, {start2}."
                    )

    return True


@stimulus_function
def draw_lines_rays_and_segments(data: PointsAndLines):
    validate_input(data)
    points = {point.label: np.array(point.coordinates) for point in data.points}
    fig, ax = plt.subplots()

    ax.axis("off")

    for label, coords in points.items():
        ax.plot(coords[0], coords[1], "o", color="black")
        ax.text(coords[0] + 0.4, coords[1] + 0.25, label, fontsize=13, ha="right")

    # Plot lines
    for line in data.lines:
        start_point = points[line.start_label]
        end_point = points[line.end_label]
        direction = end_point - start_point
        unit_direction = direction / np.linalg.norm(direction)

        if line.type == "segment":
            ax.plot(
                [start_point[0], end_point[0]], [start_point[1], end_point[1]], "k-"
            )
        elif line.type == "ray" or line.type == "line":
            ray_end = end_point + unit_direction
            ax.arrow(
                start_point[0],
                start_point[1],
                ray_end[0] - start_point[0],
                ray_end[1] - start_point[1],
                head_width=0.2,
                head_length=0.3,
                fc="k",
                ec="k",
                length_includes_head=True,
            )

        if line.type == "line":
            direction = start_point - end_point
            unit_direction = direction / np.linalg.norm(direction)
            line_end = start_point + unit_direction
            ax.arrow(
                end_point[0],
                end_point[1],
                line_end[0] - end_point[0],
                line_end[1] - end_point[1],
                head_width=0.2,
                head_length=0.3,
                fc="k",
                ec="k",
                length_includes_head=True,
            )

    ax.set_aspect("equal")
    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/lines_rays_and_segments_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_parallel_lines_cut_by_transversal(input_data: ParallelLinesTransversal):
    fig, ax = plt.subplots()

    # Extract angle and labels from input
    angle = 180 - input_data.angle
    top_left_angle_label = input_data.top_line_top_left_angle_label
    bottom_left_angle_label = input_data.bottom_line_top_left_angle_label
    top_right_angle_label = input_data.top_line_top_right_angle_label
    bottom_right_angle_label = input_data.bottom_line_top_right_angle_label

    # Define the offset for moving the parallel lines up by 25%
    vertical_offset = 0.5

    # Calculate the length of the transversal line
    trans_length = 2 / np.sin(np.radians(angle))  # Length of the transversal line

    # Ensure the minimum length is 3
    line_length = max(trans_length, 3)

    # Calculate the midpoint for centering the lines
    midpoint_x = 1.75  # Assuming the image width is 3 units
    midpoint_y = 0.5 + vertical_offset  # Centered vertically between the parallel lines

    # Define the coordinates for the parallel lines with the adjusted length
    line1_x = np.array([1.5 - line_length / 2, 1.5 + line_length / 2])
    line1_y = np.array([1 + vertical_offset, 1 + vertical_offset])
    line2_x = line1_x  # Same x-coordinates for the second parallel line
    line2_y = np.array([0 + vertical_offset, 0 + vertical_offset])

    trans_length = 2 / np.sin(np.radians(angle))  # Length of the transversal line
    trans_x = np.array(
        [
            midpoint_x - (trans_length / 2) * np.cos(np.radians(angle)),
            midpoint_x + (trans_length / 2) * np.cos(np.radians(angle)),
        ]
    )
    trans_y = np.array(
        [
            midpoint_y - (trans_length / 2) * np.sin(np.radians(angle)),
            midpoint_y + (trans_length / 2) * np.sin(np.radians(angle)),
        ]
    )

    # Plot the parallel lines with arrows
    ax.plot(
        line1_x,
        line1_y,
        color="black",
        marker=">",
        linewidth=3,
        markevery=[1],
        markersize=8,
    )
    ax.plot(
        line1_x,
        line1_y,
        color="black",
        marker="<",
        linewidth=3,
        markevery=[0],
        markersize=8,
    )
    ax.plot(
        line2_x,
        line2_y,
        color="black",
        marker=">",
        linewidth=3,
        markevery=[1],
        markersize=8,
    )
    ax.plot(
        line2_x,
        line2_y,
        color="black",
        marker="<",
        linewidth=3,
        markevery=[0],
        markersize=8,
    )

    # Plot the transversal line with arrows
    ax.plot(
        trans_x,
        trans_y,
        linestyle="-",
        color="black",
        linewidth=3,
    )
    ax.plot(
        trans_x,
        trans_y,
        linestyle="-",
        color="black",
        linewidth=3,
    )

    # Calculate the direction vector for the transversal line
    direction_x = trans_x[1] - trans_x[0]
    direction_y = trans_y[1] - trans_y[0]

    # Normalize the direction vector for consistent arrow size
    norm = np.sqrt(direction_x**2 + direction_y**2)
    direction_x /= norm
    direction_y /= norm
    direction_x *= 0.1
    direction_y *= 0.1

    # Plot the transversal line with arrows using quiver
    ax.quiver(
        trans_x[0],
        trans_y[0],
        -direction_x,
        -direction_y,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.005,
        headwidth=4.5,
        headlength=4.5,
    )
    ax.quiver(
        trans_x[1],
        trans_y[1],
        direction_x,
        direction_y,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="black",
        width=0.005,
        headwidth=4.5,
        headlength=4.5,
    )
    text_helper = TextHelper(font_size=22, dpi=600)

    # Add labels for the lines
    ax.text(
        line1_x[1] + 0.05,
        line1_y[1],
        input_data.top_line_label,
        fontsize=text_helper.font_size,
        verticalalignment="center",
        horizontalalignment="left",
    )
    ax.text(
        line2_x[1] + 0.05,
        line2_y[1],
        input_data.bottom_line_label,
        fontsize=text_helper.font_size,
        verticalalignment="center",
        horizontalalignment="left",
    )
    ax.text(
        trans_x[0] + (-0.15 if input_data.angle < 90 else +0.15),
        trans_y[0],
        input_data.transversal_line_label,
        fontsize=text_helper.font_size,
        verticalalignment="center",
        horizontalalignment="right" if input_data.angle < 90 else "left",
    )

    def calculate_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        """Calculate the intersection point of two lines (x1, y1) to (x2, y2) and (x3, y3) to (x4, y4)."""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Lines are parallel or coincident
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * y4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return px, py

    # Example coordinates for the horizontal lines
    horizontal_line1_start = (line1_x[0], line1_y[0])
    horizontal_line1_end = (line1_x[1], line1_y[0])
    horizontal_line2_start = (line2_x[0], line2_y[0])
    horizontal_line2_end = (line2_x[1], line2_y[0])

    # Example coordinates for the transversal line
    transversal_line_start = (trans_x[0], trans_y[0])
    transversal_line_end = (trans_x[1], trans_y[1])

    # Calculate intersection points
    intersection1 = calculate_intersection(
        *horizontal_line1_start,
        *horizontal_line1_end,
        *transversal_line_start,
        *transversal_line_end,
    )
    intersection2 = calculate_intersection(
        *horizontal_line2_start,
        *horizontal_line2_end,
        *transversal_line_start,
        *transversal_line_end,
    )

    if intersection1:
        inter1_x, inter1_y = intersection1
        # Get the text height using TextHelper
        if top_left_angle_label:
            text_height = text_helper.get_text_height(top_left_angle_label)
            # Calculate the offset based on the angle and text height
            if angle <= 100:
                offset_x = 0.05
            else:
                offset_x = text_height * np.cos(np.radians(180 - angle))
            ax.text(
                inter1_x - offset_x,  # Adjust x based on angle and text height
                inter1_y + 0.01,  # Adjust y based on angle and text height
                f"{top_left_angle_label}",
                fontsize=text_helper.font_size,
                verticalalignment="bottom",
                horizontalalignment="right",
            )
        if top_right_angle_label:
            text_height = text_helper.get_text_height(top_right_angle_label)
            # Calculate the offset based on the angle and text height
            if angle > 100:
                offset_x = -0.05
            else:
                offset_x = text_height * np.cos(np.radians(angle))
            # Add label for the top right angle
            ax.text(
                inter1_x + offset_x,  # Adjust x for the right side
                inter1_y + 0.01,  # Adjust y based on angle and text height
                f"{top_right_angle_label}",
                fontsize=text_helper.font_size,
                verticalalignment="bottom",
                horizontalalignment="left",
            )

    if intersection2:
        inter2_x, inter2_y = intersection2
        if bottom_left_angle_label:
            # Get the text height using TextHelper
            text_height = text_helper.get_text_height(bottom_left_angle_label)
            # Calculate the offset based on the angle and text height
            if angle <= 100:
                offset_x = 0.05
            else:
                offset_x = text_height * np.cos(np.radians(180 - angle))
            ax.text(
                inter2_x - offset_x,  # Adjust x based on angle and text height
                inter2_y + 0.01,  # Adjust y based on angle and text height
                f"{bottom_left_angle_label}",
                fontsize=text_helper.font_size,
                verticalalignment="bottom",
                horizontalalignment="right",
            )
        if bottom_right_angle_label:
            text_height = text_helper.get_text_height(bottom_right_angle_label)
            # Calculate the offset based on the angle and text height
            if angle > 100:
                offset_x = -0.05
            else:
                offset_x = text_height * np.cos(np.radians(angle))
            # Add label for the bottom right angle
            ax.text(
                inter2_x + offset_x,  # Adjust x for the right side
                inter2_y + 0.01,  # Adjust y based on angle and text height
                f"{bottom_right_angle_label}",
                fontsize=text_helper.font_size,
                verticalalignment="bottom",
                horizontalalignment="left",
            )

    # Remove the grid and axis
    ax.grid(False)
    ax.axis("off")

    fig.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/parallel_lines_cut_by_transversal_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    fig.savefig(
        file_name,
        dpi=text_helper.dpi,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_labeled_transversal_angle(params: TransversalAngleParams):
    """
    Draws a diagram of two parallel lines with a transversal, creating 8 angles.
    The given angle and the variable angle x are placed based on the provided positions.

    :param params: An instance of TransversalAngleParams containing the given angle and positions.
    """

    # Validate positions
    params.validate_positions()

    fig, ax = plt.subplots()

    # Drawing the parallel lines m and n with thinner lines
    ax.plot([-5, 5], [2, 2], color="black", label="m", linewidth=1)  # Line m
    ax.plot([-5, 5], [0, 0], color="black", label="n", linewidth=1)  # Line n

    # Drawing the transversal line l with thinner line
    x_transversal = np.linspace(-5, 5, 100)
    y_transversal = -0.5 * x_transversal + 1
    ax.plot(x_transversal, y_transversal, color="black", label="l", linewidth=1)

    # Label lines m, n, and l
    ax.text(5.1, 2, "m", verticalalignment="center", fontsize=12)
    ax.text(5.1, 0, "n", verticalalignment="center", fontsize=12)
    ax.text(5.1, -1.5, "k", verticalalignment="center", fontsize=12)

    # Positions for the 8 angles around the transversal
    angle_positions = {
        1: (-2, 2.2),
        2: (-3.8, 2.2),
        3: (-2.8, 1.7),
        4: (-1.2, 1.7),
        5: (2, 0.15),
        6: (0.3, 0.15),
        7: (1.3, -0.35),
        8: (2.8, -0.35),
    }

    # Plot the given angle at the given position
    ax.text(
        angle_positions[params.given_angle_position][0],
        angle_positions[params.given_angle_position][1],
        f"{params.given_angle}°",
        fontsize=12,
        color="black",
    )

    # Plot the variable angle x at the x_angle_position
    ax.text(
        angle_positions[params.x_angle_position][0],
        angle_positions[params.x_angle_position][1],
        "x°",
        fontsize=12,
        color="black",
    )

    # Add a note at the bottom
    ax.text(0, -2, "Note: Figure not drawn to scale", fontsize=10, ha="center")

    # Setting the limits and removing axis for a clean look
    ax.set_xlim([-6, 6])  # type: ignore
    ax.set_ylim([-2.5, 3.5])  # type: ignore
    ax.axis("off")

    # Adjust layout to add padding to the top and remove from the bottom
    plt.tight_layout(pad=1.0)

    # Add the following code to save the figure and return the file name
    file_name = f"{settings.additional_content_settings.image_destination_folder}/transversal_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
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
def draw_fractional_angle_circle(stimulus: FractionalAngle):
    """
    Draw a circle divided into equal parts with a shaded sector representing a fraction.

    Creates a visual representation of how fractions relate to angle measures
    in a complete circle (360 degrees).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    # Circle parameters
    center = (0, 0)
    radius = 1.0

    # Calculate angles
    total_parts = stimulus.denominator
    shaded_parts = stimulus.numerator
    angle_per_part = 360 / total_parts
    total_shaded_angle = angle_per_part * shaded_parts

    # Draw the main circle outline
    circle = patches.Circle(center, radius, fill=False, edgecolor="black", linewidth=3)
    ax.add_patch(circle)

    # Draw division lines for all parts
    for i in range(total_parts):
        angle_deg = i * angle_per_part
        angle_rad = np.deg2rad(angle_deg)

        # Draw line from center to edge
        x_end = center[0] + radius * np.cos(angle_rad)
        y_end = center[1] + radius * np.sin(angle_rad)
        ax.plot([center[0], x_end], [center[1], y_end], "k-", linewidth=2)

    # Draw the shaded sector
    start_angle = 0  # Start from 0 degrees (rightmost position)
    end_angle = total_shaded_angle

    sector = patches.Wedge(
        center,
        radius,
        theta1=start_angle,
        theta2=end_angle,
        facecolor=stimulus.sector_color,
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(sector)

    # Add center point
    ax.plot(center[0], center[1], "ko", markersize=8)

    # Set plot limits with padding
    padding = 0.2
    ax.set_xlim(-radius - padding, radius + padding)
    ax.set_ylim(-radius - padding, radius + padding)

    plt.tight_layout()

    # Save the image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/fractional_angle_{stimulus.numerator}_{stimulus.denominator}_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=300,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name
