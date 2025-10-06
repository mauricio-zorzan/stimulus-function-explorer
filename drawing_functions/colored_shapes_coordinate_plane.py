import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.colored_shapes_coordinate_plane import (
    ColoredShapesCoordinatePlane,
)
from content_generators.settings import settings

matplotlib.rcParams["font.family"] = "serif"


def draw_heart(ax, x, y, size=0.3, color="red"):
    """Draw a heart shape at the given coordinates."""
    t = np.linspace(0, 2 * np.pi, 100)
    # Heart parametric equations - use same scale as circle (size)
    heart_x = size * 0.05 * (16 * np.sin(t) ** 3)
    heart_y = size * 0.05 * (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
    
    ax.plot(x + heart_x, y + heart_y, color=color, linewidth=2, solid_capstyle='round')
    ax.fill(x + heart_x, y + heart_y, color=color, alpha=0.7)


def draw_star(ax, x, y, size=0.3, color="blue"):
    """Draw a star shape at the given coordinates."""
    # Create a 5-pointed star with proper proportions
    angles = np.linspace(0, 2 * np.pi, 11)  # 10 points + 1 to close the star
    outer_radius = size * 0.7
    inner_radius = size * 0.28
    
    star_x = []
    star_y = []
    
    for i, angle in enumerate(angles[:-1]):  # Exclude the last point to avoid duplication
        if i % 2 == 0:  # Outer points
            star_x.append(x + outer_radius * np.cos(angle))
            star_y.append(y + outer_radius * np.sin(angle))
        else:  # Inner points
            star_x.append(x + inner_radius * np.cos(angle))
            star_y.append(y + inner_radius * np.sin(angle))
    
    ax.plot(star_x, star_y, color=color, linewidth=2, solid_capstyle='round')
    ax.fill(star_x, star_y, color=color, alpha=0.7)


def draw_pentagon(ax, x, y, size=0.3, color="green"):
    """Draw a pentagon shape at the given coordinates."""
    angles = np.linspace(0, 2 * np.pi, 6)  # 6 points to close the pentagon
    pentagon_x = x + size * 0.7 * np.cos(angles)
    pentagon_y = y + size * 0.7 * np.sin(angles)
    
    ax.plot(pentagon_x, pentagon_y, color=color, linewidth=2, solid_capstyle='round')
    ax.fill(pentagon_x, pentagon_y, color=color, alpha=0.7)


def draw_triangle(ax, x, y, size=0.3, color="purple"):
    """Draw a triangle shape at the given coordinates."""
    # Equilateral triangle
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3, 0])  # 4 points to close the triangle
    triangle_x = x + size * 0.7 * np.cos(angles)
    triangle_y = y + size * 0.7 * np.sin(angles)
    
    ax.plot(triangle_x, triangle_y, color=color, linewidth=2, solid_capstyle='round')
    ax.fill(triangle_x, triangle_y, color=color, alpha=0.7)


def draw_square(ax, x, y, size=0.3, color="orange"):
    """Draw a square shape at the given coordinates."""
    # Square with sides parallel to axes
    square_x = [x - size * 0.7, x + size * 0.7, x + size * 0.7, x - size * 0.7, x - size * 0.7]
    square_y = [y - size * 0.7, y - size * 0.7, y + size * 0.7, y + size * 0.7, y - size * 0.7]
    
    ax.plot(square_x, square_y, color=color, linewidth=2, solid_capstyle='round')
    ax.fill(square_x, square_y, color=color, alpha=0.7)


def draw_circle(ax, x, y, size=0.3, color="pink"):
    """Draw a circle shape at the given coordinates."""
    circle = plt.Circle((x, y), size * 0.7, color=color, alpha=0.7, linewidth=2)
    ax.add_patch(circle)


def draw_letter(ax, x, y, letter, size=0.3, color="black"):
    """Draw a bold capital letter to the northeast of a small point at the given coordinates."""
    # Draw a small point at the coordinate
    ax.plot(x, y, 'o', color=color, markersize=6)
    # Draw the letter to the northeast of the point
    ax.text(x + 0.1, y + 0.1, letter, fontsize=16, fontweight='bold', 
            color=color, ha='left', va='bottom')


@stimulus_function
def draw_colored_shapes_coordinate_plane(stimulus_description: ColoredShapesCoordinatePlane) -> str:
    """
    Draw colored shapes on a coordinate plane for coordinate identification questions.
    
    This function creates a coordinate plane with various colored shapes positioned
    at specific coordinates, suitable for questions asking students to identify
    coordinates of shapes or find shapes at given coordinates.
    """
    # Validate that all coordinates are in the first quadrant (0-10 range)
    for shape in stimulus_description.shapes:
        if not (0 <= shape.x <= 10 and 0 <= shape.y <= 10):
            raise ValueError(f"Shape {shape.label} at ({shape.x}, {shape.y}) is outside the valid range (0-10)")
    
    # Check for duplicate coordinates
    seen_coords = set()
    for shape in stimulus_description.shapes:
        coord = (shape.x, shape.y)
        if coord in seen_coords:
            raise ValueError(f"Duplicate coordinates found: ({shape.x}, {shape.y})")
        seen_coords.add(coord)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    
    # Set up the coordinate plane (first quadrant only)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    # Make axes black and thicker
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(2)
    
    # Set the range for x and y axes (0-10 grid)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Ensure equal aspect ratio
    ax.set_aspect("equal", adjustable="box")
    
    # Set ticks and increase font size (integer grid only)
    ax.set_xticks(range(0, 11))
    ax.set_yticks(range(0, 11))
    ax.tick_params(
        axis="both", which="major", labelsize=16, color="black", width=2, length=6
    )
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1)
    
    # Shape drawing functions mapping
    shape_functions = {
        "heart": draw_heart,
        "star": draw_star,
        "pentagon": draw_pentagon,
        "triangle": draw_triangle,
        "square": draw_square,
        "circle": draw_circle,
        "letter": draw_letter,
    }
    
    # Draw each shape
    for shape in stimulus_description.shapes:
        if shape.shape_type in shape_functions:
            if shape.shape_type == "letter":
                if shape.letter is None:
                    raise ValueError(f"Letter shape {shape.label} must have a 'letter' field specified")
                shape_functions[shape.shape_type](ax, shape.x, shape.y, shape.letter, size=0.4, color=shape.color)
            else:
                shape_functions[shape.shape_type](ax, shape.x, shape.y, size=0.4, color=shape.color)
        else:
            # Fallback to circle for unknown shape types
            draw_circle(ax, shape.x, shape.y, size=0.4, color=shape.color)
    
    # Title removed as requested
    
    # Save the file
    file_name = f"{settings.additional_content_settings.image_destination_folder}/colored_shapes_coordinate_plane_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    
    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()
    
    return file_name
