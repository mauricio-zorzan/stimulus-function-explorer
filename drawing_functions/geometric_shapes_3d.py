import math
import random
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.right_prisms import (
    CubeRight,
    HexagonalPrism,
    IrregularPrism,
    OctagonalPrism,
    PentagonalPrism,
    RectangularPrismRight,
    RightPrismsList,
    RightPrismType,
    RightPrismUnion,
    TrapezoidalPrism,
    TriangularPrism,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.three_dimensional_objects import (
    Base3DShape,
    Cone,
    CrossSectionQuestion,
    Cylinder,
    Pyramid,
    RectangularPrism,
    Sphere,
    ThreeDimensionalObjectsList,
)
from content_generators.settings import settings
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@stimulus_function
def draw_multiple_3d_objects(shapes: ThreeDimensionalObjectsList):
    num_prisms = len(shapes.shapes)
    cols = int(np.ceil(np.sqrt(num_prisms)))
    rows = int(np.ceil(num_prisms / cols))

    fig = plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

    max_diff_ax = [0, 0, 0]
    axes = []
    shape_data = []  # Store shape data for labeling
    for index, prism in enumerate(reversed(shapes.shapes), start=1):
        ax: Axes3D = fig.add_subplot(
            rows, cols, len(shapes.shapes) - index + 1, projection="3d"
        )  # type: ignore
        draw_3d_object(ax, prism)
        max_diff_ax = [
            max(max_diff_ax[0], ax.get_xlim()[1] - ax.get_xlim()[0]),
            max(max_diff_ax[1], ax.get_ylim()[1] - ax.get_ylim()[0]),
            max(max_diff_ax[2], ax.get_zlim()[1] - ax.get_zlim()[0]),
        ]
        ax.set_title(prism.label, loc="center", fontsize=20)
        axes.append(ax)
        shape_data.append(prism)

    for i, ax in enumerate(axes):
        # Get the current max values for each axis
        current_max_x = ax.get_xlim()[1]
        current_max_y = ax.get_ylim()[1]
        current_max_z = ax.get_zlim()[1]
        current_min_x = ax.get_xlim()[0]
        current_min_y = ax.get_ylim()[0]
        current_min_z = ax.get_zlim()[0]

        # Calculate the difference with max_ax
        diff_x = max_diff_ax[0] - current_max_x
        diff_y = max_diff_ax[1] - current_max_y
        diff_z = max_diff_ax[2] - current_max_z

        # Adjust the limits by shifting both sides out by the difference
        ax.set_xlim(current_min_x - diff_x / 2, current_max_x + diff_x / 2)
        ax.set_ylim(current_min_y - diff_y / 2, current_max_y + diff_y / 2)
        ax.set_zlim(current_min_z - diff_z / 2, current_max_z + diff_z / 2)
        # Add dimensional labels for shapes that have dimensions
        add_dimensional_labels(ax, shape_data[i], shapes.units)

    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/geo_shapes_3d_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def add_dimensional_labels(ax: Axes3D, shape: "Base3DShape", units: str) -> None:
    """Add dimensional labels to 3D shapes without disrupting the shape visibility."""
    # Get current axis limits for positioning
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    # Calculate label positioning offsets based on shape size
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    max_range = max(x_range, y_range, z_range)
    # Base offset for labels (proportional to shape size)
    base_offset = max_range * 0.15
    # Font size based on figure size
    font_size = max(8, min(14, int(max_range * 2)))
    if shape.shape in ["rectangular prism", "cube"]:
        # For rectangular prisms and cubes, show length, width, height
        height = getattr(shape, "height", None)
        width = getattr(shape, "width", None)
        length = getattr(shape, "length", None)

        if height is not None:
            # Height label (red) - positioned on the left side
            ax.plot(
                [xlim[0] - base_offset * 0.5, xlim[0] - base_offset * 0.5],
                [ylim[0], ylim[0]],
                [zlim[0], height],
                "--",
                color="red",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                xlim[0] - base_offset,
                ylim[0],
                height / 2,
                f"{height} {units}",
                color="red",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

        if width is not None:
            # Width label (blue) - positioned on the right side
            ax.plot(
                [xlim[1] + base_offset * 0.3, xlim[1] + base_offset * 0.3],
                [ylim[0], width],
                [zlim[0], zlim[0]],
                "--",
                color="blue",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                xlim[1] + base_offset * 0.6,
                width / 2,
                zlim[0],
                f"{width} {units}",
                color="blue",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

        if length is not None:
            # Length label (green) - positioned at the bottom front
            ax.plot(
                [xlim[0], length],
                [ylim[0] - base_offset * 0.5, ylim[0] - base_offset * 0.5],
                [zlim[0], zlim[0]],
                "--",
                color="green",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                length / 2,
                ylim[0] - base_offset,
                zlim[0],
                f"{length} {units}",
                color="green",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

    elif shape.shape == "sphere":
        # For spheres, show radius
        radius = getattr(shape, "radius", None)
        if radius is not None:
            # Radius label - positioned to the right of center
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            center_z = (zlim[0] + zlim[1]) / 2

            ax.plot(
                [center_x, center_x + radius],
                [center_y, center_y],
                [center_z, center_z],
                "--",
                color="purple",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                center_x + radius + base_offset * 0.3,
                center_y,
                center_z,
                f"r = {radius} {units}",
                color="purple",
                fontsize=font_size,
                ha="left",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

    elif shape.shape in ["cylinder", "cone"]:
        # For cylinders and cones, show height and radius
        height = getattr(shape, "height", None)
        radius = getattr(shape, "radius", None)
        if height is not None:
            # Height label (red) - positioned on the left
            ax.plot(
                [xlim[0] - base_offset * 0.3, xlim[0] - base_offset * 0.3],
                [ylim[0], ylim[0]],
                [zlim[0], height],
                "--",
                color="red",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                xlim[0] - base_offset * 0.6,
                ylim[0],
                height / 2,
                f"h = {height} {units}",
                color="red",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

        if radius is not None:
            # Radius label (blue) - positioned at the base
            center_x = (xlim[0] + xlim[1]) / 2
            ax.plot(
                [center_x, center_x + radius],
                [ylim[0], ylim[0]],
                [zlim[0], zlim[0]],
                "--",
                color="blue",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                center_x + radius / 2,
                ylim[0] - base_offset * 0.5,
                zlim[0],
                f"r = {radius} {units}",
                color="blue",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

    elif shape.shape == "pyramid":
        # For pyramids, show side length and height
        side = getattr(shape, "side", None)
        height = getattr(shape, "height", None)

        if height is not None:
            # Height label (red) - positioned on the left
            ax.plot(
                [xlim[0] - base_offset * 0.3, xlim[0] - base_offset * 0.3],
                [ylim[0], ylim[0]],
                [zlim[0], height],
                "--",
                color="red",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                xlim[0] - base_offset * 0.6,
                ylim[0],
                height / 2,
                f"h = {height} {units}",
                color="red",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

        if side is not None:
            # Side length label (orange) - positioned at the base
            ax.plot(
                [xlim[0], side],
                [ylim[0] - base_offset * 0.5, ylim[0] - base_offset * 0.5],
                [zlim[0], zlim[0]],
                "--",
                color="orange",
                linewidth=2,
                zorder=1000,
            )
            ax.text(
                side / 2,
                ylim[0] - base_offset,
                zlim[0],
                f"side = {side} {units}",
                color="orange",
                fontsize=font_size,
                ha="center",
                va="center",
                zorder=1000,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )


def draw_pyramid(ax: Axes3D, description: Pyramid):
    base_side = description.side
    height = description.height

    half_base = base_side / 2  # type: ignore

    vertices = np.array(
        [
            [0, 0, 0],
            [base_side, 0, 0],
            [base_side, base_side, 0],
            [0, base_side, 0],
            [half_base, half_base, height],
        ]
    )

    # Define the faces of the pyramid
    faces = [
        [vertices[0], vertices[1], vertices[4]],
        [vertices[1], vertices[2], vertices[4]],
        [vertices[2], vertices[3], vertices[4]],
        [vertices[3], vertices[0], vertices[4]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
    ]

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="cyan", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_sphere(ax: Axes3D, description: Sphere):
    resolution = 20  # Number of points used to draw the sphere
    scale = 1.25
    radius = scale * (description.radius or 4.0)

    # Generate the sphere data
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    x_grid = radius * np.sin(phi_grid) * np.cos(theta_grid) + radius
    y_grid = radius * np.sin(phi_grid) * np.sin(theta_grid)
    z_grid = radius * np.cos(phi_grid) + radius

    # Draw the surface
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="red",
        edgecolor="black",
        linewidth=2,
        zorder=2,
        alpha=0.6,
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_cylinder(ax: Axes3D, description: Cylinder):
    scale = 1.25
    height = description.height * scale  # type: ignore
    radius = description.radius * scale  # type: ignore
    # Generate the cylinder data
    resolution = 20  # Number of points used to draw the circle
    shift = description.radius / 2  # type: ignore  # Shift the cylinder to the right by 2 units

    # Generate the cylinder data
    z = np.linspace(0, height, 2)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = radius * np.cos(theta_grid) + shift
    y_grid = radius * np.sin(theta_grid)

    # Draw the surfaces
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="green",
        edgecolor="black",
        linewidth=2,
        zorder=2,
        alpha=0.6,
    )

    # Draw the top and bottom circles
    top_circle = np.array(
        [
            radius * np.cos(theta) + shift,
            radius * np.sin(theta),
            height * np.ones_like(theta),
        ]
    ).T
    bottom_circle = np.array(
        [radius * np.cos(theta) + shift, radius * np.sin(theta), np.zeros_like(theta)]
    ).T

    ax.add_collection3d(
        Poly3DCollection(
            [top_circle],
            color="green",
            edgecolor="black",
            linewidth=2,
            zorder=2,
            alpha=0.6,
        )
    )
    ax.add_collection3d(
        Poly3DCollection(
            [bottom_circle],
            color="green",
            edgecolor="black",
            linewidth=2,
            zorder=2,
            alpha=0.6,
        )
    )

    # Draw black outlines (top and bottom)
    ax.plot(
        radius * np.cos(theta) + shift,
        radius * np.sin(theta),
        height * np.ones_like(theta),
        color="black",
        linewidth=2,
        zorder=10000,
    )
    ax.plot(
        radius * np.cos(theta) + shift,
        radius * np.sin(theta),
        np.zeros_like(theta),
        color="black",
        linewidth=2,
        zorder=10000,
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_rect_prism(ax: Axes3D, description: RectangularPrism):
    height = description.height
    width = description.width
    length = description.length

    # Define the vertices of the box
    vertices = np.array(
        [
            [0, 0, 0],
            [length, 0, 0],
            [length, width, 0],
            [0, width, 0],
            [0, 0, height],
            [length, 0, height],
            [length, width, height],
            [0, width, height],
        ]
    )

    # Define the faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Back face
    ]

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="cyan", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_cone(ax: Axes3D, description: Cone):
    scale = 1.25
    height = description.height * scale  # type: ignore
    radius = description.radius * scale  # type: ignore
    resolution = 20  # Number of points used to draw the base
    shift = description.radius / 2  # type: ignore

    # Generate the cone data
    z = np.linspace(0, height, 2)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = (height - z_grid) / height * radius * np.cos(theta_grid) + shift
    y_grid = (height - z_grid) / height * radius * np.sin(theta_grid)

    # Draw the surface
    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color="blue",
        edgecolor="black",
        linewidth=2,
        zorder=2,
        alpha=0.6,
    )

    # Draw the base circle
    base_circle = np.array(
        [radius * np.cos(theta) + shift, radius * np.sin(theta), np.zeros_like(theta)]
    ).T

    ax.add_collection3d(
        Poly3DCollection(
            [base_circle],
            color="blue",
            edgecolor="none",
            linewidth=0,
            zorder=2,
            alpha=0.6,
        )
    )

    # Draw black outline (base)
    ax.plot(
        radius * np.cos(theta) + shift,
        radius * np.sin(theta),
        np.zeros_like(theta),
        color="blue",
        linewidth=2,
        zorder=10000,
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_3d_object(ax: Axes3D, shape: Base3DShape):
    if shape.shape == "sphere":
        sphere = Sphere(**{"radius": 4, **shape.model_dump(exclude_none=True)})
        draw_sphere(ax, sphere)
    elif shape.shape == "pyramid":
        pyramid = Pyramid(
            **{"side": 5, "height": 6, **shape.model_dump(exclude_none=True)}
        )
        draw_pyramid(ax, pyramid)
    elif shape.shape == "cube":
        cube = RectangularPrism(
            **{
                "height": 5,
                "width": 5,
                "length": 5,
                **shape.model_dump(exclude={"shape"}, exclude_none=True),
            }
        )
        draw_rect_prism(ax, cube)
    elif shape.shape == "rectangular prism":
        rectangular_prism = RectangularPrism(
            **{
                "height": 4,
                "width": 6,
                "length": 3,
                **shape.model_dump(exclude_none=True),
            }
        )
        draw_rect_prism(ax, rectangular_prism)
    elif shape.shape == "cone":
        cone = Cone(**{"height": 6, "radius": 3, **shape.model_dump(exclude_none=True)})
        draw_cone(ax, cone)
    elif shape.shape == "cylinder":
        cylinder = Cylinder(
            **{"height": 5, "radius": 3, **shape.model_dump(exclude_none=True)}
        )
        draw_cylinder(ax, cylinder)
    else:
        raise ValueError("Unknown Shape " + shape.shape)


def draw_outline(ax, kind, params, color="#C49554"):
    """Draw a 2D outline in a flat axes"""
    if kind == "circle":
        (r,) = params
        patch = Circle((0, 0), r, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-r * 1.1, r * 1.1)
        ax.set_ylim(-r * 1.1, r * 1.1)
    elif kind == "rectangle":
        w, h = params
        patch = Rectangle((-w / 2, 0), w, h, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-w * 0.6, w * 0.6)
        ax.set_ylim(0, h * 1.1)
    elif kind == "square":
        (side,) = params
        patch = Rectangle((-side / 2, -side / 2), side, side, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-side * 0.6, side * 0.6)
        ax.set_ylim(-side * 0.6, side * 0.6)
    elif kind == "triangle":
        (pts,) = params
        patch = Polygon(pts, facecolor=color)
        ax.add_patch(patch)
        xs, ys = zip(*pts)
        ax.set_xlim(min(xs) - 0.1, max(xs) + 0.1)
        ax.set_ylim(min(ys) - 0.1, max(ys) + 0.1)
    elif kind == "ellipse":
        patch = Ellipse((0, 0), 2, 1, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 0.7)
    elif kind == "pentagon":
        theta = np.linspace(0, 2 * np.pi, 6)[:-1]
        pts = [(np.cos(t), np.sin(t)) for t in theta]
        patch = Polygon(pts, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    elif kind == "quadrilateral":
        pts = [(-1, -0.5), (1, -0.5), (0.8, 0.7), (-0.7, 0.8)]
        patch = Polygon(pts, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    elif kind == "hexagon":
        theta = np.linspace(0, 2 * np.pi, 7)[:-1]
        pts = [(np.cos(t), np.sin(t)) for t in theta]
        patch = Polygon(pts, facecolor=color)
        ax.add_patch(patch)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    ax.axis("off")


def generate_options(correct_kind):
    """Pick three wrong kinds and shuffle them with the correct one."""
    all_kinds = ["circle", "rectangle", "triangle"]
    wrong = random.sample([k for k in all_kinds if k != correct_kind], 2)
    opts = [correct_kind] + wrong
    random.shuffle(opts)
    return opts


@stimulus_function
def draw_cross_section_question(question: CrossSectionQuestion):
    # Use the specified correct cross-section instead of random choice
    correct_kind = question.correct_cross_section.lower().strip()

    # Determine orientation based on the shape and desired cross-section
    shape_type = question.shape.shape.lower().replace("_", " ").strip()

    # Map of valid shape + cross-section combinations to determine orientation
    valid_combinations = {
        ("sphere", "circle"): "horizontal",  # Any sphere cut is a circle
        ("cylinder", "circle"): "horizontal",  # Horizontal cylinder cut
        ("cylinder", "rectangle"): "vertical",  # Vertical cylinder cut
        ("cone", "circle"): "horizontal",  # Horizontal cone cut
        ("cone", "triangle"): "vertical",  # Vertical cone cut
        ("rectangular prism", "rectangle"): "horizontal",  # Horizontal prism cut
        ("rectangular prism", "square"): "vertical",  # Vertical prism cut
        ("cube", "square"): "horizontal",  # Any cube cut is a square
        ("pyramid", "square"): "horizontal",  # Horizontal pyramid cut
        ("pyramid", "triangle"): "vertical",  # Vertical pyramid cut
    }

    # Determine orientation based on the combination
    orientation = valid_combinations.get((shape_type, correct_kind), "horizontal")

    # Figure out where the slicing plane actually is
    shape = question.shape
    z = 0.0  # Initialize z for all cases
    # --- Plane position logic ---
    # Note: Cutting plane removed as requested - only the 2D cross-section options are shown
    # The plane position is still calculated for determining the correct cross-section parameters
    if orientation == "horizontal":
        base_z = 0.0
        if shape_type == "sphere":
            base_z = float(getattr(shape, "radius", 4.0))
            # Position plane at the center of the sphere so top half appears above
            z = base_z * 0.5
        elif hasattr(shape, "height") and getattr(shape, "height", None) is not None:
            base_z = 0.0
            # Position plane at 1/3 height so top 2/3 appears above
            z = base_z + 0.33 * float(getattr(shape, "height", 5.0))
        elif hasattr(shape, "side") and getattr(shape, "side", None) is not None:
            base_z = 0.0
            # Position plane at 1/3 height so top 2/3 appears above
            z = base_z + 0.33 * float(getattr(shape, "side", 5.0))
        else:
            # Position plane at 1/3 height so top 2/3 appears above
            z = 0.33 * float(
                getattr(shape, "height", 5.0) or getattr(shape, "radius", 4.0)
            )
    else:
        # For vertical cuts, we need x position but only for sphere calculations
        if shape_type == "sphere":
            # For vertical sphere cuts, z represents the height where we're cutting
            z = float(getattr(shape, "radius", 4.0)) * 0.5

    # Compute correct_params based on shape + orientation with true slice dimensions
    min_val = 0.1  # Minimum visible size
    if correct_kind == "circle":
        if shape_type == "sphere":
            R = float(getattr(shape, "radius", 4.0))
            # For sphere, slice radius = sqrt(R^2 - z^2) where z is distance from center
            # Assuming sphere is centered at origin, z is the height of the slice
            z_from_center = abs(z - R)  # Distance from sphere center
            r_slice = np.sqrt(max(min_val**2, R**2 - z_from_center**2))
        elif shape_type == "cone":
            R = float(getattr(shape, "radius", 3.0))
            H = float(getattr(shape, "height", 6.0))
            # For cone, slice radius = R * (1 - z/H) where z is height from base
            r_slice = max(min_val, R * (1 - z / H))
        elif shape_type == "cylinder":
            R = float(getattr(shape, "radius", 3.0))
            # For cylinder horizontal cut, slice radius = full radius
            r_slice = max(min_val, R)
        else:
            # For other shapes, use side length as radius approximation
            side = float(getattr(shape, "side", 3.0))
            r_slice = max(min_val, side / 2)
        correct_params = (r_slice,)
    elif correct_kind == "rectangle":
        if shape_type == "cylinder":
            # Vertical slice of cylinder: width = 2*radius, height = full height
            width = 2 * float(getattr(shape, "radius", 3.0))
            height = float(getattr(shape, "height", 5.0))
        else:  # rectangular prism
            width = float(getattr(shape, "width", 6.0))
            height = float(getattr(shape, "height", 4.0))
        correct_params = (max(min_val, width), max(min_val, height))
    elif correct_kind == "square":
        side = float(
            getattr(shape, "side", None)
            or getattr(shape, "width", None)
            or getattr(shape, "height", None)
            or 3.0
        )
        correct_params = (max(min_val, side),)
    elif correct_kind == "triangle":
        if shape_type == "cone":
            R = float(getattr(shape, "radius", 3.0))
            H = float(getattr(shape, "height", 6.0))
            pts = [(-R, 0), (R, 0), (0, H)]
        elif shape_type == "pyramid":
            S = float(getattr(shape, "side", 5.0))
            H = float(getattr(shape, "height", 6.0))
            pts = [(0, 0), (S, 0), (S / 2, H)]
        else:
            pts = [(-1, 0), (1, 0), (0, 1.5)]
        correct_params = (pts,)
    else:
        correct_params = ()

    fig = plt.figure(figsize=(14, 6), dpi=100)
    gs = gridspec.GridSpec(2, 4, height_ratios=[3, 1], figure=fig)

    ax3: Axes3D = fig.add_subplot(gs[0, :], projection="3d")  # type: ignore

    # Draw the 3D object first
    draw_3d_object(ax3, question.shape)

    # Get the bounds of the shape for proper plane sizing
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    zlim = ax3.get_zlim()

    # Create a more sophisticated cutting plane visualization
    if orientation == "horizontal":
        # Create a plane that extends beyond the shape bounds
        margin = 0.5
        # Create mesh for the plane
        xx = np.array(
            [[xlim[0] - margin, xlim[1] + margin], [xlim[0] - margin, xlim[1] + margin]]
        )
        yy = np.array(
            [[ylim[0] - margin, ylim[0] - margin], [ylim[1] + margin, ylim[1] + margin]]
        )
        zz = np.full_like(xx, z)

    else:  # vertical
        # For vertical cuts, position the plane at the center of the shape
        x_center = (xlim[0] + xlim[1]) / 2
        margin = 0.5

        # Create mesh for vertical plane
        yy = np.array(
            [[ylim[0] - margin, ylim[1] + margin], [ylim[0] - margin, ylim[1] + margin]]
        )
        zz = np.array(
            [[zlim[0] - margin, zlim[0] - margin], [zlim[1] + margin, zlim[1] + margin]]
        )
        xx = np.full_like(yy, x_center)

    # Draw the cutting plane with specific z-order
    plane = ax3.plot_surface(
        xx,
        yy,
        zz,
        color="darkblue",
        alpha=0.4,  # Make it more transparent
        linewidth=0,
        antialiased=True,
        rcount=2,
        ccount=2,
        zorder=1,  # Put it behind the shape
    )
    plane.set_edgecolor("none")

    # Add a border/outline to the cutting plane to make it more visible
    if orientation == "horizontal":
        # Draw the outline of the plane
        outline_x = [
            xlim[0] - margin,
            xlim[1] + margin,
            xlim[1] + margin,
            xlim[0] - margin,
            xlim[0] - margin,
        ]
        outline_y = [
            ylim[0] - margin,
            ylim[0] - margin,
            ylim[1] + margin,
            ylim[1] + margin,
            ylim[0] - margin,
        ]
        outline_z = [z] * 5
        ax3.plot(
            outline_x, outline_y, outline_z, color="darkblue", linewidth=2, alpha=0.8
        )
    else:
        # Draw the outline of the vertical plane
        outline_x = [x_center] * 5
        outline_y = [
            ylim[0] - margin,
            ylim[1] + margin,
            ylim[1] + margin,
            ylim[0] - margin,
            ylim[0] - margin,
        ]
        outline_z = [
            zlim[0] - margin,
            zlim[0] - margin,
            zlim[1] + margin,
            zlim[1] + margin,
            zlim[0] - margin,
        ]
        ax3.plot(
            outline_x, outline_y, outline_z, color="darkblue", linewidth=2, alpha=0.8
        )

    # Determine the color based on the 3D shape type
    shape_colors = {
        "sphere": "red",
        "cylinder": "green",
        "cone": "blue",
        "pyramid": "cyan",
        "rectangular prism": "cyan",
        "cube": "cyan",
    }
    shape_color = shape_colors.get(shape_type, "#C49554")  # Default color if not found

    all_kinds = [
        "circle",
        "rectangle",
        "square",
        "triangle",
        "ellipse",
        "pentagon",
        "quadrilateral",
        "hexagon",
    ]
    # Remove the correct kind from distractors and ensure uniqueness
    distractors = [k for k in all_kinds if k != correct_kind]
    # Exclude 'rectangle' if correct is 'square', and exclude 'square' if correct is 'rectangle'
    if correct_kind == "square":
        distractors = [k for k in distractors if k != "rectangle"]
    elif correct_kind == "rectangle":
        distractors = [k for k in distractors if k != "square"]
    # Sample 3 unique distractors
    chosen_distractors = random.sample(distractors, 3)
    letter_to_index = {"a": 0, "b": 1, "c": 2, "d": 3}
    idx = letter_to_index.get(getattr(question, "correct_letter", "a"), 0)
    opts = chosen_distractors.copy()
    opts.insert(idx, correct_kind)
    # Ensure all options are unique and length 4
    seen = set()
    unique_opts = []
    for o in opts:
        if o not in seen:
            unique_opts.append(o)
            seen.add(o)
    # If less than 4, fill with more distractors
    for k in all_kinds:
        if len(unique_opts) >= 4:
            break
        if k not in seen:
            unique_opts.append(k)
            seen.add(k)
    opts = unique_opts[:4]

    for i, kind in enumerate(opts):
        ax2 = fig.add_subplot(gs[1, i])
        if i == idx:
            draw_outline(ax2, correct_kind, correct_params, shape_color)
        else:
            # Use fixed visible parameters for distractors with the same color as the 3D shape
            if kind == "circle":
                draw_outline(ax2, "circle", (1.0,), shape_color)
            elif kind == "rectangle":
                draw_outline(ax2, "rectangle", (1.2, 0.8), shape_color)
            elif kind == "square":
                draw_outline(ax2, "square", (0.9,), shape_color)
            elif kind == "triangle":
                tri = [(-1, 0), (1, 0), (0, 1.5)]
                draw_outline(ax2, "triangle", (tri,), shape_color)
            elif kind == "ellipse":
                draw_outline(ax2, "ellipse", (), shape_color)
            elif kind == "pentagon":
                draw_outline(ax2, "pentagon", (), shape_color)
            elif kind == "quadrilateral":
                draw_outline(ax2, "quadrilateral", (), shape_color)
            elif kind == "hexagon":
                draw_outline(ax2, "hexagon", (), shape_color)
        ax2.set_title(chr(97 + i), loc="left", fontsize=14)
        ax2.set_aspect("equal")

    plt.tight_layout()
    fname = (
        f"{settings.additional_content_settings.image_destination_folder}"
        f"/crosssec_{int(time.time())}."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fname


@stimulus_function
def draw_right_prisms(prisms: RightPrismsList):
    """Draw multiple right prisms in a grid layout"""
    num_prisms = len(prisms.prisms)
    cols = int(np.ceil(np.sqrt(num_prisms)))
    rows = int(np.ceil(num_prisms / cols))

    fig = plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

    max_diff_ax = [0, 0, 0]
    axes = []
    prism_data = []  # Store prism data for labeling
    for index, prism in enumerate(reversed(prisms.prisms), start=1):
        ax: Axes3D = fig.add_subplot(
            rows, cols, len(prisms.prisms) - index + 1, projection="3d"
        )  # type: ignore
        draw_right_prism(ax, prism, prisms.units)
        max_diff_ax = [
            max(max_diff_ax[0], ax.get_xlim()[1] - ax.get_xlim()[0]),
            max(max_diff_ax[1], ax.get_ylim()[1] - ax.get_ylim()[0]),
            max(max_diff_ax[2], ax.get_zlim()[1] - ax.get_zlim()[0]),
        ]
        ax.set_title(prism.label, loc="center", fontsize=20)
        axes.append(ax)
        prism_data.append(prism)

    for i, ax in enumerate(axes):
        # Add dimensional labels for prisms that have dimensions
        add_prism_dimensional_labels(ax, prism_data[i], prisms.units, prisms)

    plt.tight_layout()

    file_name = f"{settings.additional_content_settings.image_destination_folder}/right_prisms_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=800,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


def add_prism_dimensional_labels(
    ax: Axes3D, prism: RightPrismUnion, units: str, prisms_list=None
):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    z0, z1 = ax.get_zlim()
    mx = max(x1 - x0, y1 - y0, z1 - z0)

    # VERY SMALL OFFSET - even smaller
    base_offset = mx * 0.02
    fs = 14
    bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=1)

    # Get show flags from prisms_list
    show_h = getattr(prisms_list, "show_height", True) if prisms_list else True
    show_area = getattr(prisms_list, "show_base_area", True) if prisms_list else True

    # 1) HEIGHT (red) - positioned very close to the prism
    h = prism.height
    if show_h:
        if prism.shape in [
            RightPrismType.HEXAGONAL,
            RightPrismType.PENTAGONAL,
            RightPrismType.OCTAGONAL,
        ]:
            height_x = x0 - base_offset * 0.3
        else:
            height_x = x0  # - base_offset  # * 0.4

        ax.text(
            height_x,
            y0 - base_offset * 0.1,
            h / 2,
            f"{h} {units}",
            color="red",
            ha="center",
            bbox=bbox,
            fontsize=fs,
            zorder=1000,
        )

    # 2) rectangular prism
    if prism.shape == RightPrismType.RECTANGULAR:
        if prism.width is not None and prism.length is not None:
            w, L = prism.width, prism.length
            # width (blue) - positioned very close to the right side
            ax.text(
                x1 + base_offset * 0.1,
                w / 2,
                z0 - base_offset * 0.1,
                f"{w} {units}",
                color="blue",
                ha="left",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )
            # length (green) - positioned very close to the bottom front
            ax.text(
                L / 2,
                y0 - base_offset * 0.1,
                z0 - base_offset * 0.1,
                f"{L} {units}",
                color="green",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )
        elif prism.base_area is not None and show_area:
            # Only show base area, not individual dimensions
            area = prism.base_area
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(
                cx,
                cy,
                z0 + base_offset * 0.05,
                f"Base Area = {area:.1f} {units}²",
                color="blue",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )

    # 3) cube
    elif prism.shape == RightPrismType.CUBE:
        if prism.side_length is not None:
            s = prism.side_length
            ax.text(
                s / 2,
                y0 - base_offset * 0.1,
                z0 - base_offset * 0.1,
                f"{s} {units}",
                color="purple",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )
        elif prism.base_area is not None and show_area:
            # Only show base area, not side length
            area = prism.base_area
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(
                cx,
                cy,
                z0 + base_offset * 0.05,
                f"Base Area = {area:.1f} {units}²",
                color="purple",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )

    # 4) triangular prism
    elif prism.shape == RightPrismType.TRIANGULAR:
        a, b, c = prism.side_a, prism.side_b, prism.side_c

        # Get the calculated triangle properties
        if hasattr(prism, "_triangle_vertices") and hasattr(prism, "_triangle_height"):
            vertices = prism._triangle_vertices  # type: ignore

            # Position labels better using actual triangle geometry
            # Side a (base) - positioned along the base
            ax.text(
                a / 2,
                y0 - base_offset * 0.3,
                z0 - base_offset * 0.1,
                f"{a} {units}",
                color="green",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )

            # Side b - positioned near the middle of side b with better spacing
            mid_b_x = vertices[2][0] / 2
            mid_b_y = vertices[2][1] / 2
            ax.text(
                mid_b_x - base_offset * 0.2,
                mid_b_y + base_offset * 0.1,
                z0 - base_offset * 0.1,
                f"{b} {units}",
                color="blue",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )

            # Side c - positioned near the middle of side c with better spacing
            mid_c_x = (a + vertices[2][0]) / 2
            mid_c_y = vertices[2][1] / 2
            ax.text(
                mid_c_x + base_offset * 0.2,
                mid_c_y + base_offset * 0.1,
                z0 - base_offset * 0.1,
                f"{c} {units}",
                color="orange",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )

        else:
            # Fallback to original positioning if triangle properties not available
            # a along x - positioned very close
            ax.text(
                a / 2,
                y0 - base_offset * 0.1,
                z0 - base_offset * 0.1,
                f"{a} {units}",
                color="green",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )
            # b slanted - positioned very close
            ax.text(
                b * 0.25,
                b * 0.433,
                z0 - base_offset * 0.1,
                f"{b} {units}",
                color="blue",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )
            # c - positioned very close
            ax.text(
                a * 0.75,
                c * 0.433,
                z0 - base_offset * 0.1,
                f"{c} {units}",
                color="orange",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )

    # 5) trapezoidal prism
    elif prism.shape == RightPrismType.TRAPEZOIDAL:
        t, bb, hb = prism.top_base, prism.bottom_base, prism.trapezoid_height

        # Show the two bases and height of trapezoid (removed non-existent base_area)
        # top - positioned very close
        ax.text(
            0,
            hb,
            z0 - base_offset * 0.1,
            f"top = {t} {units}",
            color="green",
            ha="center",
            bbox=bbox,
            fontsize=fs,
            zorder=1000,
        )
        # bottom - positioned very close
        ax.text(
            0,
            y0,
            z0 - base_offset * 0.1,
            f"bottom = {bb} {units}",
            color="blue",
            ha="center",
            bbox=bbox,
            fontsize=fs,
            zorder=1000,
        )
        # trapezoid height - positioned very close
        ax.text(
            x0 - base_offset * 0.3,
            hb / 2,
            z0 - base_offset * 0.1,
            f"h_base = {hb} {units}",
            color="purple",
            ha="center",
            bbox=bbox,
            fontsize=fs,
            zorder=1000,
        )

    # 6) regular polygons: always compute base area from side_length
    elif prism.shape in [
        RightPrismType.HEXAGONAL,
        RightPrismType.PENTAGONAL,
        RightPrismType.OCTAGONAL,
    ]:
        # always compute base area from side_length
        if hasattr(prism, "side_length"):
            s = prism.side_length  # type: ignore
            if prism.shape == RightPrismType.HEXAGONAL:
                base_area = (3 * np.sqrt(3) / 2) * s**2
            elif prism.shape == RightPrismType.PENTAGONAL:
                base_area = (5 / 4) * s**2 * (1 / np.tan(np.pi / 5))
            else:  # octagonal
                base_area = 2 * (1 + np.sqrt(2)) * s**2

            if show_area:
                # Base area (blue), positioned at center of base
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                ax.text(
                    cx,
                    cy,
                    z0 + base_offset * 0.05,
                    f"Base Area = {base_area:.1f} {units}²",
                    color="blue",
                    ha="center",
                    bbox=bbox,
                    fontsize=fs,
                    zorder=1000,
                )

    # 7) irregular prism: ONLY base area and height
    elif prism.shape == RightPrismType.IRREGULAR:
        verts = prism.base_vertices
        # shoelace
        area = (
            abs(
                sum(
                    verts[i][0] * verts[(i + 1) % len(verts)][1]
                    - verts[(i + 1) % len(verts)][0] * verts[i][1]
                    for i in range(len(verts))
                )
            )
            / 2
        )
        if show_area:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(
                cx,
                cy,
                z0 + base_offset * 0.05,
                f"Base Area = {area:.1f} {units}²",
                color="blue",
                ha="center",
                bbox=bbox,
                fontsize=fs,
                zorder=1000,
            )


def draw_right_prism(ax: Axes3D, prism: RightPrismUnion, units: str = ""):
    """Draw a single right prism based on its type"""
    if prism.shape == RightPrismType.IRREGULAR:
        draw_irregular_prism(ax, prism)
    elif prism.shape == RightPrismType.OCTAGONAL:
        draw_octagonal_prism(ax, prism)
    elif prism.shape == RightPrismType.HEXAGONAL:
        draw_hexagonal_prism(ax, prism)
    elif prism.shape == RightPrismType.PENTAGONAL:
        draw_pentagonal_prism(ax, prism)
    elif prism.shape == RightPrismType.TRAPEZOIDAL:
        draw_trapezoidal_prism(ax, prism)
    elif prism.shape == RightPrismType.TRIANGULAR:
        draw_triangular_prism(ax, prism, units)
    elif prism.shape == RightPrismType.RECTANGULAR:
        draw_rectangular_prism_right(ax, prism)
    elif prism.shape == RightPrismType.CUBE:
        draw_cube_right(ax, prism)
    else:
        raise ValueError(f"Unknown prism type: {prism.shape}")


def draw_irregular_prism(ax: Axes3D, prism: IrregularPrism):
    """Draw an irregular prism with custom base vertices"""
    height = prism.height
    base_vertices = prism.base_vertices

    # Convert 2D vertices to 3D by adding z=0 for bottom and z=height for top
    bottom_vertices = np.array([[v[0], v[1], 0] for v in base_vertices])
    top_vertices = np.array([[v[0], v[1], height] for v in base_vertices])

    # Create faces: bottom, top, and lateral faces
    faces = []

    # Bottom face
    faces.append(bottom_vertices)

    # Top face
    faces.append(top_vertices)

    # Lateral faces (connecting bottom to top)
    n = len(base_vertices)
    for i in range(n):
        next_i = (i + 1) % n
        face = np.array(
            [
                bottom_vertices[i],
                bottom_vertices[next_i],
                top_vertices[next_i],
                top_vertices[i],
            ]
        )
        faces.append(face)

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="lightblue", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_octagonal_prism(ax: Axes3D, prism: OctagonalPrism):
    """Draw a regular octagonal prism"""
    height = prism.height
    side_length = prism.side_length

    # Calculate octagon vertices (regular octagon)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 vertices
    radius = side_length / (2 * np.sin(np.pi / 8))  # Circumradius of regular octagon

    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # Create bottom and top vertices
    bottom_vertices = np.array([[x, y, 0] for x, y in zip(x_coords, y_coords)])
    top_vertices = np.array([[x, y, height] for x, y in zip(x_coords, y_coords)])

    # Create faces
    faces = []

    # Bottom face
    faces.append(bottom_vertices)

    # Top face
    faces.append(top_vertices)

    # Lateral faces
    n = 8
    for i in range(n):
        next_i = (i + 1) % n
        face = np.array(
            [
                bottom_vertices[i],
                bottom_vertices[next_i],
                top_vertices[next_i],
                top_vertices[i],
            ]
        )
        faces.append(face)

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="lightgreen", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_hexagonal_prism(ax: Axes3D, prism: HexagonalPrism):
    """Draw a regular hexagonal prism"""
    height = prism.height
    side_length = prism.side_length

    # Calculate hexagon vertices (regular hexagon)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
    radius = side_length  # For regular hexagon, radius = side length

    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # Create bottom and top vertices
    bottom_vertices = np.array([[x, y, 0] for x, y in zip(x_coords, y_coords)])
    top_vertices = np.array([[x, y, height] for x, y in zip(x_coords, y_coords)])

    # Create faces
    faces = []

    # Bottom face
    faces.append(bottom_vertices)

    # Top face
    faces.append(top_vertices)

    # Lateral faces
    n = 6
    for i in range(n):
        next_i = (i + 1) % n
        face = np.array(
            [
                bottom_vertices[i],
                bottom_vertices[next_i],
                top_vertices[next_i],
                top_vertices[i],
            ]
        )
        faces.append(face)

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="lightcoral", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_pentagonal_prism(ax: Axes3D, prism: PentagonalPrism):
    """Draw a regular pentagonal prism"""
    height = prism.height
    side_length = prism.side_length

    # Calculate pentagon vertices (regular pentagon)
    angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # 5 vertices
    radius = side_length / (2 * np.sin(np.pi / 5))  # Circumradius of regular pentagon

    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # Create bottom and top vertices
    bottom_vertices = np.array([[x, y, 0] for x, y in zip(x_coords, y_coords)])
    top_vertices = np.array([[x, y, height] for x, y in zip(x_coords, y_coords)])

    # Create faces
    faces = []

    # Bottom face
    faces.append(bottom_vertices)

    # Top face
    faces.append(top_vertices)

    # Lateral faces
    n = 5
    for i in range(n):
        next_i = (i + 1) % n
        face = np.array(
            [
                bottom_vertices[i],
                bottom_vertices[next_i],
                top_vertices[next_i],
                top_vertices[i],
            ]
        )
        faces.append(face)

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d,
            facecolors="lightyellow",
            linewidths=2,
            edgecolors="black",
            alpha=0.6,
        )
    )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_trapezoidal_prism(ax: Axes3D, prism: TrapezoidalPrism):
    """Draw a trapezoidal prism"""
    height = prism.height
    top_base = prism.top_base
    bottom_base = prism.bottom_base
    trapezoid_height = prism.trapezoid_height

    # Calculate trapezoid vertices
    # Bottom base centered at origin
    bottom_left = [-bottom_base / 2, 0, 0]
    bottom_right = [bottom_base / 2, 0, 0]

    # Top base centered at origin
    top_left = [-top_base / 2, trapezoid_height, 0]
    top_right = [top_base / 2, trapezoid_height, 0]

    # Create faces
    faces = []

    # Bottom face
    faces.append(
        np.array(
            [
                bottom_left,
                bottom_right,
                [bottom_right[0], bottom_right[1], height],
                [bottom_left[0], bottom_left[1], height],
            ]
        )
    )

    # Top face
    faces.append(
        np.array(
            [
                top_left,
                top_right,
                [top_right[0], top_right[1], height],
                [top_left[0], top_left[1], height],
            ]
        )
    )

    # Lateral faces
    faces.append(
        np.array(
            [
                bottom_left,
                top_left,
                [top_left[0], top_left[1], height],
                [bottom_left[0], bottom_left[1], height],
            ]
        )
    )  # Left face
    faces.append(
        np.array(
            [
                bottom_right,
                top_right,
                [top_right[0], top_right[1], height],
                [bottom_right[0], bottom_right[1], height],
            ]
        )
    )  # Right face

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="lightpink", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Add dashed line to indicate the height of the trapezoid base
    ax.plot(
        [top_left[0], top_left[0]],  # x coordinates
        [
            top_left[1],
            bottom_right[1],
        ],  # y coordinates
        [0, 0],  # z coordinates
        linestyle="--",
        color="red",
        linewidth=2,
        alpha=0.8,
        zorder=1000,
    )
    ax.plot(
        [top_left[0] + 0.25, top_left[0] + 0.25],  # x coordinates
        [
            bottom_right[1],
            bottom_right[1] + 0.25,
        ],  # y coordinates
        [0, 0],  # z coordinates
        linestyle="--",
        color="red",
        linewidth=2,
        alpha=0.8,
        zorder=1000,
    )
    ax.plot(
        [top_left[0] + 0.25, top_left[0]],  # x coordinates
        [
            bottom_right[1] + 0.25,
            bottom_right[1] + 0.25,
        ],  # y coordinates
        [0, 0],  # z coordinates
        linestyle="--",
        color="red",
        linewidth=2,
        alpha=0.8,
        zorder=1000,
    )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def _is_right_triangle(
    side_a: float, side_b: float, side_c: float, *, rel_tol: float = 1e-3
) -> bool:
    """Return ``True`` when the three side lengths satisfy the Pythagorean theorem.

    A *relative* tolerance is used instead of a fixed absolute tolerance so the
    check is robust for both very small and very large triangles.
    """
    sides = sorted([side_a, side_b, side_c])
    lhs = sides[0] ** 2 + sides[1] ** 2  # sum of squares of the two shorter sides
    rhs = sides[2] ** 2  # square of the longest side
    # math.isclose handles both relative and absolute tolerance checks
    return math.isclose(lhs, rhs, rel_tol=rel_tol, abs_tol=rel_tol)


def _find_right_angle_vertex(
    side_a: float, side_b: float, side_c: float, vertices: np.ndarray
) -> tuple[int, float, float]:
    """Identify which vertex of ``vertices`` (shape (3,2)) contains the right angle.

    The decision is made geometrically by checking the dot-product of the two
    edge vectors that emanate from each vertex.  The vertex whose adjacent
    edges are closest to being perpendicular (dot ≈ 0) is considered the right
    angle.
    """

    # Work fully in the 2-D plane of the base
    dots: list[tuple[int, float]] = []  # (vertex_index, |dot|)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        vec_ij = vertices[j] - vertices[i]
        vec_ik = vertices[k] - vertices[i]
        dot = abs(np.dot(vec_ij, vec_ik))
        dots.append((i, dot))

    # The right angle corresponds to the *smallest* absolute dot-product
    right_idx, _ = min(dots, key=lambda t: t[1])

    return right_idx, vertices[right_idx, 0], vertices[right_idx, 1]


def _calculate_marker_scale_factor(marker_size: float, leg_length: float) -> float:
    """Calculate scaling factor for angle markers that are too small to be visible."""
    # Define minimum visible size threshold
    min_leg_length = 0.2  # Minimum visible leg length
    min_marker_size = 0.15  # Minimum visible marker size

    # Check if either dimension would be too small
    if leg_length < min_leg_length or marker_size < min_marker_size:
        return 2.0  # Scale up by 2x
    return 1.0  # No scaling needed


def _draw_square_angle_marker(
    ax: Axes3D,
    vertex_point: np.ndarray,
    direction1: np.ndarray,
    direction2: np.ndarray,
    square_size: float,
    color: str = "red",
    linewidth: float = 1.5,
    zorder: int = 1004,
    auto_scale: bool = True,
):
    """Draw a square right angle marker at the vertex, transformed to align with the two directions."""

    # Auto-scale the marker if it would be too small
    scaled_square_size = square_size
    if auto_scale:
        # Calculate scaling factor based on square size
        scale_factor = _calculate_marker_scale_factor(square_size, square_size)
        scaled_square_size = square_size * scale_factor

    # Reverse and normalize direction vectors to point inward into the angle
    # This ensures the square appears inside the shape, not outside
    direction1_inward = -direction1 / np.linalg.norm(direction1)
    direction2_inward = -direction2 / np.linalg.norm(direction2)

    # Create the four corners of the square
    # All sides will be equal length since we use normalized directions
    corner1 = vertex_point  # The actual vertex
    corner2 = (
        vertex_point + direction1_inward * scaled_square_size
    )  # Along first edge (inward)
    corner3 = (
        vertex_point
        + direction1_inward * scaled_square_size
        + direction2_inward * scaled_square_size
    )  # Opposite corner (both directions inward)
    corner4 = (
        vertex_point + direction2_inward * scaled_square_size
    )  # Along second edge (inward)

    # Create the square/parallelogram by drawing all four edges
    corners = [corner1, corner2, corner3, corner4, corner1]  # Close the shape

    for i in range(len(corners) - 1):
        ax.plot(
            [corners[i][0], corners[i + 1][0]],
            [corners[i][1], corners[i + 1][1]],
            [corners[i][2], corners[i + 1][2]],
            color=color,
            linewidth=linewidth,
            zorder=zorder,
        )


def _draw_3d_right_angle_marker(ax, x, y, z, adjacent_vertices, marker_size=0.5):
    """Inset right-angle marker that sits inside the triangle and ends on the two edges."""
    p = np.array([float(x), float(y)], dtype=float)
    a = np.array(adjacent_vertices[0], dtype=float)  # along edge 1 from vertex
    b = np.array(adjacent_vertices[1], dtype=float)  # along edge 2 from vertex

    u = a - p
    v = b - p
    nu = np.linalg.norm(u) or 1.0
    nv = np.linalg.norm(v) or 1.0
    u /= nu
    v /= nv

    size = marker_size * 0.6  # inset and leg length
    p1 = p + u * size  # on edge 1
    p3 = p + v * size  # on edge 2
    p2 = p + u * size + v * size  # inside corner

    ax.plot(
        [p1[0], p2[0]], [p1[1], p2[1]], [z, z], color="red", linewidth=2.0, zorder=12
    )
    ax.plot(
        [p2[0], p3[0]], [p2[1], p3[1]], [z, z], color="red", linewidth=2.0, zorder=12
    )


def _draw_height_indicator(
    ax: Axes3D,
    vertices: np.ndarray,
    triangle_height: float,
    side_a: float,
    z: float = 0,
    units: str = "",
):
    """Draw a height indicator with dashed line and label for non-right triangles."""
    # The height is perpendicular to side_a (the base from (0,0) to (side_a,0))
    # Height line goes from vertex 0 (x3, y3) perpendicular to the base

    # Find the foot of the perpendicular from vertex 0 to the base (side_a)
    base_x = vertices[2, 0]  # x3 coordinate
    foot_x = base_x  # perpendicular drops straight down to x-axis
    foot_y = 0  # on the base line (y=0)

    # Draw dashed line from vertex to foot of perpendicular
    ax.plot(
        [vertices[2, 0], foot_x],
        [vertices[2, 1], foot_y],
        [z, z],
        linestyle="--",
        color="blue",
        linewidth=2,
        alpha=1.0,
        zorder=1001,
    )

    # Add right angle indicator at the base of the height line
    marker_size = 0.3

    # Create the right angle marker at the foot of the perpendicular
    # The marker shows the perpendicular relationship between height line and base

    # Vector along the base (side_a)
    # base_vector = np.array([1, 0])  # horizontal along x-axis

    # Vector along the height line
    # height_vector = np.array([0, 1])  # vertical along y-axis

    # Create the L-shaped marker using the same approach as right triangle markers
    # This ensures consistency and robustness

    # For height line indicators, use exactly perpendicular unit vectors
    # This ensures perfect square proportions

    # Edge 1: horizontal direction along the base (always unit length)
    edge1_unit = np.array([1, 0])

    # Edge 2: vertical direction (perpendicular to base, unit length)
    # The height line is by definition perpendicular to the base
    edge2_raw = np.array([vertices[2, 0] - foot_x, vertices[2, 1] - foot_y])
    # Create a perpendicular vector to the base that points toward the vertex
    # Since base is horizontal [1,0], perpendicular is [0,1] or [0,-1]
    if edge2_raw[1] > 0:  # vertex is above the base
        edge2_unit = np.array([0, 1])  # point up
    else:  # vertex is below the base
        edge2_unit = np.array([0, -1])  # point down

    # Calculate the angle bisector direction (same as right triangle logic)
    # This points into the triangle interior
    bisector_direction = edge1_unit + edge2_unit
    bisector_direction = bisector_direction / np.linalg.norm(bisector_direction)

    # Move the corner inward along the bisector (same offset approach as right triangles)
    # base_corner_offset = marker_size * 0.8  # Use same offset as right triangles
    base_leg_length = marker_size * 0.44  # Same as right triangle markers

    # Apply auto-scaling if the marker would be too small
    scale_factor = _calculate_marker_scale_factor(marker_size, base_leg_length)
    # corner_offset = base_corner_offset * scale_factor
    leg_length = base_leg_length * scale_factor

    # corner_point = np.array(
    #     [
    #         foot_x + bisector_direction[0] * corner_offset,
    #         foot_y + bisector_direction[1] * corner_offset,
    #         z,
    #     ]
    # )

    # Create perpendicular vectors to each edge (same logic as right triangles)
    # For a 2D vector [x, y], perpendiculars are [-y, x] and [y, -x]

    # Perpendicular to edge1 (base direction)
    edge1_perp_option1 = np.array([-edge1_unit[1], edge1_unit[0]])
    edge1_perp_option2 = np.array([edge1_unit[1], -edge1_unit[0]])

    # Perpendicular to edge2 (height direction)
    edge2_perp_option1 = np.array([-edge2_unit[1], edge2_unit[0]])
    edge2_perp_option2 = np.array([edge2_unit[1], -edge2_unit[0]])

    # Choose the perpendicular that points away from the triangle interior (same logic as right triangles)
    # Use dot product to determine which perpendicular points away from interior direction
    if np.dot(edge1_perp_option1, bisector_direction) < 0:
        edge1_perp = edge1_perp_option1
    else:
        edge1_perp = edge1_perp_option2

    if np.dot(edge2_perp_option1, bisector_direction) < 0:
        edge2_perp = edge2_perp_option1
    else:
        edge2_perp = edge2_perp_option2

    # Use the exact same process as right triangle markers (which work correctly)
    # Calculate the direction vectors exactly like right triangles do
    direction1 = np.array([edge1_perp[0], edge1_perp[1], 0])
    direction2 = np.array([edge2_perp[0], edge2_perp[1], 0])

    # Use the reusable helper function to draw the square marker
    _draw_square_angle_marker(
        ax,
        np.array([foot_x, foot_y, z]),  # Pass the intersection point as vertex
        direction1,
        direction2,
        leg_length,  # Use as square size
        color="blue",
        linewidth=1.5,
        zorder=1003,
        auto_scale=False,  # Already scaled above
    )

    # Add height label at the midpoint of the height line
    mid_x = (vertices[2, 0] + foot_x) / 2
    mid_y = (vertices[2, 1] + foot_y) / 2

    ax.text(
        mid_x - 0.3,
        mid_y,
        z + 0.2,
        f"{triangle_height:.1f} {units}",
        color="blue",
        fontsize=12,
        ha="center",
        va="bottom",
        zorder=1002,
        bbox=dict(facecolor="white", edgecolor="blue", alpha=1.0, pad=3),
    )


def draw_triangular_prism(ax: Axes3D, prism: TriangularPrism, units: str = ""):
    """Draw a triangular prism"""
    height = prism.height
    side_a = prism.side_a
    side_b = prism.side_b
    side_c = prism.side_c

    # Calculate triangle vertices using Heron's formula to find area and height
    # For simplicity, we'll create a triangle with given sides
    # Place the triangle with side_a along x-axis from origin
    x1, y1 = 0, 0
    x2, y2 = side_a, 0

    # Calculate third vertex using law of cosines
    cos_C = (side_a**2 + side_b**2 - side_c**2) / (2 * side_a * side_b)
    sin_C = np.sqrt(1 - cos_C**2)
    x3 = side_b * cos_C
    y3 = side_b * sin_C

    # Calculate the height of the triangular base using area formula
    # Area = (1/2) * base * height, where base is side_a
    # First calculate area using Heron's formula
    s = (side_a + side_b + side_c) / 2  # semi-perimeter
    area = np.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
    triangle_height = 2 * area / side_a  # height = 2*area/base

    # Store triangle properties for labeling
    prism._triangle_vertices = np.array([[x1, y1], [x2, y2], [x3, y3]])  # type: ignore
    prism._triangle_height = triangle_height  # type: ignore

    # Create bottom and top vertices
    bottom_vertices = np.array([[x1, y1, 0], [x2, y2, 0], [x3, y3, 0]])
    top_vertices = np.array([[x1, y1, height], [x2, y2, height], [x3, y3, height]])

    # Create faces
    faces = []

    # Bottom face
    faces.append(bottom_vertices)

    # Top face
    faces.append(top_vertices)

    # Lateral faces
    for i in range(3):
        next_i = (i + 1) % 3
        face = np.array(
            [
                bottom_vertices[i],
                bottom_vertices[next_i],
                top_vertices[next_i],
                top_vertices[i],
            ]
        )
        faces.append(face)

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d,
            facecolors="lightsteelblue",
            linewidths=2,
            edgecolors="black",
            alpha=0.6,
        )
    )

    # Check if triangle is a right triangle and add appropriate indicators
    vertices_2d = np.array([[x1, y1], [x2, y2], [x3, y3]])

    if _is_right_triangle(side_a, side_b, side_c):
        # Add right angle marker on bottom face
        vertex_idx, vertex_x, vertex_y = _find_right_angle_vertex(
            side_a, side_b, side_c, vertices_2d
        )

        # Get adjacent vertices for the right angle marker
        adjacent_indices = [(vertex_idx + 1) % 3, (vertex_idx + 2) % 3]
        adjacent_vertices = [
            (vertices_2d[i, 0], vertices_2d[i, 1]) for i in adjacent_indices
        ]

        # Calculate appropriate marker size based on triangle dimensions
        triangle_scale = min(side_a, side_b, side_c) * 0.2  # Increased from 0.15 to 0.2
        # Adjust the upper limit based on dimensions with a higher threshold
        max_size = (
            0.6 if min(side_a, side_b, side_c) < 10 else 3.0
        )  # Increased from 2.0 to 3.0
        marker_size = max(0.25, min(max_size, triangle_scale))

        _draw_3d_right_angle_marker(
            ax, vertex_x, vertex_y, 0, adjacent_vertices, marker_size=marker_size
        )

        # Also add right angle marker on top face
        _draw_3d_right_angle_marker(
            ax, vertex_x, vertex_y, height, adjacent_vertices, marker_size=marker_size
        )
    else:
        # Add height indicator on top face
        _draw_height_indicator(
            ax, vertices_2d, triangle_height, side_a, z=height, units=units
        )

    # Scale the axes equally
    ax.set_box_aspect([1, 1, 1])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_rectangular_prism_right(ax: Axes3D, prism: RectangularPrismRight):
    """Draw a rectangular prism"""
    height = prism.height

    # Handle both input formats: width+length OR base_area
    if prism.width is not None and prism.length is not None:
        width = prism.width
        length = prism.length
    elif prism.base_area is not None:
        # Calculate width and length from base area (make it square-like)
        base_area = prism.base_area
        # Use square root to get approximately equal dimensions
        side_length = np.sqrt(base_area)
        width = side_length
        length = side_length
    else:
        raise ValueError("Must provide either width+length OR base_area")

    # Define the vertices of the box
    vertices = np.array(
        [
            [0, 0, 0],
            [length, 0, 0],
            [length, width, 0],
            [0, width, 0],
            [0, 0, height],
            [length, 0, height],
            [length, width, height],
            [0, width, height],
        ]
    )

    # Define the faces of the box
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Back face
    ]

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d, facecolors="lightcyan", linewidths=2, edgecolors="black", alpha=0.6
        )
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()


def draw_cube_right(ax: Axes3D, prism: CubeRight):
    """Draw a cube"""
    # Handle both input formats: side_length OR base_area
    if prism.side_length is not None:
        side_length = prism.side_length
    elif prism.base_area is not None:
        # Calculate side length from base area
        side_length = np.sqrt(prism.base_area)
    else:
        raise ValueError("Must provide either side_length OR base_area")

    # Define the vertices of the cube
    vertices = np.array(
        [
            [0, 0, 0],
            [side_length, 0, 0],
            [side_length, side_length, 0],
            [0, side_length, 0],
            [0, 0, side_length],
            [side_length, 0, side_length],
            [side_length, side_length, side_length],
            [0, side_length, side_length],
        ]
    )

    # Define the faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom face
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top face
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left face
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Front face
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Back face
    ]

    # Create a Poly3DCollection
    poly3d = [np.array(face) for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            poly3d,
            facecolors="lightgoldenrodyellow",
            linewidths=2,
            edgecolors="black",
            alpha=0.6,
        )
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    ax.set_axis_off()
