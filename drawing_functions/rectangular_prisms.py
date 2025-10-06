import time

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.rectangular_prisms import (
    BaseAreaRectangularPrism,
    BaseAreaRectangularPrismList,
    FillState,
    RectangularPrism,
    RectangularPrismList,
    UnitCubeFigure,
)
from content_generators.settings import settings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_rectangular_prism(
    ax, description: RectangularPrism, max_dimension: int, count: int
):
    height = description.height
    width = description.width
    length = description.length
    fill = description.fill

    # Draw black outlines (bottom and back)
    ax.plot(
        [0, length, length, 0, 0],
        [0, 0, width, width, 0],
        [0, 0, 0, 0, 0],
        color="black",
        linewidth=1,
        zorder=1,
        clip_on=False,
    )
    ax.plot(
        [0, 0], [0, 0], [0, height], color="black", linewidth=1, zorder=1, clip_on=False
    )
    ax.plot(
        [0, 0],
        [width, width],
        [0, height],
        color="black",
        linewidth=1,
        zorder=1,
        clip_on=False,
    )

    if fill == FillState.FULL:
        # Create a 3D grid of voxels filled to the brim
        for k in range(height):
            for j in reversed(range(width)):
                for i in range(length):
                    ax.bar3d(
                        i,
                        j,
                        k,
                        1,
                        1,
                        1,
                        shade=True,
                        color="cyan",
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=2,
                        clip_on=False,
                    )
    elif fill == FillState.PARTIAL or fill == FillState.BOTTOM:
        # Create a 2D frame of voxels at the bottom for both states
        for j in reversed(range(width)):
            for i in range(length):
                ax.bar3d(
                    i,
                    j,
                    0,
                    1,
                    1,
                    1,
                    shade=True,
                    color="cyan",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=2,
                    clip_on=False,
                )
        if fill == FillState.PARTIAL:
            # Include a column of blocks at the back for partial fill
            for k in range(height):
                ax.bar3d(
                    0,
                    width - 1,
                    k,
                    1,
                    1,
                    1,
                    shade=True,
                    color="cyan",
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=2,
                    clip_on=False,
                )

    # Draw black outlines (top and front)
    ax.plot(
        [0, length, length, 0, 0],
        [0, 0, width, width, 0],
        [height, height, height, height, height],
        color="black",
        linewidth=1,
        zorder=10000,
        clip_on=False,
    )
    ax.plot(
        [length, length],
        [0, 0],
        [0, height],
        color="black",
        linewidth=1,
        zorder=10000,
        clip_on=False,
    )
    ax.plot(
        [length, length],
        [width, width],
        [0, height],
        color="black",
        linewidth=1,
        zorder=10000,
        clip_on=False,
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    bbox_props = dict(
        boxstyle="round,pad=0.1", edgecolor="none", facecolor="white", alpha=1
    )
    font_size = 14

    if fill == FillState.EMPTY or fill == FillState.BOTTOM:
        if description.prism_unit_label is None:
            description.prism_unit_label = "units"

        # Draw measurement line for length
        if description.show_length:
            ax.plot(
                [0, length],
                [-0.5, -0.5],
                [0, 0],
                "--",
                color="green",
                linewidth=1,
                clip_on=False,
            )
            ax.plot(
                [0, 0], [0, -1], [0, 0], "--", color="green", linewidth=1, clip_on=False
            )
            ax.plot(
                [length, length],
                [0, -1],
                [0, 0],
                "--",
                color="green",
                linewidth=1,
                clip_on=False,
            )
            ax.text(
                length / 2,
                0 - 0.3,  # inside lower margin
                0 - 0.1,  # inside lower margin
                f"{length} {description.prism_unit_label}",
                color="green",
                ha="center",
                bbox=bbox_props,
                fontsize=font_size,
                zorder=10000,
                clip_on=False,
            )

        # Draw measurement line for width
        if description.show_width:
            ax.plot(
                [length + 0.5, length + 0.5],
                [0, width],
                [0, 0],
                "--",
                color="blue",
                linewidth=1,
            )
            ax.plot(
                [length, length + length * 0.2],
                [0, 0],
                [0, 0],
                "--",
                color="blue",
                linewidth=1,
                clip_on=False,
            )
            ax.plot(
                [length, length + length * 0.2],
                [width, width],
                [0, 0],
                "--",
                color="blue",
                linewidth=1,
                clip_on=False,
            )
            ax.text(
                length + length * 0.25,  # Relative to length
                width / 2,
                0,
                f"{width} {description.prism_unit_label}",
                color="blue",
                ha="center",
                bbox=bbox_props,
                fontsize=font_size,
                zorder=10000,
                clip_on=False,
            )

        # Draw measurement line for height
        if description.show_height:
            ax.plot(
                [-0.5, -0.5],
                [0, 0],
                [0, height],
                "--",
                color="red",
                linewidth=1,
                clip_on=False,
            )
            ax.plot(
                [0, -length * 0.2],
                [0, 0],
                [0, 0],
                "--",
                color="red",
                linewidth=1,
                clip_on=False,
            )
            ax.plot(
                [0, -length * 0.2],
                [0, 0],
                [height, height],
                "--",
                color="red",
                linewidth=1,
                clip_on=False,
            )
            ax.text(
                0 - 0.3,  # inside left margin
                0 - 0.3,  # inside lower margin
                height / 2,
                f"{height} {description.prism_unit_label}",
                color="red",
                ha="center",
                bbox=bbox_props,
                fontsize=font_size,
                zorder=10000,
                clip_on=False,
            )

    # Add key for unit cube with size label
    if description.unit_cube_unit_size_and_label:
        offset = length + 2
        key_size = 0.8
        ax.bar3d(
            offset,
            -1,
            0,
            key_size,
            key_size,
            key_size,
            shade=True,
            color="cyan",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
            clip_on=False,
        )
        ax.text(
            offset + 1,
            0,
            0,
            f"${description.unit_cube_unit_size_and_label.replace(" ", "\ ")}$",
            color="black",
            ha="left",
            fontsize=16,
            zorder=10000,
            clip_on=False,
        )


def draw_base_area_rectangular_prism(
    ax, description: BaseAreaRectangularPrism, max_dimension: int, count: int
):
    base_area = description.base_area
    height = description.height

    # Calculate length and width from base area
    # We'll use the closest factors to make it look more square-like
    length = int(np.ceil(np.sqrt(base_area)))
    width = int(np.ceil(base_area / length))

    # Draw black outlines (bottom and back)
    ax.plot(
        [0, length, length, 0, 0],
        [0, 0, width, width, 0],
        [0, 0, 0, 0, 0],
        color="black",
        linewidth=1,
        zorder=1,
        clip_on=False,
    )
    ax.plot(
        [0, 0], [0, 0], [0, height], color="black", linewidth=1, zorder=1, clip_on=False
    )
    ax.plot(
        [0, 0],
        [width, width],
        [0, height],
        color="black",
        linewidth=1,
        zorder=1,
        clip_on=False,
    )

    # Draw black outlines (top and front)
    ax.plot(
        [0, length, length, 0, 0],
        [0, 0, width, width, 0],
        [height, height, height, height, height],
        color="black",
        linewidth=1,
        zorder=10000,
        clip_on=False,
    )
    ax.plot(
        [length, length],
        [0, 0],
        [0, height],
        color="black",
        linewidth=1,
        zorder=10000,
        clip_on=False,
    )
    ax.plot(
        [length, length],
        [width, width],
        [0, height],
        color="black",
        linewidth=1,
        zorder=10000,
        clip_on=False,
    )

    # Draw green base
    ax.plot_surface(
        np.array([[0, length], [0, length]]),
        np.array([[0, 0], [width, width]]),
        np.array([[0, 0], [0, 0]]),
        color="green",
        alpha=0.3,
        zorder=2,
    )

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    bbox_props = dict(
        boxstyle="round,pad=0.1", edgecolor="none", facecolor="white", alpha=1
    )
    font_size = 14

    # Calculate axis limits with margin
    margin_factor = 0.3
    max_dim = max(length, width, height)
    x_margin = margin_factor * max_dim
    y_margin = margin_factor * max_dim
    z_margin = margin_factor * max_dim

    xlim = [-x_margin, length + x_margin]
    ylim = [-y_margin, width + y_margin]
    zlim = [-z_margin, height + z_margin]

    # Only call these on 3D axes
    if hasattr(ax, "set_zlim") and hasattr(ax, "set_box_aspect"):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])

    # Base area label (green)
    if description.show_base_area:
        ax.text(
            length / 2,
            ylim[0] + y_margin * 0.3,
            zlim[0] + z_margin * 0.1,
            f"Area of Base = {base_area} {description.prism_unit_label}$^2$",
            color="green",
            ha="center",
            bbox=bbox_props,
            fontsize=font_size,
            zorder=10000,
            clip_on=False,
        )

    # Draw measurement line for height
    if description.show_height:
        ax.plot(
            [-0.5, -0.5],
            [0, 0],
            [0, height],
            "--",
            color="red",
            linewidth=1,
            clip_on=False,
        )
        ax.plot(
            [0, -length * 0.2],
            [0, 0],
            [0, 0],
            "--",
            color="red",
            linewidth=1,
            clip_on=False,
        )
        ax.plot(
            [0, -length * 0.2],
            [0, 0],
            [height, height],
            "--",
            color="red",
            linewidth=1,
            clip_on=False,
        )
        ax.text(
            0 - 0.3,  # inside left margin
            0 - 0.3,  # inside lower margin
            height / 2,
            f"{height} {description.prism_unit_label}",
            color="red",
            ha="center",
            bbox=bbox_props,
            fontsize=font_size,
            zorder=10000,
            clip_on=False,
        )


@stimulus_function
def draw_unit_cube_figure(figure: UnitCubeFigure):  # ← CHANGED
    """
    Render either:
      • a right rectangular prism      (figure.shape.kind == 'rectangular'), or
      • an irregular set of unit cubes (figure.shape.kind == 'custom'),
    save the figure, and return the image path.
    """
    # ── build a figure & 3-D axis ──────────────────────────────────────────
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    # pleasant violet gradient for the three visible faces
    top_color = "#d6b4fc"
    front_color = "#b38dfc"
    right_color = "#7a5df5"
    edge_color = "black"
    edge_width = 0.5

    # ── iterate over every cube we need to draw ───────────────────────────
    if figure.shape.kind == "rectangular":
        L, W, H = (
            figure.shape.length,
            figure.shape.width,
            figure.shape.height,
        )
        cubes = [(i, j, k) for k in range(H) for j in range(W) for i in range(L)]
    else:  # 'custom'
        cubes = figure.shape.cubes
        # compute bounding-box dimensions for axis limits
        xs = [cube.x for cube in cubes]
        ys = [cube.y for cube in cubes]
        zs = [cube.z for cube in cubes]
        L, W, H = max(xs) + 1, max(ys) + 1, max(zs) + 1

    for cube in cubes:
        if isinstance(cube, tuple):
            i, j, k = cube
        else:
            i, j, k = cube.x, cube.y, cube.z

        # eight vertices of cube at (i, j, k)
        v0, v1, v2 = (i, j, k), (i + 1, j, k), (i + 1, j + 1, k)
        v4, v5, v6, v7 = (
            (i, j, k + 1),
            (i + 1, j, k + 1),
            (i + 1, j + 1, k + 1),
            (i, j + 1, k + 1),
        )

        for verts, colour in (
            ([v4, v5, v6, v7], top_color),  # +z top
            ([v1, v2, v6, v5], right_color),  # +x right
            ([v0, v1, v5, v4], front_color),  #  y front (toward viewer)
        ):
            ax.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=colour,
                    edgecolors=edge_color,
                    linewidths=edge_width,
                )
            )

    # ── tidy up view & save ──────────────────────────────────────────────
    max_dim = max(L, W, H)
    ax.set_xlim(-2, max_dim)
    ax.set_ylim(-2, max_dim)
    ax.set_zlim(-0.5, max_dim)  # type: ignore
    ax.set_box_aspect((1, 1, 0.8))  # type: ignore
    ax.set_axis_off()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/rectangular_prism_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_multiple_rectangular_prisms(prisms: RectangularPrismList):
    num_prisms = len(prisms)
    cols = int(np.ceil(np.sqrt(num_prisms)))  # Determine grid size
    rows = int(np.ceil(num_prisms / cols))

    # Calculate the maximum dimensions across all prisms
    max_length = max(prism.length for prism in prisms)
    max_width = max(prism.width for prism in prisms)
    max_height = max(prism.height for prism in prisms)
    additional_limit = (
        2 if any(prism.unit_cube_unit_size_and_label for prism in prisms) else 0.5
    )

    adjusted_length = max_length + additional_limit
    adjusted_width = max_width + additional_limit
    adjusted_height = max_height + additional_limit
    max_dimension = max(adjusted_length, adjusted_width, adjusted_height)

    fig = plt.figure(figsize=(cols * 5, rows * 5), dpi=100)

    scale = 1  # + max_dimension*(0.01*(len(prisms)//2))

    for index, prism in enumerate(reversed(prisms), start=1):
        ax = fig.add_subplot(rows, cols, len(prisms) - index + 1, projection="3d")

        # If this prism is a cube, switch to an orthographic, equal-aspect view
        if int(prism.height) == int(prism.width) == int(prism.length):
            s = float(prism.length)
            m = 0.6
            ax.set_proj_type("ortho")
            ax.view_init(elev=25, azim=-35)
            ax.set_xlim(-m, s + m)
            ax.set_ylim(-m, s + m)
            ax.set_zlim(-m, s + m)  # type: ignore
            ax.set_box_aspect((1, 1, 1))  # type: ignore
            draw_rectangular_prism(
                ax, prism, max_dimension=int(s + 2 * m), count=len(prisms)
            )
        else:
            # Rectangular prisms: keep existing perspective/aspect logic
            ax.set_xlim(-0.5, adjusted_length / scale - 0.5)
            ax.set_ylim(-1.5, adjusted_width / scale - 0.5)
            ax.set_zlim(-0.5, adjusted_height / scale - 0.5)  # type: ignore
            x_scale = adjusted_length * scale / max_dimension
            y_scale = adjusted_width * scale / max_dimension
            z_scale = adjusted_height * scale / max_dimension
            ax.set_box_aspect((x_scale, y_scale, z_scale))  # type: ignore
            draw_rectangular_prism(
                ax, prism, max_dimension=int(max_dimension), count=len(prisms)
            )

        if num_prisms > 1:
            ax.set_title(prism.title, loc="center")

    plt.tight_layout()
    file_name = f"{settings.additional_content_settings.image_destination_folder}/rectangular_prism_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_multiple_base_area_rectangular_prisms(prisms: BaseAreaRectangularPrismList):
    num_prisms = len(prisms)
    cols = int(np.ceil(np.sqrt(num_prisms)))  # Determine grid size
    rows = int(np.ceil(num_prisms / cols))

    # Calculate the maximum dimensions across all prisms
    max_base_area = max(prism.base_area for prism in prisms)
    max_height = max(prism.height for prism in prisms)

    # Calculate approximate length and width from max base area
    max_length = int(np.ceil(np.sqrt(max_base_area)))
    max_width = int(np.ceil(max_base_area / max_length))

    additional_limit = 0.5

    adjusted_length = max_length + additional_limit
    adjusted_width = max_width + additional_limit
    adjusted_height = max_height + additional_limit
    max_dimension = max(adjusted_length, adjusted_width, adjusted_height)

    fig = plt.figure(figsize=(cols * 5, rows * 5), dpi=100)
    scale = 1

    for index, prism in enumerate(reversed(prisms), start=1):
        ax = fig.add_subplot(rows, cols, len(prisms) - index + 1, projection="3d")
        # Set the same limits and aspect ratio for all subplots (before drawing)
        ax.set_xlim(-2.5, adjusted_length / scale - 0.5)
        ax.set_ylim(-2.5, adjusted_width / scale - 0.5)
        ax.set_zlim(-1.0, adjusted_height / scale - 0.5)  # type: ignore
        x_scale = adjusted_length * scale / max_dimension
        y_scale = adjusted_width * scale / max_dimension
        z_scale = adjusted_height * scale / max_dimension
        ax.set_box_aspect((x_scale, y_scale, z_scale))  # type: ignore
        draw_base_area_rectangular_prism(
            ax, prism, max_dimension=int(max_dimension), count=len(prisms)
        )
        if num_prisms > 1:
            ax.set_title(prism.title, loc="center")

    plt.tight_layout(pad=3.0)
    file_name = f"{settings.additional_content_settings.image_destination_folder}/base_area_rectangular_prism_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


if __name__ == "__main__":
    # Testing function
    stimulus_description = [
        {"title": "Prism A", "height": 9, "width": 3, "length": 6, "fill": "empty"},
    ]
    print(draw_multiple_rectangular_prisms(stimulus_description))
