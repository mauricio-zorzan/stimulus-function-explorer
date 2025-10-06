import time
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.drawing_functions.images.drawing_image import (
    DrawingImage,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.measurement_comparison import (
    MeasuredItemName,
    MeasuredObject,
    MeasurementComparison,
    MeasurementUnitImage,
    UnitDisplayError,
)
from content_generators.settings import settings
from content_generators.utils import Lerp, Math
from matplotlib.axes._axes import Axes
from PIL import Image

matplotlib.use("Agg")

BASE_TEXT_WIDTH = 8
LABEL_SPACE = 1


@stimulus_function
def draw_measurement_comparison(stimulus: MeasurementComparison):
    """
    Draw a measurement comparison stimulus.
    """
    has_units = stimulus[0].unit is not None
    height_per_object = 1 if has_units else 2.1
    total_height = len(stimulus) * height_per_object
    total_width = max(stimulus.lengths) + LABEL_SPACE

    has_small_widths = min(stimulus.lengths) < 4

    fig, axs = plt.subplots(
        len(stimulus),
        1,
        figsize=(total_width, total_height),
    )
    plt.subplots_adjust(
        hspace=0.1 if has_units else (-0.6 if has_small_widths else -0.4),
    )

    if len(stimulus) == 1:
        axs = [axs]
    else:
        axs = axs[::-1]

    max_length = max(obj.length for obj in stimulus)

    for i, obj in enumerate(reversed(stimulus)):
        ax: Axes = axs[i]
        y_limit = 1 if has_units else 0.5
        ax.set_xlim(-LABEL_SPACE / 2, max_length)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_aspect("equal")
        ax.axis("off")  # Remove axes

        # Wrap the label text
        wrapped_label = "\n".join(wrap(obj.label, width=BASE_TEXT_WIDTH))

        # Get figure x_limit dimension
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min

        y_aspect = x_range / (x_range + 0.05)

        # Define a common width for the label area (in axes coordinates)
        label_area_width = LABEL_SPACE  # Adjust this value as needed

        # Calculate font size based on figure height
        figure_height = fig.get_size_inches()[1] * fig.dpi / 5
        font_size = figure_height * (
            (0.05 + (0.01 if len(stimulus) == 2 else 0))
            if has_units
            else 0.07 - (0.02 if has_small_widths else 0)
        )  # Adjust the multiplier as needed

        # Wrap the label text to fit within the label area
        wrapped_label = "\n".join(
            wrap(obj.label, width=int(label_area_width * BASE_TEXT_WIDTH))
        )

        # Add the text within the label area
        ax.text(
            0,  # Position the text within the label area
            0.5,
            wrapped_label,
            ha="right",  # Align text to the right within the label area
            va="center",
            fontsize=font_size,
            transform=ax.transAxes,
        )

        image = DrawingImage(obj.object_name.value)
        image_height = Lerp.linear(
            image.aspect_ratio * 3, image.aspect_ratio * 12, obj.length / 12
        )

        if obj.unit is None:
            extent = (
                0,
                obj.length,
                Math.clamp(-image_height / 2, -0.5, 0),
                Math.clamp(image_height / 2, 0, 0.5),
            )
        else:
            extent = (
                0,
                obj.length,
                0.1,
                Math.clamp(image_height + 0.1, 0, 0.99),
            )
        ax.imshow(image.image, extent=extent, aspect="equal")

        # Add unit images if specified
        if obj.unit:
            if obj.unit_display_error == UnitDisplayError.GAP:
                gap_count = 2 if obj.length > 5 else 1
                unit_count = obj.length - gap_count
                positions = np.linspace(0, obj.length - 1, unit_count)
            elif obj.unit_display_error == UnitDisplayError.OVERLAP:
                overlap_count = 2 if obj.length > 5 else 1
                unit_count = obj.length + overlap_count
                positions = np.linspace(0, obj.length - 1, unit_count)
                np.random.shuffle(positions)
            else:
                unit_count = obj.length
                positions = np.linspace(0, obj.length - 1, unit_count)

            if obj.unit == MeasurementUnitImage.UNIT_SQUARES:
                for pos in positions:
                    square_width = 1
                    square_height = y_aspect
                    square = plt.Rectangle(
                        (pos, -1),
                        square_width,
                        square_height,
                        fill=True,
                        edgecolor="black",
                        linewidth=2,
                    )
                    ax.add_patch(square)
            else:
                unit_image = DrawingImage(obj.unit.value)
                for pos in positions:
                    extent = (
                        pos,
                        pos + 1,
                        -y_aspect,
                        0,
                    )
                    ax.imshow(unit_image.image, extent=extent, aspect="equal")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/measurement_comparison_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=800,
    )
    plt.close()

    return file_name


if __name__ == "__main__":
    # Test the function with a sample stimulus
    sample_stimulus = MeasurementComparison(
        [
            MeasuredObject(
                object_name=MeasuredItemName.STRAW,
                length=1,
                label="Pencil A",
                unit=MeasurementUnitImage.UNIT_SQUARES,
            ),
            MeasuredObject(
                object_name=MeasuredItemName.STRAW,
                length=12,
                label="Pencil B",
                unit=MeasurementUnitImage.UNIT_SQUARES,
            ),
            MeasuredObject(
                object_name=MeasuredItemName.ARROW,
                length=12,
                label="Pencil B",
                unit=MeasurementUnitImage.UNIT_SQUARES,
            ),
        ],
    )

    file_path = draw_measurement_comparison(sample_stimulus)
    print(f"Image saved to: {file_path}")

    # Open the image file
    img = Image.open(file_path)
    matplotlib.use("TkAgg")
    # Display the image
    plt.imshow(img)
    plt.axis("off")  # Turn off axis
    plt.show()
