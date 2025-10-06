import time

import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.drawing_functions.images.drawing_image import (
    DrawingImage,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.ruler import (
    MeasuredItem,
    MeasuredItemName,
    MeasurementUnit,
    Ruler,
    RulerStimulus,
)
from content_generators.settings import settings
from content_generators.utils import Lerp
from matplotlib.patches import Rectangle

DPI = 450


@stimulus_function
def draw_ruler_measured_objects(stimulus: RulerStimulus) -> str:
    fig, ax = plt.subplots(figsize=(12, len(stimulus.items) * 2))

    # Calculate the maximum length in centimeters for scaling
    max_length = max(item.ruler.length_in_cm or 0 for item in stimulus.items)

    # Set up the plot
    ax.axis("off")
    y_pos = 0
    ax.figure.dpi = DPI
    # Draw rulers and items
    for index, item in enumerate(reversed(stimulus.items)):
        y_pos += draw_ruler(ax, item.ruler, y_pos + 1)
        y_pos += draw_item(
            ax, item, y_pos, y_padding=0 if index == len(stimulus.items) - 1 else 0.4
        )

    ax.set_xlim(0, max_length + 1)
    ax.set_ylim(0, y_pos)
    # Save and return the file path
    file_name = f"{settings.additional_content_settings.image_destination_folder}/ruler_measured_objects_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
        dpi=DPI,
    )
    plt.close()

    return file_name


def draw_ruler(ax, ruler: Ruler, y_pos: float) -> float:
    length = ruler.length_in_cm or 0
    height = 1
    # Draw the ruler rectangle with rounded edges
    # Calculate ruler width to match the actual tick span
    if ruler.unit == MeasurementUnit.INCHES:
        # For inch rulers, calculate the actual span of tick marks including offset and small buffer
        ruler_length_inches = ruler.length or 0
        ruler_width = (
            ruler_length_inches * 2.54 + 0.5 + 0.3
        )  # Convert inches to cm, add x_offset + small buffer
    else:
        # Centimeter rulers work fine with small buffer
        ruler_width = length + 1

    rect = Rectangle(  # type: ignore
        (0, y_pos),  # Shifted 1 unit to the right
        ruler_width,
        -height,
        linewidth=3,
        edgecolor="black",
        facecolor="none",
        capstyle="round",
    )
    ax.add_patch(rect)

    # Draw ticks and numbers
    tick_heights = [height / (i + 3) for i in range(3)]  # Major, half, quarter ticks
    tick_heights[0] = tick_heights[0] * 0.9
    x_offset = 0.5
    increment = 2.54
    if ruler.unit == MeasurementUnit.CENTIMETERS:
        increment = 1
        # Add millimeter ticks for centimeters
        for i in range(int(length * 10) + 1):
            pos = i * 0.1  # 0.1 cm = 1 mm
            if i % 10 != 0:  # Skip positions where we'll draw centimeter ticks
                # Make half-centimeter (5mm) marks longer and thicker
                if i % 5 == 0:  # Half-centimeter marks
                    tick_height = tick_heights[2] * 1.1  # Longer than before
                    line_width = 1.5  # Thicker line
                else:
                    tick_height = tick_heights[2] * 0.7  # Regular millimeter marks
                    line_width = 1  # Regular line width
                ax.plot(
                    [pos + x_offset, pos + x_offset],
                    [y_pos, y_pos - tick_height],
                    "k-",
                    linewidth=line_width,
                )

    for i in range(int(length / increment) + 1):
        pos = i * increment
        ax.plot(
            [pos + x_offset, pos + x_offset],
            [y_pos, y_pos - tick_heights[0]],
            "k-",
            linewidth=2,
        )  # Shifted 1 unit to the right
        if pos < length:
            if ruler.unit == MeasurementUnit.INCHES:
                # Only draw half and quarter ticks for inches
                ax.plot(
                    [
                        pos + x_offset + increment * 0.5,
                        pos + x_offset + increment * 0.5,
                    ],
                    [y_pos, y_pos - tick_heights[1]],
                    "k-",
                    linewidth=2,
                )
                ax.plot(
                    [
                        pos + x_offset + increment * 0.25,
                        pos + x_offset + increment * 0.25,
                    ],
                    [y_pos, y_pos - tick_heights[2]],
                    "k-",
                    linewidth=2,
                )
                ax.plot(
                    [
                        pos + x_offset + increment * 0.75,
                        pos + x_offset + increment * 0.75,
                    ],
                    [y_pos, y_pos - tick_heights[2]],
                    "k-",
                    linewidth=2,
                )

        # Add numbers to the main delimiters
        if (
            abs(pos % increment) < 0.01 * increment
            or abs(pos % increment) > 0.99 * increment
        ):
            ax.text(
                pos + x_offset,
                y_pos - (tick_heights[0] * 1.2),
                str(i),
                ha="center",
                va="top",
                fontsize=14,
            )  # Shifted 1 unit to the right

    # Add unit label
    ax.text(
        length / 2,
        y_pos - height,
        ruler.unit.value,
        ha="center",
        va="bottom",
        fontsize=18,
    )  # Shifted 1 unit to the right

    return height


def draw_item(
    ax: plt.Axes,  # type: ignore
    item: MeasuredItem,
    y_pos: float,
    x_offset: float = 0.5,
    y_padding: float = 0.2,
) -> float:
    length = item.length_in_cm or 0
    padding = 0.1
    image = DrawingImage(item.name.value)
    image_height = Lerp.linear(image.aspect_ratio, image.aspect_ratio * 10, length / 30)

    # Convert start_position to centimeters to match the coordinate system
    start_position_cm = Ruler.convert_to_unit(
        item.start_position, item.ruler.unit, MeasurementUnit.CENTIMETERS
    )

    # Calculate the actual start position based on item's start_position (in cm)
    actual_start_x = start_position_cm + x_offset
    actual_end_x = actual_start_x + length

    # Display the image using ax.imshow
    ax.imshow(
        image.image,
        extent=(
            actual_start_x,
            actual_end_x,
            y_pos + padding,
            y_pos + image_height + padding,
        ),  # Adjust the extent to center the image around y_pos + padding
        aspect="auto",
    )

    # Draw measurement lines
    measurement_line_height = image_height + padding * 2
    ax.plot(
        [actual_start_x, actual_end_x],
        [y_pos + measurement_line_height, y_pos + measurement_line_height],
        "r--",
    )  # Shifted 1 unit to the right
    ax.plot(
        [actual_start_x, actual_start_x], [y_pos, y_pos + measurement_line_height], "r-"
    )  # Shifted 1 unit to the right
    ax.plot(
        [actual_end_x, actual_end_x],
        [y_pos, y_pos + measurement_line_height],
        "r-",
    )  # Shifted 1 unit to the right

    # Add item label
    label = ax.text(
        length / 2 + actual_start_x,
        y_pos + measurement_line_height + padding,
        item.name.value if item.label is None else item.label,
        ha="center",
        va="bottom",
        fontsize=20,
    )  # Shifted 1 unit to the right

    return (
        measurement_line_height
        + padding
        + float(
            label.get_window_extent(renderer=ax.figure.canvas.get_renderer()).height  # type: ignore
            / ax.figure.dpi
        )
        + y_padding
    )


if __name__ == "__main__":
    # Create test cases with non-zero start positions
    test_cases = [
        # Test case 1: Centimeter ruler with object starting at 1 cm
        RulerStimulus(
            items=[
                MeasuredItem(
                    name=MeasuredItemName.PENCIL,
                    length=4.0,  # 4 cm long
                    start_position=1.0,  # starts at 1 cm mark
                    ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
                ),
            ],
        ),
        # Test case 2: Inch ruler with object starting at 0.5 inches
        RulerStimulus(
            items=[
                MeasuredItem(
                    name=MeasuredItemName.ARROW,
                    length=2.5,  # 2.5 inches long
                    start_position=0.5,  # starts at 0.5 inch mark
                    ruler=Ruler(unit=MeasurementUnit.INCHES),
                ),
            ],
        ),
        # Test case 3: Multiple objects with different start positions
        RulerStimulus(
            items=[
                MeasuredItem(
                    name=MeasuredItemName.STRAW,
                    length=3.0,  # 3 cm long
                    start_position=2.0,  # starts at 2 cm mark
                    ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
                ),
                MeasuredItem(
                    name=MeasuredItemName.PENCIL,
                    length=1.75,  # 1.75 inches long
                    start_position=1.25,  # starts at 1.25 inch mark
                    ruler=Ruler(unit=MeasurementUnit.INCHES),
                ),
            ],
        ),
        # Test case 4: Fractional start positions
        RulerStimulus(
            items=[
                MeasuredItem(
                    name=MeasuredItemName.ARROW,
                    length=2.5,  # 2.5 cm long
                    start_position=1.5,  # starts at 1.5 cm mark
                    ruler=Ruler(unit=MeasurementUnit.CENTIMETERS),
                ),
            ],
        ),
    ]

    print("Generating test images with non-zero start positions...")

    for i, stimulus in enumerate(test_cases, 1):
        # Generate the image
        file_path = draw_ruler_measured_objects(stimulus)
        print(f"Test case {i}: Generated image: {file_path}")

        # Print details for each item
        for j, item in enumerate(stimulus.items):
            print(f"  Item {j+1}: {item.name.value}")
            print(f"    Length: {item.length} {item.ruler.unit.value}")
            print(f"    Start position: {item.start_position} {item.ruler.unit.value}")
            print(
                f"    End position: {item.start_position + item.length} {item.ruler.unit.value}"
            )
            print(f"    Ruler length: {item.ruler.length} {item.ruler.unit.value}")
            print(f"    Ruler length in cm: {item.ruler.length_in_cm} cm")
        print()

    print("All test cases completed successfully!")
