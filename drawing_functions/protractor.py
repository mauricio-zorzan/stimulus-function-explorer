import os
import random  # Add import for random
import time

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.protractor import (
    Protractor,
    ProtractorPoint,
)
from content_generators.settings import settings
from PIL import Image

matplotlib.use("Agg")

IMAGE_WIDTH = 11.05
RAY_LENGTH = IMAGE_WIDTH / 2 + 0.3


@stimulus_function
def draw_protractor(stimulus: Protractor):
    """
    Draw a protractor stimulus with points and rays.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Load the protractor image
    img_path = os.path.join(
        os.path.dirname(__file__),
        "images",
        "protractor.png",
    )
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        ax.imshow(img, extent=(0, IMAGE_WIDTH, 0, 6))
    else:
        raise FileNotFoundError(f"Protractor image not found at {img_path}")

    # Set the origin for the polar coordinate system
    origin = (IMAGE_WIDTH / 2 - 0.005, 0.445)

    # Randomly decide whether to mirror the structure
    mirror = random.choice([True, False])

    # Adjust origin and end points for mirroring
    if mirror:
        origin = (IMAGE_WIDTH - origin[0], origin[1])
        end_x = -0.3
        arrow_marker = "<"
        arrow_x = -0.8
    else:
        end_x = IMAGE_WIDTH + 0.3
        arrow_marker = ">"
        arrow_x = IMAGE_WIDTH + 0.5

    # Draw ray PQ (base ray)
    ax.plot(
        [origin[0], end_x],
        [origin[1], origin[1]],
        color="black",
        linewidth=2,
        marker="None",
    )

    # Draw arrow at the end of the PQ ray
    ax.plot(
        [end_x],
        [origin[1]],
        color="black",
        linewidth=2,
        marker=arrow_marker,
        markersize=10,
    )
    ax.text(arrow_x, origin[1], "Q", fontsize=16, verticalalignment="center")

    # Draw points and rays
    for point in stimulus.root:
        angle = np.radians(point.degree)
        x = origin[0] + RAY_LENGTH * np.cos(angle)
        y = origin[1] + RAY_LENGTH * np.sin(angle)

        if mirror:
            x = IMAGE_WIDTH - x  # Mirror the x-coordinate

        # Draw ray
        ax.plot(
            [origin[0], x], [origin[1], y], color="red", linewidth=1, linestyle="--"
        )

        # Draw point
        ax.plot(x, y, "ro", markersize=6)

        # Calculate new label position
        label_distance = 0.3  # Distance to move the label away from the point
        if mirror:
            label_angle = np.pi - angle  # Flip the angle for mirrored labels
        else:
            label_angle = angle

        label_x = x + label_distance * np.cos(label_angle)
        label_y = y + label_distance * np.sin(label_angle)

        # Add label
        ax.text(
            label_x,
            label_y,
            point.label,
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Add P label at the origin
    ax.text(
        origin[0],
        origin[1] - 0.1,
        "P",
        fontsize=16,
        horizontalalignment="center",
        verticalalignment="top",
    )
    ax.plot(origin[0], origin[1], "o", markersize=6, color="black")

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set aspect ratio to equal
    ax.set_aspect("equal")

    # Remove axes
    ax.axis("off")

    file_name = f"{settings.additional_content_settings.image_destination_folder}/protractor_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


if __name__ == "__main__":
    # Test the function with a sample stimulus
    sample_stimulus = Protractor(
        root=[
            ProtractorPoint(label="R", degree=115),
            ProtractorPoint(label="S", degree=65),
            ProtractorPoint(label="T", degree=90),
            ProtractorPoint(label="V", degree=180),
        ]
    )

    file_path = draw_protractor(sample_stimulus)
    print(f"Image saved to: {file_path}")

    # Open the image file
    img = Image.open(file_path)
    matplotlib.use("TkAgg")
    # Display the image
    plt.imshow(img)
    plt.axis("off")  # Turn off axis
    plt.show()
