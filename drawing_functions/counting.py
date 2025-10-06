import os
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.counting import (
    Counting,
    DrawableItem,
)
from content_generators.settings import settings
from PIL import Image


@stimulus_function
def draw_counting(stimulus: Counting):
    # Load the image for the drawable item
    img_path = os.path.join(
        os.path.dirname(__file__),
        "images",
        f"{stimulus.object_name.value}.png",
    )
    if os.path.exists(img_path):
        img = plt.imread(img_path)
    else:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Calculate base number of items per row
    base_items_per_row = int(np.ceil(np.sqrt(stimulus.count)))

    # Create positions with variable items per row
    positions = []
    max_items_in_row = 0
    rows = 0
    remaining_items = stimulus.count
    while remaining_items > 0:
        items_in_row = min(base_items_per_row + random.randint(-1, 1), remaining_items)
        max_items_in_row = max(max_items_in_row, items_in_row)
        start_x = (max_items_in_row - items_in_row) / 2  # Center the row

        for i in range(items_in_row):
            x = start_x + i
            positions.append((x, rows))

        rows += 1  # Move to next row
        remaining_items -= items_in_row

    # Set figure size based on the layout
    fig_width = max_items_in_row  # Add a small margin
    fig_height = rows  # Add a small margin
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Shuffle the positions within each row
    row_positions = [
        positions[i : i + max_items_in_row]
        for i in range(0, len(positions), max_items_in_row)
    ]
    for row in row_positions:
        random.shuffle(row)
    positions = [pos for row in row_positions for pos in row]

    # Plot the items
    for x, y in positions:
        ax.imshow(img, extent=(x, x + 1, fig_height - y - 1, fig_height - y))

    file_name = f"{settings.additional_content_settings.image_destination_folder}/counting_stimulus_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.1,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


if __name__ == "__main__":
    # Test the function with a sample stimulus
    sample_stimulus = Counting(object_name=DrawableItem.SUN, count=12)

    file_path = draw_counting(sample_stimulus)
    print(f"Image saved to: {file_path}")

    # Open the image file
    img = Image.open(file_path)
    matplotlib.use("TkAgg")
    # Display the image
    plt.imshow(img)
    plt.axis("off")  # Turn off axis
    plt.show()
