import os
import time

import matplotlib
import matplotlib.pyplot as plt
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.object_array import (
    ObjectArray,
)
from content_generators.settings import settings


@stimulus_function
def draw_object_array(stimulus: ObjectArray):
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

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(stimulus.columns, stimulus.rows))
    ax.set_xlim(0, stimulus.columns)
    ax.set_ylim(0, stimulus.rows)
    ax.axis("off")

    # Plot the items in a grid
    for row in range(stimulus.rows):
        for col in range(stimulus.columns):
            ax.imshow(
                img, extent=(col, col + 1, stimulus.rows - row - 1, stimulus.rows - row)
            )

    # Save the figure
    file_name = f"{settings.additional_content_settings.image_destination_folder}/object_array_stimulus_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
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
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.object_array import (
        DrawableItem,
    )

    sample_stimulus = ObjectArray(object_name=DrawableItem.SUN, rows=3, columns=4)

    file_path = draw_object_array(sample_stimulus)
    print(f"Image saved to: {file_path}")

    # Open the image file
    from PIL import Image

    img = Image.open(file_path)
    matplotlib.use("TkAgg")
    # Display the image
    plt.imshow(img)
    plt.axis("off")  # Turn off axis
    plt.show()
