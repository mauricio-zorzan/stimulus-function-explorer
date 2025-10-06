import time

import matplotlib.pyplot as plt
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.spinner import (
    Spinner,
)
from content_generators.settings import settings
from matplotlib.patches import Circle


@stimulus_function
def generate_spinner(stimulus_description: Spinner):
    """
    Function to generate a carnival spinner visualization
    :param stimulus_description: Contains title and section details
    """
    # Parse stimulus description
    title = stimulus_description.title
    sections = stimulus_description.sections

    # Initialize data structures
    labels = []
    sizes = []
    section_colors = {}

    # Generate a unique color for each label
    label_color_mapping = {
        "Red": "#FF0000",
        "Green": "#00FF00",
        "Blue": "#0000FF",
        "Yellow": "#FFFF00",
        "Purple": "#800080",
        "Pink": "#FFC0CB",
    }

    is_colored = False

    # Assign colors to each label and prepare data for plotting
    for label in sections:
        labels.append(label)
        sizes.append(100 / len(sections))

        # assign color if label matches known colors
        if label in label_color_mapping:
            section_colors[label] = label_color_mapping[label]
            is_colored = True
        else:
            section_colors[label] = "#00000000"  # colorless section

    colors = [section_colors[label] for label in labels]

    # Create figure and axis objects
    fig, ax = plt.subplots()

    # Create a pie
    wedges, txt, autotexts = ax.pie(  # type: ignore
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%",
        wedgeprops=dict(width=0.2),
    )

    # draw lines from the center to the edge
    for wedge in wedges:
        # center = wedge.center
        radius = wedge.r
        theta1, theta2 = wedge.theta1, wedge.theta2
        circle_radius = 0  # the extent to which lines should reach the center.
        ax.plot(
            [0, radius * (1 - circle_radius) * np.cos(np.deg2rad(theta1))],
            [0, radius * (1 - circle_radius) * np.sin(np.deg2rad(theta1))],
            "k",
        )
        ax.plot(
            [0, radius * (1 - circle_radius) * np.cos(np.deg2rad(theta2))],
            [0, radius * (1 - circle_radius) * np.sin(np.deg2rad(theta2))],
            "k",
        )

    # draw a black border around the circle
    circle = Circle((0, 0), 1, edgecolor="black", facecolor="none")
    ax.add_artist(circle)

    if is_colored:
        # draw a black border around the inner spinner
        circle = Circle((0, 0), 0.8, edgecolor="black", facecolor="none")
        ax.add_artist(circle)

    # Set equal aspect ratio to make it circular
    ax.axis("equal")

    # Set the autotext labels to their respective labels from our input
    for i, autotext in enumerate(autotexts):
        autotext.set_text("{}".format(labels[i]))

    # Set the title
    plt.title(title, fontsize=14)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/spinner_{int(time.time() * 1000000)}.{settings.additional_content_settings.stimulus_image_format}"
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
