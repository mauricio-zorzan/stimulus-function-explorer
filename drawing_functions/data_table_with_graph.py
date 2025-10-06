import textwrap
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Pydantic models (imported from or defined in data_table_with_graph.py):
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.table_with_graph import (
    # The model that enforces data_table + graph with strict graph_type
    DrawTableAndGraph,
    LineGraphs,
    MultiBarChart,
)

# Local settings import
from content_generators.settings import settings
from PIL import Image, ImageDraw, ImageFont

matplotlib.rcParams["font.family"] = "serif"


@stimulus_function
def draw_table_and_graph(stimulus_description: DrawTableAndGraph) -> str:
    """
    Creates a data table image (using create_table) and displays it side by side
    with either a bar graph or a line graph, based on the validated Pydantic model.
    """

    # Access data_table and graph using dot notation
    data_table = stimulus_description.data_table
    graph = stimulus_description.graph

    # Create the data table image
    table_img = create_table(data_table.dict())

    # Set up figure with 2 subplots: one for the table, one for the chart
    fig, (ax_table, ax_chart) = plt.subplots(1, 2, figsize=(20, 6))

    # Display the PIL image of the table in ax_table
    ax_table.imshow(table_img)
    ax_table.axis("off")  # Hide axes for the table

    # Render the chart on ax_chart
    if graph.graph_type == "bar_graph":
        draw_multi_bar_chart(graph, ax_chart)  # Ensure this function uses dot notation
    elif graph.graph_type == "line_graph":
        draw_line_graphs(graph, ax_chart)  # Ensure this function uses dot notation
    else:
        raise ValueError(f"Unknown graph_type: {graph.graph_type}")

    # Generate file_name for output
    file_name = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"draw_table_with_graph{int(time.time())}." 
        f"{settings.additional_content_settings.stimulus_image_format}"
    )

    plt.tight_layout()
    plt.savefig(
        file_name,
        dpi=600,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )

    return file_name


def create_table(json_input: dict) -> Image.Image:
    """
    Generates a PIL image containing a table, given JSON-style input in the form:
      {
        "title":    (optional) str
        "metadata": (optional) str
        "headers":  list[str],
        "data":     list[list[str]]
      }

    This is your original function that actually draws rows and columns
    with text-wrapping, so we get a real visible table.
    """
    table_data = json_input

    # Validate required fields
    if not all(key in table_data for key in ["headers", "data"]):
        raise ValueError("Missing required fields in JSON input")

    # Constants
    FONT_SIZE = 14
    TITLE_FONT_SIZE = FONT_SIZE + 2
    CELL_PADDING = 14
    MAX_CELL_WIDTH = 180
    LINE_SPACING = 5
    TITLE_TABLE_PADDING = 8

    # Load fonts
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
        bold_font = ImageFont.truetype("arialbd.ttf", FONT_SIZE)
        title_font = ImageFont.truetype("arialbd.ttf", TITLE_FONT_SIZE)
    except IOError:
        # Fallback if Arial is not available
        font = ImageFont.load_default()
        bold_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    def wrap_text(text, the_font, max_width):
        """
        Splits the text into multiple lines so that no line's pixel width
        exceeds max_width minus cell padding.
        """
        words = text.split()
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}" if line else word
            w = the_font.getlength(test_line)
            if w <= max_width - 2 * CELL_PADDING:
                line = test_line
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    # Create a dummy image and drawing context for measuring text
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    # Calculate column widths
    col_widths = []
    for i in range(len(table_data["headers"])):
        header_lines = wrap_text(
            str(table_data["headers"][i]), bold_font, MAX_CELL_WIDTH
        )
        header_width = (
            max(bold_font.getlength(line) for line in header_lines) + 2 * CELL_PADDING
        )

        max_data_width = 0
        for row in table_data["data"]:
            if i < len(row):
                cell_lines = wrap_text(str(row[i]), font, MAX_CELL_WIDTH)
                cell_width = (
                    max(font.getlength(line) for line in cell_lines) + 2 * CELL_PADDING
                )
                max_data_width = max(max_data_width, cell_width)
        col_width = min(max(header_width, max_data_width), MAX_CELL_WIDTH)
        col_widths.append(col_width)

    # Calculate row heights
    header_height = max(
        sum(
            (bold_font.getbbox(line)[3] - bold_font.getbbox(line)[1]) + LINE_SPACING
            for line in wrap_text(str(header), bold_font, col_widths[idx])
        )
        - LINE_SPACING
        + 2 * CELL_PADDING
        for idx, header in enumerate(table_data["headers"])
    )

    row_heights = []
    for row in table_data["data"]:
        row_height = 0
        for idx, cell in enumerate(row):
            if idx < len(col_widths):
                cell_lines = wrap_text(str(cell), font, col_widths[idx])
                cell_height = (
                    sum(
                        (font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING
                        for line in cell_lines
                    )
                    - LINE_SPACING
                    + 2 * CELL_PADDING
                )
                row_height = max(row_height, cell_height)
        row_heights.append(row_height)

    # Calculate total table dimensions
    table_width = sum(col_widths) + 1  # +1 for rightmost grid line
    table_height = header_height + sum(row_heights) + 1  # +1 for bottom grid line

    # Add space for title + metadata if present
    title_height = 0
    if "title" in table_data and table_data["title"]:
        title_lines = wrap_text(str(table_data["title"]), title_font, table_width)
        title_height = (
            sum(
                (title_font.getbbox(line)[3] - title_font.getbbox(line)[1])
                + LINE_SPACING
                for line in title_lines
            )
            - LINE_SPACING
            + 2 * CELL_PADDING
        )

    metadata_height = 0
    if "metadata" in table_data and table_data["metadata"]:
        metadata_lines = wrap_text(str(table_data["metadata"]), font, table_width)
        metadata_height = (
            sum(
                (font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING
                for line in metadata_lines
            )
            - LINE_SPACING
            + 2 * CELL_PADDING
        )

    total_height = title_height + TITLE_TABLE_PADDING + table_height + metadata_height

    # Create new image
    img = Image.new("RGB", (int(table_width), int(total_height)), "white")
    draw = ImageDraw.Draw(img)

    # Draw title if present
    current_y = 0
    if "title" in table_data and table_data["title"]:
        y_text = current_y + CELL_PADDING
        title_lines = wrap_text(str(table_data["title"]), title_font, table_width)
        for line in title_lines:
            draw.text((CELL_PADDING, y_text), line, font=title_font, fill="black")
            line_height = title_font.getbbox(line)[3] - title_font.getbbox(line)[1]
            y_text += line_height + LINE_SPACING
        current_y += title_height + TITLE_TABLE_PADDING

    # Draw headers
    current_x = 0
    for i, header in enumerate(table_data["headers"]):
        # Header background
        draw.rectangle(
            (
                current_x,
                current_y,
                current_x + col_widths[i],
                current_y + header_height,
            ),
            fill="lightgray",
        )
        # Draw horizontal grid line at top of headers
        draw.line([(0, current_y), (table_width, current_y)], fill="black")
        # Draw vertical grid line at the start of each header cell
        draw.line(
            [(current_x, current_y), (current_x, current_y + header_height)],
            fill="black",
        )

        # Wrap header text
        header_lines = wrap_text(str(header), bold_font, col_widths[i])
        y_text = current_y + CELL_PADDING
        for line in header_lines:
            draw.text(
                (current_x + CELL_PADDING, y_text), line, font=bold_font, fill="black"
            )
            line_height = bold_font.getbbox(line)[3] - bold_font.getbbox(line)[1]
            y_text += line_height + LINE_SPACING

        current_x += col_widths[i]
        # Draw vertical grid line at end of each header cell
        draw.line(
            [(current_x, current_y), (current_x, current_y + header_height)],
            fill="black",
        )

    current_y += header_height
    # Draw horizontal line below headers
    draw.line([(0, current_y), (table_width, current_y)], fill="black")

    # Draw data rows
    for row_idx, row in enumerate(table_data["data"]):
        current_x = 0
        row_height = row_heights[row_idx]
        for col_idx in range(len(table_data["headers"])):
            # Draw vertical grid line at start of each cell
            draw.line(
                [(current_x, current_y), (current_x, current_y + row_height)],
                fill="black",
            )

            cell_text = str(row[col_idx]) if col_idx < len(row) else ""

            cell_lines = wrap_text(cell_text, font, col_widths[col_idx])
            y_text = current_y + CELL_PADDING
            for line in cell_lines:
                draw.text(
                    (current_x + CELL_PADDING, y_text), line, font=font, fill="black"
                )
                line_height = font.getbbox(line)[3] - font.getbbox(line)[1]
                y_text += line_height + LINE_SPACING

            current_x += col_widths[col_idx]
            # Draw vertical grid line at end of each cell
            draw.line(
                [(current_x, current_y), (current_x, current_y + row_height)],
                fill="black",
            )

        current_y += row_height
        # Draw horizontal grid line at end of the row
        draw.line([(0, current_y), (table_width, current_y)], fill="black")

    # Draw metadata if present
    if "metadata" in table_data and table_data["metadata"]:
        metadata_lines = wrap_text(str(table_data["metadata"]), font, table_width)
        y_text = current_y + CELL_PADDING
        for line in metadata_lines:
            draw.text((CELL_PADDING, y_text), line, font=font, fill="gray")
            line_height = font.getbbox(line)[3] - font.getbbox(line)[1]
            y_text += line_height + LINE_SPACING
        current_y += metadata_height

    # Add a margin around the final image
    IMAGE_MARGIN = 10
    new_width = img.width + IMAGE_MARGIN * 2
    new_height = img.height + IMAGE_MARGIN * 2
    new_img = Image.new("RGB", (int(new_width), int(new_height)), "white")
    new_img.paste(img, (IMAGE_MARGIN, IMAGE_MARGIN))

    return new_img


def draw_multi_bar_chart(graph: MultiBarChart, ax: plt.Axes):
    """
    Draws a multi-bar (grouped bar) chart on the given Axes object
    based on the data fields of the MultiBarChart model.
    """
    data_list = graph.data
    groups = sorted({d["group"] for d in data_list})
    conditions = list(dict.fromkeys(d["condition"] for d in data_list))

    values = []
    errors = []

    for group in groups:
        group_values = []
        group_errors = []
        for condition in conditions:
            entry = next(
                (
                    d
                    for d in data_list
                    if d["group"] == group and d["condition"] == condition
                ),
                None,
            )
            if entry:
                group_values.append(entry["value"])
                group_errors.append(entry.get("error", 0))
            else:
                group_values.append(0)
                group_errors.append(0)
        values.append(group_values)
        errors.append(group_errors)

    num_groups = len(groups)
    num_conditions = len(conditions)

    total_width = 0.8
    bar_width = total_width / num_groups

    # Handle single-condition case to avoid negative spacing
    if num_conditions > 1:
        condition_spacing = max(
            0.2, (1 - num_conditions * total_width) / (num_conditions - 1)
        )
    else:
        condition_spacing = 0.2

    x = np.arange(num_conditions) * (num_groups * bar_width + condition_spacing)
    values = np.array(values, dtype=float)
    errors = np.array(errors, dtype=float)

    colors = [
        "#1f77b4",  # blue
        "#aec7e8",  # light blue
        "#2ca02c",  # green
        "#FFD700",  # gold
        "#800080",  # purple
        "#ffa500",  # orange
        "#a52a2a",  # brown
    ]

    for i, group in enumerate(groups):
        x_coords = x + i * bar_width
        ax.bar(
            x_coords,
            values[i],
            bar_width,
            label=group,
            yerr=errors[i],
            capsize=5,
            color=colors[i % len(colors)],
        )

    ax.set_ylabel(graph.y_label, fontweight="bold")
    ax.set_xlabel(graph.x_label, fontweight="bold")

    wrapped_title = "\n".join(textwrap.wrap(graph.title, width=100))
    ax.set_title(wrapped_title, pad=20, loc="center", fontweight="bold")

    offset = bar_width * (num_groups / 2)
    ax.set_xticks(x + offset - bar_width / 2)
    ax.set_xticklabels(conditions, fontweight="bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.legend(loc="upper right", prop={"weight": "bold"})

    # Adjust Y-limits for the largest error bar
    max_with_error = 0
    for val_row, err_row in zip(values, errors):
        for v, e in zip(val_row, err_row):
            max_with_error = max(max_with_error, v + e)
    if max_with_error == 0:
        max_with_error = 1  # avoid zero-range
    ax.set_ylim(0, max_with_error * 1.2)


def draw_line_graphs(graph: LineGraphs, ax: plt.Axes):
    """
    Draws one or more line plots on the given Axes object
    based on the data fields of the LineGraphs model.
    """

    def use_latex(text):
        return "$" in text

    data_series = graph.data_series
    colors = [
        "#1f77b4",
        "#2ca02c",
        "#800080",
        "#ffa500",
        "#a52a2a",
        "#aec7e8",
        "#FFD700",
    ]

    has_labels = False
    for i, series in enumerate(data_series):
        ax.plot(
            series["x_values"],
            series["y_values"],
            label=series.get("label", None),
            marker=series.get("marker", None),
            color=colors[i % len(colors)],
        )
        if "label" in series:
            has_labels = True

    wrapped_title = "\n".join(textwrap.wrap(graph.title, width=100))
    ax.set_title(
        wrapped_title,
        pad=20,
        loc="center",
        usetex=use_latex(graph.title),
        fontweight="bold",
    )
    ax.set_xlabel(
        graph.x_axis["label"],
        usetex=use_latex(graph.x_axis["label"]),
        fontweight="bold",
    )
    ax.set_ylabel(
        graph.y_axis["label"],
        usetex=use_latex(graph.y_axis["label"]),
        fontweight="bold",
    )

    if "range" in graph.x_axis:
        ax.set_xlim(graph.x_axis["range"])
    if "range" in graph.y_axis:
        ax.set_ylim(graph.y_axis["range"])

    # Bold x-axis tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    # Bold y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True)
    if has_labels:
        ax.legend(prop={"weight": "bold"})
