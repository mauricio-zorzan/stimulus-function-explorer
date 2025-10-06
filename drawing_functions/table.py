import os  # noqa: I001
import time

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.drawing_functions.common import (
    TextHelper,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.probability_diagrams import (
    ProbabilityDiagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.table import (
    DataTable,
    DataTableGroup,
    Table,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.table_two_way import (
    TableTwoWay,
)
from content_generators.settings import settings


###########################
# Table #
###########################
def generate_horizontal_table(table_data: Table):
    # Settings
    dpi = 600
    font_size = 32
    text_helper = TextHelper(font_size=font_size, dpi=dpi)

    # Extract and wrap data similar to generate_table function
    columns, data_rows = table_data.extract_and_wrap_data(text_helper, wrap_length=20)

    # Transpose data for horizontal table
    df = pd.DataFrame(data_rows, columns=columns).transpose()

    # Calculate max cell dimensions with transposed data
    max_cell_width, max_cell_height = table_data.calculate_max_cell_dimensions(
        text_helper, data_rows, columns
    )

    # Calculate total width and height based on the transposed data
    width = max_cell_width * len(data_rows)
    height = max_cell_height * len(columns)

    _, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")

    table = plt.table(
        cellText=df.values,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
        linewidth=2,
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(text_helper.font_size)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_horizontal_table.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=text_helper.dpi,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def generate_table(table_data: Table):
    dpi = 600
    # start with some reasonable upper bound
    base_font_pt = 32
    text_helper = TextHelper(font_size=base_font_pt, dpi=dpi)

    cols, rows = table_data.extract_and_wrap_data(text_helper, wrap_length=30)
    max_w, max_h = table_data.calculate_max_cell_dimensions(text_helper, rows, cols)

    # account for header + data rows
    fig_w = max_w * len(cols)
    fig_h = max_h * (len(rows) + 1)

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(True)
    table.set_fontsize(base_font_pt)

    plt.tight_layout(pad=0.5)
    out = (
        f"{settings.additional_content_settings.image_destination_folder}/"
        f"{int(time.time())}_table."
        f"{settings.additional_content_settings.stimulus_image_format}"
    )
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out


def extract_and_wrap_data(columns, rows, text_helper: TextHelper, wrap_length: int):
    # Extract column labels dynamically (handles variable number of columns)
    column_labels = [list(col.values())[0] for col in columns]

    # Extract data rows dynamically (handles variable number of columns in rows)
    data_rows = [[row[key] for key in row] for row in rows]

    # Wrap text for columns
    wrapped_columns = [
        text_helper.wrap_text(label, wrap_length) for label in column_labels
    ]

    # Wrap text for data rows
    wrapped_data_rows = [
        [text_helper.wrap_text(str(item), wrap_length) for item in row]
        for row in data_rows
    ]

    # Create data rows as a list of dictionaries with wrapped column labels as keys
    data_rows = [dict(zip(wrapped_columns, row)) for row in wrapped_data_rows]

    return wrapped_columns, data_rows


def calculate_max_cell_dimensions(text_helper: TextHelper, data_rows, columns):
    max_cell_width = 0
    max_cell_height = 0
    for row in data_rows:
        for value in row.values():
            cell_width = text_helper.get_text_width(str(value), pad=2)
            cell_height = text_helper.get_text_height(str(value), pad=0.1)
            max_cell_width = max(max_cell_width, cell_width)
            max_cell_height = max(max_cell_height, cell_height)

    for col in columns:
        cell_width = text_helper.get_text_width(col, pad=2)
        cell_height = text_helper.get_text_height(col, pad=0.5)
        max_cell_width = max(max_cell_width, cell_width)
        max_cell_height = max(max_cell_height, cell_height)
    return max_cell_width, max_cell_height


@stimulus_function
def draw_table_two_way(input_dict: TableTwoWay):
    # Settings for text helper
    dpi = 600
    font_size = 32
    text_helper = TextHelper(font_size=font_size, dpi=dpi)

    # Extract and wrap data using the helper function
    columns, data_rows = extract_and_wrap_data(
        columns=input_dict.columns_title,
        rows=input_dict.data,
        text_helper=text_helper,
        wrap_length=12,
    )

    # Calculate max cell dimensions using the helper function
    max_cell_width, max_cell_height = calculate_max_cell_dimensions(
        text_helper=text_helper, data_rows=data_rows, columns=columns
    )

    # Create a DataFrame for the table
    df = pd.DataFrame(data_rows)

    # Define the width and height of the table based on the calculated cell dimensions
    table_width = max_cell_width * len(columns)
    table_height = max_cell_height * len(data_rows) + 2

    fig, ax = plt.subplots(figsize=(table_width, table_height))
    ax.axis("off")

    # Plotting the table
    table = plt.table(
        cellText=df.values,
        colLabels=columns,
        rowLabels=[list(row.values())[0] for row in input_dict.rows_title],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],  # Adjust this to shift the table as needed
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # Adjust cell sizes to fit the text appropriately based on max dimensions
    for key, cell in table.get_celld().items():
        cell.set_width(max_cell_width / len(columns))
        cell.set_height(max_cell_height / len(data_rows))
        cell.set_linewidth(2)  # Set thicker line width for the table cells

    plt.title(
        input_dict.table_title,
        y=1.05,
        fontsize=font_size + 6,
    )

    file_name = f"{settings.additional_content_settings.image_destination_folder}/{int(time.time())}_table_two_way.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        dpi=dpi,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.2,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.tight_layout()
    plt.close()

    return file_name


@stimulus_function
def create_probability_diagram(stimulus_description: ProbabilityDiagram):
    # Extract row and column titles
    row_titles = [list(item.values())[0] for item in stimulus_description.rows_title]
    column_titles = [
        list(item.values())[0] for item in stimulus_description.columns_title
    ]

    # Extract data
    data = pd.DataFrame.from_records(
        [item.dict() for item in stimulus_description.data]
    )
    data.columns = column_titles  # Set DataFrame columns to the provided titles
    data.index = row_titles  # type: ignore # Set DataFrame index to the provided row titles

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Hide axes
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(
        cellText=data.values,  # type: ignore
        rowLabels=data.index,
        colLabels=data.columns,  # type: ignore
        cellLoc="center",
        loc="center",
    )

    # Set the font size and scale the table
    table.set_fontsize(14)
    table.scale(0.5, 2.5)

    file_name = f"{settings.additional_content_settings.image_destination_folder}/prob_diagram_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0,
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_data_table(table_data: DataTable):
    # Extracting headers, data, title, and metadata from the table_data object
    td_headers = table_data.headers
    td_data = table_data.data
    td_title = getattr(table_data, "title", None)
    td_metadata = getattr(table_data, "metadata", None)

    # Constants
    FONT_SIZE = 14
    TITLE_FONT_SIZE = FONT_SIZE + 2
    CELL_PADDING = 14
    MAX_CELL_WIDTH = 180
    LINE_SPACING = 5
    TITLE_TABLE_PADDING = 8

    # Load font
    try:
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_dir = os.path.join(current_dir, "additional_fonts")
        font = ImageFont.truetype(
            os.path.join(font_dir, "arial.ttf"),
            FONT_SIZE,
        )
        bold_font = ImageFont.truetype(
            os.path.join(font_dir, "arial_bold.ttf"),
            FONT_SIZE,
        )
        title_font = ImageFont.truetype(
            os.path.join(font_dir, "arial_bold.ttf"),
            TITLE_FONT_SIZE,
        )
    except IOError:
        font = ImageFont.load_default()
        bold_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Wrap text into multiple lines based on a maximum pixel width
    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        line = ""
        for word in words:
            test_line = f"{line} {word}" if line else word
            w = font.getlength(test_line)
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
    for i in range(len(td_headers)):
        header_lines = wrap_text(str(td_headers[i]), bold_font, MAX_CELL_WIDTH)
        header_width = (
            max(bold_font.getlength(line) for line in header_lines) + 2 * CELL_PADDING
        )
        max_data_width = 0
        for row in td_data:
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
        for idx, header in enumerate(td_headers)
    )
    row_heights = []
    for row in td_data:
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

    # Calculate total dimensions
    table_width = sum(col_widths) + 1  # +1 for gridline at the end
    table_height = header_height + sum(row_heights) + 1  # +1 for gridline at the bottom

    # Add space for title and metadata if present
    title_height = 0
    if td_title:
        title_lines = wrap_text(str(td_title), title_font, table_width)
        title_height = (
            sum(
                (title_font.getbbox(line)[3] - title_font.getbbox(line)[1])
                + LINE_SPACING
                for line in title_lines
            )
            - LINE_SPACING
            + CELL_PADDING * 2
        )
        # Calculate the width of the longest word in the title, including padding
        longest_word_width = (
            max(title_font.getlength(word) for word in td_title.split())
            + 2 * CELL_PADDING
        )
        # Adjust table width if the longest word in the title is wider
        if longest_word_width > table_width:
            scale_factor = longest_word_width / table_width
            col_widths = [int(width * scale_factor) for width in col_widths]
            table_width = sum(col_widths) + 1
    metadata_height = 0
    if td_metadata:
        metadata_lines = wrap_text(str(td_metadata), font, table_width)
        metadata_height = (
            sum(
                (font.getbbox(line)[3] - font.getbbox(line)[1]) + LINE_SPACING
                for line in metadata_lines
            )
            - LINE_SPACING
            + CELL_PADDING * 2
        )

    # Total image height includes title, padding between title and table, table, and metadata
    total_height = title_height + TITLE_TABLE_PADDING + table_height + metadata_height
    # Create image
    img = Image.new("RGB", (int(table_width), int(total_height)), "white")
    draw = ImageDraw.Draw(img)

    # Draw title if present
    current_y = 0
    if td_title:
        y_text = current_y + CELL_PADDING
        for line in title_lines:
            draw.text((CELL_PADDING, y_text), line, font=title_font, fill="black")
            line_height = title_font.getbbox(line)[3] - title_font.getbbox(line)[1]
            y_text += line_height + LINE_SPACING
        current_y += (
            title_height + TITLE_TABLE_PADDING
        )  # Add padding between title and table

    # Draw headers
    current_x = 0
    for i, header in enumerate(td_headers):
        # Draw header background
        draw.rectangle(
            [
                current_x,
                current_y,
                current_x + col_widths[i],
                current_y + header_height,
            ],
            fill="lightgray",
        )
        # Draw horizontal & vertical gridline at the top of headers
        draw.line([(0, current_y), (table_width, current_y)], fill="black")
        draw.line(
            [(current_x, current_y), (current_x, current_y + header_height)],
            fill="black",
        )

        # Draw header text with wrapping
        header_lines = wrap_text(str(header), bold_font, col_widths[i])
        y_text = current_y + CELL_PADDING
        for line in header_lines:
            draw.text(
                (current_x + CELL_PADDING, y_text), line, font=bold_font, fill="black"
            )
            line_height = bold_font.getbbox(line)[3] - bold_font.getbbox(line)[1]
            y_text += line_height + LINE_SPACING
        current_x += col_widths[i]

        # Draw vertical gridline at the end of each header cell
        draw.line(
            [(current_x, current_y), (current_x, current_y + header_height)],
            fill="black",
        )
    current_y += header_height

    # Draw horizontal gridline below headers
    draw.line([(0, current_y), (table_width, current_y)], fill="black")

    # Draw data rows
    for row_idx, row in enumerate(td_data):
        current_x = 0
        row_height = row_heights[row_idx]
        for col_idx in range(len(td_headers)):
            # Draw vertical gridline at the start of each cell
            draw.line(
                [(current_x, current_y), (current_x, current_y + row_height)],
                fill="black",
            )
            if col_idx < len(row):
                cell_text = str(row[col_idx])
            else:
                cell_text = ""

            # Draw cell text with wrapping
            cell_lines = wrap_text(cell_text, font, col_widths[col_idx])
            y_text = current_y + CELL_PADDING
            for line in cell_lines:
                draw.text(
                    (current_x + CELL_PADDING, y_text), line, font=font, fill="black"
                )
                line_height = font.getbbox(line)[3] - font.getbbox(line)[1]
                y_text += line_height + LINE_SPACING
            current_x += col_widths[col_idx]

            # Draw vertical gridline at the end of each cell
            draw.line(
                [(current_x, current_y), (current_x, current_y + row_height)],
                fill="black",
            )
        current_y += row_height

        # Draw horizontal gridline at the end of the row
        draw.line([(0, current_y), (table_width, current_y)], fill="black")

    # Draw metadata if present
    if td_metadata:
        y_text = current_y + CELL_PADDING
        for line in metadata_lines:
            draw.text((CELL_PADDING, y_text), line, font=font, fill="gray")
            line_height = font.getbbox(line)[3] - font.getbbox(line)[1]
            y_text += line_height + LINE_SPACING
        current_y += metadata_height

    # Add whitespace padding around the image
    IMAGE_MARGIN = 10  # Whitespace around the image
    new_width = img.width + IMAGE_MARGIN * 2
    new_height = img.height + IMAGE_MARGIN * 2
    new_img = Image.new("RGB", (int(new_width), int(new_height)), "white")
    new_img.paste(img, (IMAGE_MARGIN, IMAGE_MARGIN))

    # Save image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/data_table{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    new_img.save(file_name, format="WEBP")
    return file_name


@stimulus_function
def draw_data_table_group(table_group_data: DataTableGroup):
    """Generate a group of data tables arranged in a grid layout."""
    import math
    
    # Generate individual table images first
    table_images = []
    for table in table_group_data.tables:
        table_file = draw_data_table(table)
        table_images.append(Image.open(table_file))
        # Clean up the temporary individual table file
        os.remove(table_file)
    
    # Determine layout
    num_tables = len(table_images)
    if table_group_data.layout == "horizontal":
        rows, cols = 1, num_tables
    elif table_group_data.layout == "vertical":
        rows, cols = num_tables, 1
    else:  # auto layout
        # Calculate optimal grid layout
        cols = math.ceil(math.sqrt(num_tables))
        rows = math.ceil(num_tables / cols)
    
    # Calculate dimensions for the combined image
    max_table_width = max(img.width for img in table_images)
    max_table_height = max(img.height for img in table_images)
    
    # Constants
    TABLE_MARGIN = 20  # Space between tables
    TITLE_MARGIN = 30  # Space for group title
    BORDER_MARGIN = 20  # Border around entire image
    
    # Calculate total dimensions
    total_width = cols * max_table_width + (cols - 1) * TABLE_MARGIN + 2 * BORDER_MARGIN
    
    # Calculate title height if present
    title_height = 0
    if table_group_data.group_title:
        # Use same font loading logic as draw_data_table
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            font_dir = os.path.join(current_dir, "additional_fonts")
            title_font = ImageFont.truetype(
                os.path.join(font_dir, "arial_bold.ttf"), 18
            )
        except IOError:
            title_font = ImageFont.load_default()
        
        title_height = title_font.getbbox(table_group_data.group_title)[3] + TITLE_MARGIN
    
    total_height = (
        rows * max_table_height 
        + (rows - 1) * TABLE_MARGIN 
        + 2 * BORDER_MARGIN 
        + title_height
    )
    
    # Create the combined image
    combined_img = Image.new("RGB", (int(total_width), int(total_height)), "white")
    draw = ImageDraw.Draw(combined_img)
    
    # Draw group title if present
    if table_group_data.group_title:
        title_x = (total_width - title_font.getlength(table_group_data.group_title)) // 2
        title_y = BORDER_MARGIN
        draw.text((title_x, title_y), table_group_data.group_title, font=title_font, fill="black")
    
    # Place individual tables in the grid
    current_table_idx = 0
    start_y = BORDER_MARGIN + title_height
    
    for row in range(rows):
        for col in range(cols):
            if current_table_idx >= num_tables:
                break
                
            # Calculate position for this table
            x = BORDER_MARGIN + col * (max_table_width + TABLE_MARGIN)
            y = start_y + row * (max_table_height + TABLE_MARGIN)
            
            # Center the table within its allocated space
            table_img = table_images[current_table_idx]
            center_x = x + (max_table_width - table_img.width) // 2
            center_y = y + (max_table_height - table_img.height) // 2
            
            # Paste the table image
            combined_img.paste(table_img, (int(center_x), int(center_y)))
            current_table_idx += 1
        
        if current_table_idx >= num_tables:
            break
    
    # Save the combined image
    file_name = f"{settings.additional_content_settings.image_destination_folder}/data_table_group_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    combined_img.save(file_name, format="WEBP")
    
    # Close all opened images to free memory
    for img in table_images:
        img.close()
    combined_img.close()
    
    return file_name
