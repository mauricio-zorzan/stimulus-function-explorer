import textwrap
from functools import lru_cache

import matplotlib.pyplot as plt


class TextHelper:
    def __init__(self, *, font_size, font_properties=None, dpi=None):
        self.font_properties = font_properties
        self.font_size = font_size
        self.dpi = dpi or self.fig.dpi
        self.fig, self.ax = plt.subplots(
            figsize=(1, 1), dpi=self.dpi
        )  # Create a new figure for calculations

    def get_text_width(self, text: str, pad=0):
        # Get bounding box of the text
        bbox = self.get_text_bbox(text)
        # Calculate the width of the text
        width = bbox.width

        char_width = width / len(text)

        # Add padding based on character width
        width += pad * char_width

        return width

    def get_text_height(self, text: str, pad: float = 0):
        # Get bounding box of the text
        bbox = self.get_text_bbox(text)
        # Calculate the height of the text
        height = bbox.height

        # Get the number of lines in the text
        num_lines = len(text.split("\n"))

        # Calculate the character height based on the total height and number of lines
        char_height = height / num_lines

        # Add padding based on character height
        height += pad * char_height

        return height

    def wrap_text(self, text, line_length) -> str:
        return "\n".join(textwrap.wrap(text, width=line_length))

    @lru_cache(maxsize=None)
    def get_text_bbox(self, text):
        text_obj = self.ax.text(
            0.5,
            0.5,
            text,
            fontproperties=self.font_properties,
            fontsize=self.font_size,
            horizontalalignment="center",
            verticalalignment="center",
        )

        self.fig.canvas.draw()

        # Get the renderer
        renderer = self.fig.canvas.get_renderer()  # type: ignore

        # Get the bounding box of the text
        return text_obj.get_window_extent(renderer=renderer).transformed(
            self.fig.gca().transData.inverted()
        )

    def __del__(self):
        # Close the figure when the object is deleted
        plt.close(self.fig)


# Usage
if __name__ == "__main__":
    helper = TextHelper(font_size=20, dpi=600)
    text_width = helper.get_text_height("Number of \n Tow Trucks")
    print(text_width)
    plt.show()
