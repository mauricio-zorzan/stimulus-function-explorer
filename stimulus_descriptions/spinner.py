from pydantic import ConfigDict, Field

from .stimulus_description import StimulusDescription


class Spinner(StimulusDescription):
    """
    The Stimulus is a circular spinner divided into equally sized sections denoting a probability distribution of categories.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "requirements": {
                "Question Phrasing": "Clear, concise and omit describing the spinner distribution.",
                "Theme diversity": "The spinner should have an interesting theme that is suitable for the grade",
            }
        }
    )

    title: str = Field(
        ...,
        description="A clear title that relates to the content or purpose of the spinner.",
    )
    sections: list[str] = Field(
        ...,
        description="Each section of the spinner is represented by a label of a category.",
        json_schema_extra={
            "features": {
                "colors": "You may use color names as labels for sections out of the available colors and they will color the wheel.",
                "availableColors": ["Red", "Blue", "Green", "Yellow", "Pink", "Purple"],
                "order": "The order of the sections in the array are the order of the items on the spinner in clockwise order.",
            },
            "requirements": {
                "uniqueSections": "At-least one category labels 2 or more distinct sections.",
                "maximumCategories": 5,
                "maximumSections": "You may have up to 10 sections if you use colors or words with less than 7 letters as labels, otherwise you may have up to 8 sections.",
                "minimumSections": 4,
            },
        },
    )


if __name__ == "__main__":
    Spinner.generate_assistant_function_schema("mcq4")
