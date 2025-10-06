from typing import List

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_validator, model_validator


class DataProperties(BaseModel):
    category: str
    x_axis_subcategory_data: List[float]


class MultipleBarGraph(StimulusDescription):
    graph_type: str
    title: str
    x_axis_label: str
    y_axis_label: str
    x_axis_subcategories: List[str] = Field(default_factory=list, max_length=3)
    data: List[DataProperties] = Field(default_factory=list, max_length=3)

    def num_x_axis_labels(self) -> int:
        return len(self.x_axis_subcategories)

    def num_words_x_axis_label(self) -> int:
        return len(self.x_axis_label.split())

    def num_words_y_axis_label(self) -> int:
        return len(self.y_axis_label.split())

    @model_validator(mode="after")
    def validate_graph(self):
        self.validate_label_length()
        return self

    def validate_label_length(self):
        x_axis_words = len(self.x_axis_label.split())
        y_axis_words = len(self.y_axis_label.split())

        if x_axis_words > 6:
            raise ValueError(
                f"X-axis label is too long: {x_axis_words} words. Maximum allowed is 6."
            )
        if y_axis_words > 6:
            raise ValueError(
                f"Y-axis label is too long: {y_axis_words} words. Maximum allowed is 6."
            )

    @field_validator("x_axis_subcategories", "data")
    def validate_complexity(cls, value, info):
        x_axis_subcategories = info.data.get("x_axis_subcategories")
        data = value if info.field_name == "data" else info.data.get("data")

        if x_axis_subcategories is None or data is None:
            return value

        num_x_axis_subcategories = len(x_axis_subcategories)
        num_categories = len(data)

        if num_x_axis_subcategories > 3 or num_categories > 3:
            raise ValueError(
                "The graph is too complex. Maximum 3 x-axis subcategories and 3 categories allowed."
            )

        return value


if __name__ == "__main__":
    MultipleBarGraph.generate_assistant_function_schema("sat-math")
