from typing import List, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescriptionList,
)
from pydantic import BaseModel, Field, model_validator


class DataPoint(BaseModel):
    category: str = Field(..., description="The category name for the data point.")
    frequency: float = Field(
        ...,
        description="The frequency or count for the category. Can be a whole number or half (e.g., 3.5 for 3 and a half stars).",
    )

    @model_validator(mode="after")
    def validate_frequency(self):
        """Validate that frequency is a valid value (whole number or half)."""
        if self.frequency < 0:
            raise ValueError(
                f"Frequency cannot be negative. Got {self.frequency} for category '{self.category}'"
            )

        # Check if it's a valid half-star value (whole number or .5)
        if (
            self.frequency != int(self.frequency)
            and self.frequency != int(self.frequency) + 0.5
        ):
            raise ValueError(
                f"Frequency must be a whole number or half (e.g., 3.5). Got {self.frequency} for category '{self.category}'"
            )

        return self


class PictureGraphConfig(BaseModel):
    """Configuration for picture graph display options."""

    star_value: int = Field(
        default=1,
        description="The value that each star represents in a picture graph. For example, if star_value=2, then 1 star = 2 items.",
        gt=0,
    )
    star_unit: str = Field(
        default="items",
        description="The unit/name for what each star represents in a picture graph. Examples: 'books', 'pencils', 'students', 'votes', etc.",
    )
    show_half_star_value: bool = Field(
        default=False,
        description="Whether to show an additional legend for half star values in picture graphs.",
    )


class CategoricalGraph(BaseModel):
    graph_type: Literal["bar_graph", "histogram", "picture_graph"] = Field(
        ...,
        description="Either 'bar_graph', 'histogram', or 'picture_graph'. The picture graph uses stars as the picture.",
    )
    title: str = Field(..., description="The title of the graph.")
    x_axis_label: str = Field(..., description="The label for the x-axis of the graph.")
    y_axis_label: str = Field(..., description="The label for the y-axis of the graph.")
    data: List[DataPoint] = Field(
        ..., description="A list of data points, each with a category and frequency."
    )
    picture_graph_config: Optional[PictureGraphConfig] = Field(
        default=None,
        description="Configuration for picture graph display options. Only used when graph_type is 'picture_graph'. If None, default values will be used.",
    )

    @property
    def _picture_config(self) -> PictureGraphConfig:
        """Get picture graph config with defaults if None."""
        return self.picture_graph_config or PictureGraphConfig()

    @model_validator(mode="after")
    def validate_picture_graph_frequencies(self):
        """Validate that frequencies in data points respect picture graph limits."""
        if self.graph_type == "picture_graph":
            for data_point in self.data:
                if data_point.frequency > 35:
                    raise ValueError(
                        f"Frequency cannot exceed 35 for picture graphs. Got {data_point.frequency} for category '{data_point.category}'"
                    )
        return self


class CategoricalGraphList(StimulusDescriptionList[CategoricalGraph]):
    root: list[CategoricalGraph] = Field(
        ...,
        description="A list of categorical graphs to be displayed.",
        min_length=1,
        max_length=1,
    )


class MultiGraphList(StimulusDescriptionList[CategoricalGraph]):
    root: list[CategoricalGraph] = Field(
        ...,
        description="A list of exactly 4 categorical graphs to be displayed in a 2x2 grid.",
        min_length=4,
        max_length=4,
    )


if __name__ == "__main__":
    CategoricalGraphList.generate_assistant_function_schema("mcq4")
    MultiGraphList.generate_assistant_function_schema("mcq4")
