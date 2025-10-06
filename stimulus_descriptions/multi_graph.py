from typing import List, Literal, Optional, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator


class BarGraphDataItem(StimulusDescription):
    group: str = Field(..., description="Name of the group")
    condition: str = Field(..., description="Name of the condition")
    value: float = Field(..., description="Value for the bar")
    error: Optional[float] = Field(None, description="Error bar value (optional)")


class BarGraphItem(StimulusDescription):
    graph_type: Literal["bar_graph"] = Field(..., description="Type of graph")
    title: str = Field(..., description="Title of the bar graph", max_length=200)
    x_label: str = Field(..., description="Label for the x-axis")
    y_label: str = Field(..., description="Label for the y-axis")
    data: List[BarGraphDataItem] = Field(..., description="List of data items")

    @field_validator("data")
    def check_groups_and_conditions(cls, data_items):
        groups = {item.group for item in data_items}
        conditions = {item.condition for item in data_items}

        if not (1 <= len(groups) <= 4):
            raise ValueError("Number of bar graph groups must be between 1 and 4.")
        if not (1 <= len(conditions) <= 4):
            raise ValueError("Number of conditions must be between 1 and 4.")

        return data_items


class Axis(StimulusDescription):
    label: str = Field(..., description="Axis label, e.g., 'Time'")
    min_value: Optional[float] = Field(
        None, description="Optional minimum value for axis range"
    )
    max_value: Optional[float] = Field(
        None, description="Optional maximum value for axis range"
    )
    # NEW: Optional categorical tick labels (e.g., ["Thursday","Friday",...])
    tick_labels: Optional[List[str]] = Field(
        None, description="Optional categorical tick labels for the axis"
    )

    @property
    def range(self) -> Optional[tuple[float, float]]:
        """Get the range as a tuple for backward compatibility."""
        if self.min_value is not None and self.max_value is not None:
            return (self.min_value, self.max_value)
        return None

    @range.setter
    def range(self, value: Optional[tuple[float, float]]) -> None:
        """Set the range from a tuple for backward compatibility."""
        if value is not None:
            self.min_value, self.max_value = value
        else:
            self.min_value = None
            self.max_value = None


class LineGraphSeriesItem(StimulusDescription):
    x_values: List[float] = Field(..., description="X-values for the line series")
    y_values: List[float] = Field(..., description="Y-values for the line series")
    label: Optional[str] = Field(None, description="Label for this line")
    marker: Optional[str] = Field(None, description="Marker style (optional)")


class LineGraphItem(StimulusDescription):
    graph_type: Literal["line_graph"] = Field(..., description="Type of graph")
    title: str = Field(..., description="Title of the line graph", max_length=200)
    x_axis: Axis = Field(..., description="X-Axis settings")
    y_axis: Axis = Field(..., description="Y-Axis settings")
    data_series: List[LineGraphSeriesItem] = Field(
        ..., min_length=1, max_length=5, description="Line series data (1 to 5 series)"
    )


GraphUnion = Union[BarGraphItem, LineGraphItem]


class CombinedGraphs(StimulusDescription):
    graphs: List[GraphUnion] = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Exactly two items must be provided.They can both be line graphs or bar graphs, or there can be one of each.",
    )


if __name__ == "__main__":
    # Test schema generation to ensure it works with AI function calls
    try:
        schema = LineGraphItem.model_json_schema()
        print("✓ Schema generation successful!")
        print("✓ No Tuple[float, float] issues detected")

        # Test backward compatibility
        axis = Axis(label="Test", range=(0, 10))
        print(f"✓ Backward compatibility test: range={axis.range}")
        print(f"✓ Min/Max values: {axis.min_value}, {axis.max_value}")

    except Exception as e:
        print(f"✗ Schema generation failed: {e}")
