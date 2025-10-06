from typing import List, Literal, Optional

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field


class Point(BaseModel):
    label: str = Field(..., description="The label for the point")
    x: float = Field(..., description="The x-coordinate of the point")
    y: float = Field(..., description="The y-coordinate of the point")


class TableData(BaseModel):
    headers: List[str] = Field(..., description="The column headers for the table.")
    rows: List[List[str]] = Field(..., description="The rows of data in the table.")
    title: Optional[str] = Field(
        default=None, description="Optional title for the table."
    )


class GraphSpec(BaseModel):
    type: Literal[
        "line",
        "curve",
        "scatter",
        "circle",
        "sideways_parabola",
        "hyperbola",
        "ellipse",
        "quadratic",
        "cubic",
        "sqrt",
        "rational",
    ] = Field(..., description="The type of graph to draw.")
    equation: Optional[str] = Field(
        default=None, description="The equation of the line/curve (for display)."
    )
    slope: Optional[float] = Field(
        default=None, description="The slope of the line (for line type)."
    )
    y_intercept: Optional[float] = Field(
        default=None, description="The y-intercept of the line (for line type)."
    )
    points: Optional[List[Point]] = Field(
        default=None, description="Points to plot for scatter type."
    )
    color: str = Field(default="blue", description="The color of the graph.")
    label: Optional[str] = Field(
        default=None, description="Label for the graph in legend."
    )
    line_style: str = Field(default="-", description="Line style: '-', '--', '-.', ':'")
    line_width: float = Field(default=2.0, description="Width of the line.")

    # Parameters for non-linear functions (assessment boundary requirements)
    a: Optional[float] = Field(default=1.0, description="Coefficient 'a' for curves")
    b: Optional[float] = Field(default=0.0, description="Coefficient 'b' for curves")
    c: Optional[float] = Field(default=0.0, description="Coefficient 'c' for curves")
    h: Optional[float] = Field(default=0.0, description="Horizontal shift")
    k: Optional[float] = Field(default=0.0, description="Vertical shift")
    radius: Optional[float] = Field(default=5.0, description="Radius for circles")
    center_x: Optional[float] = Field(
        default=0.0, description="Circle center x-coordinate"
    )
    center_y: Optional[float] = Field(
        default=0.0, description="Circle center y-coordinate"
    )


class AxisSpec(BaseModel):
    label: str = Field(..., description="The label for the axis.")
    min_value: float = Field(..., description="The minimum value for the axis.")
    max_value: float = Field(..., description="The maximum value for the axis.")
    tick_interval: Optional[float] = Field(
        default=None, description="Interval between ticks."
    )


class ComboPointsTableGraph(StimulusDescription):
    table: Optional[TableData] = Field(
        default=None, description="The table data to display."
    )
    points: Optional[List[Point]] = Field(
        default=None, description="Points to plot and label."
    )
    graphs: Optional[List[GraphSpec]] = Field(
        default=None, description="Graphs/lines to draw."
    )
    x_axis: AxisSpec = Field(..., description="X-axis specification.")
    y_axis: AxisSpec = Field(..., description="Y-axis specification.")
    highlight_points: Optional[List[str]] = Field(
        default=None, description="Point labels to highlight with special styling."
    )
    show_grid: bool = Field(
        default=True, description="Whether to show grid on the graph."
    )
    graph_title: Optional[str] = Field(default=None, description="Title for the graph.")
    legend_position: str = Field(
        default="upper right", description="Position of the legend."
    )

    def validate_pipeline(self) -> None:
        """Validate that at least one of table, points, or graphs is provided."""
        if not any([self.table, self.points, self.graphs]):
            raise ValueError(
                "At least one of table, points, or graphs must be provided"
            )

        if self.highlight_points and not self.points:
            raise ValueError(
                "points must be provided when highlight_points is specified"
            )


if __name__ == "__main__":
    ComboPointsTableGraph.generate_assistant_function_schema("mcq4")
