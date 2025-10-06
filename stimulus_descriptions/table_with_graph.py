"""
This module enforces generating either a bar graph or a line graph only.
Any attempt to produce another type of graph (e.g., "pie_graph") will
fail the Pydantic validation and is disallowed by design.
"""

from typing import Literal, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.table import (
    DataTable,
)
from pydantic import Field


class MultiBarChart(StimulusDescription):
    """
    Because we rely on 'graph_type' to discriminate,
    this MUST be 'bar_graph' every time.
    """

    graph_type: Literal["bar_graph"] = Field(
        "bar_graph", description="Must be bar_graph"
    )
    title: str
    x_label: str
    y_label: str
    data: list[dict]


class LineGraphs(StimulusDescription):
    """
    Because we rely on 'graph_type' to discriminate,
    this MUST be 'line_graph' every time.
    """

    graph_type: Literal["line_graph"] = Field(
        "line_graph", description="Must be line_graph"
    )
    title: str
    x_axis: dict
    y_axis: dict
    data_series: list[dict]


class DrawTableAndGraph(StimulusDescription):
    """
    A Pydantic model ensuring:
      • data_table is a DataTable
      • graph is either MultiBarChart (graph_type='bar_graph')
          or LineGraphs (graph_type='line_graph').

    Because we used a union with Literal-based 'graph_type',
    any other type (like "pie_graph") is forbidden. The generator
    must never produce such an invalid value, or validation will fail.

    Example usage:
    {
      "data_table": { ... },
      "graph": {
        "graph_type": "bar_graph",
        "title": "...",
        "x_label": "...",
        "y_label": "...",
        "data": [...]
      }
    }
    or
    {
      "data_table": { ... },
      "graph": {
        "graph_type": "line_graph",
        "title": "...",
        "x_axis": {...},
        "y_axis": {...},
        "data_series": [...]
      }
    }
    """

    data_table: DataTable = Field(
        ...,
        description="DataTable specifying headers, data rows, optional title/metadata.",
    )

    # Strict union: must be one or the other. No pie_graph or anything else allowed.
    graph: Union[MultiBarChart, LineGraphs] = Field(
        ...,
        description="Must have 'graph_type' in {'bar_graph','line_graph'} only.",
    )
