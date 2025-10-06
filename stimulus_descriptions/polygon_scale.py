from typing import List, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import Field, field_validator


class PolygonScale(StimulusDescription):
    """
    Stimulus description for a polygon and its scale factor.
    Shows an original polygon and generates its scaled copy for scale factor determination exercises.
    Measurements are explicitly specified and displayed in the image.
    Original and scaled polygons can show different numbers of side measurements for educational purposes.
    """
    
    polygon_type: str = Field(
        ...,
        description="Type of polygon to draw (e.g., 'triangle', 'quadrilateral', 'pentagon', 'hexagon', 'irregular')"
    )
    
    original_polygon_label: str = Field(
        ...,
        description="Label for the original polygon (e.g., 'Polygon P')"
    )
    
    scaled_polygon_label: str = Field(
        ...,
        description="Label for the scaled polygon (e.g., 'Polygon Q')"
    )
    
    scale_factor: float = Field(
        ...,
        description="The scale factor used to transform the original polygon",
        gt=0,
        le=5
    )
    
    original_vertex_labels: List[str] = Field(
        ...,
        description="Vertex labels for the original polygon (e.g., ['A', 'B', 'C', 'D']). Required to know which sides to reference in questions."
    )
    
    scaled_vertex_labels: List[str] = Field(
        ...,
        description="Vertex labels for the scaled polygon (e.g., ['A'', 'B'', 'C'', 'D'']). Required to know which sides to reference in questions."
    )
    
    original_visible_sides: List[str] = Field(
        ...,
        description="Names of sides whose measurements are displayed on the original polygon (e.g., ['AB', 'BC', 'CD'])"
    )
    
    scaled_visible_sides: List[str] = Field(
        ...,
        description="Names of sides whose measurements are displayed on the scaled polygon (e.g., ['A'B'', 'B'C''] - can be fewer than original for educational purposes)"
    )
    
    original_measurements: List[Union[int, float]] = Field(
        ...,
        description="All side measurements for the original polygon in order (AB, BC, CD, DA for quadrilateral). Used for proper coordinate generation. Example: [30, 20, 30, 20]"
    )
    
    scaled_measurements: List[Union[int, float]] = Field(
        ...,
        description="All side measurements for the scaled polygon in order (A'B', B'C', C'D', D'A' for quadrilateral). Used for proper coordinate generation. Example: [4.5, 3.0, 4.5, 3.0]"
    )
    
    measurement_unit: str = Field(
        default="units",
        description="The unit of measurement (e.g., 'feet', 'cm', 'units')"
    )

    @field_validator("original_visible_sides")
    @classmethod
    def validate_original_visible_sides(cls, v, values):
        if "original_vertex_labels" in values.data:
            vertex_labels = values.data["original_vertex_labels"]
            # Check that each side name uses valid vertex labels
            for side in v:
                # Parse side name into two vertex labels
                vertices_in_side = []
                for vertex in vertex_labels:
                    if side.startswith(vertex):
                        vertices_in_side.append(vertex)
                        remaining_side = side[len(vertex):]
                        # Check if remaining part is also a valid vertex
                        if remaining_side in vertex_labels:
                            vertices_in_side.append(remaining_side)
                            break
                
                if len(vertices_in_side) != 2:
                    raise ValueError(f"Side name '{side}' must consist of exactly 2 vertex labels from {vertex_labels}")
                    
        return v

    @field_validator("scaled_visible_sides")
    @classmethod
    def validate_scaled_visible_sides(cls, v, values):
        if "scaled_vertex_labels" in values.data:
            vertex_labels = values.data["scaled_vertex_labels"]
            
            # Check that each side name uses valid vertex labels
            for side in v:
                # Parse side name into two vertex labels
                # Need to handle cases like "A'B'" where the labels are "A'" and "B'"
                vertices_in_side = []
                for vertex in vertex_labels:
                    if side.startswith(vertex):
                        vertices_in_side.append(vertex)
                        remaining_side = side[len(vertex):]
                        # Check if remaining part is also a valid vertex
                        if remaining_side in vertex_labels:
                            vertices_in_side.append(remaining_side)
                            break
                
                if len(vertices_in_side) != 2:
                    raise ValueError(f"Scaled side name '{side}' must consist of exactly 2 vertex labels from {vertex_labels}")
                    
        return v

    @field_validator("original_measurements")
    @classmethod
    def validate_original_measurements(cls, v, values):
        if "original_vertex_labels" in values.data:
            # Should have measurements for all sides of the polygon
            expected_count = len(values.data["original_vertex_labels"])
            if len(v) != expected_count:
                raise ValueError(f"Original measurements must have exactly {expected_count} values for all polygon sides")
        return v

    @field_validator("scaled_measurements")
    @classmethod
    def validate_scaled_measurements(cls, v, values):
        if "original_measurements" in values.data:
            # Should have same number of measurements as original
            if len(v) != len(values.data["original_measurements"]):
                raise ValueError("Scaled measurements must have same number of values as original measurements")
        return v

    @field_validator("scaled_vertex_labels")
    @classmethod
    def validate_scaled_vertex_labels(cls, v, values):
        if "original_vertex_labels" in values.data:
            if len(v) != len(values.data["original_vertex_labels"]):
                raise ValueError("Number of scaled vertex labels must match number of original vertex labels")
        return v


if __name__ == "__main__":
    PolygonScale.generate_assistant_function_schema("mcq4") 