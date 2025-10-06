import logging
import math
from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Union

import matplotlib.path as mpath
from pydantic import Field, field_validator, model_validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


def contains_subunit_values(labels, shapes) -> bool:
    """Check if any point has a coordinate value between 0 and 1."""
    for shape in shapes:
        for point in shape:
            if 0 < point[0] < 1 or 0 < point[1] < 1:
                return True
    for label in labels:
        for point in label:
            if 0 < point[0] < 1 or 0 < point[1] < 1:
                return True
    return False


class ShapeDecomposition(StimulusDescription):
    title: str = Field("", description="The title of the diagram.")
    units: str = Field("", description="must be cm, m, in, or ft.")
    gridlines: bool = Field(..., description="a lowercase boolean value")
    shapes: List[List[List[Union[int, float]]]] = Field(
        [],
        description="List of shapes represented as lists of coordinates. Each coordinate can be an integer or a float.",
    )
    shaded: Optional[List[int]] = Field(
        [], description="Index of the shape to be shaded."
    )
    labels: List[List[List[Union[int, float]]]] = Field(
        [], description="Labels for the shapes represented as lists of coordinates."
    )

    @model_validator(mode="before")
    def check_overlap(cls, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        labels = stimulus.get("labels")
        shapes = stimulus.get("shapes")
        if not labels or contains_subunit_values(labels, shapes):  # type: ignore
            return stimulus  # Bypass logic if labels are None or empty

        def generate_points(
            label: List[Tuple[Union[int, float], Union[int, float]]],
        ) -> List[Tuple[Union[int, float], Union[int, float]]]:
            (x1, y1), (x2, y2) = label
            points = []
            if x1 == x2:  # Vertical line
                # For float coordinates, use a different approach
                if isinstance(y1, float) or isinstance(y2, float):
                    # Sample points along the line for float coordinates
                    min_y, max_y = min(y1, y2), max(y1, y2)
                    points = [(x1, min_y), (x1, max_y)]
                else:
                    points = [(x1, y) for y in range(min(y1, y2), max(y1, y2) + 1)]
            elif y1 == y2:  # Horizontal line
                # For float coordinates, use a different approach
                if isinstance(x1, float) or isinstance(x2, float):
                    # Sample points along the line for float coordinates
                    min_x, max_x = min(x1, x2), max(x1, x2)
                    points = [(min_x, y1), (max_x, y1)]
                else:
                    points = [(x, y1) for x in range(min(x1, x2), max(x1, x2) + 1)]
            return points

        def share_common_points(
            label1: List[Tuple[Union[int, float], Union[int, float]]],
            label2: List[Tuple[Union[int, float], Union[int, float]]],
        ) -> bool:
            points1 = generate_points(label1)
            points2 = generate_points(label2)
            common_points = set(points1) & set(points2)
            return len(common_points) >= 2

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if share_common_points(labels[i], labels[j]):
                    raise ValueError(
                        f"Labels {labels[i]} and {labels[j]} share two or more common points."
                    )

        return stimulus

    @model_validator(mode="before")
    def check_unique_distances(cls, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        labels = stimulus.get("labels")
        shapes = stimulus.get("shapes")
        if not labels or contains_subunit_values(labels, shapes):  # type: ignore
            return stimulus  # Bypass logic if labels are None or empty

        def calculate_distance(
            p1: Tuple[Union[int, float], Union[int, float]],
            p2: Tuple[Union[int, float], Union[int, float]],
        ) -> float:
            """Calculate Euclidean distance between two points."""
            return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        def is_horizontal(
            label: List[Tuple[Union[int, float], Union[int, float]]],
        ) -> bool:
            """Check if the label is horizontal (same y coordinates)."""
            return label[0][1] == label[1][1]

        # Check for restricted variety in x (for horizontal) or y (for vertical) coordinates in pairs
        for label1, label2 in combinations(labels, 2):
            # Check if both labels are either horizontal or vertical and compare the relevant coordinates
            if is_horizontal(label1) == is_horizontal(label2):
                if is_horizontal(label1):  # Both labels are horizontal
                    relevant_values = [
                        label1[0][0],
                        label1[1][0],
                        label2[0][0],
                        label2[1][0],
                    ]
                else:  # Both labels are vertical
                    relevant_values = [
                        label1[0][1],
                        label1[1][1],
                        label2[0][1],
                        label2[1][1],
                    ]

                # Calculate distances for both labels
                dist1 = calculate_distance(label1[0], label1[1])
                dist2 = calculate_distance(label2[0], label2[1])

                # Only proceed if the labels are the same length
                if dist1 == dist2:
                    # Create a counter to count occurrences of each relevant coordinate
                    counts = Counter(relevant_values)

                    # Check if there are exactly two distinct values and each appears twice
                    if len(counts) == 2 and all(
                        count == 2 for count in counts.values()
                    ):
                        if is_horizontal(label1):
                            raise ValueError(
                                f"Horizontal labels {label1} and {label2} are of the same length and have exactly two identical x values repeated: {relevant_values}"
                            )
                        else:
                            raise ValueError(
                                f"Vertical labels {label1} and {label2} are of the same length and have exactly two identical y values repeated: {relevant_values}"
                            )

        return stimulus

    @model_validator(mode="before")
    def check_units_and_title(cls, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        units = stimulus.get("units")
        title = stimulus.get("title")
        max_unit_length = 3
        max_title_length = 40
        if not units:
            return stimulus  # Bypass logic if units are None or empty
        if units and len(units) > max_unit_length:
            raise ValueError("Units must be abbreviated to less than 3 characters.")
        if title and len(title) > max_title_length:
            raise ValueError("Title must be less than 40 characters.")

        return stimulus

    @model_validator(mode="before")
    def check_labels(cls, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        labels = stimulus.get("labels")
        shapes = stimulus.get("shapes")

        if not labels or not shapes or contains_subunit_values(labels, shapes):  # type: ignore
            return stimulus  # Bypass logic if labels or shapes are None or empty

        def is_point_outside_shapes(
            shapes: List[List[Tuple[Union[int, float], Union[int, float]]]],
            point1: Tuple[Union[int, float], Union[int, float]],
        ) -> bool:
            # Convert point into a format usable by matplotlib (x, y)
            point: Tuple[Union[int, float], Union[int, float]] = (
                point1[0],
                point1[1],
            )  # Ensure point is a tuple of two coordinates

            # Check each shape in the shapes object
            for shape in shapes:
                # Create a Path object from the current shape
                path = mpath.Path(shape)
                # Use the contains_point() method to determine if the point is inside the shape
                if path.contains_point(point):
                    return False

            # If no shapes contain the point, return True
            return True

        for label in labels:
            point1 = label[0]
            point2 = label[1]
            midpoint_x = (point1[0] + point2[0]) / 2
            midpoint_y = (point1[1] + point2[1]) / 2
            inside_point_found = False
            increment = 0

            while not inside_point_found:
                increment += 1
                # find the two points one unit away in perpendicular direction
                if point1[0] == point2[0]:
                    perpendicular_plus = [midpoint_x + increment, midpoint_y]
                    perpendicular_minus = [midpoint_x - increment, midpoint_y]
                elif point1[1] == point2[1]:
                    perpendicular_plus = [midpoint_x, midpoint_y + increment]
                    perpendicular_minus = [midpoint_x, midpoint_y - increment]
                else:
                    raise ValueError("Points must be vertical or horizontal")

                # check which of the two points is inside a shape
                if is_point_outside_shapes(
                    shapes, (perpendicular_plus[0], perpendicular_plus[1])
                ) and not is_point_outside_shapes(
                    shapes, (perpendicular_minus[0], perpendicular_minus[1])
                ):
                    inside_point_found = True
                elif is_point_outside_shapes(
                    shapes, (perpendicular_minus[0], perpendicular_minus[1])
                ) and not is_point_outside_shapes(
                    shapes, (perpendicular_plus[0], perpendicular_plus[1])
                ):
                    inside_point_found = True
                elif not is_point_outside_shapes(
                    shapes, (perpendicular_minus[0], perpendicular_minus[1])
                ) and not is_point_outside_shapes(
                    shapes, (perpendicular_plus[0], perpendicular_plus[1])
                ):
                    inside_point_found = True
                    raise ValueError("Both perpendicular points are inside a shape.")
                elif increment > 5:
                    inside_point_found = True
                    raise ValueError("Labels are too far from shapes.")
                else:
                    continue

        return stimulus

    def _check_coordinate_coverage(self) -> None:
        """Check that labels cover all coordinates from min to max in both x and y directions."""
        if not self.labels or not self.shapes or self.gridlines:
            return  # Bypass logic if labels or shapes are None or empty or gridlines is True

        # Collect all x and y coordinates from the shapes
        all_x = [point[0] for shape in self.shapes for point in shape]
        all_y = [point[1] for shape in self.shapes for point in shape]

        # Determine the minimum and maximum x and y coordinates from the shapes
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Sets to track covered x and y coordinates
        x_covered = set()
        y_covered = set()

        # Update the sets with the range of x and y coordinates covered by each label
        for label in self.labels:
            x_covered.update(
                [
                    x * 0.5
                    for x in range(
                        int(2 * min(label[0][0], label[1][0])),
                        int(2 * max(label[0][0], label[1][0])) + 1,
                    )
                ]
            )
            y_covered.update(
                [
                    y * 0.5
                    for y in range(
                        int(2 * min(label[0][1], label[1][1])),
                        int(2 * max(label[0][1], label[1][1])) + 1,
                    )
                ]
            )

        # Check if all x coordinates from min_x to max_x are covered
        for x in [i * 0.5 for i in range(int(2 * min_x), int(2 * max_x) + 1)]:
            if x not in x_covered:
                raise ValueError(
                    "The entire length from min x to max x is not covered by any combination of labels."
                )

        # Check if all y coordinates from min_y to max_y are covered
        for y in [i * 0.5 for i in range(int(2 * min_y), int(2 * max_y) + 1)]:
            if y not in y_covered:
                raise ValueError(
                    "The entire length from min y to max y is not covered by any combination of labels."
                )

    @model_validator(mode="before")
    def check_shapes_have_minimum_points(
        cls, stimulus: Dict[str, Any]
    ) -> Dict[str, Any]:
        shapes = stimulus.get("shapes")
        if not shapes:
            return stimulus  # Bypass logic if shapes are None or empty

        for shape in shapes:
            if len(shape) < 3:
                raise ValueError(f"Shape {shape} has less than three points.")

        return stimulus

    def check_labels_when_no_gridlines(self, stimulus):
        gridlines = stimulus.gridlines
        labels = stimulus.labels

        if not gridlines:
            horizontal_label = False
            vertical_label = False

            for label in labels:
                if label[0][0] == label[1][0]:  # x-coordinates are the same
                    vertical_label = True
                elif label[0][1] == label[1][1]:  # y-coordinates are the same
                    horizontal_label = True

                if horizontal_label and vertical_label:
                    break

            if not (horizontal_label and vertical_label):
                raise ValueError(
                    "When gridlines are off, there must be at least one vertical and one horizontal label."
                )

        return stimulus

    def check_shape_overlap(self, shapes: List[List[List[int]]]) -> Optional[bool]:
        if len(shapes) <= 1:
            return False

        def shapes_overlap(shape1: List[List[int]], shape2: List[List[int]]) -> bool:
            def project_polygon(axis, polygon):
                min_proj = float("inf")
                max_proj = float("-inf")
                for point in polygon:
                    projection = point[0] * axis[0] + point[1] * axis[1]
                    if projection < min_proj:
                        min_proj = projection
                    if projection > max_proj:
                        max_proj = projection
                return min_proj, max_proj

            def polygons_overlap_on_axis(axis, shape1, shape2):
                min1, max1 = project_polygon(axis, shape1)
                min2, max2 = project_polygon(axis, shape2)
                # Check for overlap, but not just touching
                return not (max1 <= min2 or max2 <= min1)

            def get_axes(shape):
                axes = []
                for i in range(len(shape)):
                    p1 = shape[i]
                    p2 = shape[i - 1]
                    edge = (p1[0] - p2[0], p1[1] - p2[1])
                    normal = (-edge[1], edge[0])
                    axes.append(normal)
                return axes

            axes1 = get_axes(shape1)
            axes2 = get_axes(shape2)

            for axis in axes1 + axes2:
                if not polygons_overlap_on_axis(axis, shape1, shape2):
                    return False

            return True

        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                if shapes_overlap(shapes[i], shapes[j]):
                    raise ValueError(f"Shapes {i} and {j} overlap.")

    def check_shapes_share_edge(self, shapes: List[List[List[int]]]) -> None:
        def get_edges(shape: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
            edges = []
            for i in range(len(shape)):
                start = shape[i]
                end = shape[(i + 1) % len(shape)]
                edge = []

                # Calculate the differences
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                steps = max(abs(dx), abs(dy))

                # Calculate the increment
                x_increment = dx / steps
                y_increment = dy / steps

                # Generate points on the edge
                for step in range(steps + 1):
                    x = start[0] + step * x_increment
                    y = start[1] + step * y_increment
                    edge.append((round(x), round(y)))

                edges.append(edge)
            return edges

        def edges_share_points(
            edge1: List[Tuple[int, int]], edge2: List[Tuple[int, int]]
        ) -> bool:
            return len(set(edge1) & set(edge2)) >= 2

        for i, shape1 in enumerate(shapes):
            shape1_edges = get_edges([tuple(point) for point in shape1])  # type: ignore
            shared_edge_found = False

            for j, shape2 in enumerate(shapes):
                if i == j:
                    continue
                shape2_edges = get_edges([tuple(point) for point in shape2])  # type: ignore

                for edge1 in shape1_edges:
                    for edge2 in shape2_edges:
                        if edges_share_points(edge1, edge2):
                            shared_edge_found = True
                            break
                    if shared_edge_found:
                        break
                if shared_edge_found:
                    break

            if not shared_edge_found:
                raise ValueError(
                    f"Shape {i} does not share at least two edge points with any other shape."
                )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext") -> None:
        """
        Additional validation here for substandard configurations
        """
        logging.info(self)
        super().pipeline_validate(pipeline_context)

        if pipeline_context.standard_id in {
            "CCSS.MATH.CONTENT.3.MD.C.7.D+1",
            "CCSS.MATH.CONTENT.3.MD.C.7.D+2",
            "CCSS.MATH.CONTENT.3.MD.C.7.D+3",
            "CCSS.MATH.CONTENT.3.MD.C.7.D+4",
            "CCSS.MATH.CONTENT.3.MD.C.7.C+1",
        }:
            self.check_shape_overlap(self.shapes)  # type: ignore
            self.check_labels_when_no_gridlines(self)
        if pipeline_context.standard_id in {
            "CCSS.MATH.CONTENT.1.G.A.2+1",
            "CCSS.MATH.CONTENT.1.G.A.2+3",
        }:
            self.check_shape_overlap(self.shapes)  # type: ignore
            self.check_shapes_share_edge(self.shapes)  # type: ignore

        # Coordinate coverage check - excluded for specific 6.G.A.1 standards
        excluded_standards = {
            "CCSS.MATH.CONTENT.6.G.A.1+1",
            "CCSS.MATH.CONTENT.6.G.A.1+2",
            "CCSS.MATH.CONTENT.6.G.A.1+3",
            "CCSS.MATH.CONTENT.6.G.A.1+4",
            "CCSS.MATH.CONTENT.6.G.A.1+5",
            "CCSS.MATH.CONTENT.6.G.A.1+6",
        }

        if pipeline_context.standard_id not in excluded_standards:
            self._check_coordinate_coverage()


class RhombusDiagonalsDescription(StimulusDescription):
    """
    Data model for create_rhombus_with_diagonals_figure().
    Pure container; no standard/difficulty mapping logic.
    """

    model_config = {
        "extra": "ignore",  # Ignore unexpected fields so we don't return None
    }

    units: str = Field(..., description="Units: cm, m, in, ft")
    d1: float = Field(..., ge=1, le=50, description="First (vertical) diagonal length")
    d2: Optional[float] = Field(
        None,
        ge=1,
        le=50,
        description="Second (horizontal) diagonal length (None if unknown)",
    )
    show_missing_placeholder: bool = Field(
        False, description="Whether to display a placeholder on a missing diagonal"
    )
    placeholder_text: str = Field(
        "?",
        max_length=4,
        description="Placeholder label when second diagonal is unknown",
    )
    title: Optional[str] = Field(
        default=None,
        max_length=40,
        description="Optional title displayed above the figure",
    )
    inside_labels: bool = Field(
        default=True,
        description="Labels rendered inside along diagonals (always True here)",
    )

    ALLOWED_UNITS: ClassVar[set[str]] = {"cm", "m", "in", "ft"}

    @model_validator(mode="before")
    def _inject_defaults(cls, values):
        if "units" not in values or not values["units"]:
            values["units"] = "cm"
        if "d1" not in values:
            values["d1"] = 10
        if values.get("show_missing_placeholder") and not values.get(
            "placeholder_text"
        ):
            values["placeholder_text"] = "?"
        return values

    @field_validator("units")
    @classmethod
    def _check_units(cls, v: str) -> str:
        if v not in cls.ALLOWED_UNITS:
            raise ValueError(f"units must be one of {sorted(cls.ALLOWED_UNITS)}")
        return v

    @model_validator(mode="after")
    def _validate_numbers(self):
        def ok_decimals(val: float) -> bool:
            s = f"{val}"
            return "." not in s or len(s.split(".")[1]) <= 1

        if not ok_decimals(self.d1):
            raise ValueError("d1 must have at most 1 decimal place")  # CHANGED
        if self.d2 is not None and not ok_decimals(self.d2):
            raise ValueError("d2 must have at most 1 decimal place")  # CHANGED
        if (
            self.d2 is None
            and self.show_missing_placeholder
            and not self.placeholder_text
        ):
            raise ValueError(
                "placeholder_text required when show_missing_placeholder is True"
            )
        return self

    @property
    def both_diagonals_known(self) -> bool:
        return self.d2 is not None

    def area_if_known(self) -> Optional[float]:
        if self.d2 is None:
            return None
        return (self.d1 * self.d2) / 2
