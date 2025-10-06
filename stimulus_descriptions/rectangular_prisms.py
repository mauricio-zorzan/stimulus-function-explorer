from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, List, Literal, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
    StimulusDescriptionProtocol,
)
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class FillState(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    BOTTOM = "bottom"
    EMPTY = "empty"


class RectangularPrism(BaseModel):
    title: str = Field(description="The title of the figure.")
    height: int = Field(
        description="The height in unit cubes or optional units of measure of the rectangular prism.",
        ge=1,
        le=40,  # Changed from 10 to 40
    )
    width: int = Field(
        description="The width in unit cubes or optional units of measure of the rectangular prism.",
        ge=1,
        le=40,  # Changed from 10 to 40
    )
    length: int = Field(
        description="The length in unit cubes or optional units of measure of the rectangular prism.",
        ge=1,
        le=40,  # Changed from 10 to 40
    )
    fill: FillState = Field(
        description="The fill state of the rectangular prism.",
        json_schema_extra={
            "fill descriptions": {
                "full": "The rectangular prism is full of unit cubes with a total count of (height * width * length).",
                "partial": "The rectangular prism has the bottom layer and one column filled with unit cubes. Total fill is (width * length) + height - 1.",
                "bottom": "The rectangular prism has the bottom layer filled with unit cubes and includes measurements. Total fill is (width * length).",
                "empty": "The rectangular prism is empty but displays measurements for height, width, and length.",
            }
        },
    )
    prism_unit_label: str | None = Field(
        None,
        description="The optional label for the unit of measure for the prism measurements.",
    )
    unit_cube_unit_size_and_label: str | None = Field(
        None,
        description="The optional label and size for the unit of measure for the unit cube measurements.",
    )
    show_length: bool = Field(
        default=True, description="Whether to draw the prism length label"
    )
    show_width: bool = Field(
        default=True, description="Whether to draw the prism width label"
    )
    show_height: bool = Field(
        default=True, description="Whether to draw the prism height label"
    )

    volume: int | None = Field(
        description="The volume of the rectangular prism in unit cubes.",
        exclude=True,
        ge=1,
    )

    @model_validator(mode="before")
    def pre_calculate_values(cls, values: dict[str, Any]):
        """
        Calculate and set the volume before the model is loaded.
        """
        values["volume"] = values["height"] * values["width"] * values["length"]
        return values

    @model_validator(mode="before")
    def check_prism_unit_label(cls, values):
        fill = values.get("fill")
        prism_unit_label = values.get("prism_unit_label")
        if prism_unit_label is not None and fill in {FillState.PARTIAL, FillState.FULL}:
            raise AttributeError(
                "prism_unit_label must be None if fill state is 'partial' or 'full'."
            )
        return values

    @model_validator(mode="after")
    def check_at_least_two_measurements_visible(self):
        """
        Ensure that at least two measurements are visible when measurements are shown.
        Only applies to EMPTY and BOTTOM fill states where measurements are displayed.
        """
        if self.fill in {FillState.EMPTY, FillState.BOTTOM}:
            visible_count = sum([self.show_length, self.show_width, self.show_height])
            if visible_count < 2:
                raise ValueError(
                    "At least two measurements (show_length, show_width, or show_height) must be True for rectangular prisms with measurements."
                )
        return self


class RectangularPrismList(StimulusDescriptionList[RectangularPrism]):
    def assert_all_same_fill(
        self, pipeline_context: "QuestionGeneratorContext", fill: FillState
    ):
        assert all(
            prism.fill == fill for prism in self.root
        ), f"{pipeline_context.standard_id}: All rectangular prisms must be {fill}."

    def assert_correct_volume_is_unique_for_select(
        self, pipeline_context: "QuestionGeneratorContext"
    ):
        """
        Ensure that the correct volume is unique for each rectangular prism in the list.
        """
        if not hasattr(pipeline_context.question, "correct_answer"):
            return
        correct_answer_text = pipeline_context.question.correct_answer.strip()  # type: ignore
        figure = next(
            (
                prism
                for prism in self.root
                if prism.title.strip() == correct_answer_text
            ),
            None,
        )
        if figure is None:
            raise ValueError(
                f"No figure found with title matching the correct answer: {correct_answer_text}"
            )

        volumes = [prism.volume for prism in self.root]
        if volumes.count(figure.volume) > 1:
            raise ValueError(
                f"The volume {figure.volume} is not unique among the rectangular prisms."
            )

    def assert_all_same_height(
        self, pipeline_context: "QuestionGeneratorContext", height: int
    ):
        assert all(
            prism.height == height for prism in self.root
        ), f"{pipeline_context.standard_id}: All rectangular prisms must have the same height of {height}."

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """
        Additional validation here for substandard configurations
        """
        super().pipeline_validate(pipeline_context)
        match pipeline_context.standard_id:
            case "CCSS.MATH.CONTENT.8.G.C.9+2":
                self.assert_all_same_fill(pipeline_context, FillState.EMPTY)
            case "CCSS.MATH.CONTENT.5.MD.C.3.A":
                self.assert_all_same_fill(pipeline_context, FillState.FULL)
                self.assert_correct_volume_is_unique_for_select(pipeline_context)
            case "CCSS.MATH.CONTENT.5.MD.C.3.B":  # Expression questions need to have expressions in the answers
                pipeline_context.question.assert_all_answers_contain(r"\times")
            case "CCSS.MATH.CONTENT.5.MD.C.4+4":  # This sub standard always has height 1 full prisms
                self.assert_all_same_fill(pipeline_context, FillState.FULL)
                self.assert_all_same_height(pipeline_context, 1)

    @model_validator(mode="after")
    def check_all_figures_same_fill(self):
        """
        Ensure that all figures have the same fill state.
        """
        if not len(self):
            raise ValueError("No figures found in the list.")

        first_fill_state = self[0].fill
        for prism in self:
            if prism.fill != first_fill_state:
                raise ValueError(
                    f"Not all figures have the same fill state. "
                    f"Expected {first_fill_state}, but found {prism.fill}."
                )
        return self


class RectangularPrismShape(BaseModel):
    kind: Literal["rectangular"]  # ← discriminator
    length: int = Field(ge=1, le=10)
    width: int = Field(ge=1, le=10)
    height: int = Field(ge=1, le=10)

    @property
    def volume(self) -> int:
        return self.length * self.width * self.height


class Point3d(BaseModel):
    x: int = Field(ge=0, le=10)
    y: int = Field(ge=0, le=10)
    z: int = Field(ge=0, le=10)


class CustomCubeShape(BaseModel):
    """
    Single-layer, face-connected arrangement of unit cubes.
    Every cube shares the same y-coordinate, so the shape is only one cube thick.
    """

    kind: Literal["custom"] = "custom"
    cubes: List[Point3d] = Field(
        ...,
        description=(
            "Distinct (x, y, z) integer coordinates for each unit cube. "
            "• All y values must match (single layer). "
            "• Cubes must be face-connected (one 6-neighbourhood)."
        ),
    )

    # ── full validation ────────────────────────────────────────────────────
    @model_validator(mode="before")
    def _validate_cubes(cls, values):
        if not isinstance(values, dict):
            return values

        coords = values.get("cubes", [])
        if not isinstance(coords, list):
            return values

        # Convert dict coordinates to Point3d objects if needed
        processed_coords = []
        for coord in coords:
            if isinstance(coord, dict):
                processed_coords.append(Point3d(**coord))
            else:
                processed_coords.append(coord)

        values["cubes"] = processed_coords
        coords = processed_coords

        # at least one cube
        if not coords:
            raise ValueError("'cubes' must contain at least one coordinate")

        # duplicates & non-negative check
        for i, coord1 in enumerate(coords):
            for coord2 in coords[i + 1 :]:
                if (
                    coord1.x == coord2.x
                    and coord1.y == coord2.y
                    and coord1.z == coord2.z
                ):
                    raise ValueError("Duplicate coordinates found in 'cubes'")
        if any(min(coord.x, coord.y, coord.z) < 0 for coord in coords):
            raise ValueError("All cube coordinates must be non-negative")

        # single-layer check
        y_vals = {coord.y for coord in coords}
        if len(y_vals) != 1:
            raise ValueError(
                "Custom shapes must be exactly one cube thick: "
                "all cubes must have the same y-coordinate"
            )

        # connectivity check (6-neighbour BFS)
        cube_set = {(coord.x, coord.y, coord.z) for coord in coords}
        to_visit = deque([(coords[0].x, coords[0].y, coords[0].z)])
        visited = set()

        neighbours = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 0, 1),
            (0, 0, -1),
            (0, 1, 0),
            (0, -1, 0),
        ]

        while to_visit:
            x, y, z = to_visit.popleft()
            if (x, y, z) in visited:
                continue
            visited.add((x, y, z))

            for dx, dy, dz in neighbours:
                n = (x + dx, y + dy, z + dz)
                if n in cube_set and n not in visited:
                    to_visit.append(n)

        if len(visited) != len(cube_set):
            raise ValueError(
                "Cubes are not all face-connected; the shape must be a single connected cluster"
            )

        return values

    # convenience property
    @property
    def volume(self) -> int:
        return len(self.cubes)


# One type that can be *either* of the above – Pydantic 2 style
ShapeType = Annotated[
    Union[RectangularPrismShape, CustomCubeShape], Field(discriminator="kind")
]


# ─────────────────────────── TOP-LEVEL MODEL ───────────────────────────
class UnitCubeFigure(StimulusDescription, StimulusDescriptionProtocol):
    """
    Generic figure of unit cubes: either a right rectangular prism
    *or* an irregular connected shape.
    """

    title: str
    shape: ShapeType

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """
        Implement pipeline validation for the unit cube figure.
        """
        if pipeline_context is None:
            raise ValueError("Validation context is None")

    def generate_image(self):
        """
        Implement image generation logic.
        """
        pass


class BaseAreaRectangularPrism(BaseModel):
    title: str = Field(description="The title of the figure.")
    base_area: int = Field(
        description="The area of the base of the rectangular prism in square units.",
        ge=1,
        le=100,
    )
    height: int = Field(
        description="The height of the rectangular prism in units.",
        ge=1,
        le=10,
    )

    prism_unit_label: str | None = Field(
        None,
        description="The optional label for the unit of measure for the prism measurements.",
    )
    show_base_area: bool = Field(
        default=True, description="Whether to draw the base area label"
    )
    show_height: bool = Field(
        default=True, description="Whether to draw the prism height label"
    )

    volume: int | None = Field(
        description="The volume of the rectangular prism in cubic units.",
        exclude=True,
        ge=1,
    )

    @model_validator(mode="before")
    def pre_calculate_values(cls, values: dict[str, Any]):
        """
        Calculate and set the volume before the model is loaded.
        """
        values["volume"] = values["base_area"] * values["height"]
        return values

    @model_validator(mode="after")
    def check_at_least_one_measurement_visible(self):
        """
        Ensure that at least one measurement is visible for base area rectangular prisms.
        """
        if not (self.show_base_area or self.show_height):
            raise ValueError(
                "At least one measurement (show_base_area or show_height) must be True for base area rectangular prisms."
            )
        return self


class BaseAreaRectangularPrismList(StimulusDescriptionList[BaseAreaRectangularPrism]):
    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """
        Additional validation here for substandard configurations
        """
        super().pipeline_validate(pipeline_context)

    @model_validator(mode="after")
    def check_all_figures_same_height(self):
        """
        Ensure that all figures have the same height.
        """
        if not len(self):
            raise ValueError("No figures found in the list.")

        first_height = self[0].height
        for prism in self:
            if prism.height != first_height:
                raise ValueError(
                    f"Not all figures have the same height. "
                    f"Expected {first_height}, but found {prism.height}."
                )
        return self


if __name__ == "__main__":
    RectangularPrismList.generate_assistant_function_schema("mcq4")
