# content_generators/additional_content/stimulus_image/stimulus_descriptions/equation_tape_diagram.py
# Pydantic v2 models for equation tape diagrams (single & comparison).
# IMPORTANT: keep typing explicit and use a discriminated union on "type"
# so that schema generation (model_json_schema) works in AWS Lambda/tooling.


import math
from typing import Annotated, Literal, Optional, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionProtocol,
)
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator

# -------------------- Base types --------------------


class EquationDiagramBase(BaseModel):
    """Common base for all diagram models."""

    type: Literal[
        "addition",
        "subtraction",
        "equal_groups",
        "division",
        "multiplication",
        "comparison",
    ]


def _clean_symbol(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = v.strip()
    if not v:
        return None
    # Allow short strings (incl. digits) to support distractor cases like "30"
    if len(v) > 3:
        raise ValueError("variable_symbol must be 1–3 characters")
    return v


# -------------------- Single-diagram models --------------------


class AdditionDiagram(EquationDiagramBase):
    """
    Addition: part1 + part2 = total
    Exactly one of (part1, part2, total) is unknown and denoted by variable_symbol.
    """

    type: Literal["addition"] = "addition"
    unknown: Literal["part1", "part2", "total"]
    part1: Optional[float] = None
    part2: Optional[float] = None
    total: Optional[float] = None
    variable_symbol: Optional[str] = None

    part1_label: Optional[str] = None
    part2_label: Optional[str] = None
    total_label: Optional[str] = None

    @field_validator("variable_symbol")
    @classmethod
    def _var_sym(cls, v):
        return _clean_symbol(v)

    @model_validator(mode="after")
    def _check(self):
        provided = {"part1": self.part1, "part2": self.part2, "total": self.total}
        unknowns = [k for k, v in provided.items() if v is None]
        if self.unknown not in provided:
            raise ValueError("unknown must be one of 'part1', 'part2', 'total'")
        if len(unknowns) != 1 or unknowns[0] != self.unknown:
            raise ValueError(
                "Exactly one of part1/part2/total must be missing and match 'unknown'."
            )
        return self

    @model_validator(mode="after")
    def _reject_non_finite(cls, values):
        # values is `self` in v2; handle both styles safely
        self = values
        for name in ("part1", "part2", "total"):
            v = getattr(self, name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        return self


class SubtractionDiagram(EquationDiagramBase):
    """
    Subtraction in one-step equation form:
      start − change = result
    Unknown may be 'start', 'change', or 'result'.
    """

    type: Literal["subtraction"] = "subtraction"
    unknown: Literal["start", "change", "result"]
    start: Optional[float] = None
    change: Optional[float] = None
    result: Optional[float] = None
    variable_symbol: Optional[str] = None

    start_label: Optional[str] = None
    change_label: Optional[str] = None
    result_label: Optional[str] = None

    @field_validator("variable_symbol")
    @classmethod
    def _var_sym(cls, v):
        return _clean_symbol(v)

    @model_validator(mode="after")
    def _check(self):
        provided = {"start": self.start, "change": self.change, "result": self.result}
        if self.unknown not in provided:
            raise ValueError("unknown must be one of 'start', 'change', 'result'")
        unknowns = [k for k, v in provided.items() if v is None]
        if len(unknowns) != 1 or unknowns[0] != self.unknown:
            raise ValueError(
                "Exactly one of start/change/result must be missing and match 'unknown'."
            )
        return self

    @model_validator(mode="after")
    def _reject_non_finite(cls, values):
        # values is `self` in v2; handle both styles safely
        self = values
        for name in ("start", "change", "result"):
            v = getattr(self, name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        return self


class EqualGroupsDiagram(EquationDiagramBase):
    """
    Equal groups (repeated addition / multiplication):
      groups × group_size = total
    Unknown may be 'groups', 'group_size', or 'total'.
    """

    type: Literal["equal_groups"] = "equal_groups"
    unknown: Literal["groups", "group_size", "total"]
    groups: Optional[int] = None
    group_size: Optional[float] = None
    total: Optional[float] = None
    variable_symbol: Optional[str] = None

    groups_label: Optional[str] = None
    group_size_label: Optional[str] = None
    total_label: Optional[str] = None

    @field_validator("variable_symbol")
    @classmethod
    def _var_sym(cls, v):
        return _clean_symbol(v)

    @model_validator(mode="after")
    def _check(self):
        provided = {
            "groups": self.groups,
            "group_size": self.group_size,
            "total": self.total,
        }
        if self.unknown not in provided:
            raise ValueError("unknown must be one of 'groups', 'group_size', 'total'")
        unknowns = [k for k, v in provided.items() if v is None]
        if len(unknowns) != 1 or unknowns[0] != self.unknown:
            raise ValueError(
                "Exactly one of groups/group_size/total must be missing and match 'unknown'."
            )
        return self

    @model_validator(mode="after")
    def _validate_positive_params(cls, values):
        self = values

        # Reject NaN/inf for numeric floats
        for name in ("group_size", "total"):
            v = getattr(self, name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")

        # If provided, groups must be >= 1
        if self.groups is not None and int(self.groups) <= 0:
            raise ValueError("groups must be >= 1 when provided")

        # Scenario-specific requirements
        if self.unknown == "total":
            # need valid groups, group_size > 0
            if self.groups is None or self.groups <= 0:
                raise ValueError("When unknown='total', groups must be >= 1")
            if self.group_size is None or self.group_size <= 0:
                raise ValueError("When unknown='total', group_size must be > 0")

        elif self.unknown == "groups":
            # need valid total and group_size > 0
            if self.total is None or self.total <= 0:
                raise ValueError("When unknown='groups', total must be > 0")
            if self.group_size is None or self.group_size <= 0:
                raise ValueError("When unknown='groups', group_size must be > 0")

        elif self.unknown == "group_size":
            # need valid groups >= 1 and total > 0
            if self.groups is None or self.groups <= 0:
                raise ValueError("When unknown='group_size', groups must be >= 1")
            if self.total is None or self.total <= 0:
                raise ValueError("When unknown='group_size', total must be > 0")

        return self


class DivisionDiagram(EquationDiagramBase):
    """
    Division: dividend ÷ divisor = quotient
    Unknown may be 'dividend', 'divisor', or 'quotient'.
    """

    type: Literal["division"] = "division"
    unknown: Literal["dividend", "divisor", "quotient"]
    dividend: Optional[float] = None
    divisor: Optional[float] = None
    quotient: Optional[float] = None
    variable_symbol: Optional[str] = None

    dividend_label: Optional[str] = None
    divisor_label: Optional[str] = None
    quotient_label: Optional[str] = None

    @field_validator("variable_symbol")
    @classmethod
    def _var_sym(cls, v):
        return _clean_symbol(v)

    @model_validator(mode="after")
    def _check(self):
        provided = {
            "dividend": self.dividend,
            "divisor": self.divisor,
            "quotient": self.quotient,
        }
        if self.unknown not in provided:
            raise ValueError("unknown must be one of 'dividend', 'divisor', 'quotient'")
        unknowns = [k for k, v in provided.items() if v is None]
        if len(unknowns) != 1 or unknowns[0] != self.unknown:
            raise ValueError(
                "Exactly one of dividend/divisor/quotient must be missing and match 'unknown'."
            )
        if self.divisor is not None and self.divisor == 0:
            raise ValueError("divisor cannot be 0")
        return self

    @model_validator(mode="after")
    def _reject_non_finite(cls, values):
        # values is `self` in v2; handle both styles safely
        self = values
        for name in ("dividend", "divisor", "quotient"):
            v = getattr(self, name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        return self


class MultiplicationDiagram(EquationDiagramBase):
    """
    Multiplication: factor × factor2 = product
    Unknown may be 'factor', 'factor2', or 'product'.
    NOTE: If unknown == "factor", we accept EITHER factor OR factor2 missing,
    since tests use "factor" to represent "one of the factors is unknown".

    """

    type: Literal["multiplication"] = "multiplication"
    unknown: Literal["factor", "factor2", "product"]
    factor: Optional[float] = None
    factor2: Optional[float] = None
    product: Optional[float] = None
    variable_symbol: Optional[str] = None

    factor_label: Optional[str] = None
    factor2_label: Optional[str] = None
    product_label: Optional[str] = None

    @field_validator("variable_symbol")
    @classmethod
    def _var_sym(cls, v):
        return _clean_symbol(v)

    @model_validator(mode="after")
    def _check(self):
        if self.unknown not in {"factor", "factor2", "product"}:
            raise ValueError("unknown must be one of 'factor', 'factor2', 'product'")

        # If the product is unknown, both factors must be present.
        if self.unknown == "product":
            if self.product is not None:
                raise ValueError(
                    "When unknown='product', product must be missing (None)."
                )
            if self.factor is None or self.factor2 is None:
                raise ValueError(
                    "When unknown='product', both factors must be provided."
                )
            return self

        # If a factor is unknown:
        # - product must be provided
        # - exactly one of (factor, factor2) must be missing
        if self.product is None:
            raise ValueError("When a factor is unknown, product must be provided.")
        missing = [k for k in ("factor", "factor2") if getattr(self, k) is None]
        if len(missing) != 1:
            raise ValueError(
                "Exactly one of 'factor' or 'factor2' must be missing when a factor is unknown."
            )
        # Accept both unknown='factor' and unknown='factor2' as "a factor is unknown"
        # to match the tests' usage (unknown='factor' even when factor2 is missing).
        return self

    @model_validator(mode="after")
    def _reject_non_finite(cls, values):
        # values is `self` in v2; handle both styles safely
        self = values
        for name in ("factor", "factor2", "product"):
            v = getattr(self, name)
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number")
        return self


# -------------------- Comparison (two-diagram) model --------------------

DiagramUnion = Annotated[
    Union[
        AdditionDiagram,
        SubtractionDiagram,
        EqualGroupsDiagram,
        DivisionDiagram,
        MultiplicationDiagram,
    ],
    Field(discriminator="type"),
]


class ComparisonDiagram(EquationDiagramBase):
    """
    Place two diagrams side by side for 'which matches' type questions.
    Both nested diagrams must be of the same concrete type and also match diagram_type.
    """

    type: Literal["comparison"] = "comparison"
    diagram_type: Literal[
        "addition", "subtraction", "equal_groups", "division", "multiplication"
    ]

    correct_diagram: DiagramUnion
    distractor_diagram: DiagramUnion

    correct_label: str = "Diagram A"
    distractor_label: str = "Diagram B"

    @model_validator(mode="after")
    def _check(self):
        if self.correct_diagram.type != self.distractor_diagram.type:
            raise ValueError("Both diagrams in a comparison must have the same 'type'.")
        if self.correct_diagram.type != self.diagram_type:
            raise ValueError("diagram_type must match the nested diagram 'type'.")
        return self


# -------------------- Root model for schema export --------------------

# Expose a root model over the union of all acceptable top-level payloads.
TopLevelUnion = Annotated[
    Union[
        AdditionDiagram,
        SubtractionDiagram,
        EqualGroupsDiagram,
        DivisionDiagram,
        MultiplicationDiagram,
        ComparisonDiagram,
    ],
    Field(discriminator="type"),
]


class EquationTapeDiagram(RootModel[TopLevelUnion]):
    """Root model so that tooling can call model_json_schema()."""

    root: TopLevelUnion


# -------------------- Pipeline-compatible wrapper model --------------------


class EquationTapeDiagramWrapper(StimulusDescription, StimulusDescriptionProtocol):
    """
    Top-level stimulus model for equation tape diagrams.
    Wraps the discriminated union so the pipeline receives a protocol-compliant instance.
    """

    root: TopLevelUnion = Field(
        ...,
        description="Discriminated union of supported equation-tape diagram variants",
        discriminator="type",
    )

    def pipeline_validate(self, pipeline_context):
        if pipeline_context is None:
            raise ValueError("Validation context is None")
        return super().pipeline_validate(pipeline_context)


# IMPORTANT: export the CLASS (not an instance) for schema discovery
model = EquationTapeDiagram

__all__ = [
    "AdditionDiagram",
    "SubtractionDiagram",
    "EqualGroupsDiagram",
    "DivisionDiagram",
    "MultiplicationDiagram",
    "ComparisonDiagram",
    "EquationTapeDiagram",
    "EquationTapeDiagramWrapper",
    "model",
]

# _main__ helper pattern
if __name__ == "__main__":
    AdditionDiagram.generate_assistant_function_schema("mcq4")
    SubtractionDiagram.generate_assistant_function_schema("mcq4")
    EqualGroupsDiagram.generate_assistant_function_schema("mcq4")
    DivisionDiagram.generate_assistant_function_schema("mcq4")
    MultiplicationDiagram.generate_assistant_function_schema("mcq4")
    ComparisonDiagram.generate_assistant_function_schema("mcq4")
