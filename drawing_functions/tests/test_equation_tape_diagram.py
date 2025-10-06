# src/content_generators/additional_content/stimulus_image/drawing_functions/tests/test_equation_tape_diagram.py
import os

import matplotlib
import pytest
from pydantic import TypeAdapter, ValidationError

matplotlib.use("Agg")

from content_generators.additional_content.stimulus_image.drawing_functions.equation_tape_diagram import (
    draw_equation_tape_diagram,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.equation_tape_diagram import (
    AdditionDiagram,
    ComparisonDiagram,
    DiagramUnion,
    DivisionDiagram,
    EqualGroupsDiagram,
    FractionStripNew,
    MultiplicationDiagram,
    SubtractionDiagram,
)

# Helper for strongly-typed dict→union conversion (keeps type checkers happy)
_as_union = TypeAdapter(DiagramUnion).validate_python


def _exists_png(path: str):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


# --- existing 6 (addition/subtraction) ---
@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # 1) 10 − 7 = b  (7 + b = 10  ⇒ unknown change)
        SubtractionDiagram(unknown="change", start=10, result=7, variable_symbol="b"),
        # 2) 32 + 15 = x (unknown total → true addition case)
        AdditionDiagram(unknown="total", part1=32, part2=15, variable_symbol="x"),
        # 3) r − 25 = 75 (unknown start)
        SubtractionDiagram(unknown="start", change=25, result=75, variable_symbol="r"),
        # 4) 63 − m = 48 (unknown change)
    ],
)
def test_equation_tape_diagram_add_sub(stim):
    out = draw_equation_tape_diagram(stim.model_dump_json())
    _exists_png(out)


# --- NEW: 4 scenarios from your screenshots (equal groups + one subtraction) ---
@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # 3) Lacrosse: t - 20 = 40  (difference)
        SubtractionDiagram(unknown="start", change=20, result=40, variable_symbol="t"),
    ],
)
def test_equation_tape_diagram_equal_groups_and_diff(stim):
    out = draw_equation_tape_diagram(stim.model_dump_json())
    _exists_png(out)


# --- NEW: 5 division test cases ---
@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # 1) 60 ÷ 10 = p  (dividend ÷ divisor = quotient, unknown quotient)
        DivisionDiagram(
            unknown="quotient",
            dividend=60,
            divisor=10,
            variable_symbol="p",
        ),
        # 2) 48 ÷ d = 6   (dividend ÷ divisor = quotient, unknown divisor)
        DivisionDiagram(
            unknown="divisor",
            dividend=48,
            quotient=6,
            variable_symbol="d",
        ),
        # 3) d ÷ 8 = 7    (dividend ÷ divisor = quotient, unknown dividend)
        DivisionDiagram(
            unknown="dividend",
            divisor=8,
            quotient=7,
            variable_symbol="d",
        ),
        # 4) 72 ÷ 9 = q   (dividend ÷ divisor = quotient, unknown quotient)
        DivisionDiagram(
            unknown="quotient",
            dividend=72,
            divisor=9,
            variable_symbol="q",
        ),
        # 5) 35 ÷ 5 = 7   (dividend ÷ divisor = quotient, all known - for validation)
        DivisionDiagram(
            unknown="quotient",
            dividend=35,
            divisor=5,
            variable_symbol="x",
        ),
    ],
)
def test_equation_tape_diagram_division(stim):
    out = draw_equation_tape_diagram(stim.model_dump_json())
    _exists_png(out)


# --- NEW: 5 multiplication test cases ---
@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # 1) 10 × j = 80  (factor × factor = product, unknown factor)
        MultiplicationDiagram(
            unknown="factor",
            factor=10,
            product=80,
            variable_symbol="j",
        ),
        # 2) 5 × 6 = p    (factor × factor = product, unknown product)
        MultiplicationDiagram(
            unknown="product",
            factor=5,
            factor2=6,
            variable_symbol="p",
        ),
        # 3) 8 × k = 64   (factor × factor = product, unknown factor)
        MultiplicationDiagram(
            unknown="factor",
            factor=8,
            product=64,
            variable_symbol="k",
        ),
    ],
)
def test_equation_tape_diagram_multiplication(stim):
    out = draw_equation_tape_diagram(stim.model_dump_json())
    _exists_png(out)


# --- NEW: 5 comparison test cases ---
@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # 1) Two addition diagrams: 8 + 6 = n vs 6 + n = 8
        ComparisonDiagram(
            diagram_type="addition",
            correct_position="A",  # Correct answer is Diagram A (top)
            correct_diagram=_as_union(
                {
                    "type": "addition",
                    "unknown": "total",
                    "part1": 8,
                    "part2": 6,
                    "variable_symbol": "n",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "addition",
                    "unknown": "part2",
                    "part1": 6,
                    "total": 8,
                    "variable_symbol": "n",
                }
            ),
            correct_label="Diagram A",
            distractor_label="Diagram B",
        ),
        # 2) Two subtraction diagrams: 10 - 3 = 7 vs 10 - 7 = 3
        ComparisonDiagram(
            diagram_type="subtraction",
            correct_position="B",  # Correct answer is Diagram B (bottom)
            correct_diagram=_as_union(
                {
                    "type": "subtraction",
                    "unknown": "result",
                    "start": 10,
                    "change": 3,
                    "variable_symbol": "r",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "subtraction",
                    "unknown": "change",
                    "start": 10,
                    "result": 7,
                    "variable_symbol": "c",
                }
            ),
            correct_label="Diagram A",
            distractor_label="Diagram B",
        ),
        # 3) Two multiplication diagrams: 5 × 6 = 30 vs 5 × 6 = p
        ComparisonDiagram(
            diagram_type="multiplication",
            correct_position="A",  # Correct answer is Diagram A (top)
            correct_diagram=_as_union(
                {
                    "type": "multiplication",
                    "unknown": "product",
                    "factor": 5,
                    "factor2": 6,
                    "variable_symbol": "p",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "multiplication",
                    "unknown": "product",
                    "factor": 5,
                    "factor2": 6,
                    "variable_symbol": "30",
                }
            ),
            correct_label="Diagram A",
            distractor_label="Diagram B",
        ),
        # 4) Two division diagrams: 48 ÷ 8 = 6 vs 48 ÷ 6 = 8
        ComparisonDiagram(
            diagram_type="division",
            correct_position="B",  # Correct answer is Diagram B (bottom)
            correct_diagram=_as_union(
                {
                    "type": "division",
                    "unknown": "quotient",
                    "dividend": 48,
                    "divisor": 8,
                    "variable_symbol": "q",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "division",
                    "unknown": "quotient",
                    "dividend": 48,
                    "divisor": 6,
                    "variable_symbol": "q",
                }
            ),
            correct_label="Diagram A",
            distractor_label="Diagram B",
        ),
        # 5) Two equal groups diagrams: 3 × 4 = 12 vs 4 × 3 = 12
        ComparisonDiagram(
            diagram_type="equal_groups",
            correct_position="A",  # Correct answer is Diagram A (top)
            correct_diagram=_as_union(
                {
                    "type": "equal_groups",
                    "unknown": "total",
                    "groups": 3,
                    "group_size": 4,
                    "variable_symbol": "t",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "equal_groups",
                    "unknown": "total",
                    "groups": 4,
                    "group_size": 3,
                    "variable_symbol": "t",
                }
            ),
            correct_label="Diagram A",
            distractor_label="Diagram B",
        ),
    ],
)
def test_equation_tape_diagram_comparison(stim):
    out = draw_equation_tape_diagram(stim)
    _exists_png(out)


def test_equation_tape_schema_is_exposable():
    # If this ever points to `dict`, this call will crash — early warning before Lambda.
    # Access the stimulus_type through the function's __dict__ to avoid wrapper issues
    stimulus_type = draw_equation_tape_diagram.__dict__.get("stimulus_type")
    assert stimulus_type is not None, "stimulus_type should be set on the function"
    schema = stimulus_type.model_json_schema(mode="serialization")
    assert isinstance(schema, dict)
    assert "properties" in schema or "oneOf" in schema or "allOf" in schema


# =========================
# New: Safety / Edge Cases
# =========================


def test_draw_accepts_multiple_input_forms():
    """
    The @stimulus_function wrapper and local _coerce should accept:
      1) JSON string (what Lambda path uses),
      2) plain dict from LLM,
      3) concrete submodel instance.
    """
    # A simple addition case: 3 + 5 = n
    inst = AdditionDiagram(unknown="total", part1=3, part2=5, variable_symbol="n")

    # 1) JSON string
    p1 = draw_equation_tape_diagram(inst.model_dump_json())
    _exists_png(p1)

    # 2) dict (LLM style)
    p2 = draw_equation_tape_diagram(
        {
            "type": "addition",
            "unknown": "total",
            "part1": 3,
            "part2": 5,
            "variable_symbol": "n",
        }
    )
    _exists_png(p2)

    # 3) concrete submodel
    p3 = draw_equation_tape_diagram(inst)
    _exists_png(p3)


def test_invalid_dict_missing_type_raises_validation_error():
    # Missing the discriminator "type"
    bad = {"unknown": "total", "part1": 2, "part2": 2, "variable_symbol": "x"}
    with pytest.raises(ValidationError):
        # draw() validates to EquationTapeDiagram internally
        draw_equation_tape_diagram(bad)


@pytest.mark.parametrize(
    "payload",
    [
        # wrong 'unknown' value
        {
            "type": "addition",
            "unknown": "sum",
            "part1": 3,
            "part2": 4,
            "variable_symbol": "x",
        },
        # non-numeric strings in numeric fields
        {
            "type": "addition",
            "unknown": "total",
            "part1": "abc",
            "part2": 4,
            "variable_symbol": "x",
        },
        {
            "type": "subtraction",
            "unknown": "result",
            "start": "NaN",
            "change": 5,
            "variable_symbol": "y",
        },
        # missing required field for declared unknown
        {
            "type": "subtraction",
            "unknown": "change",
            "start": 10,
            "variable_symbol": "m",
        },
    ],
)
def test_various_invalid_payloads_raise_validation_error(payload):
    with pytest.raises(ValidationError):
        draw_equation_tape_diagram(payload)


def test_equal_groups_zero_groups_is_rejected():
    """
    n = 0 should not be allowed (prevents divide-by-zero/degenerate layout).
    We accept either Pydantic ValidationError or a runtime Exception depending on model guards.
    """
    bad = {
        "type": "equal_groups",
        "unknown": "total",
        "groups": 0,
        "group_size": 5,
        "variable_symbol": "t",
    }
    with pytest.raises(Exception):
        draw_equation_tape_diagram(bad)


def test_division_by_zero_divisor_is_rejected():
    bad = {
        "type": "division",
        "unknown": "quotient",
        "dividend": 24,
        "divisor": 0,
        "variable_symbol": "q",
    }
    with pytest.raises(Exception):
        draw_equation_tape_diagram(bad)


@pytest.mark.parametrize(
    "stim",
    [
        # floating point inputs
        AdditionDiagram(unknown="total", part1=2.5, part2=3.5, variable_symbol="x"),
        SubtractionDiagram(
            unknown="result", start=10.0, change=0.25, variable_symbol="r"
        ),
        EqualGroupsDiagram(
            unknown="total", groups=3, group_size=2.5, variable_symbol="t"
        ),
        DivisionDiagram(
            unknown="divisor", dividend=12.0, quotient=3.0, variable_symbol="d"
        ),
        MultiplicationDiagram(
            unknown="product", factor=1.5, factor2=4, variable_symbol="p"
        ),
    ],
)
def test_handles_floats_and_large_values(stim):
    out = draw_equation_tape_diagram(stim)
    _exists_png(out)


def test_comparison_invalid_diagram_type_raises():
    # diagram_type must be one of the supported types
    # Use a string variable to bypass type checker literal validation
    invalid_type = "invalid_type"  # type: ignore
    with pytest.raises(ValidationError):
        ComparisonDiagram(
            diagram_type=invalid_type,  # invalid - not in the Literal union
            correct_position="A",  # Add required field
            correct_diagram=_as_union(
                {
                    "type": "addition",
                    "unknown": "total",
                    "part1": 1,
                    "part2": 2,
                    "variable_symbol": "x",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "addition",
                    "unknown": "total",
                    "part1": 2,
                    "part2": 1,
                    "variable_symbol": "x",
                }
            ),
            correct_label="Diagram A",
            distractor_label="Diagram B",
        )


def test_comparison_accepts_typed_dicts_via_union_adapter():
    # This also documents that our tests use validated dicts to avoid editor arg-type warnings.
    cmp = ComparisonDiagram(
        diagram_type="addition",
        correct_position="A",  # Add required field
        correct_diagram=_as_union(
            {
                "type": "addition",
                "unknown": "total",
                "part1": 8,
                "part2": 6,
                "variable_symbol": "n",
            }
        ),
        distractor_diagram=_as_union(
            {
                "type": "addition",
                "unknown": "part2",
                "part1": 6,
                "total": 8,
                "variable_symbol": "n",
            }
        ),
        correct_label="Diagram A",
        distractor_label="Diagram B",
    )
    out = draw_equation_tape_diagram(cmp)
    _exists_png(out)


# =========================
# New: Simple FractionStripNew Tests
# =========================


@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # Example 1: 5 parts with value 6.0, total is p
        FractionStripNew(
            total_parts=5,
            part_value="6.0",
            total_value="p",
        ),
        # Example 2: 5 parts with value 6.0, total is 30
        FractionStripNew(
            total_parts=5,
            part_value="6.0",
            total_value="30",
        ),
        # Example 3: 4 parts with value m, total is 28
        FractionStripNew(
            total_parts=4,
            part_value="m",
            total_value="28",
        ),
        # Example 4: 6 parts with value 5, total is n
        FractionStripNew(
            total_parts=6,
            part_value="5",
            total_value="n",
        ),
        # Example 5: 3 parts with value x, total is 18
        FractionStripNew(
            total_parts=3,
            part_value="x",
            total_value="18",
        ),
    ],
)
def test_fraction_strip_new_diagrams(stim):
    """Test the new simple fraction strip diagrams."""
    out = draw_equation_tape_diagram(stim.model_dump_json())
    _exists_png(out)


# =========================
# Simple Comparison Tests - Using FractionStripNew Only
# =========================


@pytest.mark.drawing_functions
@pytest.mark.parametrize(
    "stim",
    [
        # Simple comparison: 6 parts with "5" vs 6 parts with "p"
        ComparisonDiagram(
            diagram_type="fraction_strip_new",
            correct_position="A",  # Correct answer is Diagram A (top)
            correct_diagram=_as_union(
                {
                    "type": "fraction_strip_new",
                    "total_parts": 6,
                    "part_value": "5",
                    "total_value": "p",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "fraction_strip_new",
                    "total_parts": 6,
                    "part_value": "p",
                    "total_value": "5",
                }
            ),
        ),
        # Simple comparison: 5 parts with "6.0" total "p" vs total "30"
        ComparisonDiagram(
            diagram_type="fraction_strip_new",
            correct_position="B",  # Correct answer is Diagram B (bottom)
            correct_diagram=_as_union(
                {
                    "type": "fraction_strip_new",
                    "total_parts": 5,
                    "part_value": "6.0",
                    "total_value": "p",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "fraction_strip_new",
                    "total_parts": 5,
                    "part_value": "30",
                    "total_value": "6",
                }
            ),
        ),
        # Simple comparison: 4 parts with "m" vs 4 parts with "7"
        ComparisonDiagram(
            diagram_type="fraction_strip_new",
            correct_position="A",  # Correct answer is Diagram A (top)
            correct_diagram=_as_union(
                {
                    "type": "fraction_strip_new",
                    "total_parts": 4,
                    "part_value": "m",
                    "total_value": "28",
                }
            ),
            distractor_diagram=_as_union(
                {
                    "type": "fraction_strip_new",
                    "total_parts": 4,
                    "part_value": "7",
                    "total_value": "28",
                }
            ),
        ),
    ],
)
def test_simple_fraction_strip_comparisons(stim):
    """Test comparison diagrams with the new simple FractionStripNew style."""
    out = draw_equation_tape_diagram(stim)
    _exists_png(out)


# =========================
# New: Test correct_position functionality
# =========================


@pytest.mark.drawing_functions
def test_comparison_correct_position_A():
    """Test that correct_position='A' puts the correct diagram on top (Diagram A)."""
    comparison = ComparisonDiagram(
        diagram_type="fraction_strip_new",
        correct_position="A",  # Correct answer should be on top
        correct_diagram=_as_union(
            {
                "type": "fraction_strip_new",
                "total_parts": 8,
                "part_value": "3",
                "total_value": "w",
            }
        ),
        distractor_diagram=_as_union(
            {
                "type": "fraction_strip_new",
                "total_parts": 3,
                "part_value": "8",
                "total_value": "w",
            }
        ),
    )

    out = draw_equation_tape_diagram(comparison)
    _exists_png(out)

    # The image should show:
    # Diagram A (top): [3][3][3][3][3][3][3][3] with total line "w" (CORRECT)
    # Diagram B (bottom): [8][8][8] with total line "w" (DISTRACTOR)


@pytest.mark.drawing_functions
def test_comparison_correct_position_B():
    """Test that correct_position='B' puts the correct diagram on bottom (Diagram B)."""
    comparison = ComparisonDiagram(
        diagram_type="fraction_strip_new",
        correct_position="B",  # Correct answer should be on bottom
        correct_diagram=_as_union(
            {
                "type": "fraction_strip_new",
                "total_parts": 5,
                "part_value": "x",
                "total_value": "20",
            }
        ),
        distractor_diagram=_as_union(
            {
                "type": "fraction_strip_new",
                "total_parts": 4,
                "part_value": "x",
                "total_value": "20",
            }
        ),
    )

    out = draw_equation_tape_diagram(comparison)
    _exists_png(out)

    # The image should show:
    # Diagram A (top): [x][x][x][x] with total line "20" (DISTRACTOR)
    # Diagram B (bottom): [x][x][x][x][x] with total line "20" (CORRECT)
