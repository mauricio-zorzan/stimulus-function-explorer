import pytest
from content_generators.additional_content.stimulus_image.stimulus_descriptions.graphing_piecewise_model import (
    GraphingPiecewise,
    Segment,
)
from pydantic import ValidationError


def test_graphing_piecewise_example1():  # CCSS.MATH.CONTENT.8.F.B.5+1 Example Stimulus
    example1 = {
        "segments": [
            {
                "start_coordinate": (-7, -4),
                "end_coordinate": (-4, 2),
                "linear": False,
            },
            {
                "start_coordinate": (-4, 2),
                "end_coordinate": (1, 2),
                "linear": True,
            },
            {
                "start_coordinate": (1, 2),
                "end_coordinate": (4, 9),
                "linear": False,
            },
            {
                "start_coordinate": (4, 9),
                "end_coordinate": (10, 1),
                "linear": False,
            },
        ]
    }

    graph = GraphingPiecewise(**example1)  # type: ignore
    assert len(graph.segments) == 4
    assert graph.x_axis_label is None
    assert graph.y_axis_label is None
    assert graph.segments[0].start_coordinate == (-7, -4)
    assert graph.segments[-1].end_coordinate == (10, 1)


def test_graphing_piecewise_example2():  # CCSS.MATH.CONTENT.8.F.B.5+2 Example Stimulus
    example2 = {
        "x_axis_label": "Time",
        "y_axis_label": "Distance",
        "segments": [
            {
                "start_coordinate": (0, 0),
                "end_coordinate": (3, 1),
                "linear": True,
            },
            {
                "start_coordinate": (3, 1),
                "end_coordinate": (5, 1),
                "linear": True,
            },
            {
                "start_coordinate": (5, 1),
                "end_coordinate": (6, 5),
                "linear": True,
            },
            {
                "start_coordinate": (6, 5),
                "end_coordinate": (8, 6),
                "linear": True,
            },
        ],
    }

    graph = GraphingPiecewise(**example2)
    assert len(graph.segments) == 4
    assert graph.x_axis_label == "Time"
    assert graph.y_axis_label == "Distance"
    assert graph.segments[0].start_coordinate == (0, 0)
    assert graph.segments[-1].end_coordinate == (8, 6)


def test_disconnected_segments():
    with pytest.raises(ValidationError) as exc_info:
        GraphingPiecewise(
            segments=[
                Segment(
                    start_coordinate=(0, 0),
                    end_coordinate=(2, 2),
                    linear=True,
                ),
                Segment(
                    start_coordinate=(2, 2),
                    end_coordinate=(4, 2),
                    linear=True,
                ),
                Segment(
                    start_coordinate=(5, 2),
                    end_coordinate=(7, 0),
                    linear=True,
                ),
            ]
        )  # type: ignore
    assert "Segment 2 does not connect with the previous segment" in str(exc_info.value)


def test_too_few_segments():
    with pytest.raises(ValidationError) as exc_info:
        GraphingPiecewise(
            segments=[
                Segment(
                    start_coordinate=(0, 0),
                    end_coordinate=(2, 2),
                    linear=True,
                ),
                Segment(
                    start_coordinate=(2, 2),
                    end_coordinate=(4, 2),
                    linear=True,
                ),
            ]
        )  # type: ignore
    assert "Number of segments must be between 3 and 5" in str(exc_info.value)


def test_too_many_segments():
    with pytest.raises(ValidationError) as exc_info:
        GraphingPiecewise(
            segments=[
                Segment(start_coordinate=(0, 0), end_coordinate=(1, 2), linear=True),
                Segment(start_coordinate=(1, 2), end_coordinate=(2, 2), linear=True),
                Segment(start_coordinate=(2, 2), end_coordinate=(3, 4), linear=True),
                Segment(start_coordinate=(3, 4), end_coordinate=(4, 6), linear=False),
                Segment(start_coordinate=(4, 6), end_coordinate=(5, 8), linear=True),
                Segment(start_coordinate=(5, 8), end_coordinate=(6, 8), linear=True),
            ]
        )  # type: ignore
    assert "Number of segments must be between 3 and 5" in str(exc_info.value)


def test_float_coordinates():
    with pytest.raises(ValidationError):
        GraphingPiecewise(
            segments=[
                Segment(
                    start_coordinate=(0.5, 1),  # type: ignore
                    end_coordinate=(2, 3),
                    linear=True,
                ),
                Segment(
                    start_coordinate=(2, 3),
                    end_coordinate=(4, 3),
                    linear=True,
                ),
                Segment(
                    start_coordinate=(4, 3),
                    end_coordinate=(6, 1),
                    linear=True,
                ),
            ]
        )


def test_coordinates_within_range():
    # Valid case: all coordinates within 10 units of origin
    valid_graph = GraphingPiecewise(
        segments=[
            Segment(start_coordinate=(0, 0), end_coordinate=(5, 5), linear=True),
            Segment(start_coordinate=(5, 5), end_coordinate=(8, 3), linear=True),
            Segment(start_coordinate=(8, 3), end_coordinate=(10, 0), linear=True),
        ]
    )  # type: ignore
    assert isinstance(valid_graph, GraphingPiecewise)

    # Invalid case: coordinate outside 10 units of origin
    with pytest.raises(ValidationError) as exc_info:
        GraphingPiecewise(
            segments=[
                Segment(start_coordinate=(0, 0), end_coordinate=(5, 5), linear=True),
                Segment(start_coordinate=(5, 5), end_coordinate=(8, 3), linear=True),
                Segment(start_coordinate=(8, 3), end_coordinate=(11, 0), linear=True),
            ]
        )  # type: ignore
    assert "All coordinates must be within 10 units of the origin" in str(
        exc_info.value
    )

    # Invalid case: negative coordinate outside 10 units of origin
    with pytest.raises(ValidationError) as exc_info:
        GraphingPiecewise(
            segments=[
                Segment(start_coordinate=(-11, 0), end_coordinate=(5, 5), linear=True),
                Segment(start_coordinate=(5, 5), end_coordinate=(8, 3), linear=True),
                Segment(start_coordinate=(8, 3), end_coordinate=(10, 0), linear=True),
            ]
        )  # type: ignore
    assert "All coordinates must be within 10 units of the origin" in str(
        exc_info.value
    )
