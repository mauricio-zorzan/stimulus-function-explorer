import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.angles import (
    draw_labeled_transversal_angle,
    draw_lines_rays_and_segments,
    draw_parallel_lines_cut_by_transversal,
    draw_triangle_with_ray,
    generate_angle_diagram,
    generate_angle_types,
    generate_angles,
    generate_single_angle_type,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.angles import (
    SingleAngle,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.parallel_lines_transversal import (
    ParallelLinesTransversal,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.points_and_lines import (
    Line,
    Point,
    PointsAndLines,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.transversal import (
    TransversalAngleParams,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.triangle import (
    Angle,
    Ray,
    RightTriangle,
    RightTriangleWithRay,
    Triangle,
    TrianglePoint,
    TriangleStimulusDescription,
)
from pydantic import ValidationError


@pytest.mark.drawing_functions
def test_180_and_acutes_angles():
    stimulus_description = [
        {"label": "L", "measure": 30},
        {"label": "M", "measure": 180},
        {"label": "N", "measure": 45},
        {"label": "O", "measure": 165},
    ]
    file_name = generate_angle_types(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_obtuses_and_acutes_angles():
    stimulus_description = [
        {"label": "L", "measure": 170},
        {"label": "M", "measure": 105},
        {"label": "N", "measure": 25},
        {"label": "O", "measure": 150},
    ]
    file_name = generate_angle_types(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_obtuses_and_acutes_angles_2():
    stimulus_description = [
        {"label": "L", "measure": 170},
        {"label": "M", "measure": 130},
        {"label": "N", "measure": 15},
        {"label": "O", "measure": 90},
    ]
    file_name = generate_angle_types(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_obtuses_and_acutes_angles_3():
    stimulus_description = [
        {"label": "L", "measure": 170},
        {"label": "M", "measure": 105},
        {"label": "N", "measure": 25},
    ]
    file_name = generate_angle_types(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angles_0():
    stimulus_description = TriangleStimulusDescription(
        triangle=Triangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure=110),
                Angle(vertex="B", measure="(2x + 10)"),
                Angle(vertex="C", measure="(4x - 20)"),
            ],
        )
    )
    file_name = generate_angles(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angles_1():
    stimulus_description = TriangleStimulusDescription(
        triangle=Triangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure=110),
                Angle(vertex="B", measure="(2x + 10)"),
                Angle(vertex="C", measure="(4x - 20)"),
            ],
        )
    )
    file_name = generate_angles(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angles_2():
    stimulus_description = TriangleStimulusDescription(
        triangle=Triangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure=70),
                Angle(vertex="B", measure=50),
                Angle(vertex="C", measure="(4x - 10)"),
            ],
        )
    )
    file_name = generate_angles(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angles_3():
    stimulus_description = TriangleStimulusDescription(
        triangle=Triangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure=110),
                Angle(vertex="B", measure=50),
                Angle(vertex="C", measure="(4x - 10)"),
            ],
        )
    )
    file_name = generate_angles(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angles_4():
    stimulus_description = TriangleStimulusDescription(
        triangle=Triangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure="(x - 7)"),
                Angle(vertex="B", measure=50),
                Angle(vertex="C", measure="(4x - 10)"),
            ],
        )
    )
    file_name = generate_angles(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_basic():
    input_data = ParallelLinesTransversal(
        angle=45,
        top_line_top_left_angle_label="(x + 20) / 3",
        bottom_line_top_left_angle_label="x - 80",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_basic_both_labels():
    input_data = ParallelLinesTransversal(
        angle=45,
        top_line_top_left_angle_label="(x + 20) / 3",
        bottom_line_top_left_angle_label="x - 80",
        top_line_top_right_angle_label="Top-Right",
        bottom_line_top_right_angle_label="Bottom-Right",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_basic_both_labels_big_angle():
    input_data = ParallelLinesTransversal(
        angle=150,
        top_line_top_left_angle_label="(x + 20) / 3",
        bottom_line_top_left_angle_label="x - 80",
        top_line_top_right_angle_label="Top-Right",
        bottom_line_top_right_angle_label="Bottom-Right",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_different_angle():
    input_data = ParallelLinesTransversal(
        angle=60,
        top_line_top_left_angle_label="X",
        bottom_line_top_left_angle_label="Y",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_no_labels():
    input_data = ParallelLinesTransversal(
        angle=30, top_line_top_left_angle_label="", bottom_line_top_left_angle_label=""
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_extreme_angle():
    input_data = ParallelLinesTransversal(
        angle=30,
        top_line_top_left_angle_label="(x + 20) / 3",
        bottom_line_top_left_angle_label="(x + 20) / 3",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_perpendicular_angle():
    input_data = ParallelLinesTransversal(
        angle=90,
        top_line_top_left_angle_label="Top",
        bottom_line_top_left_angle_label="Bottom",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_extreme_angle_max():
    input_data = ParallelLinesTransversal(
        angle=150,
        top_line_top_left_angle_label="Top",
        bottom_line_top_left_angle_label="Bottom",
    )
    file_name = draw_parallel_lines_cut_by_transversal(input_data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_parallel_lines_cut_by_transversal_invalid_angle():
    with pytest.raises(ValueError):
        input_data = ParallelLinesTransversal(
            angle=200,
            top_line_top_left_angle_label="Invalid",
            bottom_line_top_left_angle_label="Angle",
        )
        draw_parallel_lines_cut_by_transversal(input_data)


@pytest.mark.drawing_functions
def test_draw_lines_rays_and_segments_basic():
    data = PointsAndLines(
        points=[
            Point(label="A", coordinates=[0, 0]),
            Point(label="B", coordinates=[1, 1]),
            Point(label="C", coordinates=[2, 0]),
        ],
        lines=[
            Line(start_label="A", end_label="B", type="segment"),
            Line(start_label="B", end_label="C", type="ray"),
        ],
    )
    file_name = draw_lines_rays_and_segments(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_lines_rays_and_segments_with_line():
    data = PointsAndLines(
        points=[
            Point(label="A", coordinates=[0, 0]),
            Point(label="B", coordinates=[1, 1]),
            Point(label="C", coordinates=[2, 0]),
        ],
        lines=[
            Line(start_label="A", end_label="B", type="line"),
            Line(start_label="B", end_label="C", type="line"),
        ],
    )
    file_name = draw_lines_rays_and_segments(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_lines_rays_and_segments_invalid_data():
    with pytest.raises(KeyError):
        data = PointsAndLines(
            points=[
                Point(label="A", coordinates=[0, 0]),
                Point(label="B", coordinates=[1, 1]),
            ],
            lines=[
                Line(
                    start_label="A", end_label="C", type="segment"
                ),  # Invalid end_label
            ],
        )
        draw_lines_rays_and_segments(data)


@pytest.mark.drawing_functions
def test_draw_triangle_with_ray_basic():
    triangle = RightTriangle(
        points=[
            TrianglePoint(label="A"),
            TrianglePoint(label="B"),
            TrianglePoint(label="C"),
        ],
        angles=[
            Angle(vertex="A", measure=90),
            Angle(vertex="B", measure=45),
            Angle(vertex="C", measure=45),
        ],
        rays=[Ray(start_label="A", measures=[45, "Ray 1"])],
    )
    data = RightTriangleWithRay(triangle=triangle)
    file_name = draw_triangle_with_ray(data)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_triangle_with_ray_no_rays():
    with pytest.raises(ValidationError):
        triangle = RightTriangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure=90),
                Angle(vertex="B", measure=45),
                Angle(vertex="C", measure=45),
            ],
            rays=[],
        )
        RightTriangleWithRay(triangle=triangle)


@pytest.mark.drawing_functions
def test_draw_triangle_with_ray_invalid_angle():
    with pytest.raises(ValidationError):
        triangle = RightTriangle(
            points=[
                TrianglePoint(label="A"),
                TrianglePoint(label="B"),
                TrianglePoint(label="C"),
            ],
            angles=[
                Angle(vertex="A", measure=100),
                Angle(vertex="B", measure=40),
                Angle(vertex="C", measure=40),
            ],
            rays=[Ray(start_label="A", measures=[30, "Ray 3"])],
        )
        data = RightTriangleWithRay(triangle=triangle)
        draw_triangle_with_ray(data)


@pytest.mark.drawing_functions
def test_draw_labeled_transversal_angle_basic():
    params = TransversalAngleParams(
        given_angle=100, given_angle_position=6, x_angle_position=4
    )
    file_name = draw_labeled_transversal_angle(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_labeled_transversal_angle_edge_case_min():
    params = TransversalAngleParams(
        given_angle=30, given_angle_position=1, x_angle_position=8
    )
    file_name = draw_labeled_transversal_angle(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_labeled_transversal_angle_edge_case_max():
    params = TransversalAngleParams(
        given_angle=60, given_angle_position=8, x_angle_position=1
    )
    file_name = draw_labeled_transversal_angle(params)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_basic():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 45, "points": ["A", "B", "C"]},
                {"measure": 45, "points": ["C", "B", "D"]},
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_obtuse_angle():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 150, "points": ["A", "B", "C"]},
                {"measure": 30, "points": ["C", "B", "D"]},
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_acute_angles():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 30, "points": ["A", "B", "C"]},
                {"measure": 60, "points": ["C", "B", "D"]},
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_should_share_common_point():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 100, "points": ["A", "B", "C"]},
                {"measure": 50, "points": ["B", "C", "D"]},
                {"measure": 30, "points": ["C", "D", "E"]},
            ]
        }
    }
    with pytest.raises(ValueError):
        generate_angle_diagram(stimulus_description)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_single_angle():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 90, "points": ["A", "B", "C"]},
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_angle_measures():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 20, "points": ["A", "O", "B"]},
                {"measure": 70, "points": ["B", "O", "C"]},
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_unknown_angle():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 50.0, "points": ["A", "B", "C"]},
                {
                    "measure": "?",
                    "positioning_measure": 50.0,
                    "points": ["C", "B", "D"],
                },
                {"measure": 80.0, "points": ["D", "B", "E"]},
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_variables():
    stimulus_description = {
        "diagram": {
            "angles": [
                {
                    "measure": "x",
                    "positioning_measure": 40.0,
                    "points": ["A", "B", "C"],
                },
                {"measure": 60.0, "points": ["C", "B", "D"]},
                {
                    "measure": "y",
                    "positioning_measure": 80.0,
                    "points": ["D", "B", "E"],
                },
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_with_expressions():
    stimulus_description = {
        "diagram": {
            "angles": [
                {
                    "measure": "2x",
                    "positioning_measure": 45.0,
                    "points": ["A", "B", "C"],
                },
                {"measure": 95.0, "points": ["C", "B", "D"]},
                {
                    "measure": "3x-5",
                    "positioning_measure": 40.0,
                    "points": ["D", "B", "E"],
                },
            ]
        }
    }
    file_name = generate_angle_diagram(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_angle_diagram_should_share_common_point_2():
    stimulus_description = {
        "diagram": {
            "angles": [
                {"measure": 100, "points": ["A", "B", "C"]},
                {"measure": 50, "points": ["C", "B", "D"]},
                {"measure": 50, "points": ["C", "B", "E"]},
            ]
        }
    }
    with pytest.raises(ValueError):
        generate_angle_diagram(stimulus_description)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_acute():
    """Test generating a single acute angle."""
    stimulus_description = SingleAngle(measure=45)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_right():
    """Test generating a single right angle (should show square marker)."""
    stimulus_description = SingleAngle(measure=90)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_obtuse():
    """Test generating a single obtuse angle."""
    stimulus_description = SingleAngle(measure=135)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_large_obtuse():
    """Test generating a single large obtuse angle."""
    stimulus_description = SingleAngle(measure=170)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_small_acute():
    """Test generating a single small acute angle."""
    stimulus_description = SingleAngle(measure=15)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_reflex():
    """Test generating a reflex angle (> 180°)."""
    stimulus_description = SingleAngle(measure=270)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_near_complete():
    """Test generating a near-complete angle."""
    stimulus_description = SingleAngle(measure=350)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_complete():
    """Test generating a complete angle (360°)."""
    stimulus_description = SingleAngle(measure=360)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_straight():
    """Test generating a straight angle (180°)."""
    stimulus_description = SingleAngle(measure=180)
    file_name = generate_single_angle_type(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_generate_single_angle_type_invalid_angle():
    """Test that invalid angle measures raise ValidationError."""
    with pytest.raises(ValidationError):
        SingleAngle(measure=0)  # Should fail gt=0 constraint

    with pytest.raises(ValidationError):
        SingleAngle(measure=361)  # Should fail le=360 constraint
