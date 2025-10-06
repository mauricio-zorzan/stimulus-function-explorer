import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.graphing import (
    draw_blank_coordinate_plane,
    plot_line,
    plot_nonlinear,
    plot_par_and_perp_lines,
    plot_points,
    plot_points_four_quadrants,
    plot_points_four_quadrants_with_label,
    plot_points_quadrant_one,
    plot_points_quadrant_one_with_context,
    plot_polygon_dilation,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.blank_coordinate_plane import (
    BlankCoordinatePlane,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.nonlinear_graph import (
    NonlinearEquationParameters,
    NonlinearGraph,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_lines import (
    Line,
    PlotLines,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.plot_points import (
    Point,
    PointList,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_dilation import (
    PolygonDilationStimulus,
    create_advanced_dilation_stimulus,
    create_basic_dilation_stimulus,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    Point as PolygonPoint,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.polygon_list import (
    Polygon,
    PolygonDilation,
)


@pytest.mark.drawing_functions
def test_plot_line_with_overlapping_axis_ticks_and_slope_line():
    stimulus_description = {
        "title": "Tank Volume over Time",
        "x_axis": {"label": "Minutes", "range": [0, 127]},
        "y_axis": {"label": "Volume (Gallons)", "range": [0, 533]},
        "line": {"intercept": 25, "slope": 5},
        "point": [0, 25],
    }
    file_name = plot_line(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_line_with_overlapping_point_and_ticks():
    stimulus_description = {
        "title": "Reading Speed",
        "x_axis": {"label": "Time (hours)", "range": [0, 24]},
        "y_axis": {"label": "Pages read", "range": [0, 120]},
        "line": {"intercept": 0, "slope": 7},
        "point": [17, 119],
    }
    file_name = plot_line(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_point_plots_with_context():
    stimulus_description = {
        "x_title": "Number of Pine Trees",
        "y_title": "Number of Mable Trees",
        "points": [
            {"label": "A", "x": 100, "y": 200},
            {"label": "B", "x": 200, "y": 100},
            {"label": "C", "x": 120, "y": 180},
            {"label": "D", "x": 180, "y": 120},
        ],
    }

    file_name = plot_points_quadrant_one_with_context(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_point_plots_with_context_2():
    stimulus_description = {
        "x_title": "Number of Pine Trees",
        "y_title": "Number of Mable Trees",
        "points": [
            {"label": "A", "x": 100, "y": 20},
            {"label": "B", "x": 200, "y": 10},
            {"label": "C", "x": 120, "y": 18},
            {"label": "D", "x": 180, "y": 12},
        ],
    }

    file_name = plot_points_quadrant_one_with_context(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_point_plots_with_context_3():
    stimulus_description = {
        "x_title": "Number of Pine Trees",
        "y_title": "Number of Mable Trees",
        "points": [
            {"label": "A", "x": 170, "y": 240},
            {"label": "B", "x": 140, "y": 260},
            {"label": "C", "x": 171, "y": 230},
            {"label": "D", "x": 190, "y": 210},
        ],
    }

    file_name = plot_points_quadrant_one_with_context(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_single_point():
    points = {"points": [{"x": 0, "y": 0, "label": "Origin"}]}
    file_name = plot_points(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_large_values():
    points = {
        "points": [
            {"x": 1e6, "y": 1e6, "label": "Large"},
            {"x": -1e6, "y": -1e6, "label": "Large Negative"},
        ]
    }
    file_name = plot_points(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_mixed_quadrants():
    points = {
        "points": [
            {"x": 1, "y": 1, "label": "Q1"},
            {"x": -1, "y": 1, "label": "Q2"},
            {"x": -1, "y": -1, "label": "Q3"},
            {"x": 1, "y": -1, "label": "Q4"},
        ]
    }
    file_name = plot_points(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_all_zero():
    points = {
        "points": [
            {"x": 0, "y": 0, "label": "Zero1"},
            {"x": 0, "y": 0, "label": "Zero2"},
        ]
    }
    with pytest.raises(ValueError):
        plot_points(points)


@pytest.mark.drawing_functions
def test_plot_points_fractional_values():
    points = {
        "points": [
            {"x": 0.1, "y": 0.2, "label": "Frac1"},
            {"x": -0.3, "y": 0.4, "label": "Frac2"},
        ]
    }
    file_name = plot_points(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_quadrant_one():
    stimulus_description = {
        "x_title": "X-axis Title",
        "y_title": "Y-axis Title",
        "points": [
            {"label": "A", "x": 1, "y": 2},
            {"label": "B", "x": 3, "y": 4},
            {"label": "C", "x": 5, "y": 6},
        ],
    }

    file_name = plot_points_quadrant_one(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_quadrant_one_large_values():
    stimulus_description = {
        "x_title": "X-axis Title",
        "y_title": "Y-axis Title",
        "points": [
            {"label": "A", "x": 100, "y": 200},
            {"label": "B", "x": 300, "y": 400},
            {"label": "C", "x": 500, "y": 600},
        ],
    }

    file_name = plot_points_quadrant_one(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_quadrant_one_intermediate_values():
    stimulus_description = {
        "x_title": "Miles East of Town Center",
        "y_title": "Miles North of Town Center",
        "points": [
            {"label": "Museum", "x": 7.0, "y": 22.0},
            {"label": "Library", "x": 12.0, "y": 5.0},
            {"label": "Cafe", "x": 19.0, "y": 3.0},
            {"label": "Theater", "x": 5.0, "y": 18.0},
            {"label": "Stadium", "x": 15.0, "y": 8.0},
        ],
    }

    file_name = plot_points_quadrant_one(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants():
    points = PointList(
        root=[
            Point(label="A", x=1, y=1),
            Point(label="B", x=-1, y=1),
            Point(label="C", x=-1, y=-1),
            Point(label="D", x=1, y=-1),
        ]
    )

    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_single_point():
    points = PointList(
        root=[
            Point(label="OnlyOne", x=0, y=0),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)
    # The axis limits should be [-1, 1] padded, but clamped to [-20, 20]
    # (in this case, should be [-1, 1] since it's within the cap)
    # Visual inspection or further matplotlib inspection could be added here


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_high_variance():
    points = PointList(
        root=[
            Point(label="A", x=-100, y=100),
            Point(label="B", x=100, y=-100),
            Point(label="C", x=0, y=0),
            Point(label="D", x=50, y=75),
            Point(label="E", x=-75, y=-50),
        ]
    )
    # The axis limits should be capped at [-20, 20], but if any point is outside this range, raise an error
    with pytest.raises(ValueError):
        plot_points_four_quadrants(points)


@pytest.mark.drawing_functions
def test_plot_par_and_perp_lines():
    # Create a sample list of lines with valid letter coordinate labels
    line_list = PlotLines(
        [
            Line(slope=1, y_intercept=0, label="AB"),
            Line(slope=-1, y_intercept=0, label="CD"),
            Line(slope=0.5, y_intercept=1, label="EF"),
        ]
    )

    # Call the function and get the file name
    file_name = plot_par_and_perp_lines(line_list)

    # Check if the file was created
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_par_and_perp_lines_parallel():
    # Two parallel lines (same slope, different intercepts) with valid labels
    line_list = PlotLines(
        [
            Line(slope=2, y_intercept=1, label="GH"),
            Line(slope=2, y_intercept=-3, label="IJ"),
        ]
    )
    file_name = plot_par_and_perp_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_par_and_perp_lines_single():
    # Single line with valid label
    line_list = PlotLines(
        [
            Line(slope=0.75, y_intercept=2, label="KL"),
        ]
    )
    file_name = plot_par_and_perp_lines(line_list)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_par_and_perp_lines_letter_coordinates():
    # Test with letter coordinate labels like "AB", "CD" (typical geometry case)
    line_list = PlotLines(
        [
            Line(slope=1, y_intercept=0, label="AB"),
            Line(slope=-1, y_intercept=0, label="CD"),
            Line(slope=0.5, y_intercept=2, label="EF"),
        ]
    )

    # Call the function and get the file name
    file_name = plot_par_and_perp_lines(line_list)

    # Check if the file was created
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_par_and_perp_lines_invalid_labels():
    # Test that invalid labels raise ValueError
    invalid_labels = ["Line 1", "Parallel 1", "AB1", "abc", "A", "ABC"]

    for invalid_label in invalid_labels:
        line_list = PlotLines([Line(slope=1, y_intercept=0, label=invalid_label)])

        with pytest.raises(ValueError, match="Line label .* is not valid"):
            plot_par_and_perp_lines(line_list)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_scale_2():
    # Test basic dilation with scale factor 2
    preimage = Polygon(
        points=[
            PolygonPoint(x=1, y=1, label="A"),
            PolygonPoint(x=2, y=1, label="B"),
            PolygonPoint(x=2, y=2, label="C"),
            PolygonPoint(x=1, y=2, label="D"),
        ],
        label="ABCD",
    )

    image = Polygon(
        points=[
            PolygonPoint(x=2, y=2, label="A"),
            PolygonPoint(x=4, y=2, label="B"),
            PolygonPoint(x=4, y=4, label="C"),
            PolygonPoint(x=2, y=4, label="D"),
        ],
        label="A'B'C'D'",
    )

    center = PolygonPoint(x=0, y=0, label="O")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=2.0,
        center_of_dilation=center,
        show_center=True,
    )

    # Call the function
    file_name = plot_polygon_dilation(dilation)

    # Check if the file was created
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_scale_half():
    # Test dilation with scale factor 0.5 (reduction)
    preimage = Polygon(
        points=[
            PolygonPoint(x=2, y=2, label="P"),
            PolygonPoint(x=4, y=2, label="Q"),
            PolygonPoint(x=4, y=4, label="R"),
            PolygonPoint(x=2, y=4, label="S"),
        ],
        label="PQRS",
    )

    image = Polygon(
        points=[
            PolygonPoint(x=1, y=1, label="P"),
            PolygonPoint(x=2, y=1, label="Q"),
            PolygonPoint(x=2, y=2, label="R"),
            PolygonPoint(x=1, y=2, label="S"),
        ],
        label="P'Q'R'S'",
    )

    center = PolygonPoint(x=0, y=0, label="O")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=0.5,
        center_of_dilation=center,
        show_center=True,
    )

    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_triangle():
    # Test dilation with a triangle (using smaller coordinates to avoid label overlap)
    preimage = Polygon(
        points=[
            PolygonPoint(x=1, y=1, label="A"),
            PolygonPoint(x=2, y=1, label="B"),
            PolygonPoint(x=1, y=2, label="C"),
        ],
        label="ABC",
    )

    image = Polygon(
        points=[
            PolygonPoint(x=3, y=3, label="A"),
            PolygonPoint(x=6, y=3, label="B"),
            PolygonPoint(x=3, y=6, label="C"),
        ],
        label="A'B'C'",
    )

    center = PolygonPoint(x=0, y=0, label="O")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=3.0,
        center_of_dilation=center,
        show_center=True,
        preimage_color="green",
        image_color="purple",
    )

    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_non_origin_center():
    # Test that validation prevents center from coinciding with polygon points
    # This case has center P(1,1) coinciding with preimage point X(1,1)
    preimage = Polygon(
        points=[
            PolygonPoint(x=1, y=1, label="X"),  # This point coincides with center!
            PolygonPoint(x=2, y=1, label="Y"),
            PolygonPoint(x=2, y=2, label="Z"),
        ],
        label="XYZ",
    )

    # Dilation with center at (1, 1) and scale factor 2
    # Point (1,1) stays at (1,1) - creating visual ambiguity
    # Point (2,1) becomes (1,1) + 2*((2,1)-(1,1)) = (1,1) + 2*(1,0) = (3,1)
    # Point (2,2) becomes (1,1) + 2*((2,2)-(1,1)) = (1,1) + 2*(1,1) = (3,3)
    image = Polygon(
        points=[
            PolygonPoint(x=1, y=1, label="X"),
            PolygonPoint(x=3, y=1, label="Y"),
            PolygonPoint(x=3, y=3, label="Z"),
        ],
        label="X'Y'Z'",
    )

    center = PolygonPoint(x=1, y=1, label="P")

    # This should now raise a ValueError due to center coinciding with point X
    with pytest.raises(
        ValueError,
        match="Center of dilation .* cannot coincide with.*preimage point X.*visual ambiguity",
    ):
        PolygonDilation(
            preimage=preimage,
            image=image,
            scale_factor=2.0,
            center_of_dilation=center,
            show_center=True,
        )


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_square_enlargement():
    # Test dilation with square enlargement showing center and rays
    preimage = Polygon(
        points=[
            PolygonPoint(x=-1, y=-1, label="A"),
            PolygonPoint(x=1, y=-1, label="B"),
            PolygonPoint(x=1, y=1, label="C"),
            PolygonPoint(x=-1, y=1, label="D"),
        ],
        label="Square",
    )

    image = Polygon(
        points=[
            PolygonPoint(x=-1.5, y=-1.5, label="A"),
            PolygonPoint(x=1.5, y=-1.5, label="B"),
            PolygonPoint(x=1.5, y=1.5, label="C"),
            PolygonPoint(x=-1.5, y=1.5, label="D"),
        ],
        label="Square'",
    )

    center = PolygonPoint(x=0, y=0, label="O")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=1.5,
        center_of_dilation=center,
        show_center=True,  # Show center and rays for educational value
    )

    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_pentagon():
    # Test dilation with a pentagon
    preimage = Polygon(
        points=[
            PolygonPoint(x=1, y=0, label="A"),
            PolygonPoint(x=2, y=1, label="B"),
            PolygonPoint(x=1, y=2, label="C"),
            PolygonPoint(x=-1, y=2, label="D"),
            PolygonPoint(x=-1, y=0, label="E"),
        ],
        label="Pentagon",
    )

    image = Polygon(
        points=[
            PolygonPoint(x=2, y=0, label="A"),
            PolygonPoint(x=4, y=2, label="B"),
            PolygonPoint(x=2, y=4, label="C"),
            PolygonPoint(x=-2, y=4, label="D"),
            PolygonPoint(x=-2, y=0, label="E"),
        ],
        label="Pentagon'",
    )

    center = PolygonPoint(x=0, y=0, label="O")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=2.0,
        center_of_dilation=center,
        show_center=True,
    )

    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_hexagon_reduction():
    # Test dilation with a hexagon and fractional scale factor (reduction)
    preimage = Polygon(
        points=[
            PolygonPoint(x=2, y=0, label="A"),
            PolygonPoint(x=4, y=0, label="B"),
            PolygonPoint(x=5, y=2, label="C"),
            PolygonPoint(x=4, y=4, label="D"),
            PolygonPoint(x=2, y=4, label="E"),
            PolygonPoint(x=1, y=2, label="F"),
        ],
        label="Hexagon",
    )

    # Scale factor 0.75 (3/4 reduction)
    image = Polygon(
        points=[
            PolygonPoint(x=1.5, y=0, label="A"),
            PolygonPoint(x=3, y=0, label="B"),
            PolygonPoint(x=3.75, y=1.5, label="C"),
            PolygonPoint(x=3, y=3, label="D"),
            PolygonPoint(x=1.5, y=3, label="E"),
            PolygonPoint(x=0.75, y=1.5, label="F"),
        ],
        label="Hexagon'",
    )

    center = PolygonPoint(x=0, y=0, label="O")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=0.75,
        center_of_dilation=center,
        show_center=True,
        preimage_color="orange",
        image_color="purple",
    )

    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_four_quadrants_with_dilation_negative_quadrants():
    # Test dilation with polygon in negative coordinates and non-origin center
    preimage = Polygon(
        points=[
            PolygonPoint(x=-3, y=-1, label="M"),
            PolygonPoint(x=-1, y=-1, label="N"),
            PolygonPoint(x=-1, y=-3, label="O"),
            PolygonPoint(x=-3, y=-3, label="P"),
        ],
        label="MNOP",
    )

    # Dilation with center at (-2, -2) and scale factor 1.5
    # Point (-3,-1) becomes (-2,-2) + 1.5*((-3,-1)-(-2,-2)) = (-2,-2) + 1.5*(-1,1) = (-3.5,-0.5)
    # Point (-1,-1) becomes (-2,-2) + 1.5*((-1,-1)-(-2,-2)) = (-2,-2) + 1.5*(1,1) = (-0.5,-0.5)
    # Point (-1,-3) becomes (-2,-2) + 1.5*((-1,-3)-(-2,-2)) = (-2,-2) + 1.5*(1,-1) = (-0.5,-3.5)
    # Point (-3,-3) becomes (-2,-2) + 1.5*((-3,-3)-(-2,-2)) = (-2,-2) + 1.5*(-1,-1) = (-3.5,-3.5)
    image = Polygon(
        points=[
            PolygonPoint(x=-3.5, y=-0.5, label="M"),
            PolygonPoint(x=-0.5, y=-0.5, label="N"),
            PolygonPoint(x=-0.5, y=-3.5, label="O"),
            PolygonPoint(x=-3.5, y=-3.5, label="P"),
        ],
        label="M'N'O'P'",
    )

    center = PolygonPoint(x=-2, y=-2, label="P")

    dilation = PolygonDilation(
        preimage=preimage,
        image=image,
        scale_factor=1.5,
        center_of_dilation=center,
        show_center=True,
        preimage_color="green",
        image_color="red",
    )

    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_nonlinear_quadratic():
    stimulus_description = NonlinearGraph(
        equation_type="quadratic",
        parameters=NonlinearEquationParameters(coef1=1, coef2=2),
    )
    file_name = plot_nonlinear(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_nonlinear_cubic():
    stimulus_description = NonlinearGraph(
        equation_type="cubic",
        parameters=NonlinearEquationParameters(coef1=1, coef2=2, coef3=3),
    )
    file_name = plot_nonlinear(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_nonlinear_quartic():
    stimulus_description = NonlinearGraph(
        equation_type="quartic",
        parameters=NonlinearEquationParameters(coef1=1, coef2=5, coef3=4, coef4=3),
    )
    file_name = plot_nonlinear(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_nonlinear_quintic():
    stimulus_description = NonlinearGraph(
        equation_type="quintic",
        parameters=NonlinearEquationParameters(
            coef1=1, coef2=5, coef3=4, coef4=3, coef5=4
        ),
    )
    file_name = plot_nonlinear(stimulus_description)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_basic_triangle_enlargement():
    """Test basic complexity triangle with enlargement (scale > 1)."""
    stimulus = create_basic_dilation_stimulus(
        polygon_type="triangle", scale_factor=2.0, quadrant="I"
    )

    # Generate the dilation data from stimulus
    dilation = stimulus.generate_polygon_dilation()

    # Use the main drawing function
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_basic_quadrilateral_reduction():
    """Test basic complexity quadrilateral with reduction (scale < 1)."""
    stimulus = create_basic_dilation_stimulus(
        polygon_type="quadrilateral", scale_factor=0.5, quadrant="I"
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_first_quadrant():
    points = PointList(
        root=[
            Point(label="A", x=2, y=3),
            Point(label="B", x=4, y=5),
            Point(label="C", x=6, y=7),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_advanced_pentagon_enlargement():
    """Test advanced complexity pentagon with enlargement and non-origin center."""
    stimulus = create_advanced_dilation_stimulus(
        polygon_type="pentagon", scale_factor=1.5, quadrant="I"
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_on_axes():
    points = PointList(
        root=[
            Point(label="X0", x=0, y=5),
            Point(label="Y0", x=5, y=0),
            Point(label="O", x=0, y=0),
            Point(label="Xneg", x=-5, y=0),
            Point(label="Yneg", x=0, y=-5),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_quadrant_ii_placement():
    """Test triangle in quadrant II with custom colors."""
    stimulus = PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type="triangle",
        scale_factor=2.0,
        preimage_quadrant="II",
        show_center=True,
        preimage_color="green",
        image_color="purple",
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_decimals():
    points = PointList(
        root=[
            Point(label="P1", x=1.5, y=-2.3),
            Point(label="P2", x=-3.7, y=4.2),
            Point(label="P3", x=0.0, y=0.0),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_quadrant_iii_no_center():
    """Test quadrilateral in quadrant III with center hidden."""
    stimulus = PolygonDilationStimulus(
        complexity_level="ADVANCED",
        polygon_type="quadrilateral",
        scale_factor=0.75,
        preimage_quadrant="III",
        show_center=False,
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_near_limits():
    points = PointList(
        root=[
            Point(label="MaxX", x=19, y=0),
            Point(label="MaxY", x=0, y=19),
            Point(label="MinX", x=-19, y=0),
            Point(label="MinY", x=0, y=-19),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_quadrant_iv_large_scale():
    """Test triangle in quadrant IV with large scale factor."""
    stimulus = PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type="triangle",
        scale_factor=3.0,
        preimage_quadrant="IV",
        show_center=True,
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_diagonal():
    points = PointList(
        root=[
            Point(label="D1", x=-4, y=-4),
            Point(label="D2", x=-2, y=-2),
            Point(label="D3", x=0, y=0),
            Point(label="D4", x=2, y=2),
            Point(label="D5", x=4, y=4),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_mixed_quadrants():
    """Test polygon spanning multiple quadrants."""
    stimulus = PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type="triangle",
        scale_factor=2.0,
        preimage_quadrant="mixed",
        show_center=True,
        preimage_color="blue",
        image_color="red",
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


# New tests focused on plot_points_four_quadrants formatting and bounds


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_decimal_rounding_edge_cases():
    # Values chosen to exercise rounding to a single decimal (0.05 -> 0.1; 0.04 -> 0.0)
    points = PointList(
        root=[
            Point(label="E1", x=2.05, y=-3.04),
            Point(label="E2", x=-7.25, y=1.66),
            Point(label="E3", x=19.95, y=-19.95),  # stays within [-20, 20]
        ]
    )
    file_name = plot_points_four_quadrants_with_label(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_bounds_rejection():
    # Any point outside [-20, 20] in either axis should raise
    points = PointList(
        root=[
            Point(label="OutX", x=21, y=0),
            Point(label="In", x=0, y=0),
        ]
    )
    with pytest.raises(ValueError):
        plot_points_four_quadrants_with_label(points)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_dense_origin_neighbors():
    # Several points around the origin including decimals
    points = PointList(
        root=[
            Point(label="N1", x=0.0, y=0.0),
            Point(label="N2", x=0.1, y=0.1),
            Point(label="N3", x=-0.1, y=0.1),
            Point(label="N4", x=0.1, y=-0.1),
            Point(label="N5", x=-0.1, y=-0.1),
        ]
    )
    file_name = plot_points_four_quadrants_with_label(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_with_label_basic():
    points = PointList(
        root=[
            Point(label="A", x=1.2, y=-3.7),
            Point(label="B", x=0.0, y=0.0),
        ]
    )
    file_name = plot_points_four_quadrants_with_label(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_small_reduction():
    """Test quadrilateral with small reduction factor."""
    stimulus = PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type="quadrilateral",
        scale_factor=0.25,
        preimage_quadrant="I",
        show_center=True,
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_fractional_scale():
    """Test pentagon with fractional scale factor."""
    stimulus = PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type="pentagon",
        scale_factor=1.5,
        preimage_quadrant="I",
        show_center=True,
        preimage_color="orange",
        image_color="blue",
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_polygon_dilation_stimulus_custom_colors():
    """Test triangle with custom color scheme."""
    stimulus = PolygonDilationStimulus(
        complexity_level="BASIC",
        polygon_type="triangle",
        scale_factor=2.0,
        preimage_quadrant="I",
        show_center=True,
        preimage_color="green",
        image_color="purple",
        center_color="orange",
    )

    dilation = stimulus.generate_polygon_dilation()
    file_name = plot_polygon_dilation(dilation)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_plot_points_four_quadrants_duplicate_labels():
    points = PointList(
        root=[
            Point(label="Same", x=1, y=2),
            Point(label="Same", x=-2, y=-1),
            Point(label="Same", x=3, y=-3),
        ]
    )
    file_name = plot_points_four_quadrants(points)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_standard_default():
    """Test standard 4-quadrant coordinate plane with clean appearance"""
    fn = draw_blank_coordinate_plane(BlankCoordinatePlane())
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_small_range():
    """Test small range with dense grid - should be readable"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-5, x_max=5, y_min=-5, y_max=5)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_medium_range():
    """Test medium range that balances detail and readability"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-8, x_max=8, y_min=-8, y_max=8)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_asymmetric_practical():
    """Test asymmetric range commonly used in math problems"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-5, x_max=10, y_min=-3, y_max=8)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_first_quadrant_focus():
    """Test range that emphasizes first quadrant"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-2, x_max=10, y_min=-2, y_max=10)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_without_titles():
    """Test clean coordinate plane without axis titles"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-6, x_max=6, y_min=-6, y_max=6)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_minimal_range():
    """Test minimal range around origin"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-3, x_max=3, y_min=-3, y_max=3)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_classroom_standard():
    """Test typical classroom coordinate plane size"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-12, x_max=12, y_min=-12, y_max=12)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_rectangular():
    """Test rectangular (non-square) coordinate plane"""
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=-8, x_max=8, y_min=-5, y_max=5)
    )
    assert os.path.exists(fn)


@pytest.mark.drawing_functions
def test_draw_blank_coordinate_plane_validation_errors():
    """Test that invalid ranges are handled properly"""
    # Test where origin would be excluded (should auto-include it)
    fn = draw_blank_coordinate_plane(
        BlankCoordinatePlane(x_min=5, x_max=15, y_min=3, y_max=10)
    )
    # Should auto-expand to include origin
    assert os.path.exists(fn)
