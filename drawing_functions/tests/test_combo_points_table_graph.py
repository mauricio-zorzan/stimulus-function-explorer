import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.combo_points_table_graph import (
    draw_combo_points_table_graph,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.combo_points_table_graph import (
    AxisSpec,
    ComboPointsTableGraph,
    GraphSpec,
    Point,
    TableData,
)

# =====================================================
# Function Identification Tests
# =====================================================


@pytest.mark.drawing_functions
def test_linear_function_with_table_and_points():
    """Function identification with table, points, and graph - clear linear function."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"],
            rows=[["1", "3"], ["2", "6"], ["3", "9"], ["4", "12"]],
            title="Function Table",
        ),
        points=[
            Point(label="A", x=1, y=3),
            Point(label="B", x=2, y=6),
            Point(label="C", x=3, y=9),
            Point(label="D", x=4, y=12),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=3,
                y_intercept=0,
                equation="y = 3x",
                color="blue",
                label="Linear Function",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=15, tick_interval=3),
        graph_title="Is this a function?",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_non_function_with_repeated_inputs():
    """Non-function identification with table and points - repeated input values."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Input", "Output"],
            rows=[["1", "2"], ["2", "4"], ["2", "-1"], ["3", "6"]],
            title="Relation Table",
        ),
        points=[
            Point(label="A", x=1, y=2),
            Point(label="B", x=2, y=4),
            Point(label="C", x=2, y=-1),
            Point(label="D", x=3, y=6),
        ],
        x_axis=AxisSpec(label="Input", min_value=0, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="Output", min_value=-2, max_value=7, tick_interval=1),
        highlight_points=["B", "C"],
        graph_title="Not a Function - Repeated Input",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_real_world_car_trip_function():
    """Function identification with coordinate pairs and real-world context."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Time (hours)", "Distance (miles)"],
            rows=[["0", "0"], ["1", "50"], ["2", "100"], ["3", "150"]],
            title="Car Trip Data",
        ),
        points=[
            Point(label="Start", x=0, y=0),
            Point(label="1hr", x=1, y=50),
            Point(label="2hr", x=2, y=100),
            Point(label="3hr", x=3, y=150),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=50,
                y_intercept=0,
                equation="Distance = 50 × Time",
                color="green",
                label="Constant Speed Function",
            )
        ],
        x_axis=AxisSpec(
            label="Time (hours)", min_value=0, max_value=4, tick_interval=1
        ),
        y_axis=AxisSpec(
            label="Distance (miles)", min_value=0, max_value=200, tick_interval=50
        ),
        graph_title="Real-World Function: Car Trip",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_circle_equation_non_function():
    """Circle equation - clear non-function example."""
    stimulus = ComboPointsTableGraph(
        points=[
            Point(label="A", x=0, y=3),
            Point(label="B", x=3, y=0),
            Point(label="C", x=0, y=-3),
            Point(label="D", x=-3, y=0),
            Point(label="E", x=2, y=2.2),
            Point(label="F", x=2, y=-2.2),
        ],
        graphs=[
            GraphSpec(
                type="circle",
                radius=3,
                center_x=0,
                center_y=0,
                equation="x² + y² = 9",
                color="red",
                label="Circle (Non-Function)",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-4, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-4, max_value=4, tick_interval=1),
        highlight_points=["E", "F"],
        graph_title="Circle: x² + y² = 9 (Not a Function)",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Proportional Relationships Comparison Tests
# =====================================================


@pytest.mark.drawing_functions
def test_compare_table_vs_equation_rates():
    """Compare table vs equation - significant rate difference."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A"],
            rows=[["1", "5"], ["2", "10"], ["3", "15"], ["4", "20"]],
            title="Function A: Table (Rate = 5)",
        ),
        points=[
            Point(label="(1,5)", x=1, y=5),
            Point(label="(2,10)", x=2, y=10),
            Point(label="(3,15)", x=3, y=15),
            Point(label="(4,20)", x=4, y=20),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=5,
                y_intercept=0,
                equation="Function A: y = 5x",
                color="blue",
                label="Function A (Rate: 5)",
            ),
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=0,
                equation="Function B: y = 2x",
                color="red",
                label="Function B (Rate: 2)",
                line_style="--",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=25, tick_interval=5),
        graph_title="Compare Rates: Table vs Equation",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_compare_job_hourly_rates():
    """Compare table vs graph - job hourly rate comparison."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Hours", "Earnings ($)"],
            rows=[["2", "15"], ["4", "30"], ["6", "45"], ["8", "60"]],
            title="Job A: $7.50/hour",
        ),
        points=[
            Point(label="A", x=2, y=15),
            Point(label="B", x=4, y=30),
            Point(label="C", x=6, y=45),
            Point(label="D", x=8, y=60),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=7.5,
                y_intercept=0,
                color="blue",
                label="Job A: $7.50/hour",
            ),
            GraphSpec(
                type="line",
                slope=8.25,
                y_intercept=0,
                color="green",
                label="Job B: $8.25/hour",
                line_style="-.",
            ),
        ],
        x_axis=AxisSpec(label="Hours", min_value=0, max_value=10, tick_interval=2),
        y_axis=AxisSpec(
            label="Earnings ($)", min_value=0, max_value=80, tick_interval=10
        ),
        graph_title="Compare Job Rates: Table vs Graph",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_compare_recipe_flour_usage():
    """Compare equation vs graph - proportional relationships."""
    stimulus = ComboPointsTableGraph(
        graphs=[
            GraphSpec(
                type="line",
                slope=3.5,
                y_intercept=0,
                equation="Recipe A: y = 3.5x",
                color="purple",
                label="Recipe A: 3.5 cups flour per batch",
            ),
            GraphSpec(
                type="line",
                slope=4.2,
                y_intercept=0,
                equation="Recipe B: y = 4.2x",
                color="orange",
                label="Recipe B: 4.2 cups flour per batch",
                line_style=":",
            ),
        ],
        x_axis=AxisSpec(
            label="Number of Batches", min_value=0, max_value=6, tick_interval=1
        ),
        y_axis=AxisSpec(
            label="Cups of Flour", min_value=0, max_value=25, tick_interval=5
        ),
        graph_title="Compare Recipes: Which uses more flour per batch?",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Function Properties Comparison Tests
# =====================================================


@pytest.mark.drawing_functions
def test_compare_function_slopes():
    """Compare slopes - positive slope comparison."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A"],
            rows=[["0", "2"], ["1", "5"], ["2", "8"], ["3", "11"]],
            title="Function A: Table",
        ),
        points=[
            Point(label="(0,2)", x=0, y=2),
            Point(label="(1,5)", x=1, y=5),
            Point(label="(2,8)", x=2, y=8),
            Point(label="(3,11)", x=3, y=11),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=3,
                y_intercept=2,
                equation="Function A: y = 3x + 2",
                color="blue",
                label="Function A (slope = 3)",
            ),
            GraphSpec(
                type="line",
                slope=1,
                y_intercept=4,
                equation="Function B: y = x + 4",
                color="red",
                label="Function B (slope = 1)",
                line_style="--",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=15, tick_interval=2),
        graph_title="Compare Slopes: Which function increases faster?",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_compare_y_intercepts():
    """Compare y-intercepts - mix of positive and negative."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A"],
            rows=[["0", "-3"], ["1", "-1"], ["2", "1"], ["3", "3"]],
            title="Function A: y-intercept = -3",
        ),
        points=[
            Point(label="(0,-3)", x=0, y=-3),
            Point(label="(1,-1)", x=1, y=-1),
            Point(label="(2,1)", x=2, y=1),
            Point(label="(3,3)", x=3, y=3),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=-3,
                equation="Function A: y = 2x - 3",
                color="blue",
                label="Function A (y-int = -3)",
            ),
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=1,
                equation="Function B: y = 2x + 1",
                color="green",
                label="Function B (y-int = 1)",
                line_style="-.",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=-1, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-5, max_value=8, tick_interval=2),
        highlight_points=["(0,-3)"],
        graph_title="Compare Y-Intercepts: Same slope, different starting points",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_compare_decreasing_functions():
    """Compare y-values at specific x - negative slopes."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A"],
            rows=[["0", "8"], ["1", "6"], ["2", "4"], ["3", "2"]],
            title="Function A: Decreasing",
        ),
        points=[
            Point(label="A(0,8)", x=0, y=8),
            Point(label="A(1,6)", x=1, y=6),
            Point(label="A(2,4)", x=2, y=4),
            Point(label="A(3,2)", x=3, y=2),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=-2,
                y_intercept=8,
                equation="Function A: y = -2x + 8",
                color="red",
                label="Function A (slope = -2)",
            ),
            GraphSpec(
                type="line",
                slope=-1,
                y_intercept=5,
                equation="Function B: y = -x + 5",
                color="purple",
                label="Function B (slope = -1)",
                line_style=":",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
        highlight_points=["A(2,4)"],
        graph_title="Compare Functions: Which decreases faster?",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_comprehensive_function_analysis():
    """Compare all properties - slope, y-intercept, and specific values."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A"],
            rows=[["0", "5"], ["1", "2"], ["2", "-1"], ["3", "-4"]],
            title="Function A: Complete Analysis",
        ),
        points=[
            Point(label="A(0,5)", x=0, y=5),
            Point(label="A(1,2)", x=1, y=2),
            Point(label="A(2,-1)", x=2, y=-1),
            Point(label="A(3,-4)", x=3, y=-4),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=-3,
                y_intercept=5,
                equation="Function A: y = -3x + 5",
                color="blue",
                label="Function A (m=-3, b=5)",
            ),
            GraphSpec(
                type="line",
                slope=1.5,
                y_intercept=-2,
                equation="Function B: y = 1.5x - 2",
                color="orange",
                label="Function B (m=1.5, b=-2)",
                line_style="--",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=-1, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-6, max_value=6, tick_interval=2),
        highlight_points=["A(0,5)", "A(1,2)", "A(2,-1)"],
        graph_title="Complete Function Analysis: Compare all properties",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Edge Cases and Special Scenarios
# =====================================================


@pytest.mark.drawing_functions
def test_points_only_scatter_plot():
    """Test with only points - scatter plot scenario."""
    stimulus = ComboPointsTableGraph(
        points=[
            Point(label="A", x=1, y=2),
            Point(label="B", x=2, y=3),
            Point(label="C", x=2, y=5),
            Point(label="D", x=3, y=1),
            Point(label="E", x=4, y=4),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=6, tick_interval=1),
        highlight_points=["B", "C"],
        graph_title="Scatter Plot: Data Points Only",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_table_only_no_graph():
    """Test with only table - no visual graph."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Input", "Output"],
            rows=[["1", "4"], ["2", "8"], ["3", "12"], ["4", "16"], ["5", "20"]],
            title="Multiplication by 4 Table",
        ),
        x_axis=AxisSpec(label="Input", min_value=0, max_value=6, tick_interval=1),
        y_axis=AxisSpec(label="Output", min_value=0, max_value=25, tick_interval=5),
        graph_title="Empty Coordinate Plane",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_multiple_function_types():
    """Test with multiple graphs for complex comparison."""
    stimulus = ComboPointsTableGraph(
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=1,
                equation="Linear: y = 2x + 1",
                color="blue",
                label="Linear Function",
            ),
            GraphSpec(
                type="scatter",
                points=[
                    Point(label="", x=1, y=1),
                    Point(label="", x=2, y=4),
                    Point(label="", x=3, y=9),
                    Point(label="", x=4, y=16),
                ],
                color="red",
                label="Quadratic Points",
            ),
            GraphSpec(
                type="line",
                slope=-1,
                y_intercept=10,
                equation="Decreasing: y = -x + 10",
                color="green",
                label="Decreasing Function",
                line_style=":",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=20, tick_interval=5),
        graph_title="Multiple Function Types Comparison",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_age_height_relationship():
    """Test real-world context for function identification."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Person", "Age", "Height (cm)"],
            rows=[
                ["Alice", "25", "165"],
                ["Bob", "30", "175"],
                ["Charlie", "25", "180"],
                ["Diana", "35", "160"],
            ],
            title="Age vs Height: Is this a function?",
        ),
        points=[
            Point(label="Alice", x=25, y=165),
            Point(label="Bob", x=30, y=175),
            Point(label="Charlie", x=25, y=180),
            Point(label="Diana", x=35, y=160),
        ],
        x_axis=AxisSpec(
            label="Age (years)", min_value=20, max_value=40, tick_interval=5
        ),
        y_axis=AxisSpec(
            label="Height (cm)", min_value=150, max_value=190, tick_interval=10
        ),
        highlight_points=["Alice", "Charlie"],
        graph_title="Real World: Age → Height (Not a Function)",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Advanced Non-Function Examples
# =====================================================


@pytest.mark.drawing_functions
def test_circle_with_radius_five():
    """Circle x² + y² = 25 (clear non-function example)."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"],
            rows=[
                ["0", "5"],
                ["0", "-5"],
                ["3", "4"],
                ["3", "-4"],
                ["5", "0"],
                ["-5", "0"],
            ],
            title="Circle: x² + y² = 25",
        ),
        points=[
            Point(label="A", x=0, y=5),
            Point(label="B", x=0, y=-5),
            Point(label="C", x=3, y=4),
            Point(label="D", x=3, y=-4),
            Point(label="E", x=5, y=0),
            Point(label="F", x=-5, y=0),
        ],
        graphs=[
            GraphSpec(
                type="circle",
                radius=5,
                center_x=0,
                center_y=0,
                equation="x² + y² = 25",
                color="red",
                label="Circle (Non-Function)",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-6, max_value=6, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-6, max_value=6, tick_interval=1),
        highlight_points=["A", "B", "C", "D"],
        graph_title="Circle: x² + y² = 25",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_sideways_parabola_equation():
    """Sideways parabola x = y² (clear non-function example)."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["y", "x"],
            rows=[["0", "0"], ["1", "1"], ["-1", "1"], ["2", "4"], ["-2", "4"]],
            title="Sideways Parabola: x = y²",
        ),
        points=[
            Point(label="A", x=0, y=0),
            Point(label="B", x=1, y=1),
            Point(label="C", x=1, y=-1),
            Point(label="D", x=4, y=2),
            Point(label="E", x=4, y=-2),
        ],
        graphs=[
            GraphSpec(
                type="sideways_parabola",
                a=1,
                k=0,
                h=0,
                equation="x = y²",
                color="purple",
                label="Sideways Parabola (Non-Function)",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-1, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-3, max_value=3, tick_interval=1),
        highlight_points=["B", "C", "D", "E"],
        graph_title="Sideways Parabola: x = y²",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_hyperbola_equation():
    """Basic hyperbola x² - y² = 9 (clear non-function example)."""
    stimulus = ComboPointsTableGraph(
        points=[
            Point(label="A", x=3, y=0),
            Point(label="B", x=-3, y=0),
            Point(label="C", x=4, y=2.6),
            Point(label="D", x=4, y=-2.6),
            Point(label="E", x=-4, y=2.6),
            Point(label="F", x=-4, y=-2.6),
        ],
        graphs=[
            GraphSpec(
                type="hyperbola",
                a=3,
                equation="x² - y² = 9",
                color="orange",
                label="Hyperbola (Non-Function)",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-5, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-4, max_value=4, tick_interval=1),
        highlight_points=["C", "D", "E", "F"],
        graph_title="Hyperbola: x² - y² = 9",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_ellipse_equation():
    """Simple ellipse x²/4 + y²/4 = 1 (clear non-function example)."""
    stimulus = ComboPointsTableGraph(
        points=[
            Point(label="A", x=2, y=0),
            Point(label="B", x=-2, y=0),
            Point(label="C", x=0, y=2),
            Point(label="D", x=0, y=-2),
            Point(label="E", x=1.4, y=1.4),
            Point(label="F", x=1.4, y=-1.4),
        ],
        graphs=[
            GraphSpec(
                type="ellipse",
                a=2,
                b=2,
                center_x=0,
                center_y=0,
                equation="x²/4 + y²/4 = 1",
                color="green",
                label="Ellipse (Non-Function)",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-3, max_value=3, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-3, max_value=3, tick_interval=1),
        highlight_points=["E", "F"],
        graph_title="Ellipse: x²/4 + y²/4 = 1",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_table_with_repeated_inputs():
    """Table with repeated inputs/different outputs (clear non-function example)."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Input", "Output"],
            rows=[
                ["1", "2"],
                ["2", "4"],
                ["2", "6"],
                ["3", "8"],
                ["3", "10"],
                ["4", "12"],
            ],
            title="Repeated Inputs → Different Outputs",
        ),
        points=[
            Point(label="A", x=1, y=2),
            Point(label="B", x=2, y=4),
            Point(label="C", x=2, y=6),
            Point(label="D", x=3, y=8),
            Point(label="E", x=3, y=10),
            Point(label="F", x=4, y=12),
        ],
        x_axis=AxisSpec(label="Input", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="Output", min_value=0, max_value=14, tick_interval=2),
        highlight_points=["B", "C", "D", "E"],
        graph_title="Table with Repeated Inputs",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Clear Function Examples
# =====================================================


@pytest.mark.drawing_functions
def test_basic_linear_function():
    """Clear linear function y = 2x + 3 example."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"],
            rows=[["0", "3"], ["1", "5"], ["2", "7"], ["3", "9"], ["4", "11"]],
            title="Linear Function: y = 2x + 3",
        ),
        points=[
            Point(label="A", x=0, y=3),
            Point(label="B", x=1, y=5),
            Point(label="C", x=2, y=7),
            Point(label="D", x=3, y=9),
            Point(label="E", x=4, y=11),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=3,
                equation="y = 2x + 3",
                color="blue",
                label="Linear Function",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=12, tick_interval=2),
        graph_title="Linear Function: y = 2x + 3",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_basic_quadratic_function():
    """Clear quadratic function y = x² example."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"],
            rows=[
                ["0", "0"],
                ["1", "1"],
                ["2", "4"],
                ["3", "9"],
                ["-1", "1"],
                ["-2", "4"],
            ],
            title="Quadratic Function: y = x²",
        ),
        points=[
            Point(label="A", x=0, y=0),
            Point(label="B", x=1, y=1),
            Point(label="C", x=2, y=4),
            Point(label="D", x=3, y=9),
            Point(label="E", x=-1, y=1),
            Point(label="F", x=-2, y=4),
        ],
        graphs=[
            GraphSpec(
                type="quadratic",
                a=1,
                b=0,
                c=0,
                equation="y = x²",
                color="red",
                label="Quadratic Function",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-3, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
        graph_title="Quadratic Function: y = x²",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_square_root_function():
    """Clear square root function y = √x example."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"],
            rows=[["0", "0"], ["1", "1"], ["4", "2"], ["9", "3"], ["16", "4"]],
            title="Square Root Function: y = √x",
        ),
        points=[
            Point(label="A", x=0, y=0),
            Point(label="B", x=1, y=1),
            Point(label="C", x=4, y=2),
            Point(label="D", x=9, y=3),
            Point(label="E", x=16, y=4),
        ],
        graphs=[
            GraphSpec(
                type="sqrt",
                a=1,
                h=0,
                k=0,
                equation="y = √x",
                color="green",
                label="Square Root Function",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=18, tick_interval=2),
        y_axis=AxisSpec(label="y", min_value=0, max_value=5, tick_interval=1),
        graph_title="Square Root Function: y = √x",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_rational_function():
    """Simple rational function y = 2/x example."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"],
            rows=[
                ["1", "2"],
                ["2", "1"],
                ["4", "0.5"],
                ["-1", "-2"],
                ["-2", "-1"],
                ["-4", "-0.5"],
            ],
            title="Rational Function: y = 2/x",
        ),
        points=[
            Point(label="A", x=1, y=2),
            Point(label="B", x=2, y=1),
            Point(label="C", x=4, y=0.5),
            Point(label="D", x=-1, y=-2),
            Point(label="E", x=-2, y=-1),
            Point(label="F", x=-4, y=-0.5),
        ],
        graphs=[
            GraphSpec(
                type="rational",
                a=2,
                k=0,
                equation="y = 2/x",
                color="purple",
                label="Rational Function",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=-5, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-3, max_value=3, tick_interval=1),
        graph_title="Rational Function: y = 2/x",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Enhanced Comparison Tests
# =====================================================


@pytest.mark.drawing_functions
def test_proportional_relationships_comparison():
    """Enhanced proportional relationships comparison (y = kx)."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Relationship A", "Relationship B"],
            rows=[
                ["1", "3", "4"],
                ["2", "6", "8"],
                ["3", "9", "12"],
                ["4", "12", "16"],
            ],
            title="Proportional Relationships: A vs B",
        ),
        points=[
            Point(label="A1", x=1, y=3),
            Point(label="A2", x=2, y=6),
            Point(label="A3", x=3, y=9),
            Point(label="A4", x=4, y=12),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=3,
                y_intercept=0,
                equation="y = 3x",
                color="blue",
                label="Relationship A (k=3)",
            ),
            GraphSpec(
                type="line",
                slope=4,
                y_intercept=0,
                equation="y = 4x",
                color="red",
                label="Relationship B (k=4)",
                line_style="--",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=20, tick_interval=4),
        graph_title="Compare Proportional Relationships",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_rate_slope_comparison():
    """Enhanced rate/slope comparison between representations."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["Time", "Distance A", "Distance B"],
            rows=[["0", "0", "5"], ["1", "2", "8"], ["2", "4", "11"], ["3", "6", "14"]],
            title="Rate Comparison: A vs B",
        ),
        points=[
            Point(label="A0", x=0, y=0),
            Point(label="A1", x=1, y=2),
            Point(label="A2", x=2, y=4),
            Point(label="A3", x=3, y=6),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=0,
                equation="Distance A = 2t",
                color="green",
                label="Rate A: 2 units/time",
            ),
            GraphSpec(
                type="line",
                slope=3,
                y_intercept=5,
                equation="Distance B = 3t + 5",
                color="orange",
                label="Rate B: 3 units/time",
                line_style="-.",
            ),
        ],
        x_axis=AxisSpec(label="Time", min_value=0, max_value=4, tick_interval=1),
        y_axis=AxisSpec(label="Distance", min_value=0, max_value=16, tick_interval=2),
        graph_title="Rate/Slope Comparison",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_y_intercept_comparison_multiple_functions():
    """Enhanced y-intercept comparison with multiple functions."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A", "Function B", "Function C"],
            rows=[["0", "2", "-1", "4"], ["1", "4", "1", "6"], ["2", "6", "3", "8"]],
            title="Y-Intercept Comparison",
        ),
        points=[
            Point(label="A(0,2)", x=0, y=2),
            Point(label="B(0,-1)", x=0, y=-1),
            Point(label="C(0,4)", x=0, y=4),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=2,
                equation="Function A: y = 2x + 2",
                color="blue",
                label="Function A (y-int = 2)",
            ),
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=-1,
                equation="Function B: y = 2x - 1",
                color="red",
                label="Function B (y-int = -1)",
                line_style="--",
            ),
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=4,
                equation="Function C: y = 2x + 4",
                color="green",
                label="Function C (y-int = 4)",
                line_style=":",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=-1, max_value=3, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=-2, max_value=10, tick_interval=2),
        highlight_points=["A(0,2)", "B(0,-1)", "C(0,4)"],
        graph_title="Y-Intercept Comparison (Same Slope)",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_coordinate_value_comparison():
    """Specific coordinate value comparison at multiple points."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "Function A", "Function B"],
            rows=[["1", "3", "2"], ["2", "5", "8"], ["3", "7", "18"], ["4", "9", "32"]],
            title="Coordinate Value Comparison",
        ),
        points=[
            Point(label="A(1,3)", x=1, y=3),
            Point(label="B(1,2)", x=1, y=2),
            Point(label="A(2,5)", x=2, y=5),
            Point(label="B(2,8)", x=2, y=8),
            Point(label="A(3,7)", x=3, y=7),
            Point(label="B(3,18)", x=3, y=18),
        ],
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=1,
                equation="Function A: y = 2x + 1",
                color="blue",
                label="Linear Function A",
            ),
            GraphSpec(
                type="quadratic",
                a=2,
                b=0,
                c=0,
                equation="Function B: y = 2x²",
                color="red",
                label="Quadratic Function B",
                line_style="--",
            ),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=35, tick_interval=5),
        highlight_points=["A(1,3)", "B(1,2)", "A(2,5)", "B(2,8)"],
        graph_title="Compare Linear vs Quadratic Functions",
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


# =====================================================
# Validation Error Tests
# =====================================================


@pytest.mark.drawing_functions
def test_validation_error_no_content():
    """Test that validation error is raised when no table, points, or graphs are provided."""
    stimulus = ComboPointsTableGraph(
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
    )

    with pytest.raises(ValueError) as exc_info:
        stimulus.validate_pipeline()

    assert "At least one of table, points, or graphs must be provided" in str(
        exc_info.value
    )


@pytest.mark.drawing_functions
def test_validation_error_highlight_points_without_points():
    """Test that validation error is raised when highlight_points is specified without points."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"], rows=[["1", "2"], ["2", "4"]], title="Test Table"
        ),
        highlight_points=["A", "B"],  # This should cause error
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
    )

    with pytest.raises(ValueError) as exc_info:
        stimulus.validate_pipeline()

    assert "points must be provided when highlight_points is specified" in str(
        exc_info.value
    )


@pytest.mark.drawing_functions
def test_validation_success_with_only_table():
    """Test that validation passes when only table is provided."""
    stimulus = ComboPointsTableGraph(
        table=TableData(
            headers=["x", "y"], rows=[["1", "2"], ["2", "4"]], title="Test Table"
        ),
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_validation_success_with_only_points():
    """Test that validation passes when only points are provided."""
    stimulus = ComboPointsTableGraph(
        points=[
            Point(label="A", x=1, y=2),
            Point(label="B", x=2, y=4),
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_validation_success_with_only_graphs():
    """Test that validation passes when only graphs are provided."""
    stimulus = ComboPointsTableGraph(
        graphs=[
            GraphSpec(
                type="line",
                slope=2,
                y_intercept=0,
                equation="y = 2x",
                color="blue",
                label="Linear Function",
            )
        ],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)


@pytest.mark.drawing_functions
def test_validation_success_with_highlight_points_and_points():
    """Test that validation passes when both highlight_points and points are provided."""
    stimulus = ComboPointsTableGraph(
        points=[
            Point(label="A", x=1, y=2),
            Point(label="B", x=2, y=4),
            Point(label="C", x=3, y=6),
        ],
        highlight_points=["A", "C"],
        x_axis=AxisSpec(label="x", min_value=0, max_value=5, tick_interval=1),
        y_axis=AxisSpec(label="y", min_value=0, max_value=10, tick_interval=2),
    )

    file_name = draw_combo_points_table_graph(stimulus)
    assert os.path.exists(file_name)
