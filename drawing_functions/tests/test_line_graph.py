import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions.graphing import (
    draw_line_graphs,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.multi_graph import (
    Axis,
    LineGraphItem,
    LineGraphSeriesItem,
)


def test_draw_line_graphs_single_series():
    """Test case 1: Single line series with markers"""
    print("Creating test case 1: Single line series with markers")

    # Create test data for a simple linear relationship
    x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    # Create LineGraphSeriesItem
    series = LineGraphSeriesItem(
        x_values=x_values, y_values=y_values, label="Linear Growth", marker="o"
    )

    # Create Axis objects
    x_axis = Axis(label="Time (months)", range=(0, 10))
    y_axis = Axis(label="Population (thousands)", range=(0, 25))

    # Create LineGraphItem
    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Population Growth Over Time",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 1 saved as: {file_name}")


def test_draw_line_graphs_multiple_series():
    """Test case 2: Multiple line series with different markers and styles"""
    print(
        "Creating test case 2: Multiple line series with different markers and styles"
    )

    # Create test data for multiple series
    x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Series 1: Exponential growth
    y1_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    series1 = LineGraphSeriesItem(
        x_values=x_values, y_values=y1_values, label="Exponential Growth", marker="s"
    )

    # Series 2: Quadratic growth
    y2_values = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    series2 = LineGraphSeriesItem(
        x_values=x_values, y_values=y2_values, label="Quadratic Growth", marker="^"
    )

    # Series 3: Linear growth
    y3_values = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    series3 = LineGraphSeriesItem(
        x_values=x_values, y_values=y3_values, label="Linear Growth", marker="o"
    )

    # Create Axis objects
    x_axis = Axis(label="Time (years)", range=(0, 10))
    y_axis = Axis(label="Value", range=(0, 1100))

    # Create LineGraphItem
    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Comparison of Different Growth Patterns",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2, series3],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 2 saved as: {file_name}")


def test_draw_line_graphs_sine_wave():
    """Test case 3: Sine wave with mathematical notation"""
    print("Creating test case 3: Sine wave with mathematical notation")

    # Create test data for sine wave
    x_values = np.linspace(0, 4 * np.pi, 50).tolist()
    y_values = np.sin(x_values).tolist()

    # Create LineGraphSeriesItem
    series = LineGraphSeriesItem(
        x_values=x_values, y_values=y_values, label="sin(x)", marker="."
    )

    # Create Axis objects with mathematical notation
    x_axis = Axis(label="$x$ (radians)", range=(0, 4 * np.pi))
    y_axis = Axis(label="$y = \\sin(x)$", range=(-1.2, 1.2))

    # Create LineGraphItem with LaTeX title
    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Sine Wave Function: $y = \\sin(x)$",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 3 saved as: {file_name}")


def test_draw_line_graphs_simple_chart():
    """Test case 4: Simple line chart like the image - monthly data with fluctuations"""
    print("Creating test case 4: Simple line chart with monthly fluctuations")

    # Create test data similar to the image (monthly values with fluctuations)
    months = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
    x_values = list(range(len(months)))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    y_values = [35, 62, 31, 34, 45, 11, 24, 53, 78]  # Approximate values from image

    # Create LineGraphSeriesItem
    series = LineGraphSeriesItem(
        x_values=x_values,
        y_values=y_values,
        label="Monthly Data",  # Optional label
        marker="o",  # Circular markers like in the image
    )

    # Create Axis objects with tick labels
    x_axis = Axis(
        label="Time",
        range=(0, 8),
        tick_labels=months,  # Use the new tick_labels feature
    )
    y_axis = Axis(label="Value", range=(0, 90))

    # Create LineGraphItem
    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Monthly Data Fluctuations",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 4 saved as: {file_name}")


def test_draw_line_graphs_two_lines():
    """Test case 5: Two lines in one graph with different colors and markers"""
    print("Creating test case 5: Two lines with different colors and markers")

    # Create test data for two different series
    months = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
    x_values = list(range(len(months)))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # Series 1: First line (similar to original data)
    y1_values = [35, 62, 31, 34, 45, 11, 24, 53, 78]
    series1 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=y1_values,
        label="Product A Sales",
        marker="o",  # Circular markers
    )

    # Series 2: Second line (different pattern)
    y2_values = [25, 45, 55, 40, 35, 50, 65, 30, 45]
    series2 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=y2_values,
        label="Product B Sales",
        marker="s",  # Square markers
    )

    # Create Axis objects with tick labels
    x_axis = Axis(
        label="Time",
        range=(0, 8),
        tick_labels=months,  # Use the new tick_labels feature
    )
    y_axis = Axis(label="Sales (thousands)", range=(0, 90))

    # Create LineGraphItem
    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Product Sales Comparison",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 5 saved as: {file_name}")


def test_draw_line_graphs_temperature_trends():
    """Test case 6: Temperature trends with seasonal data"""
    print("Creating test case 6: Temperature trends with seasonal data")

    # Create seasonal temperature data
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    x_values = list(range(len(months)))

    # Two cities with different temperature patterns
    city_a_temps = [32, 35, 45, 55, 65, 75, 80, 78, 70, 58, 45, 35]
    city_b_temps = [28, 30, 40, 50, 60, 70, 75, 73, 65, 53, 40, 30]

    series1 = LineGraphSeriesItem(
        x_values=x_values, y_values=city_a_temps, label="New York", marker="o"
    )

    series2 = LineGraphSeriesItem(
        x_values=x_values, y_values=city_b_temps, label="Boston", marker="s"
    )

    x_axis = Axis(label="Month", range=(0, 11), tick_labels=months)
    y_axis = Axis(label="Temperature (°F)", range=(20, 85))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Monthly Temperature Comparison",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 6 saved as: {file_name}")


def test_draw_line_graphs_stock_performance():
    """Test case 7: Stock performance with multiple companies"""
    print("Creating test case 7: Stock performance with multiple companies")

    # Quarterly data
    months = ["Q1", "Q2", "Q3", "Q4"]
    x_values = list(range(len(months)))

    # Different stock performance patterns
    tech_stock = [100, 105, 110, 108]
    energy_stock = [100, 95, 90, 85]
    healthcare_stock = [100, 102, 105, 107]
    finance_stock = [100, 98, 96, 94]

    series1 = LineGraphSeriesItem(
        x_values=x_values, y_values=tech_stock, label="Tech Corp", marker="o"
    )

    series2 = LineGraphSeriesItem(
        x_values=x_values, y_values=energy_stock, label="Energy Inc", marker="^"
    )

    series3 = LineGraphSeriesItem(
        x_values=x_values, y_values=healthcare_stock, label="Health Plus", marker="s"
    )

    series4 = LineGraphSeriesItem(
        x_values=x_values, y_values=finance_stock, label="Finance Group", marker="D"
    )

    x_axis = Axis(label="Quarter", range=(0, 3), tick_labels=months)
    y_axis = Axis(label="Stock Price (Index = 100)", range=(90, 130))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Quarterly Stock Performance",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2, series3, series4],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 7 saved as: {file_name}")


def test_draw_line_graphs_mathematical_functions():
    """Test case 8: Mathematical functions with LaTeX notation"""
    print("Creating test case 8: Mathematical functions with LaTeX notation")

    # Create test data for mathematical functions
    x_values = np.linspace(-3, 3, 50).tolist()
    y1_values = [x**2 for x in x_values]  # Quadratic
    y2_values = [np.sin(x) for x in x_values]  # Sine
    y3_values = [np.cos(x) for x in x_values]  # Cosine
    y4_values = [np.exp(x / 2) for x in x_values]  # Exponential

    series1 = LineGraphSeriesItem(
        x_values=x_values, y_values=y1_values, label="$y = x^2$", marker="o"
    )

    series2 = LineGraphSeriesItem(
        x_values=x_values, y_values=y2_values, label="$y = \\sin(x)$", marker="s"
    )

    series3 = LineGraphSeriesItem(
        x_values=x_values, y_values=y3_values, label="$y = \\cos(x)$", marker="^"
    )

    series4 = LineGraphSeriesItem(
        x_values=x_values, y_values=y4_values, label="$y = e^{x/2}$", marker="D"
    )

    x_axis = Axis(label="$x$", range=(-3, 3))
    y_axis = Axis(label="$y$", range=(-2, 8))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Mathematical Functions Comparison",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2, series3, series4],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 8 saved as: {file_name}")


def test_draw_line_graphs_empty_data():
    """Test case 9: Edge case - Empty data series"""
    print("Creating test case 9: Empty data series")

    # Create empty data series
    series = LineGraphSeriesItem(
        x_values=[], y_values=[], label="Empty Data", marker="o"
    )

    x_axis = Axis(label="X", range=(0, 10))
    y_axis = Axis(label="Y", range=(0, 10))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Empty Data Series Test",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 9 saved as: {file_name}")


def test_draw_line_graphs_single_point():
    """Test case 10: Edge case - Single data point"""
    print("Creating test case 10: Single data point")

    # Create single point data
    series = LineGraphSeriesItem(
        x_values=[5], y_values=[10], label="Single Point", marker="o"
    )

    x_axis = Axis(label="X", range=(0, 10))
    y_axis = Axis(label="Y", range=(0, 20))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Single Point Test",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 10 saved as: {file_name}")


def test_draw_line_graphs_mismatched_data():
    """Test case 11: Edge case - Mismatched x and y data lengths"""
    print("Creating test case 11: Mismatched data lengths")

    # Create mismatched data (this should still work)
    series1 = LineGraphSeriesItem(
        x_values=[1, 2, 3],  # 3 x values
        y_values=[10, 20, 30],  # Shorter y_values
        label="Mismatched Data",
        marker="o",
    )

    series2 = LineGraphSeriesItem(
        x_values=[1, 2, 3, 4, 5],  # 5 x values
        y_values=[15, 25, 35, 45, 55],  # Longer y_values
        label="Mismatched Data 2",
        marker="s",
    )

    x_axis = Axis(label="X", range=(0, 6))
    y_axis = Axis(label="Y", range=(0, 60))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Mismatched Data Test",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 11 saved as: {file_name}")


def test_draw_line_graphs_extreme_values():
    """Test case 12: Edge case - Extreme values requiring log scale"""
    print("Creating test case 12: Extreme values")

    # Create data with extreme values
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_values = [
        1,
        10,
        100,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
    ]

    series = LineGraphSeriesItem(
        x_values=x_values, y_values=y_values, label="Extreme Values", marker="o"
    )

    x_axis = Axis(label="X", range=(1, 10))
    y_axis = Axis(label="Y (Log Scale)", range=(1, 1000000000))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Extreme Values Test",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 12 saved as: {file_name}")


def test_draw_line_graphs_no_labels():
    """Test case 13: Edge case - No series labels"""
    print("Creating test case 13: No series labels")

    x_values = [1, 2, 3, 4, 5]
    series1 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[10, 20, 30, 40, 50],
        label=None,  # No label
        marker="o",
    )

    series2 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[15, 25, 35, 45, 55],
        label=None,  # No label
        marker="s",
    )

    x_axis = Axis(label="X", range=(0, 6))
    y_axis = Axis(label="Y", range=(0, 60))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="No Labels Test",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series1, series2],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 13 saved as: {file_name}")


def test_draw_line_graphs_very_long_title():
    """Test case 14: Edge case - Very long title"""
    print("Creating test case 14: Very long title")

    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 20, 30, 40, 50]

    series = LineGraphSeriesItem(
        x_values=x_values, y_values=y_values, label="Data Series", marker="o"
    )

    x_axis = Axis(label="X", range=(0, 6))
    y_axis = Axis(label="Y", range=(0, 60))

    # Very long title to test text wrapping
    long_title = "This is a very long title that should test the text wrapping functionality of the line graph drawing function and see how it handles extremely long titles that might be used in real-world scenarios"

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title=long_title,
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[series],
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 14 saved as: {file_name}")


def test_draw_line_graphs_random_colors():
    """Test case 15: Random color assignment demonstration"""
    print("Creating test case 15: Random color assignment demonstration")

    # Create multiple series to show random color assignment
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create 5 different series to show color variety (max allowed is 5)
    series1 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[10, 15, 12, 18, 20, 16, 22, 25, 28, 30],
        label="Series A",
        marker="o",
    )

    series2 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[5, 8, 12, 15, 18, 20, 22, 25, 28, 32],
        label="Series B",
        marker="s",
    )

    series3 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
        label="Series C",
        marker="^",
    )

    series4 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[8, 12, 15, 18, 20, 22, 25, 28, 30, 33],
        label="Series D",
        marker="D",
    )

    series5 = LineGraphSeriesItem(
        x_values=x_values,
        y_values=[12, 15, 18, 20, 22, 25, 28, 30, 32, 35],
        label="Series E",
        marker="v",
    )

    x_axis = Axis(label="Time", range=(1, 10))
    y_axis = Axis(label="Value", range=(0, 40))

    line_graph = LineGraphItem(
        graph_type="line_graph",
        title="Random Color Assignment Demo - Run Multiple Times to See Different Colors",
        x_axis=x_axis,
        y_axis=y_axis,
        data_series=[
            series1,
            series2,
            series3,
            series4,
            series5,
        ],  # Reduced to 5 series
    )

    # Call the function - it now handles everything internally
    file_name = draw_line_graphs(line_graph)

    print(f"Test case 15 saved as: {file_name}")


def test_draw_line_graphs_multiple_runs():
    """Test case 16: Multiple runs to show random color variation"""
    print("Creating test case 16: Multiple runs to show random color variation")

    # Create the same data for multiple runs
    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 20, 30, 40, 50]

    series = LineGraphSeriesItem(
        x_values=x_values, y_values=y_values, label="Same Data", marker="o"
    )

    x_axis = Axis(label="X", range=(0, 6))
    y_axis = Axis(label="Y", range=(0, 60))

    # Run the same graph multiple times to show color variation
    for run in range(1, 4):  # 3 runs
        line_graph = LineGraphItem(
            graph_type="line_graph",
            title=f"Random Colors - Run {run}",
            x_axis=x_axis,
            y_axis=y_axis,
            data_series=[series],
        )

        # Call the function - it now handles everything internally
        file_name = draw_line_graphs(line_graph)

        print(f"Run {run} saved as: {file_name}")


def test_draw_line_graphs_random_colors_extended():
    """Test case 15b: Extended random color assignment with multiple graphs"""
    print("Creating test case 15b: Extended random color assignment")

    # Create multiple series to show random color assignment
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create 8 different series to show color variety
    all_series = [
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[10, 15, 12, 18, 20, 16, 22, 25, 28, 30],
            label="Series A",
            marker="o",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[5, 8, 12, 15, 18, 20, 22, 25, 28, 32],
            label="Series B",
            marker="s",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
            label="Series C",
            marker="^",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[8, 12, 15, 18, 20, 22, 25, 28, 30, 33],
            label="Series D",
            marker="D",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[12, 15, 18, 20, 22, 25, 28, 30, 32, 35],
            label="Series E",
            marker="v",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[6, 9, 12, 15, 18, 20, 22, 25, 28, 31],
            label="Series F",
            marker="<",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[14, 17, 19, 21, 24, 27, 29, 31, 34, 37],
            label="Series G",
            marker=">",
        ),
        LineGraphSeriesItem(
            x_values=x_values,
            y_values=[7, 11, 14, 17, 19, 21, 24, 27, 29, 32],
            label="Series H",
            marker="p",
        ),
    ]

    x_axis = Axis(label="Time", range=(1, 10))
    y_axis = Axis(label="Value", range=(0, 40))

    # Create two graphs with different sets of series
    for graph_num in range(1, 3):
        # Take 5 series for each graph
        start_idx = (graph_num - 1) * 5
        end_idx = start_idx + 5
        series_subset = all_series[start_idx:end_idx]

        line_graph = LineGraphItem(
            graph_type="line_graph",
            title=f"Random Colors Graph {graph_num} - Series {start_idx+1}-{end_idx}",
            x_axis=x_axis,
            y_axis=y_axis,
            data_series=series_subset,
        )

        # Call the function - it now handles everything internally
        file_name = draw_line_graphs(line_graph)

        print(f"Extended test graph {graph_num} saved as: {file_name}")


def run_all_line_graph_tests():
    """Run all test cases for draw_line_graphs function"""
    print("Starting line graph test cases...")
    print("=" * 50)

    try:
        # Test case 1: Single series
        test_draw_line_graphs_single_series()
        print("✓ Test case 1 completed successfully")
        print()

        # Test case 2: Multiple series
        test_draw_line_graphs_multiple_series()
        print("✓ Test case 2 completed successfully")
        print()

        # Test case 3: Sine wave with LaTeX
        test_draw_line_graphs_sine_wave()
        print("✓ Test case 3 completed successfully")
        print()

        # Test case 4: Simple chart
        test_draw_line_graphs_simple_chart()
        print("✓ Test case 4 completed successfully")
        print()

        # Test case 5: Two lines
        test_draw_line_graphs_two_lines()
        print("✓ Test case 5 completed successfully")
        print()

        # Test case 6: Temperature trends
        test_draw_line_graphs_temperature_trends()
        print("✓ Test case 6 completed successfully")
        print()

        # Test case 7: Stock performance
        test_draw_line_graphs_stock_performance()
        print("✓ Test case 7 completed successfully")
        print()

        # Test case 8: Mathematical functions
        test_draw_line_graphs_mathematical_functions()
        print("✓ Test case 8 completed successfully")
        print()

        # Test case 9: Empty data
        test_draw_line_graphs_empty_data()
        print("✓ Test case 9 completed successfully")
        print()

        # Test case 10: Single point
        test_draw_line_graphs_single_point()
        print("✓ Test case 10 completed successfully")
        print()

        # Test case 11: Mismatched data
        test_draw_line_graphs_mismatched_data()
        print("✓ Test case 11 completed successfully")
        print()

        # Test case 12: Extreme values
        test_draw_line_graphs_extreme_values()
        print("✓ Test case 12 completed successfully")
        print()

        # Test case 13: No labels
        test_draw_line_graphs_no_labels()
        print("✓ Test case 13 completed successfully")
        print()

        # Test case 14: Very long title
        test_draw_line_graphs_very_long_title()
        print("✓ Test case 14 completed successfully")
        print()

        # Test case 15: Random colors (basic)
        test_draw_line_graphs_random_colors()
        print("✓ Test case 15 completed successfully")
        print()

        # Test case 16: Multiple runs
        test_draw_line_graphs_multiple_runs()
        print("✓ Test case 16 completed successfully")
        print()

        # Test case 15b: Extended random colors
        test_draw_line_graphs_random_colors_extended()
        print("✓ Test case 15b completed successfully")
        print()

        print("=" * 50)
        print("All test cases completed successfully!")
        print("Note: Colors are now randomly assigned for each generation!")

    except Exception as e:
        print(f"Error running test cases: {e}")


if __name__ == "__main__":
    # Run all test cases when script is executed directly
    run_all_line_graph_tests()
