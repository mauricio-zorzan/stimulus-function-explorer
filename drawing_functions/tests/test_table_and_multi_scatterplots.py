import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.table_and_multi_scatterplots import (
    create_table_and_multi_scatterplots,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.table_and_multi_scatterplots import (
    DataPoint,
    ScatterplotData,
    TableAndMultiScatterplots,
    TableData,
)


@pytest.mark.drawing_functions
def test_study_time_vs_test_scores():
    """Test with study time vs test scores - educational context."""
    table_data = TableData(
        headers=["Hours Studied", "Test Score (percent)"],
        rows=[
            ["1", "65"],
            ["2", "72"],
            ["3", "78"],
            ["4", "85"],
            ["5", "92"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Hours Studied",
        y_label="Test Score (percent)",
        x_min=0,
        x_max=6,
        y_min=60,
        y_max=100,
        data_points=[
            DataPoint(x=1, y=65),
            DataPoint(x=2, y=72),
            DataPoint(x=3, y=78),
            DataPoint(x=4, y=85),
            DataPoint(x=5, y=92),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="vertical"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_temperature_vs_plant_growth():
    """Test with temperature vs plant growth - science context."""
    table_data = TableData(
        headers=["Temperature (Celsius)", "Plant Height (cm)"],
        rows=[
            ["15", "8"],
            ["20", "12"],
            ["25", "18"],
            ["30", "22"],
            ["35", "15"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Temperature (Celsius)",
        y_label="Plant Height (cm)",
        x_min=10,
        x_max=40,
        y_min=5,
        y_max=25,
        data_points=[
            DataPoint(x=15, y=8),
            DataPoint(x=20, y=12),
            DataPoint(x=25, y=18),
            DataPoint(x=30, y=22),
            DataPoint(x=35, y=15),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="horizontal"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_exercise_vs_heart_rate():
    """Test with exercise time vs heart rate - health context."""
    table_data = TableData(
        headers=["Exercise Time (minutes)", "Heart Rate (beats per minute)"],
        rows=[
            ["0", "70"],
            ["5", "85"],
            ["10", "100"],
            ["15", "115"],
            ["20", "125"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Exercise Time (minutes)",
        y_label="Heart Rate (beats per minute)",
        x_min=0,
        x_max=25,
        y_min=60,
        y_max=130,
        data_points=[
            DataPoint(x=0, y=70),
            DataPoint(x=5, y=85),
            DataPoint(x=10, y=100),
            DataPoint(x=15, y=115),
            DataPoint(x=20, y=125),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="vertical"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_multiple_choice_book_pages_vs_time():
    """Test multiple choice with book reading context."""
    table_data = TableData(
        headers=["Days", "Pages Read"],
        rows=[
            ["1", "25"],
            ["2", "50"],
            ["3", "75"],
            ["4", "100"],
        ],
    )

    # Option A: Correct
    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Days",
        y_label="Pages Read",
        x_min=0,
        x_max=5,
        y_min=0,
        y_max=120,
        data_points=[
            DataPoint(x=1, y=25),
            DataPoint(x=2, y=50),
            DataPoint(x=3, y=75),
            DataPoint(x=4, y=100),
        ],
        is_correct=True,
    )

    # Option B: Swapped coordinates
    swapped_plot = ScatterplotData(
        title="Option B",
        x_label="Days",
        y_label="Pages Read",
        x_min=0,
        x_max=120,
        y_min=0,
        y_max=5,
        data_points=[
            DataPoint(x=25, y=1),
            DataPoint(x=50, y=2),
            DataPoint(x=75, y=3),
            DataPoint(x=100, y=4),
        ],
        is_correct=False,
        error_type="swapped_coordinates",
    )

    # Option C: Missing data point
    missing_plot = ScatterplotData(
        title="Option C",
        x_label="Days",
        y_label="Pages Read",
        x_min=0,
        x_max=5,
        y_min=0,
        y_max=120,
        data_points=[
            DataPoint(x=1, y=25),
            DataPoint(x=2, y=50),
            DataPoint(x=4, y=100),
        ],
        is_correct=False,
        error_type="missing_points",
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data,
        scatterplots=[correct_plot, swapped_plot, missing_plot],
        layout="vertical",
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_car_speed_vs_fuel_efficiency():
    """Test with car speed vs fuel efficiency - practical context."""
    table_data = TableData(
        headers=["Speed (mph)", "Fuel Efficiency (mpg)"],
        rows=[
            ["30", "35"],
            ["45", "42"],
            ["60", "38"],
            ["75", "32"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Speed (mph)",
        y_label="Fuel Efficiency (mpg)",
        x_min=20,
        x_max=80,
        y_min=25,
        y_max=45,
        data_points=[
            DataPoint(x=30, y=35),
            DataPoint(x=45, y=42),
            DataPoint(x=60, y=38),
            DataPoint(x=75, y=32),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="horizontal"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_age_vs_reaction_time():
    """Test with age vs reaction time - psychology context."""
    table_data = TableData(
        headers=["Age (years)", "Reaction Time (milliseconds)"],
        rows=[
            ["20", "180"],
            ["30", "190"],
            ["40", "205"],
            ["50", "220"],
            ["60", "240"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Age (years)",
        y_label="Reaction Time (milliseconds)",
        x_min=15,
        x_max=65,
        y_min=170,
        y_max=250,
        data_points=[
            DataPoint(x=20, y=180),
            DataPoint(x=30, y=190),
            DataPoint(x=40, y=205),
            DataPoint(x=50, y=220),
            DataPoint(x=60, y=240),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="vertical"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_rainfall_vs_crop_yield():
    """Test with rainfall vs crop yield - agriculture context."""
    table_data = TableData(
        headers=["Rainfall (inches)", "Crop Yield (bushels per acre)"],
        rows=[
            ["10", "45"],
            ["15", "55"],
            ["20", "65"],
            ["25", "70"],
            ["30", "68"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Rainfall (inches)",
        y_label="Crop Yield (bushels per acre)",
        x_min=5,
        x_max=35,
        y_min=40,
        y_max=75,
        data_points=[
            DataPoint(x=10, y=45),
            DataPoint(x=15, y=55),
            DataPoint(x=20, y=65),
            DataPoint(x=25, y=70),
            DataPoint(x=30, y=68),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="horizontal"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_altitude_vs_temperature():
    """Test with altitude vs temperature - geography context."""
    table_data = TableData(
        headers=["Altitude (feet)", "Temperature (Fahrenheit)"],
        rows=[
            ["0", "70"],
            ["1000", "66"],
            ["2000", "62"],
            ["3000", "58"],
            ["4000", "54"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Altitude (feet)",
        y_label="Temperature (Fahrenheit)",
        x_min=0,
        x_max=4500,
        y_min=50,
        y_max=75,
        data_points=[
            DataPoint(x=0, y=70),
            DataPoint(x=1000, y=66),
            DataPoint(x=2000, y=62),
            DataPoint(x=3000, y=58),
            DataPoint(x=4000, y=54),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="vertical"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_practice_hours_vs_basketball_shots():
    """Test with practice hours vs basketball accuracy - sports context."""
    table_data = TableData(
        headers=["Practice Hours per Week", "Free Throw Percentage"],
        rows=[
            ["2", "60"],
            ["4", "68"],
            ["6", "75"],
            ["8", "82"],
            ["10", "88"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Practice Hours per Week",
        y_label="Free Throw Percentage",
        x_min=0,
        x_max=12,
        y_min=55,
        y_max=95,
        data_points=[
            DataPoint(x=2, y=60),
            DataPoint(x=4, y=68),
            DataPoint(x=6, y=75),
            DataPoint(x=8, y=82),
            DataPoint(x=10, y=88),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="horizontal"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_screen_time_vs_sleep_hours():
    """Test with screen time vs sleep hours - health context."""
    table_data = TableData(
        headers=["Screen Time (hours)", "Sleep Hours"],
        rows=[
            ["2", "8.5"],
            ["4", "7.8"],
            ["6", "7.2"],
            ["8", "6.5"],
            ["10", "5.8"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Screen Time (hours)",
        y_label="Sleep Hours",
        x_min=0,
        x_max=12,
        y_min=5,
        y_max=9,
        data_points=[
            DataPoint(x=2, y=8.5),
            DataPoint(x=4, y=7.8),
            DataPoint(x=6, y=7.2),
            DataPoint(x=8, y=6.5),
            DataPoint(x=10, y=5.8),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="vertical"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_comprehensive_multiple_choice_pizza_sales():
    """Test comprehensive multiple choice with pizza sales context."""
    table_data = TableData(
        headers=["Temperature (Fahrenheit)", "Pizza Sales (number sold)"],
        rows=[
            ["60", "45"],
            ["70", "52"],
            ["80", "68"],
            ["90", "75"],
            ["100", "82"],
        ],
    )

    # Option A: Correct
    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Temperature (Fahrenheit)",
        y_label="Pizza Sales (number sold)",
        x_min=55,
        x_max=105,
        y_min=40,
        y_max=90,
        data_points=[
            DataPoint(x=60, y=45),
            DataPoint(x=70, y=52),
            DataPoint(x=80, y=68),
            DataPoint(x=90, y=75),
            DataPoint(x=100, y=82),
        ],
        is_correct=True,
    )

    # Option B: Swapped coordinates
    swapped_plot = ScatterplotData(
        title="Option B",
        x_label="Temperature (Fahrenheit)",
        y_label="Pizza Sales (number sold)",
        x_min=40,
        x_max=90,
        y_min=55,
        y_max=105,
        data_points=[
            DataPoint(x=45, y=60),
            DataPoint(x=52, y=70),
            DataPoint(x=68, y=80),
            DataPoint(x=75, y=90),
            DataPoint(x=82, y=100),
        ],
        is_correct=False,
        error_type="swapped_coordinates",
    )

    # Option C: Missing one data point
    missing_plot = ScatterplotData(
        title="Option C",
        x_label="Temperature (Fahrenheit)",
        y_label="Pizza Sales (number sold)",
        x_min=55,
        x_max=105,
        y_min=40,
        y_max=90,
        data_points=[
            DataPoint(x=60, y=45),
            DataPoint(x=70, y=52),
            DataPoint(x=90, y=75),
            DataPoint(x=100, y=82),
        ],
        is_correct=False,
        error_type="missing_points",
    )

    # Option D: Shifted points
    shifted_plot = ScatterplotData(
        title="Option D",
        x_label="Temperature (Fahrenheit)",
        y_label="Pizza Sales (number sold)",
        x_min=55,
        x_max=105,
        y_min=40,
        y_max=90,
        data_points=[
            DataPoint(x=65, y=50),
            DataPoint(x=75, y=57),
            DataPoint(x=85, y=73),
            DataPoint(x=95, y=80),
            DataPoint(x=105, y=87),
        ],
        is_correct=False,
        error_type="shifted_points",
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data,
        scatterplots=[correct_plot, swapped_plot, missing_plot, shifted_plot],
        layout="vertical",
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_water_consumption_vs_plant_growth():
    """Test with water consumption vs plant growth - environmental science."""
    table_data = TableData(
        headers=["Water (milliliters per day)", "Plant Growth (centimeters)"],
        rows=[
            ["50", "2.1"],
            ["100", "4.3"],
            ["150", "6.2"],
            ["200", "7.8"],
            ["250", "8.5"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Water (milliliters per day)",
        y_label="Plant Growth (centimeters)",
        x_min=0,
        x_max=300,
        y_min=0,
        y_max=10,
        data_points=[
            DataPoint(x=50, y=2.1),
            DataPoint(x=100, y=4.3),
            DataPoint(x=150, y=6.2),
            DataPoint(x=200, y=7.8),
            DataPoint(x=250, y=8.5),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="horizontal"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_monthly_savings_vs_video_game_purchases():
    """Test with monthly savings vs video game purchases - financial literacy."""
    table_data = TableData(
        headers=["Monthly Savings (amount)", "Video Games Purchased"],
        rows=[
            ["20", "1"],
            ["40", "2"],
            ["60", "3"],
            ["80", "4"],
            ["100", "5"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Monthly Savings (amount)",
        y_label="Video Games Purchased",
        x_min=0,
        x_max=120,
        y_min=0,
        y_max=6,
        data_points=[
            DataPoint(x=20, y=1),
            DataPoint(x=40, y=2),
            DataPoint(x=60, y=3),
            DataPoint(x=80, y=4),
            DataPoint(x=100, y=5),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="vertical"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_study_group_size_vs_test_performance():
    """Test with study group size vs test performance - educational research."""
    table_data = TableData(
        headers=["Study Group Size", "Average Test Score (percent)"],
        rows=[
            ["1", "78"],
            ["2", "82"],
            ["3", "85"],
            ["4", "87"],
            ["5", "84"],
            ["6", "80"],
        ],
    )

    correct_plot = ScatterplotData(
        title="Option A",
        x_label="Study Group Size",
        y_label="Average Test Score (percent)",
        x_min=0,
        x_max=7,
        y_min=75,
        y_max=90,
        data_points=[
            DataPoint(x=1, y=78),
            DataPoint(x=2, y=82),
            DataPoint(x=3, y=85),
            DataPoint(x=4, y=87),
            DataPoint(x=5, y=84),
            DataPoint(x=6, y=80),
        ],
        is_correct=True,
    )

    stimulus = TableAndMultiScatterplots(
        table=table_data, scatterplots=[correct_plot], layout="horizontal"
    )

    file_path = create_table_and_multi_scatterplots(stimulus)
    assert os.path.exists(file_path)
