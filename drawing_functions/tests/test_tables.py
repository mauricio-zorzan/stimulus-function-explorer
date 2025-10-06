import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.table import (
    draw_data_table,
    draw_data_table_group,
    draw_table_two_way,
    generate_table,
)


@pytest.mark.drawing_functions
def test_draw_data_table():
    stimulus = {
        "title": "Liquid Volumes of Drink Containers",
        "headers": ["Container Type", "Volume (ml)"],
        "data": [
            ["Bottle", "500"],
            ["Can", "330"],
            ["Cup", "250"],
            ["Jug", "1000"],
        ],
    }
    file_path = draw_data_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_data_table_long_title_small_headers():
    stimulus = {
        "headers": ["x", "y"],
        "data": [
            ["3", "12"],
            ["6", "24"],
            ["9", "?"],
            ["12", "48"],
        ],
        "title": "Proportional Relationship Table",
        "metadata": "Missing value for y.",
    }
    file_path = draw_data_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_table_two_way():
    stimulus = {
        "table_title": "Pizza Shop Preferences",
        "rows_title": [{"label_1": "Boys"}, {"label_2": "Girls"}],
        "columns_title": [{"label_1": "Store A"}, {"label_2": "Store B"}],
        "data": [{"1": 60, "2": 40}, {"1": 75, "2": 25}],
    }
    file_path = draw_table_two_way(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_table_two_way_2():
    stimulus = {
        "table_title": "Student Preferences for Various Subjects",
        "rows_title": [
            {"label_1": "Boys"},
            {"label_2": "Girls"},
            {"label_3": "Non-binary"},
            {"label_4": "Prefer not to say"},
        ],
        "columns_title": [
            {"label_1": "Math"},
            {"label_2": "Science"},
            {"label_3": "Literature"},
            {"label_4": "Art"},
            {"label_5": "Physical Education"},
        ],
        "data": [
            {"1": 75, "2": 80, "3": 60, "4": 55, "5": 85},
            {"1": 70, "2": 75, "3": 85, "4": 80, "5": 65},
            {"1": 72, "2": 78, "3": 75, "4": 82, "5": 70},
            {"1": 68, "2": 73, "3": 70, "4": 75, "5": 72},
        ],
    }
    file_path = draw_table_two_way(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table():
    stimulus = {
        "columns": [{"label": "Number of Tow Trucks"}, {"label": "Number of Cars"}],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_with_aliases():
    stimulus = {
        "columns": [{"label": "Pattern A Term"}, {"label": "Pattern B Term"}],
        "rows": [
            {"field_1": "7", "field_2": "1"},
            {"field_1": "12", "field_2": "3"},
            {"field_1": "17", "field_2": "9"},
            {"field_1": "22", "field_2": "27"},
            {"field_1": "27", "2": "81"},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_small_label():
    stimulus = {
        "columns": [{"label": "X"}, {"label": "Y"}],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_2():
    stimulus = {
        "columns": [{"label": "Two T"}, {"label": "Number of Cars"}],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_3():
    stimulus = {
        "columns": [
            {"label": "Number of Tow Trucks that have been used with cars this week"},
            {"label": "Number of Cars"},
        ],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_4():
    stimulus = {
        "columns": [{"label": "No. of Tow Trucks"}, {"label": "Number of Cars"}],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_5():
    stimulus = {
        "columns": [{"label": "No. of Tow Trucks"}, {"label": "Number of Cars"}],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_6():
    stimulus = {
        "columns": [{"label": "No. of Tow Trucks"}, {"label": "Number of Cars"}],
        "rows": [
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
            {"1": 75, "2": 25},
            {"1": 60, "2": 40},
            {"1": 75, "2": 25},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_7():
    stimulus = {
        "title": "Minutes to Seconds Equivalent",
        "columns": [{"label": "Minutes"}, {"label": "Seconds"}],
        "rows": [
            {"1": "1", "2": "60"},
            {"1": "2", "2": "120"},
            {"1": "3", "2": "180"},
            {"1": "4", "2": "240"},
            {"1": "5", "2": "300"},
            {"1": "6", "2": "360"},
            {"1": "7", "2": "420"},
            {"1": "8", "2": "?"},
        ],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_generate_table_8():
    stimulus = {
        "title": "Minutes to Seconds Equivalent",
        "columns": [{"label": "Choice"}, {"label": "Count"}],
        "rows": [{"1": "Basketball", "2": "15"}, {"1": "Soccer", "2": "5"}],
    }
    file_path = generate_table(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_data_table_group_simple():
    """Test drawing a simple group of 2 tables."""
    stimulus = {
        "group_title": "Comparative Data Analysis",
        "layout": "horizontal",
        "tables": [
            {
                "title": "Q1 Sales",
                "headers": ["Product", "Units Sold"],
                "data": [
                    ["Widget A", "150"],
                    ["Widget B", "200"],
                    ["Widget C", "120"],
                ],
            },
            {
                "title": "Q2 Sales", 
                "headers": ["Product", "Units Sold"],
                "data": [
                    ["Widget A", "180"],
                    ["Widget B", "220"],
                    ["Widget C", "140"],
                ],
            },
        ],
    }
    file_path = draw_data_table_group(stimulus)
    assert os.path.exists(file_path)


@pytest.mark.drawing_functions
def test_draw_data_table_group_auto_layout():
    """Test drawing a group of 4 tables with auto layout."""
    stimulus = {
        "layout": "auto",
        "tables": [
            {
                "title": "Table A",
                "headers": ["In", "Out"],
                "data": [["1", "3"], ["2", "6"], ["3", "9"]],
            },
            {
                "title": "Table B",
                "headers": ["In", "Out"],
                "data": [["1", "4"], ["2", "8"], ["3", "12"]],
            },
            {
                "title": "Table C", 
                "headers": ["In", "Out"],
                "data": [["1", "5"], ["2", "10"], ["3", "15"]],
            },
            {
                "title": "Table D",
                "headers": ["In", "Out"], 
                "data": [["1", "6"], ["2", "12"], ["3", "18"]],
            },
        ],
    }
    file_path = draw_data_table_group(stimulus)
    assert os.path.exists(file_path)
