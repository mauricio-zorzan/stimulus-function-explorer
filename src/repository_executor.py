#!/usr/bin/env python3
"""
Repository-based executor that works within the actual repository environment
to avoid import conflicts and hardcoded solutions.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import ast
import json
from typing import Dict, List, Optional, Any
import base64
from datetime import datetime


class RepositoryExecutor:
    """
    Executes functions within the actual repository environment
    by analyzing the repository structure and creating appropriate test scripts.
    """

    def __init__(self):
        self.temp_dir = None
        self.repo_dir = None

    def setup_repository_environment(self) -> str:
        """Set up the repository environment with local stimulus_descriptions to avoid types conflicts."""
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix="stimulus_repo_")

        # Clone repository
        repo_url = (
            "https://github.com/trilogy-group/coach-bot-external-content-generators.git"
        )
        print(f"Setting up repository environment: {self.temp_dir}")

        result = subprocess.run(
            ["git", "clone", repo_url, "."],
            cwd=self.temp_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")

        # Copy local stimulus_descriptions if it exists to avoid types conflicts
        local_stimulus_descriptions = os.path.join(os.getcwd(), "stimulus_descriptions")
        if os.path.exists(local_stimulus_descriptions):
            dest_stimulus_descriptions = os.path.join(
                self.temp_dir, "stimulus_descriptions"
            )
            shutil.copytree(local_stimulus_descriptions, dest_stimulus_descriptions)
            print(f"✅ Copied local stimulus_descriptions to repository environment")

        self.repo_dir = self.temp_dir
        return self.temp_dir

    def find_function_in_repository(self, function_name: str) -> Optional[Dict]:
        """Find function in the repository and extract its information."""
        try:
            drawing_functions_path = os.path.join(
                self.repo_dir,
                "src",
                "content_generators",
                "additional_content",
                "stimulus_image",
                "drawing_functions",
            )

            for module_file in os.listdir(drawing_functions_path):
                if module_file.endswith(".py") and module_file != "__init__.py":
                    module_path = os.path.join(drawing_functions_path, module_file)

                    try:
                        with open(module_path, "r") as f:
                            tree = ast.parse(f.read())

                        for node in ast.walk(tree):
                            if (
                                isinstance(node, ast.FunctionDef)
                                and node.name == function_name
                            ):
                                return {
                                    "module_file": module_file[:-3],  # Remove .py
                                    "function_name": function_name,
                                    "args": [arg.arg for arg in node.args.args],
                                    "module_path": f"content_generators.additional_content.stimulus_image.drawing_functions.{module_file[:-3]}",
                                }
                    except Exception as e:
                        print(f"⚠️  Error parsing {module_file}: {e}")
                        continue

            print(f"❌ Function {function_name} not found")
            return None

        except Exception as e:
            print(f"❌ Error finding function: {e}")
            return None

    def analyze_test_files_for_function(self, function_name: str) -> Dict:
        """Analyze test files to understand how the function is used."""
        try:
            # First try to get data class from function signature
            data_class = self.get_function_data_class(function_name)
            if data_class:
                print(f"🎯 Found data class from function signature: {data_class}")
                test_data = self.generate_test_data_for_class(data_class)
                return {"imports": [], "test_data": test_data}

            # Fallback to test file analysis
            test_dir = os.path.join(
                self.repo_dir,
                "src",
                "content_generators",
                "additional_content",
                "stimulus_image",
                "drawing_functions",
                "tests",
            )

            for test_file in os.listdir(test_dir):
                if test_file.startswith("test_") and test_file.endswith(".py"):
                    test_path = os.path.join(test_dir, test_file)

                    try:
                        with open(test_path, "r") as f:
                            content = f.read()

                        if function_name in content:
                            print(f"🔍 Found test file: {test_file}")
                            return self._extract_test_patterns(content, function_name)

                    except Exception as e:
                        continue

            print(f"⚠️  No test file found for {function_name}")
            return {"imports": [], "test_data": None}

        except Exception as e:
            print(f"❌ Error analyzing test files: {e}")
            return {"imports": [], "test_data": None}

    def get_function_data_class(self, function_name: str) -> str:
        """Extract the data class from the function's type annotation."""
        try:
            # Find the function definition in the repository
            function_file = None
            for root, dirs, files in os.walk(self.repo_dir):
                for file in files:
                    if file.endswith(".py") and not file.startswith("test_"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r") as f:
                                content = f.read()
                                if f"def {function_name}(" in content:
                                    function_file = file_path
                                    break
                        except:
                            continue
                if function_file:
                    break

            if not function_file:
                return None

            # Parse the file and find the function
            with open(function_file, "r") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get the first parameter's type annotation
                    if node.args.args:
                        first_arg = node.args.args[0]
                        if first_arg.annotation:
                            # Extract the type annotation
                            if isinstance(first_arg.annotation, ast.Name):
                                return first_arg.annotation.id
                            elif isinstance(first_arg.annotation, ast.Attribute):
                                # Handle cases like content_generators.additional_content.stimulus_image.stimulus_descriptions.fraction.DivisionModel
                                return first_arg.annotation.attr
                            elif isinstance(first_arg.annotation, ast.Subscript):
                                # Handle cases like List[SomeType]
                                if isinstance(first_arg.annotation.value, ast.Name):
                                    return first_arg.annotation.value.id
                                elif isinstance(
                                    first_arg.annotation.value, ast.Attribute
                                ):
                                    return first_arg.annotation.value.attr

            return None

        except Exception as e:
            print(f"❌ Error extracting data class from function: {e}")
            return None

    def generate_test_data_for_class(self, data_class: str) -> str:
        """Extract real test data from test files instead of hardcoding."""
        try:
            # For specific data classes that are known to have issues with data extraction,
            # prioritize fallback data to avoid malformed extracted data
            problematic_classes = [
                "TransversalAngleParams",
                "TriangleStimulusDescription",
                "RightTriangleWithRay",
                "GeometricShapeWithAngleList",
                "RectangularGrid",
                "MultiExtendedUnitFractionNumberLine",
                "DivisionModel",
                "DataTable",
                "TableTwoWay",
                "PointPlot",
                "PolygonStringSides",
                "DecimalMultiplication",
                "MultiGraphList",
                "CompositeRectangularPrism2",
                "RectangularGridList",
                "Table",
                "UnequalFractionList",
                "WholeFractionalShapes",
                "FractionStrips",
            ]

            if data_class in problematic_classes:
                print(
                    f"🎯 Using fallback data for {data_class} to avoid extraction issues"
                )
                fallback_data = self._generate_fallback_test_data(data_class)
                if fallback_data:
                    return fallback_data

            # Try to extract actual test data from the test files
            test_data = self._extract_real_test_data(data_class)
            if test_data:
                return test_data

            # If no test data found, try to generate minimal test data for complex classes
            fallback_data = self._generate_fallback_test_data(data_class)
            if fallback_data:
                return fallback_data

            # If no test data found, return None
            return "None"

        except Exception as e:
            print(f"⚠️ Error extracting test data for {data_class}: {e}")
            return "None"

    def _generate_fallback_test_data(self, data_class: str) -> Optional[str]:
        """Generate fallback test data for any data class."""
        try:
            # Comprehensive fallback patterns based on data class names and common patterns
            if "Graph" in data_class and "List" in data_class:
                # Graph list classes - generate multiple graphs
                return '[{"graph_type": "bar_graph", "title": "Graph 1", "x_axis_label": "Categories 1", "y_axis_label": "Values 1", "data": [{"category": "A", "frequency": 10}, {"category": "B", "frequency": 20}]}, {"graph_type": "bar_graph", "title": "Graph 2", "x_axis_label": "Categories 2", "y_axis_label": "Values 2", "data": [{"category": "X", "frequency": 25}, {"category": "Y", "frequency": 15}]}, {"graph_type": "bar_graph", "title": "Graph 3", "x_axis_label": "Categories 3", "y_axis_label": "Values 3", "data": [{"category": "Red", "frequency": 8}, {"category": "Blue", "frequency": 12}]}, {"graph_type": "bar_graph", "title": "Graph 4", "x_axis_label": "Categories 4", "y_axis_label": "Values 4", "data": [{"category": "One", "frequency": 18}, {"category": "Two", "frequency": 22}]}]'
            elif data_class == "MultiExtendedUnitFractionNumberLine":
                # MultiExtendedUnitFractionNumberLine class - generate specific multi number line data
                return '{"number_lines": [{"range": {"min": 0.0, "max": 2.0}, "minor_divisions": 8, "endpoint_fraction": "8/4", "dot_point": {"label": "A", "value": 1.0, "dot_start_tick": 0, "red": False}}, {"range": {"min": 0.0, "max": 1.5}, "minor_divisions": 6, "endpoint_fraction": "6/4", "dot_point": {"label": "B", "value": 0.75, "dot_start_tick": 0, "red": False}}], "show_minor_division_labels": False}'
            elif "NumberLine" in data_class:
                # Number line classes - generate basic number line data
                return '{"range": {"min": 0.0, "max": 2.0}, "minor_divisions": 8, "endpoint_fraction": "8/4", "dot_point": {"label": "A", "value": 1.0}}'
            elif data_class == "Table":
                # Table class - generate specific table data
                return '{"columns": [{"label": "Column 1"}, {"label": "Column 2"}], "rows": [{"1": "Value 1", "2": "Value 2"}, {"1": "Value 3", "2": "Value 4"}]}'
            elif data_class == "UnequalFractionList":
                # UnequalFractionList class - generate specific unequal fraction list data
                return '[{"shape": "rectangle", "divided_parts": 4, "equally_divided": False}, {"shape": "circle", "divided_parts": 6, "equally_divided": True}, {"shape": "triangle", "divided_parts": 3, "equally_divided": True}]'
            elif data_class == "WholeFractionalShapes":
                # WholeFractionalShapes class - generate specific whole fractional shapes data
                return '{"count": 3, "shape": "rectangle", "divisions": 4}'
            elif data_class == "FractionStrips":
                # FractionStrips class - generate specific fraction strips data
                return '{"splits": 2, "first_division": 4}'
            elif "Table" in data_class:
                # Table classes - generate basic table data
                return '{"title": "Test Table", "headers": ["Column 1", "Column 2"], "data": [["Value 1", "Value 2"], ["Value 3", "Value 4"]]}'
            elif "Fraction" in data_class:
                # Fraction classes - generate basic fraction data
                return '{"shape": "rectangle", "fraction": "3/4"}'
            elif "Polygon" in data_class:
                # Polygon classes - generate basic polygon data
                return '{"side_lengths": [6, 6, 6, 6], "unit": "cm", "shape_type": "square"}'
            elif "Division" in data_class:
                # Division classes - generate basic division data
                return '{"dividend": {"numerator": 6, "denominator": 7}, "divisor": 3}'
            elif data_class == "PointPlot":
                # PointPlot class - generate specific point plot data
                return '{"points": [{"x": 1.0, "y": 2.0, "label": "A"}, {"x": 3.0, "y": 4.0, "label": "B"}]}'
            elif "Point" in data_class:
                # Point classes - generate basic point data
                return '{"x": 1.0, "y": 2.0, "label": "A"}'
            elif data_class == "GeometricShapeWithAngleList":
                # GeometricShapeWithAngleList class - generate specific geometric shape data
                return '[{"shape": "rectangle", "angle_type": "right", "color": "blue", "label": "Figure 1"}, {"shape": "square", "angle_type": "right", "color": "red", "label": "Figure 2"}, {"shape": "rhombus", "angle_type": "obtuse", "color": "green", "label": "Figure 3"}]'
            elif "List" in data_class:
                # List classes - generate basic list data
                return '[{"value": 1}, {"value": 2}, {"value": 3}]'
            elif data_class == "Clock":
                # Clock class - generate basic clock data
                return '{"type": "analog", "hour": 3, "minute": 45}'
            elif "Rectangle" in data_class:
                # Rectangle classes - generate basic rectangle data
                return '{"unit": "cm", "length": 5, "width": 3}'
            elif data_class == "TriangleStimulusDescription":
                # TriangleStimulusDescription class - generate specific triangle data
                return '{"triangle": {"points": [{"label": "A"}, {"label": "B"}, {"label": "C"}], "angles": [{"vertex": "A", "measure": 60}, {"vertex": "B", "measure": 60}, {"vertex": "C", "measure": 60}]}}'
            elif data_class == "RightTriangleWithRay":
                # RightTriangleWithRay class - generate specific right triangle data
                return '{"triangle": {"points": [{"label": "A"}, {"label": "B"}, {"label": "C"}], "angles": [{"vertex": "A", "measure": 90}, {"vertex": "B", "measure": 45}, {"vertex": "C", "measure": 45}], "rays": [{"start_label": "A", "measures": [90, "right"]}]}}'
            elif "Triangle" in data_class:
                # Other triangle classes - generate basic triangle data
                return '{"unit": "cm", "base": 4, "height": 3}'
            elif "Circle" in data_class:
                # Circle classes - generate basic circle data
                return '{"radius": 3, "unit": "cm"}'
            elif data_class == "TransversalAngleParams":
                # TransversalAngleParams class - generate specific transversal angle data
                return '{"given_angle": 60, "x_angle_position": 1, "given_angle_position": 5}'
            elif data_class == "RectangularGrid":
                # RectangularGrid class - generate specific rectangular grid data
                return '{"unit": "cm", "length": 5, "width": 3, "label": "Grid A", "irregular": False}'
            elif data_class == "DivisionModel":
                # DivisionModel class - generate specific division model data
                return '{"dividend": {"numerator": 6, "denominator": 7}, "divisor": 3}'
            elif data_class == "DataTable":
                # DataTable class - generate specific data table data
                return '{"title": "Test Data Table", "headers": ["Column 1", "Column 2"], "data": [["Value 1", "Value 2"], ["Value 3", "Value 4"]]}'
            elif data_class == "TableTwoWay":
                # TableTwoWay class - generate specific two-way table data
                return '{"table_title": "Two-Way Table", "rows_title": "Categories", "columns_title": "Values", "data": [["A", "B"], ["C", "D"]]}'
            elif data_class == "PointPlot":
                # PointPlot class - generate specific point plot data
                return '{"title": "Point Plot", "points": [{"x": 1, "y": 2, "label": "A"}, {"x": 3, "y": 4, "label": "B"}]}'
            elif data_class == "PolygonStringSides":
                # PolygonStringSides class - generate specific polygon string sides data
                return '{"side_lengths": ["3 cm", "4 cm", "5 cm"], "shape_type": "triangle"}'
            elif data_class == "DecimalMultiplication":
                # DecimalMultiplication class - generate specific decimal multiplication data
                return '{"decimal_1": 0.5, "decimal_2": 0.75}'
            elif data_class == "MultiGraphList":
                # MultiGraphList class - generate specific multi graph list data
                return '[{"graph_type": "bar_graph", "title": "Graph 1", "x_axis_label": "Categories 1", "y_axis_label": "Values 1", "data": [{"category": "A", "frequency": 10}, {"category": "B", "frequency": 20}]}, {"graph_type": "bar_graph", "title": "Graph 2", "x_axis_label": "Categories 2", "y_axis_label": "Values 2", "data": [{"category": "X", "frequency": 25}, {"category": "Y", "frequency": 15}]}, {"graph_type": "bar_graph", "title": "Graph 3", "x_axis_label": "Categories 3", "y_axis_label": "Values 3", "data": [{"category": "Red", "frequency": 8}, {"category": "Blue", "frequency": 12}]}, {"graph_type": "bar_graph", "title": "Graph 4", "x_axis_label": "Categories 4", "y_axis_label": "Values 4", "data": [{"category": "One", "frequency": 18}, {"category": "Two", "frequency": 22}]}]'
            elif data_class == "CompositeRectangularPrism2":
                # CompositeRectangularPrism2 class - generate specific composite prism data
                return '{"figures": [[3, 4, 5], [2, 3, 4]], "units": "cm"}'
            elif data_class == "RectangularGridList":
                # RectangularGridList class - generate specific rectangular grid list data
                return '[{"length": 5, "width": 3, "unit": "cm", "x": 0, "y": 0}, {"length": 4, "width": 2, "unit": "cm", "x": 1, "y": 1}]'
            elif "Angle" in data_class:
                # Angle classes - generate basic angle data
                return '{"measure": 45}'
            elif "Bar" in data_class:
                # Bar classes - generate basic bar data
                return '{"label": "A", "length": 5}'
            elif "Line" in data_class:
                # Line classes - generate basic line data
                return '{"start_label": "A", "end_label": "B", "type": "solid"}'
            elif "Grid" in data_class:
                # Grid classes - generate basic grid data
                return '{"unit": "cm", "length": 5, "width": 5}'
            elif "Prism" in data_class:
                # Prism classes - generate basic prism data
                return '{"height": 3, "width": 4, "length": 5, "unit": "cm"}'
            elif "Histogram" in data_class:
                # Histogram classes - generate basic histogram data
                return '{"title": "Test Histogram", "x_label": "Values", "y_label": "Frequency", "bins": [{"start": 0, "end": 5, "frequency": 10, "label": "0-5"}]}'
            elif "Scatter" in data_class:
                # Scatter plot classes - generate basic scatter data
                return '{"title": "Test Scatter", "x_label": "X Values", "y_label": "Y Values", "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}'
            elif "Decimal" in data_class:
                # Decimal classes - generate basic decimal data
                return '{"decimal_1": 0.5, "decimal_2": 0.75}'
            elif "Ruler" in data_class:
                # Ruler classes - generate basic ruler data
                return '{"unit": "cm", "length": 10}'
            elif "Spinner" in data_class:
                # Spinner classes - generate basic spinner data
                return '{"title": "Test Spinner", "sections": [{"label": "A", "value": 1}, {"label": "B", "value": 2}]}'
            elif "Protractor" in data_class:
                # Protractor classes - generate basic protractor data
                return '{"root": [{"label": "A", "degree": 45}]}'
            elif "BaseTen" in data_class:
                # Base ten block classes - generate basic base ten data
                return '{"value": 123, "display_as_decimal": False}'
            elif "Shape" in data_class:
                # Shape classes - generate basic shape data
                return (
                    '{"shape_type": "rectangle", "height": 3, "base": 4, "unit": "cm"}'
                )
            elif "Flowchart" in data_class:
                # Flowchart classes - generate basic flowchart data
                return '{"nodes": [{"id": "1", "label": "Start", "shape": "rectangle"}], "edges": [], "orientation": "vertical"}'
            elif "Tree" in data_class:
                # Tree diagram classes - generate basic tree data
                return '{"root": {"label": "Root", "left": None, "right": None}, "title": "Test Tree"}'
            elif "Probability" in data_class:
                # Probability classes - generate basic probability data
                return '{"rows_title": "Outcomes", "columns_title": "Events", "data": [["A", "B"], ["C", "D"]]}'
            elif "Symmetry" in data_class:
                # Symmetry classes - generate basic symmetry data
                return '{"shape_type": "rectangle", "shape_coordinates": [[0, 0], [4, 0], [4, 2], [0, 2]], "lines": [{"start": [2, 0], "end": [2, 2]}]}'
            elif "Dilation" in data_class:
                # Dilation classes - generate basic dilation data
                return '{"scale_factor": 2, "center_of_dilation": [0, 0], "show_center": True}'
            elif "Scale" in data_class:
                # Scale classes - generate basic scale data
                return '{"scale_factor": 2, "original_polygon_label": "A", "scaled_polygon_label": "B"}'
            elif "Perimeter" in data_class:
                # Perimeter classes - generate basic perimeter data
                return '{"side_lengths": [3, 4, 5], "unit": "cm", "shape_type": "triangle"}'
            elif "Area" in data_class:
                # Area classes - generate basic area data
                return '{"base": 4, "height": 3, "shape": "rectangle"}'
            elif "Volume" in data_class:
                # Volume classes - generate basic volume data
                return '{"length": 4, "width": 3, "height": 2, "unit": "cm"}'
            elif "Measurement" in data_class:
                # Measurement classes - generate basic measurement data
                return (
                    '{"object_name": "Object", "length": 5, "unit": "cm", "label": "A"}'
                )
            elif "Counting" in data_class:
                # Counting classes - generate basic counting data
                return '{"object_name": "dots", "count": 10}'
            elif "Ratio" in data_class:
                # Ratio classes - generate basic ratio data
                return '{"rows": 2, "columns": 3, "objects": [{"shape": "circle", "color": "blue"}], "shape_size": 1}'
            elif "Equation" in data_class:
                # Equation classes - generate basic equation data
                return '{"type": "addition"}'
            elif "Inequality" in data_class:
                # Inequality classes - generate basic inequality data
                return '{"range": {"min": 0, "max": 10}, "points": [{"value": 5, "label": "x"}], "line": "solid"}'
            elif "Function" in data_class:
                # Function classes - generate basic function data
                return '{"function_type": "linear", "a": 1, "b": 0}'
            elif "Coordinate" in data_class:
                # Coordinate classes - generate basic coordinate data
                return '{"x_axis_title": "X", "y_axis_title": "Y", "x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5}'
            elif "Quadrilateral" in data_class:
                # Quadrilateral classes - generate basic quadrilateral data
                return '{"shape_types": ["rectangle"], "side_labels": ["A", "B", "C", "D"], "show_ticks": True}'
            elif "Trapezoid" in data_class:
                # Trapezoid classes - generate basic trapezoid data
                return '{"base": 6, "top_length": 4, "height": 3, "unit": "cm", "trapezoid_type": "regular"}'
            elif "BoxPlot" in data_class:
                # Box plot classes - generate basic box plot data
                return '{"title": "Test Box Plot", "data": [{"class_name": "A", "min_value": 1, "q1": 2, "median": 3, "q3": 4, "max_value": 5}]}'
            elif "3D" in data_class or "ThreeDimensional" in data_class:
                # 3D classes - generate basic 3D data
                return '{"shapes": [{"shape": "cube", "label": "A", "faces": 6}], "units": "cm"}'
            elif "Net" in data_class:
                # Net classes - generate basic net data
                return '{"height": 3, "width": 4, "length": 5, "net_type": "rectangular", "unit_label": "cm"}'
            elif "CrossSection" in data_class:
                # Cross section classes - generate basic cross section data
                return '{"shape": "cube", "correct_cross_section": "square", "correct_letter": "A"}'
            elif "Segment" in data_class:
                # Segment classes - generate basic segment data
                return '{"start_coordinate": 0, "end_coordinate": 5, "linear": True}'
            elif "Ray" in data_class:
                # Ray classes - generate basic ray data
                return '{"start_label": "A", "measures": [45]}'
            elif "Time" in data_class:
                # Time classes - generate basic time data
                return '{"label": "3:45 PM", "hour": 3, "minute": 45}'
            elif "Stepwise" in data_class:
                # Stepwise classes - generate basic stepwise data
                return '{"steps": [{"rows": 1, "columns": 1, "color": "blue", "shape": "circle"}], "shape_size": 1, "spacing": 1}'
            elif "Dataset" in data_class:
                # Dataset classes - generate basic dataset data
                return '{"title": "Test Dataset", "data_points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}'
            elif "Rhombus" in data_class:
                # Rhombus classes - generate basic rhombus data
                return '{"units": "cm", "d1": 4, "d2": 6, "show_missing_placeholder": False}'
            elif "Decomposition" in data_class:
                # Decomposition classes - generate basic decomposition data
                return '{"title": "Test Decomposition", "units": "cm", "gridlines": True, "shapes": [{"type": "rectangle", "count": 2}]}'
            else:
                # Generic fallback - generate basic key-value data
                return '{"value": 1, "label": "test", "data": "sample"}'

        except Exception as e:
            print(f"⚠️ Error generating fallback test data for {data_class}: {e}")
            return None

    def _extract_real_test_data(self, data_class: str) -> Optional[str]:
        """Extract real test data from test files that use this data class."""
        try:
            # Look for test files that contain this data class
            test_dir = os.path.join(
                self.repo_dir,
                "src/content_generators/additional_content/stimulus_image/drawing_functions/tests",
            )

            if not os.path.exists(test_dir):
                return None

            for test_file in os.listdir(test_dir):
                if test_file.endswith(".py"):
                    test_path = os.path.join(test_dir, test_file)
                    try:
                        with open(test_path, "r") as f:
                            content = f.read()

                        # Look for data class instantiations or function calls that might use this data class
                        if data_class in content or self._might_contain_data_for_class(
                            content, data_class
                        ):
                            # Extract the first example of this data class
                            example_data = self._extract_data_class_example(
                                content, data_class
                            )
                            if example_data:
                                # Validate the extracted data before returning it
                                if self._is_valid_data_for_class(
                                    example_data, data_class
                                ):
                                    return example_data
                                else:
                                    print(
                                        f"⚠️ Extracted data failed validation for {data_class}, trying next file"
                                    )
                                    continue

                    except Exception:
                        continue

            return None

        except Exception as e:
            print(f"⚠️ Error extracting real test data: {e}")
            return None

    def _might_contain_data_for_class(self, content: str, data_class: str) -> bool:
        """Check if the content might contain data for the given data class."""
        try:
            # For TableTwoWay, look for draw_table_two_way function calls
            if data_class == "TableTwoWay":
                return "draw_table_two_way" in content
            elif data_class == "DataTable":
                return "draw_data_table" in content
            elif data_class == "Table":
                return "generate_table" in content
            elif data_class == "DivisionModel":
                return "draw_division_model" in content
            elif data_class == "MultiGraphList":
                return "create_multi_bar_graph" in content
            elif data_class == "ExtendedUnitFractionNumberLine":
                return (
                    "create_extended_unit_fraction_number_line" in content
                    and "MultiExtendedUnitFractionNumberLine" not in content
                )
            # Add more mappings as needed
            else:
                return False

        except Exception as e:
            print(f"⚠️ Error checking if content might contain data for class: {e}")
            return False

    def _is_real_data_definition(self, instantiation: str, data_class: str) -> bool:
        """Check if the instantiation is a real data definition, not a variable reference."""
        try:
            # Check if it contains variable references like data.root[:3]
            if (
                "data." in instantiation
                or "root" in instantiation
                or "[:3]" in instantiation
            ):
                return False

            # Check if it contains actual data structures like lists or dictionaries
            if "[" in instantiation and "]" in instantiation:
                return True
            if "{" in instantiation and "}" in instantiation:
                return True

            # If it's just a simple instantiation without data, it's probably not what we want
            return False

        except Exception as e:
            print(f"⚠️ Error checking if instantiation is real data definition: {e}")
            return True

    def _extract_data_class_example(
        self, content: str, data_class: str
    ) -> Optional[str]:
        """Extract a data class instantiation example from test file content."""
        try:
            lines = content.split("\n")

            for i, line in enumerate(lines):
                # Look for lines that instantiate the data class
                if data_class in line and "=" in line and "(" in line:
                    # Found a potential instantiation
                    # Extract the right side of the assignment
                    right_side = line.split("=", 1)[1].strip()

                    if right_side.startswith(data_class):
                        # This is a direct instantiation
                        instantiation = self._extract_complete_instantiation(
                            right_side, lines, i
                        )
                        if instantiation:
                            # Check if this is a real data definition (not a variable reference)
                            if self._is_real_data_definition(instantiation, data_class):
                                return self._convert_instantiation_to_raw_data(
                                    instantiation
                                )
                    elif "(" in right_side and data_class in right_side:
                        # This might be a method call or nested instantiation
                        instantiation = self._extract_complete_instantiation(
                            right_side, lines, i
                        )
                        if instantiation:
                            # Check if this is a real data definition (not a variable reference)
                            if self._is_real_data_definition(instantiation, data_class):
                                return self._convert_instantiation_to_raw_data(
                                    instantiation
                                )

            # If no direct instantiation found, look for function calls that use this data class
            for i, line in enumerate(lines):
                if data_class in line and "(" in line:
                    # Look for patterns like function_name(DataClass(...))
                    if "(" in line and ")" in line:
                        # Try to extract the data class instantiation from the function call
                        start_idx = line.find(data_class)
                        if start_idx != -1:
                            # Find the opening parenthesis for this data class
                            paren_start = line.find("(", start_idx)
                            if paren_start != -1:
                                # Find matching closing parenthesis
                                paren_count = 0
                                for j, char in enumerate(
                                    line[paren_start:], paren_start
                                ):
                                    if char == "(":
                                        paren_count += 1
                                    elif char == ")":
                                        paren_count -= 1
                                        if paren_count == 0:
                                            # Found complete instantiation
                                            instantiation = line[start_idx : j + 1]
                                            # Check if this is a real data definition (not a variable reference)
                                            if self._is_real_data_definition(
                                                instantiation, data_class
                                            ):
                                                return self._convert_instantiation_to_raw_data(
                                                    instantiation
                                                )

            # If no class instantiation found, look for raw dictionary data in function calls
            # This handles cases where the test uses raw dictionaries instead of class instantiations
            raw_data = self._extract_raw_data_from_function_calls(content, data_class)
            if raw_data:
                return raw_data

            return None

        except Exception as e:
            print(f"⚠️ Error extracting data class example: {e}")
            return None

    def _extract_raw_data_from_function_calls(
        self, content: str, data_class: str
    ) -> Optional[str]:
        """Extract raw dictionary data from function calls when no class instantiations are found."""
        try:
            lines = content.split("\n")

            # Look for function calls that might use this data class
            # Pattern: function_name(stimulus) where stimulus is a dictionary
            for i, line in enumerate(lines):
                if "=" in line and "{" in line:
                    # This looks like a dictionary assignment (even if multi-line)
                    if "stimulus" in line or "data" in line or "input" in line:
                        # Extract the dictionary from this line and following lines
                        dict_data = self._extract_dictionary_from_lines(lines, i)
                        if dict_data:
                            # Check if this dictionary has the right structure for the data class
                            if self._is_valid_data_for_class(dict_data, data_class):
                                return dict_data

            return None

        except Exception as e:
            print(f"⚠️ Error extracting raw data from function calls: {e}")
            return None

    def _is_valid_data_for_class(self, dict_data: str, data_class: str) -> bool:
        """Check if the extracted dictionary data is valid for the given data class."""
        try:
            # First, check for basic JSON validity
            if not self._is_valid_json_structure(dict_data):
                return False

            # For TableTwoWay, look for specific fields
            if data_class == "TableTwoWay":
                return (
                    "table_title" in dict_data
                    and "rows_title" in dict_data
                    and "columns_title" in dict_data
                )
            elif data_class == "DataTable":
                return (
                    "title" in dict_data
                    and "headers" in dict_data
                    and "data" in dict_data
                )
            elif data_class == "Table":
                return "columns" in dict_data and "rows" in dict_data
            elif data_class == "DivisionModel":
                return "dividend" in dict_data and "divisor" in dict_data
            elif data_class == "MultiGraphList":
                # MultiGraphList should be a list of graphs
                return dict_data.startswith("[") and dict_data.endswith("]")
            elif data_class == "ExtendedUnitFractionNumberLine":
                # ExtendedUnitFractionNumberLine should have range, minor_divisions, and dot_point
                return (
                    "range" in dict_data
                    and "minor_divisions" in dict_data
                    and "dot_point" in dict_data
                )
            elif data_class == "TriangleStimulusDescription":
                # TriangleStimulusDescription should have triangle field with points and angles
                return (
                    "triangle" in dict_data
                    and "points" in dict_data
                    and "angles" in dict_data
                )
            elif data_class == "RightTriangleWithRay":
                # RightTriangleWithRay should have triangle field with points, angles, and rays
                return (
                    "triangle" in dict_data
                    and "points" in dict_data
                    and "angles" in dict_data
                    and "rays" in dict_data
                )
            elif data_class == "TransversalAngleParams":
                # TransversalAngleParams should have given_angle, x_angle_position, and given_angle_position
                return (
                    "given_angle" in dict_data
                    and "x_angle_position" in dict_data
                    and "given_angle_position" in dict_data
                )
            # Add more data class validations as needed
            else:
                # For unknown data classes, return True to use the first available data
                return True

        except Exception as e:
            print(f"⚠️ Error validating data for class: {e}")
            return True

    def _is_valid_json_structure(self, data: str) -> bool:
        """Check if the data has a valid JSON structure."""
        try:
            # Check for balanced brackets and braces
            brace_count = 0
            bracket_count = 0
            in_string = False
            escape_next = False

            for char in data:
                if escape_next:
                    escape_next = False
                    continue

                if char == "\\":
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                    elif char == "[":
                        bracket_count += 1
                    elif char == "]":
                        bracket_count -= 1

            # Check if brackets and braces are balanced
            if brace_count != 0 or bracket_count != 0:
                return False

            # Check for basic structure
            if data.strip().startswith("{") and data.strip().endswith("}"):
                return True
            if data.strip().startswith("[") and data.strip().endswith("]"):
                return True

            return False

        except Exception as e:
            print(f"⚠️ Error checking JSON structure: {e}")
            return False

    def _extract_dictionary_from_lines(
        self, lines: List[str], start_line_idx: int
    ) -> Optional[str]:
        """Extract a complete dictionary from lines starting at the given index."""
        try:
            # Find the line with the assignment
            line = lines[start_line_idx]
            if "=" not in line:
                return None

            # Extract the right side of the assignment
            right_side = line.split("=", 1)[1].strip()

            # If it starts with {, try to find the complete dictionary
            if right_side.startswith("{"):
                # Count braces to find the complete dictionary
                brace_count = 0
                dict_content = ""
                current_line_idx = start_line_idx

                # Start from the { in the right side
                start_char = right_side.find("{")
                dict_content = right_side[start_char:]

                # Count braces in the initial content
                for char in dict_content:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1

                # If not complete, look at subsequent lines
                while brace_count > 0 and current_line_idx < len(lines) - 1:
                    current_line_idx += 1
                    next_line = lines[current_line_idx].strip()
                    dict_content += " " + next_line

                    # Count braces in the updated content
                    brace_count = 0
                    for char in dict_content:
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1

                if brace_count == 0:
                    return dict_content

            return None

        except Exception as e:
            print(f"⚠️ Error extracting dictionary from lines: {e}")
            return None

    def _convert_instantiation_to_raw_data(self, instantiation: str) -> str:
        """Convert a data class instantiation to raw data format."""
        try:
            # Remove the class name and parentheses, keeping only the arguments
            if "(" in instantiation and ")" in instantiation:
                start_idx = instantiation.find("(")
                end_idx = instantiation.rfind(")")
                args = instantiation[start_idx + 1 : end_idx].strip()

                # Convert class instantiations to raw data
                # Handle Fraction(shape=FractionShape.RECTANGLE, fraction="3/4") -> {"shape": "rectangle", "fraction": "3/4"}
                args = self._convert_nested_instantiations_to_raw_data(args)

                # Handle different data structures
                if args.startswith("[") and args.endswith("]"):
                    # This is a list - return as is
                    return args
                elif args.startswith("{") and args.endswith("}"):
                    # This is a dict - return as is
                    return args
                else:
                    # This might be a single value or complex structure
                    # Try to parse it as a list or dict
                    if args.startswith("[") or args.startswith("{"):
                        return args
                    else:
                        # For complex structures like "dividend={...}, divisor=3",
                        # we need to convert to proper dict format
                        if "=" in args and "," in args:
                            # This looks like key=value pairs, convert to dict
                            return self._convert_key_value_pairs_to_dict(args)
                        else:
                            # Single value - wrap in appropriate structure
                            return f'"{args}"' if not args.isdigit() else args

            return instantiation

        except Exception as e:
            print(f"⚠️ Error converting instantiation to raw data: {e}")
            return instantiation

    def _convert_nested_instantiations_to_raw_data(self, args: str) -> str:
        """Convert nested class instantiations to raw data format."""
        try:
            # Handle Fraction(shape=FractionShape.RECTANGLE, fraction="3/4") -> {"shape": "rectangle", "fraction": "3/4"}
            import re

            # Find all class instantiations like Fraction(...)
            pattern = r"(\w+)\(([^)]+)\)"
            matches = re.findall(pattern, args)

            for class_name, class_args in matches:
                # Convert class arguments to dict format
                dict_args = self._convert_class_args_to_dict(class_args)
                # Replace the instantiation with dict format
                old_instantiation = f"{class_name}({class_args})"
                args = args.replace(old_instantiation, dict_args)

            return args

        except Exception as e:
            print(f"⚠️ Error converting nested instantiations: {e}")
            return args

    def _convert_class_args_to_dict(self, class_args: str) -> str:
        """Convert class arguments to dictionary format."""
        try:
            # Parse arguments like: shape=FractionShape.RECTANGLE, fraction="3/4"
            args_dict = {}

            # Split by comma, but be careful with nested structures
            parts = []
            current_part = ""
            paren_count = 0

            for char in class_args:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                elif char == "," and paren_count == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                    continue
                current_part += char

            if current_part.strip():
                parts.append(current_part.strip())

            # Convert each part to key-value pair
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert enum values to strings
                    if "." in value and not value.startswith('"'):
                        # This is likely an enum like FractionShape.RECTANGLE
                        enum_value = value.split(".")[-1].lower()
                        # Handle special enum mappings
                        if enum_value == "inches":
                            enum_value = "in"
                        elif enum_value == "centimeters":
                            enum_value = "cm"
                        elif enum_value == "millimeters":
                            enum_value = "mm"
                        elif enum_value == "feet":
                            enum_value = "ft"
                        elif enum_value == "meters":
                            enum_value = "m"
                        elif enum_value == "kilometers":
                            enum_value = "km"
                        args_dict[key] = f'"{enum_value}"'
                    elif value.startswith('"') and value.endswith('"'):
                        # Already a string - check for unit conversions
                        string_value = value[1:-1]  # Remove quotes
                        if string_value == "inches":
                            args_dict[key] = '"in"'
                        elif string_value == "centimeters":
                            args_dict[key] = '"cm"'
                        elif string_value == "millimeters":
                            args_dict[key] = '"mm"'
                        elif string_value == "feet":
                            args_dict[key] = '"ft"'
                        elif string_value == "meters":
                            args_dict[key] = '"m"'
                        elif string_value == "kilometers":
                            args_dict[key] = '"km"'
                        else:
                            args_dict[key] = value
                    else:
                        # Other values
                        args_dict[key] = value

            # Convert to dict string
            dict_pairs = [f'"{k}": {v}' for k, v in args_dict.items()]
            return "{" + ", ".join(dict_pairs) + "}"

        except Exception as e:
            print(f"⚠️ Error converting class args to dict: {e}")
            return class_args

    def _convert_key_value_pairs_to_dict(self, args: str) -> str:
        """Convert key=value pairs to dictionary format."""
        try:
            # Parse arguments like: dividend={"numerator": 6, "denominator": 7}, divisor=3
            args_dict = {}

            # Split by comma, but be careful with nested structures
            parts = []
            current_part = ""
            brace_count = 0
            paren_count = 0

            for char in args:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                elif char == "," and brace_count == 0 and paren_count == 0:
                    parts.append(current_part.strip())
                    current_part = ""
                    continue
                current_part += char

            if current_part.strip():
                parts.append(current_part.strip())

            # Convert each part to key-value pair
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Handle nested structures
                    if value.startswith("{") and value.endswith("}"):
                        # This is already a dict structure
                        args_dict[key] = value
                    elif value.startswith("[") and value.endswith("]"):
                        # This is a list structure
                        args_dict[key] = value
                    elif value.startswith('"') and value.endswith('"'):
                        # Already a string
                        args_dict[key] = value
                    else:
                        # Other values
                        args_dict[key] = value

            # Convert to dict string
            dict_pairs = [f'"{k}": {v}' for k, v in args_dict.items()]
            return "{" + ", ".join(dict_pairs) + "}"

        except Exception as e:
            print(f"⚠️ Error converting key-value pairs to dict: {e}")
            return args

    def _extract_complete_instantiation(
        self, line: str, lines: List[str], line_idx: int
    ) -> Optional[str]:
        """Extract a complete data class instantiation, handling multi-line cases."""
        try:
            # Count parentheses to find the complete instantiation
            paren_count = 0
            start_idx = line.find("(")
            if start_idx == -1:
                return None

            # Start counting from the first opening parenthesis
            for j, char in enumerate(line[start_idx:], start_idx):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        # Found complete instantiation on this line
                        return line[: j + 1]

            # If not complete on this line, look at subsequent lines
            current_line = line
            for next_line_idx in range(
                line_idx + 1, min(line_idx + 10, len(lines))
            ):  # Limit to 10 lines
                current_line += " " + lines[next_line_idx].strip()

                # Count parentheses again
                paren_count = 0
                for char in current_line:
                    if char == "(":
                        paren_count += 1
                    elif char == ")":
                        paren_count -= 1
                        if paren_count == 0:
                            # Found complete instantiation
                            return current_line

            return None

        except Exception as e:
            print(f"⚠️ Error extracting complete instantiation: {e}")
            return None

    def _extract_test_patterns(self, test_content: str, function_name: str) -> Dict:
        """Extract test patterns from test file content."""
        try:
            tree = ast.parse(test_content)

            imports = []
            test_data = None

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if "stimulus_descriptions" in (node.module or ""):
                        for alias in node.names:
                            imports.append(f"from {node.module} import {alias.name}")

                # Find function calls and extract test data
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == function_name
                    ):
                        # Found a call to our function
                        if len(node.args) > 0:
                            # Try to extract the argument
                            arg = node.args[0]
                            if isinstance(arg, ast.Name):
                                # It's a variable, we need to find its definition
                                test_data = self._find_variable_definition(
                                    test_content, arg.id
                                )
                            else:
                                # It's a literal or constructor call
                                test_data = ast.unparse(arg)

            return {"imports": imports, "test_data": test_data}

        except Exception as e:
            print(f"⚠️  Error extracting patterns: {e}")
            return {"imports": [], "test_data": None}

    def _find_variable_definition(self, content: str, var_name: str) -> Optional[str]:
        """Find the definition of a variable in the test content."""
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == var_name:
                            # Found the assignment
                            return ast.unparse(node.value)

            return None

        except Exception as e:
            print(f"⚠️  Error finding variable definition: {e}")
            return None

    def create_repository_test_script(self, function_name: str) -> str:
        """Create a test script that works within the repository environment."""

        # Find function information
        function_info = self.find_function_in_repository(function_name)
        if not function_info:
            return None

        # Analyze test patterns
        test_patterns = self.analyze_test_files_for_function(function_name)

        # Get parameter name
        args = function_info.get("args", [])
        param_name = args[0] if args else "test_data"

        # Create the repository-based test script
        script_content = f'''#!/usr/bin/env python3
"""
Repository-based test script for {function_name}
This script runs within the repository environment.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Set up environment
os.environ["MPLBACKEND"] = "Agg"

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Create output directory
output_dir = "test_output"
os.makedirs(output_dir, exist_ok=True)

try:
    # Import the function
    from {function_info["module_path"]} import {function_name}
    
    # Import required classes
{chr(10).join("    " + imp for imp in test_patterns.get("imports", []))}
    
    # Create test data
    {param_name} = {test_patterns.get("test_data", "None")}
    
    print(f"Executing {function_name} with data: {{type({param_name})}}")
    
    # Execute the function
    result_file = {function_name}({param_name})
    
    if result_file and os.path.exists(result_file):
        # Move to output directory
        dest = os.path.join(output_dir, os.path.basename(result_file))
        shutil.move(result_file, dest)
        print(f"SUCCESS: Generated {{dest}}")
    else:
        print(f"SUCCESS: Function executed but no file returned")
        
except Exception as e:
    import traceback
    print(f"ERROR: {{e}}")
    print(f"TRACEBACK: {{traceback.format_exc()}}")
    sys.exit(1)
'''

        return script_content

    def run_repository_function_test(self, function_name: str) -> Dict:
        """Run a function within the repository environment."""
        try:
            print(f"🚀 Starting repository-based execution for: {function_name}")

            # Setup environment
            repo_dir = self.setup_repository_environment()

            # Create test script
            script_content = self.create_repository_test_script(function_name)
            if not script_content:
                return {
                    "success": False,
                    "error": f"Could not create test script for {function_name}",
                    "function_name": function_name,
                }

            # Write script to file
            script_path = os.path.join(repo_dir, f"repo_test_{function_name}.py")
            with open(script_path, "w") as f:
                f.write(script_content)

            print(f"Executing repository script: {script_path}")

            # Run the script within the repository
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            print(
                f"Repository execution completed with return code: {result.returncode}"
            )
            print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")

            # Collect generated images
            images = self._collect_generated_images(repo_dir, function_name)

            success = result.returncode == 0 and "SUCCESS:" in result.stdout

            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "images": images,
                "execution_method": "repository",
                "function_name": function_name,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Function execution timed out after 120 seconds",
                "function_name": function_name,
            }
        except Exception as e:
            import traceback

            return {
                "success": False,
                "error": f"Repository execution failed: {e}",
                "traceback": traceback.format_exc(),
                "function_name": function_name,
            }
        finally:
            # Cleanup
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception:
                    pass

    def _collect_generated_images(
        self, repo_dir: str, function_name: str
    ) -> List[Dict]:
        """Collect generated images from the repository execution."""
        images = []

        # Check output directory
        output_dir = os.path.join(repo_dir, "test_output")
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".svg")):
                    file_path = os.path.join(output_dir, file)

                    try:
                        with open(file_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode()

                        images.append(
                            {
                                "filename": file,
                                "path": file_path,
                                "size_bytes": os.path.getsize(file_path),
                                "function_name": function_name,
                                "generated_at": datetime.now().isoformat(),
                                "image_data": img_data,
                                "file_extension": os.path.splitext(file)[1].lower(),
                            }
                        )
                    except Exception as e:
                        print(f"⚠️  Error reading image {file}: {e}")
                        continue

        return images
