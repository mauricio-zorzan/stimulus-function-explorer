from typing import Annotated, Callable, Optional  # noqa: I001

from pydantic import PlainValidator, WithJsonSchema

from content_generators.additional_content.stimulus_image.drawing_functions.bar_models import (
    draw_bar_model_stimulus,
)
from content_generators.additional_content.stimulus_image.drawing_functions.colored_shapes_coordinate_plane import (
    draw_colored_shapes_coordinate_plane,
)
from content_generators.additional_content.stimulus_image.drawing_functions.flow_chart import (
    create_flowchart,
)
from content_generators.additional_content.stimulus_image.drawing_functions.prism_nets import (
    draw_custom_triangular_prism_net,
    draw_dual_nets,
    draw_prism_net,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusFunctionProtocol,
)

from .angles import (
    draw_fractional_angle_circle,
    draw_labeled_transversal_angle,
    draw_lines_rays_and_segments,
    draw_parallel_lines_cut_by_transversal,
    draw_triangle_with_ray,
    generate_angle_diagram,
    generate_angle_diagram_360,
    generate_angle_supplementary,
    generate_angle_types,
    generate_angles,
    generate_single_angle_type,
)
from .angles_on_circle import draw_circle_angle_measurement
from .area_models import create_area_model, unit_square_decomposition
from .base_ten_blocks import (
    draw_base_ten_blocks,
    draw_base_ten_blocks_grid,
    draw_base_ten_blocks_division,
    draw_base_ten_blocks_division_grid,
)
from .box_plots import draw_bar_models, draw_box_plots
from .categorical_graphs import (
    create_categorical_graph,
    create_multi_bar_graph,
    create_multi_picture_graph,
    create_stats_diagram_bar,
    draw_tree_diagram,
)
from .clocks import create_clock
from .combo_points_table_graph import draw_combo_points_table_graph
from .counting import draw_counting
from .data_table_with_graph import draw_table_and_graph
from .decimal_grid import (
    draw_decimal_comparison,
    draw_decimal_grid,
    draw_decimal_multiplication,
)
from .divide_into_equal_groups import draw_divide_into_equal_groups
from .divide_items_into_array import draw_divide_items_into_array
from .equation_tape_diagram import draw_equation_tape_diagram
from .fraction_addition import draw_fraction_addition_model
from .fraction_models import (
    draw_division_model,
    draw_fraction_strips,
    draw_fractional_models,
    draw_fractional_models_full_shade,
    draw_fractional_models_labeled,
    draw_fractional_models_multiplication_units,
    draw_fractional_models_no_shade,
    draw_fractional_models_unequal,
    draw_fractional_pair_models,
    draw_fractional_sets_models,
    draw_mixed_fractional_models,
    draw_whole_fractional_models,
)
from .geometric_shapes import (
    create_parallel_quadrilateral,
    draw_circle_diagram,
    draw_circle_with_arcs,
    draw_circle_with_radius,
    draw_composite_rectangular_grid,
    draw_composite_rectangular_grid_with_dashed_lines,
    draw_composite_rectangular_triangular_grid,
    draw_geometric_shapes,
    draw_geometric_shapes_no_indicators,
    draw_geometric_shapes_no_indicators_with_rotation,
    draw_geometric_shapes_with_angles,
    draw_geometric_shapes_with_rotation,
    draw_regular_irregular_polygons,
    draw_polygons_bare,
    draw_parallelogram_with_height,
    draw_polygon_fully_labeled,
    draw_polygon_perimeter,
    draw_polygon_perimeter_with_all_sides_labeled,
    draw_polygon_with_string_sides,
    draw_quadrilateral_figures,
    draw_quadrilateral_venn_diagram,
    draw_single_quadrilateral_stimulus,
    draw_shape_with_right_angles,
    draw_similar_right_triangles,
    generate_area_stimulus,
    generate_composite_rect_prism,
    generate_composite_rect_prism_v2,
    generate_geometric_shapes_transformations,
    generate_multiple_grids,
    generate_rect_with_side_len,
    generate_single_rect_with_side_len_stimulus,
    generate_rect_with_side_len_and_area,
    generate_shape_with_base_and_height,
    generate_stimulus_with_grid,
    generate_trapezoid_with_side_len,
    generate_triangle_with_opt_side_len,
    generate_triangle_with_side_len,
    generate_unit_squares_unitless,
)
from .geometric_shapes_3d import (
    draw_cross_section_question,
    draw_multiple_3d_objects,
    draw_right_prisms,
)
from .graphing import (
    create_polygon,
    create_scatterplot,
    draw_blank_coordinate_plane,
    draw_combined_graphs,
    draw_grouped_bar_chart,
    draw_line_graphs,
    draw_linear_diagram,
    draw_multi_line_graph,
    draw_stats_scatterplot,
    multiple_bar_graph,
    plot_line,
    plot_lines,
    plot_nonlinear,
    plot_par_and_perp_lines,
    plot_points,
    plot_points_four_quadrants,
    plot_points_four_quadrants_with_label,
    plot_points_quadrant_one,
    plot_points_quadrant_one_with_context,
    plot_polygon_dilation,
    plot_polygon_four_quadrants,
)
from .graphing_function import (
    draw_graphing_function,
    draw_graphing_function_quadrant_one,
)
from .graphing_piecewise import generate_piecewise_graph
from .histogram import (
    draw_histogram,
    draw_histogram_pair,
    draw_histogram_with_dotted_bin,
)
from .line_graph import create_line_graph
from .line_plots import (
    generate_double_line_plot,
    generate_single_line_plot,
    generate_stacked_line_plots,
)
from .lines_of_best_fit import draw_lines_of_best_fit
from .measurement_comparison import draw_measurement_comparison
from .measurements import draw_measurement
from .number_lines import (
    create_decimal_comparison_number_line,
    create_dot_plot,
    create_dual_dot_plot,
    create_extended_unit_fraction_number_line,
    create_fixed_step_number_line,
    create_inequality_number_line,
    create_multi_extended_unit_fraction_number_line_with_bar,
    create_multi_extended_unit_fraction_number_line_with_bar_v2,
    create_multi_extended_unit_fraction_number_line_with_dots,
    create_multi_inequality_number_line,
    create_multi_labeled_unit_fraction_number_line,
    create_number_line,
    create_unit_fraction_number_line,
    create_vertical_number_line,
)
from .number_lines_clock import create_clock_number_line
from .object_array import draw_object_array
from .pedigree_chart import draw_pedigree_chart
from .polygon_scales import draw_polygon_scale
from .protractor import draw_protractor
from .ratio_object_array import draw_ratio_object_array
from .rectangular_prisms import (
    draw_multiple_base_area_rectangular_prisms,
    draw_multiple_rectangular_prisms,
    draw_unit_cube_figure,
)
from .rulers import draw_ruler_measured_objects
from .shapes_decomposition import (
    create_dimensional_compound_area_figure,
    create_rhombus_with_diagonals_figure,
    create_shape_decomposition,
    create_shape_decomposition_decimal_only,
)
from .triangles_decomposition import create_triangle_decomposition_decimal_only
from .trapezoids_decomposition import create_trapezoid_decomposition_decimal_only
from .spinners import generate_spinner
from .stepwise_dot_pattern import draw_stepwise_shape_pattern
from .symmetry_lines import (
    generate_lines_of_symmetry,
    generate_symmetry_identification_task,
)
from .table import (
    create_probability_diagram,
    draw_data_table,
    draw_data_table_group,
    draw_table_two_way,
    generate_table,
)
from .table_and_multi_scatterplots import create_table_and_multi_scatterplots

STIMULUS_DRAWER: list[Callable | StimulusFunctionProtocol] = [
    draw_fractional_models,
    plot_points,
    draw_colored_shapes_coordinate_plane,
    draw_geometric_shapes,
    draw_geometric_shapes_no_indicators,
    draw_geometric_shapes_with_rotation,
    draw_geometric_shapes_no_indicators_with_rotation,
    draw_geometric_shapes_with_angles,
    draw_regular_irregular_polygons,
    draw_polygons_bare,
    draw_quadrilateral_figures,
    draw_quadrilateral_venn_diagram,
    draw_single_quadrilateral_stimulus,
    draw_parallelogram_with_height,
    generate_rect_with_side_len,
    generate_single_rect_with_side_len_stimulus,
    generate_rect_with_side_len_and_area,
    draw_composite_rectangular_grid,
    draw_composite_rectangular_grid_with_dashed_lines,
    draw_composite_rectangular_triangular_grid,
    draw_polygon_fully_labeled,
    draw_polygon_with_string_sides,
    draw_polygon_perimeter,
    draw_polygon_perimeter_with_all_sides_labeled,
    generate_shape_with_base_and_height,
    generate_trapezoid_with_side_len,
    generate_triangle_with_side_len,
    generate_triangle_with_opt_side_len,
    plot_lines,
    create_number_line,
    create_decimal_comparison_number_line,
    create_fixed_step_number_line,
    create_vertical_number_line,
    create_unit_fraction_number_line,
    create_extended_unit_fraction_number_line,
    create_inequality_number_line,
    create_multi_extended_unit_fraction_number_line_with_bar,
    create_multi_extended_unit_fraction_number_line_with_bar_v2,
    create_multi_extended_unit_fraction_number_line_with_dots,
    create_multi_inequality_number_line,
    create_multi_labeled_unit_fraction_number_line,
    plot_par_and_perp_lines,
    generate_table,
    create_scatterplot,
    generate_spinner,
    generate_angles,
    generate_angle_diagram_360,
    draw_circle_angle_measurement,
    draw_fractional_angle_circle,
    draw_table_two_way,
    generate_stimulus_with_grid,
    generate_multiple_grids,
    create_polygon,
    plot_line,
    generate_angle_types,
    generate_single_angle_type,
    generate_angle_diagram,
    generate_angle_supplementary,
    generate_composite_rect_prism,
    generate_composite_rect_prism_v2,
    draw_multiple_base_area_rectangular_prisms,
    generate_stacked_line_plots,
    generate_double_line_plot,
    generate_single_line_plot,
    create_categorical_graph,
    draw_tree_diagram,
    draw_multiple_rectangular_prisms,
    draw_unit_cube_figure,
    generate_geometric_shapes_transformations,
    generate_unit_squares_unitless,
    draw_fractional_models_no_shade,
    draw_fractional_models_full_shade,
    draw_fractional_models_unequal,
    draw_fractional_models_labeled,
    draw_mixed_fractional_models,
    draw_whole_fractional_models,
    draw_lines_rays_and_segments,
    create_area_model,
    unit_square_decomposition,
    plot_points_quadrant_one,
    plot_points_four_quadrants,
    plot_polygon_dilation,
    plot_polygon_four_quadrants,
    create_clock,
    draw_multiple_3d_objects,
    draw_cross_section_question,
    create_stats_diagram_bar,
    create_probability_diagram,
    create_dot_plot,
    create_dual_dot_plot,
    create_shape_decomposition,
    create_shape_decomposition_decimal_only,
    create_triangle_decomposition_decimal_only,
    create_trapezoid_decomposition_decimal_only,
    create_rhombus_with_diagonals_figure,
    draw_stats_scatterplot,
    draw_linear_diagram,
    draw_box_plots,
    draw_bar_models,
    draw_bar_model_stimulus,
    draw_ruler_measured_objects,
    draw_triangle_with_ray,
    draw_base_ten_blocks,
    draw_base_ten_blocks_grid,
    draw_base_ten_blocks_division,
    draw_base_ten_blocks_division_grid,
    draw_decimal_grid,
    draw_decimal_multiplication,
    generate_piecewise_graph,
    draw_measurement_comparison,
    draw_protractor,
    draw_counting,
    create_parallel_quadrilateral,
    draw_circle_with_radius,
    draw_circle_diagram,
    draw_measurement,
    draw_object_array,
    generate_lines_of_symmetry,
    generate_symmetry_identification_task,
    plot_nonlinear,
    draw_parallel_lines_cut_by_transversal,
    draw_graphing_function,
    draw_graphing_function_quadrant_one,
    draw_fractional_pair_models,
    draw_fractional_sets_models,
    draw_fractional_models_multiplication_units,
    plot_points_quadrant_one_with_context,
    create_clock_number_line,
    multiple_bar_graph,
    create_multi_bar_graph,
    create_multi_picture_graph,
    draw_prism_net,
    draw_dual_nets,
    draw_custom_triangular_prism_net,
    draw_lines_of_best_fit,
    generate_area_stimulus,
    draw_labeled_transversal_angle,
    draw_similar_right_triangles,
    draw_shape_with_right_angles,
    draw_grouped_bar_chart,
    draw_circle_with_arcs,
    draw_multi_line_graph,
    draw_data_table,
    draw_data_table_group,
    create_flowchart,
    draw_pedigree_chart,
    draw_combined_graphs,
    draw_table_and_graph,
    draw_stepwise_shape_pattern,
    draw_polygon_scale,
    create_table_and_multi_scatterplots,
    draw_combo_points_table_graph,
    draw_right_prisms,
    draw_divide_into_equal_groups,
    draw_divide_items_into_array,
    create_table_and_multi_scatterplots,
    create_dimensional_compound_area_figure,
    draw_ratio_object_array,
    draw_decimal_comparison,
    draw_division_model,
    draw_fraction_strips,
    draw_histogram,
    draw_histogram_pair,
    draw_equation_tape_diagram,
    draw_line_graphs,
    create_line_graph,
    draw_histogram_with_dotted_bin,
    plot_points_four_quadrants_with_label,
    draw_blank_coordinate_plane,
    draw_fraction_addition_model,
]


def convert_to_stimulus_function(v: str | None) -> StimulusFunctionProtocol | None:
    if isinstance(v, str):
        for func in STIMULUS_DRAWER:
            if func.__name__ == v or (
                func.__name__ == "wrapper" and func.__wrapped__.__name__ == v
            ):
                return func
        return None
    elif v is None:
        return None
    else:
        raise TypeError(
            "Invalid type for stimulus function, expecting str or None, got "
            + str(type(v))
        )


StimulusFunction = Annotated[
    Optional[StimulusFunctionProtocol],
    PlainValidator(lambda s: convert_to_stimulus_function(s)),
    WithJsonSchema(
        {
            "type": "string",
            "enum": [
                sf.__name__ if sf.__name__ != "wrapper" else sf.__wrapped__.__name__
                for sf in STIMULUS_DRAWER
            ],
        },
    ),
]
