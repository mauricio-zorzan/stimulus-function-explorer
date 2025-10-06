import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.graphing_function_model import (
    GraphingFunction,
    GraphingFunctionQuadrantOne,
)
from content_generators.settings import settings


def _calculate_relation_values(x, graphing_function: GraphingFunction):
    """Calculate y values for different relation types."""
    if graphing_function.function_type == "linear":
        return graphing_function.a * x + (graphing_function.b or 0)
    elif graphing_function.function_type == "quadratic":
        return (
            graphing_function.a * x**2
            + (graphing_function.b or 0) * x
            + (graphing_function.c or 0)
        )
    elif graphing_function.function_type == "exponential":
        return graphing_function.a * np.exp((graphing_function.b or 0) * x)
    elif graphing_function.function_type == "cubic":
        return (
            graphing_function.a * x**3
            + (graphing_function.b or 0) * x**2
            + (graphing_function.c or 0) * x
            + (graphing_function.d or 0)
        )
    elif graphing_function.function_type == "square_root":
        domain_shift = graphing_function.b or 0
        # Ensure we only evaluate where x + domain_shift >= 0
        valid_mask = (x + domain_shift) >= 0
        y = np.full_like(x, np.nan)
        y[valid_mask] = graphing_function.a * np.sqrt(x[valid_mask] + domain_shift) + (
            graphing_function.c or 0
        )
        return y
    elif graphing_function.function_type == "rational":
        # Return special marker for rational functions to be handled in main function
        return "rational_function"
    elif graphing_function.function_type == "circle":
        # For circles, we need to plot parametrically
        theta = np.linspace(0, 2 * np.pi, len(x))
        x_circle = graphing_function.radius * np.cos(theta)
        y_circle = graphing_function.radius * np.sin(theta)
        return x_circle, y_circle
    elif graphing_function.function_type == "sideways_parabola":
        # x = a(y-k)² + h - solve for x given y
        # We'll use y as our parameter instead of x
        y_param = np.linspace(-10, 10, len(x))
        x_param = graphing_function.a * y_param**2 + (graphing_function.b or 0)
        return x_param, y_param
    elif graphing_function.function_type == "hyperbola":
        # Return special marker for hyperbola functions to be handled in main function
        return "hyperbola_function"
    elif graphing_function.function_type == "ellipse":
        # (x-h)²/a² + (y-k)²/b² = 1 - plot parametrically
        theta = np.linspace(0, 2 * np.pi, len(x))
        x_ellipse = graphing_function.x_radius * np.cos(theta)
        y_ellipse = graphing_function.y_radius * np.sin(theta)
        return x_ellipse, y_ellipse
    else:
        raise ValueError(
            f"Unsupported function type: {graphing_function.function_type}"
        )


@stimulus_function
def draw_graphing_function(graphing_function: GraphingFunction):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Determine the range for x values
    max_val = 10  # Set to exactly 10
    x = np.linspace(
        -max_val, max_val, 1000
    )  # Increased number of points for smoother curve

    # Calculate y values based on function type
    result = _calculate_relation_values(x, graphing_function)

    # Handle different return types from _calculate_relation_values
    if isinstance(result, str) and result == "rational_function":
        # IMPROVED: Explicitly split branches and draw dashed asymptotes for rational functions
        # Split x values into negative and positive branches
        mask_neg = x < -0.01  # Avoid values near zero
        mask_pos = x > 0.01  # Avoid values near zero

        # Calculate y values for each branch separately and use the same color
        color = plt.cm.tab10(0)  # Use first color from default color cycle
        if np.any(mask_neg):
            x_neg = x[mask_neg]
            y_neg = graphing_function.a / x_neg + (graphing_function.b or 0)
            ax.plot(x_neg, y_neg, linewidth=2, color=color)

        if np.any(mask_pos):
            x_pos = x[mask_pos]
            y_pos = graphing_function.a / x_pos + (graphing_function.b or 0)
            ax.plot(x_pos, y_pos, linewidth=2, color=color)

        # Draw dashed asymptotes for educational clarity
        ax.axvline(
            0, ls="--", lw=1, color="gray", alpha=0.7
        )  # Vertical asymptote at x=0
        ax.axhline(
            graphing_function.b or 0, ls="--", lw=2, color="gray", alpha=0.7
        )  # Horizontal asymptote (thicker for vertical shift emphasis)
    elif isinstance(result, str) and result == "hyperbola_function":
        # FIXED: Plot hyperbola branches separately with proper domain control
        # x²/a² - y²/b² = 1
        a = graphing_function.x_radius
        b = graphing_function.y_radius

        color = plt.cm.tab10(0)  # Use first color from default color cycle

        # Right branch: create dedicated x array from vertex to right boundary
        if a < max_val:  # Only plot if vertex is within view
            x_right = np.linspace(a, max_val, 500)  # Start precisely at vertex x=a
            y_right_pos = b * np.sqrt(x_right**2 / a**2 - 1)
            y_right_neg = -y_right_pos
            # Plot positive and negative parts of right branch
            ax.plot(x_right, y_right_pos, linewidth=2, color=color)
            ax.plot(x_right, y_right_neg, linewidth=2, color=color)

        # Left branch: create dedicated x array from left boundary to vertex
        if a < max_val:  # Only plot if vertex is within view
            x_left = np.linspace(-max_val, -a, 500)  # End precisely at vertex x=-a
            y_left_pos = b * np.sqrt(x_left**2 / a**2 - 1)
            y_left_neg = -y_left_pos
            # Plot positive and negative parts of left branch
            ax.plot(x_left, y_left_pos, linewidth=2, color=color)
            ax.plot(x_left, y_left_neg, linewidth=2, color=color)
    elif isinstance(result, tuple):
        # Functions that return (x, y) pairs (circle, etc.)
        x_plot, y_plot = result
        ax.plot(x_plot, y_plot, linewidth=2)
    else:
        # Functions that return y values for the original x array
        y = result
        ax.plot(x, y, linewidth=2)

    # Set up the coordinate plane
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)

    # Set integer ticks counting by 2s
    ticks = range(-10, 11, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Increase font size of tick labels and remove zero labels
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xticklabels([str(t) if t != 0 else "" for t in ticks])
    ax.set_yticklabels([str(t) if t != 0 else "" for t in ticks])

    # Add grid with darker gray lines
    ax.grid(True, linewidth=0.5, color="#808080")

    # Center the axes and make them thinner
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Set tick positions
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.tight_layout()

    # Save the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_function_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name


@stimulus_function
def draw_graphing_function_quadrant_one(graphing_function: GraphingFunctionQuadrantOne):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use a reasonable x range for quadrant I
    x = np.linspace(
        0.01, 10, 1000
    )  # Start from 0.01 to avoid division by zero for rational functions

    # Calculate y values for plotting
    if graphing_function.function_type == "linear":
        y = graphing_function.a * x + (graphing_function.b or 0)
    elif graphing_function.function_type == "quadratic":
        y = (
            graphing_function.a * x**2
            + (graphing_function.b or 0) * x
            + (graphing_function.c or 0)
        )
    elif graphing_function.function_type == "exponential":
        y = graphing_function.a * np.exp((graphing_function.b or 0) * x)
    elif graphing_function.function_type == "cubic":
        y = (
            graphing_function.a * x**3
            + (graphing_function.b or 0) * x**2
            + (graphing_function.c or 0) * x
            + (graphing_function.d or 0)
        )
    elif graphing_function.function_type == "square_root":
        domain_shift = graphing_function.b or 0
        y = graphing_function.a * np.sqrt(x + domain_shift) + (graphing_function.c or 0)
    elif graphing_function.function_type == "rational":
        # For quadrant I, only plot positive x values with improved asymptote handling
        y = graphing_function.a / x + (graphing_function.b or 0)
        ax.plot(x, y, linewidth=2)

        # Draw dashed asymptotes for educational clarity (only positive quadrant)
        ax.axhline(
            graphing_function.b or 0, ls="--", lw=2, color="gray", alpha=0.7
        )  # Horizontal asymptote (thicker for vertical shift emphasis)
    else:
        raise ValueError(
            f"Unsupported function type: {graphing_function.function_type}"
        )

    # Plot the function with a thicker line (for non-rational functions)
    if graphing_function.function_type != "rational":
        ax.plot(x, y, linewidth=2)

    # Let Matplotlib autoscale to your data (with default small margin)
    ax.relim()
    ax.autoscale_view()

    # Force "nice" integer ticks at reasonable intervals
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

    # Ensure we stay in quadrant I (positive values only)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(0, max(5, x_lim[1]))  # At least 5, but can go higher
    ax.set_ylim(0, max(5, y_lim[1]))  # At least 5, but can go higher

    # Increase font size of tick labels
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Add grid with darker gray lines
    ax.grid(True, linewidth=0.5, color="#808080")

    # Set up axes for quadrant I (bottom-left corner)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Set tick positions
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.tight_layout()

    # Save the plot
    file_name = f"{settings.additional_content_settings.image_destination_folder}/graphing_function_quadrant_one_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"
    plt.savefig(
        file_name,
        transparent=False,
        bbox_inches="tight",
        format=settings.additional_content_settings.stimulus_image_format,
    )
    plt.close()

    return file_name
