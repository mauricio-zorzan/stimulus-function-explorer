import time

from content_generators.additional_content.stimulus_image.drawing_functions import (
    stimulus_function,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.flowchart import (
    Flowchart,
)
from content_generators.settings import settings
from graphviz import Digraph
from IPython.display import HTML, display


@stimulus_function
def create_flowchart(stimulus_description: Flowchart):
    flowchart_data = stimulus_description.flowchart
    orientation = flowchart_data.orientation

    dot = Digraph(
        format=settings.additional_content_settings.stimulus_image_format,
        graph_attr={"fontsize": "10"},
    )

    if orientation == "vertical":
        dot.graph_attr["rankdir"] = "TB"  # Top to Bottom for vertical
    else:
        dot.graph_attr["rankdir"] = "LR"  # Left to Right for horizontal

    # Add nodes
    for node in flowchart_data.nodes:
        dot.node(node.id, node.label or "", shape=node.shape.value)

    # Add edges
    for edge in flowchart_data.edges:
        edge_label = edge.label or ""
        dot.edge(edge.from_, edge.to, label=edge_label)

    # Create the base filename without extension
    base_name = f"{settings.additional_content_settings.image_destination_folder}/flowchart_{int(time.time())}"

    # Render will add the extension automatically
    dot.render(
        filename=base_name,  # Don't include the extension here
        format=settings.additional_content_settings.stimulus_image_format,
        cleanup=True,
    )

    # Render the graph to a buffer and display the centered SVG
    svg_data = dot.pipe(format="svg").decode("utf-8")  # Get SVG data as string
    centered_svg = (
        f'<div style="display: flex; justify-content: center;">{svg_data}</div>'
    )
    display(HTML(centered_svg))

    # Return the complete filename (with extension) for consistency
    return f"{base_name}.{settings.additional_content_settings.stimulus_image_format}"
