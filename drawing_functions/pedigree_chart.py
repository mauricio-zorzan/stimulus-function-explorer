import json
import textwrap
import time

import pandas as pd
from content_generators.settings import settings
from graphviz import Digraph


def draw_pedigree_chart(stimulus_description):
    # Handle both JSON strings and dictionaries
    if isinstance(stimulus_description, str):
        data = json.loads(stimulus_description)
    else:
        data = stimulus_description

    ancestry = pd.DataFrame(data["ancestry"])

    dot = Digraph(
        comment="Ancestry",
        graph_attr={
            "splines": "ortho",
            "nodesep": "0.5",
            "ranksep": "0.5",
            "margin": "0.3",
            "pad": "0.2",
            "ratio": "fill",
        },
    )

    # Add spacing subgraph before the caption
    with dot.subgraph(name="spacing") as s:  # type: ignore[attr-defined]
        s.attr(rank="sink")
        s.node("spacing", "", shape="none", height="0.02")

    # Add caption as a label at the bottom with padding and centering
    caption = data.get("caption", "")
    wrapped_caption = "\n".join(textwrap.wrap(caption, width=120))
    dot.attr(
        label=wrapped_caption,
        labelloc="b",
        fontsize="14",
        labeljust="c",
        labeldistance="2",
    )

    # Add legend subgraph
    with dot.subgraph(name="cluster_legend") as legend:  # type: ignore[attr-defined]
        legend.attr(label="", style="invis")

        # Create a single HTML-like label for the entire legend
        legend_label = """<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2">
            <TR>
                <TD><TABLE BORDER="0" CELLBORDER="1"><TR><TD WIDTH="15" HEIGHT="15" BGCOLOR="white" STYLE="filled" FIXEDSIZE="TRUE"></TD></TR></TABLE></TD>
                <TD ALIGN="left" WIDTH="100"><FONT POINT-SIZE="12">Unaffected male</FONT></TD>
            </TR>
            <TR>
                <TD><TABLE BORDER="0" CELLBORDER="0"><TR><TD WIDTH="15" HEIGHT="15" BGCOLOR="black" STYLE="filled" FIXEDSIZE="TRUE"></TD></TR></TABLE></TD>
                <TD ALIGN="left" WIDTH="100"><FONT POINT-SIZE="12">Affected male</FONT></TD>
            </TR>
            <TR>
                <TD><TABLE BORDER="0" CELLBORDER="1"><TR><TD WIDTH="15" HEIGHT="15" BGCOLOR="white" STYLE="rounded,filled" FIXEDSIZE="TRUE"></TD></TR></TABLE></TD>
                <TD ALIGN="left" WIDTH="100"><FONT POINT-SIZE="12">Unaffected female</FONT></TD>
            </TR>
            <TR>
                <TD><TABLE BORDER="0" CELLBORDER="0"><TR><TD WIDTH="15" HEIGHT="15" BGCOLOR="black" STYLE="rounded,filled" FIXEDSIZE="TRUE"></TD></TR></TABLE></TD>
                <TD ALIGN="left" WIDTH="100"><FONT POINT-SIZE="12">Affected female</FONT></TD>
            </TR>
        </TABLE>
        >"""

        # Create a single node for the legend
        legend.node("legend", legend_label, shape="none")

    def add_individual(person_id, gender, affected):
        shape = "rect" if gender == "M" else "ellipse"
        fillcolor = "black" if affected else "white"
        dot.node(
            person_id,
            "",
            shape=shape,
            style="filled",
            fillcolor=fillcolor,
            width="0.5",
            height="0.5",
            fixedsize="true",
        )

    earliest_ancestor = ancestry.query("Relation == 'Earliest Ancestor'")[
        "Person_1"
    ].iloc[0]
    earliest_ancestor_gender = ancestry.query("Relation == 'Earliest Ancestor'")[
        "Gender"
    ].iloc[0]

    earliest_ancestor_affected = ancestry.query("Relation == 'Earliest Ancestor'")[
        "affected"
    ].iloc[0]
    add_individual(
        earliest_ancestor, earliest_ancestor_gender, earliest_ancestor_affected
    )

    processed = [earliest_ancestor]

    for person in processed:
        spouse_info = ancestry[
            (ancestry["Relation"] == "Spouse") & (ancestry["Person_2"] == person)
        ]
        children = ancestry[
            (ancestry["Relation"] == "Child") & (ancestry["Person_2"] == person)
        ]

        if not spouse_info.empty:
            spouse_id = spouse_info["Person_1"].values[0]
            add_individual(
                spouse_id,
                spouse_info["Gender"].values[0],
                spouse_info["affected"].values[0],
            )

            # Create spouse connection
            with dot.subgraph() as s:  # type: ignore[attr-defined]
                s.attr(rank="same")
                s.edge(person, spouse_id, dir="none", weight="2", constraint="true")

            if not children.empty:
                # Create single vertical connector
                vertical_connector = f"vert_{person}_{spouse_id}"
                dot.node(
                    vertical_connector,
                    "",
                    shape="point",
                    style="invisible",
                    width="0",
                    height="0",
                )

                # Create horizontal children connector
                children_connector = f"child_{person}_{spouse_id}"
                dot.node(
                    children_connector,
                    "",
                    shape="point",
                    style="invisible",
                    width="0",
                    height="0",
                )

                # Connect vertical line from spouse connection to children connector
                mid_point = f"{person}_{spouse_id}_mid"
                dot.node(
                    mid_point,
                    "",
                    shape="point",
                    style="invisible",
                    width="0",
                    height="0",
                )

                dot.edge(person, mid_point, dir="none", weight="2", constraint="true")
                dot.edge(
                    mid_point,
                    vertical_connector,
                    dir="none",
                    weight="2",
                    constraint="true",
                )
                dot.edge(
                    vertical_connector,
                    children_connector,
                    dir="none",
                    weight="2",
                    constraint="true",
                )

                # Connect children to the horizontal connector
                for child in children.itertuples():
                    add_individual(child[1], child.Gender, child.affected)
                    dot.edge(
                        children_connector,
                        child[1],
                        dir="none",
                        weight="2",  # Changed from headport/tailport approach
                        constraint="true",
                    )
                    processed.append(child[1])

    dot.format = "png"
    file_name = f"{settings.additional_content_settings.image_destination_folder}/pedigree_chart_{int(time.time())}.{settings.additional_content_settings.stimulus_image_format}"

    # Add rendering specifications
    dot.attr(dpi="600")  # Set DPI to 600
    dot.attr(bgcolor="transparent")  # Make background transparent if needed
    dot.attr(margin="0.5")  # Add margins (in inches)

    output_path = dot.render(file_name, cleanup=True)
    return output_path
