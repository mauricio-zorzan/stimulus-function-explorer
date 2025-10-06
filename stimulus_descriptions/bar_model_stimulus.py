from typing import TYPE_CHECKING, List, Literal, Optional, Union

from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
    StimulusDescriptionList,
    StimulusDescriptionProtocol,
)
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class BarSection(StimulusDescription):
    """
    Represents a single section of the bar model.
    """
    value: Optional[Union[int, float, str]] = Field(
        None, 
        description="The value for this section (can be number or variable like 'x')"
    )
    label: Optional[str] = Field(
        None,
        description="Optional label for this section"
    )
    color: Optional[str] = Field(
        None,
        description="Color for this section (hex code or color name)"
    )
    pattern: Optional[Literal["solid", "hatched", "dotted"]] = Field(
        "solid",
        description="Fill pattern for this section"
    )
    highlighted: bool = Field(
        False,
        description="Whether this section should be highlighted/emphasized"
    )


class BarModel(StimulusDescription):
    """
    Represents a bar model (tape diagram) for mathematical problem solving.
    """
    title: Optional[str] = Field(
        None,
        description="Optional title for the bar model"
    )
    total_value: Optional[Union[int, float, str]] = Field(
        None,
        description="The total value represented by the entire bar"
    )
    total_label: Optional[str] = Field(
        None,
        description="Label for the total (e.g., 'Total candies', '9 candies')"
    )
    sections: List[BarSection] = Field(
        ...,
        description="The sections that make up the bar model",
        min_length=1,
        max_length=8
    )
    orientation: Literal["horizontal", "vertical"] = Field(
        "horizontal",
        description="Orientation of the bar model"
    )
    show_section_divisions: bool = Field(
        True,
        description="Whether to show lines dividing the sections"
    )
    show_section_values: bool = Field(
        True,
        description="Whether to show values inside or near each section"
    )
    show_total_value: bool = Field(
        True,
        description="Whether to show the total value above/beside the bar"
    )
    equal_sections: bool = Field(
        False,
        description="Whether all sections represent equal parts (for fraction problems)"
    )
    section_width_mode: Literal["proportional", "equal"] = Field(
        "equal",
        description="Whether sections are sized proportionally to their values or equally"
    )
    problem_context: Optional[str] = Field(
        None,
        description="Brief context about what the bar model represents (e.g., 'candies', 'dollars', 'students')"
    )

    @field_validator("sections")
    def validate_sections(cls, v):
        """Validate the sections."""
        if len(v) < 1:
            raise ValueError("Bar model must have at least 1 section")
        if len(v) > 8:
            raise ValueError("Bar model cannot have more than 8 sections for clarity")
        return v


class ComparisonBarModel(StimulusDescription):
    """
    Represents multiple bar models for comparison problems.
    """
    title: Optional[str] = Field(
        None,
        description="Title for the comparison"
    )
    bars: List[BarModel] = Field(
        ...,
        description="List of bar models to compare",
        min_length=2,
        max_length=4
    )
    comparison_type: Literal["difference", "ratio", "part_whole"] = Field(
        "difference",
        description="Type of comparison being shown"
    )
    alignment: Literal["left", "center"] = Field(
        "left",
        description="How to align the bars"
    )


class BarModelStimulus(StimulusDescription, StimulusDescriptionProtocol):
    """
    Main stimulus for bar model problems.
    """
    model_type: Literal["single_bar", "comparison_bars"] = Field(
        ...,
        description="Type of bar model stimulus"
    )
    single_bar: Optional[BarModel] = Field(
        None,
        description="Single bar model (required if model_type is 'single_bar')"
    )
    comparison_bars: Optional[ComparisonBarModel] = Field(
        None,
        description="Comparison bar models (required if model_type is 'comparison_bars')"
    )
    instruction_text: Optional[str] = Field(
        None,
        description="Instruction text to display with the model"
    )
    not_to_scale_note: str = Field(
        "Figure not drawn to scale.",
        description="Note indicating the figure is not to scale"
    )

    @field_validator("single_bar", "comparison_bars")
    def validate_model_consistency(cls, v, info):
        """Ensure the correct model is provided based on model_type."""
        model_type = info.data.get("model_type")
        field_name = info.field_name
        
        if model_type == "single_bar" and field_name == "single_bar" and v is None:
            raise ValueError("single_bar is required when model_type is 'single_bar'")
        elif model_type == "comparison_bars" and field_name == "comparison_bars" and v is None:
            raise ValueError("comparison_bars is required when model_type is 'comparison_bars'")
        
        return v

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        """
        Override this method to implement additional pipeline validation here
        This is called before the assistant validators
        """
        if pipeline_context is None:
            raise ValueError("Validation context is None")
        super().pipeline_validate(pipeline_context)

    def generate_image(self):
        """
        Override this method to implement custom image generation logic.
        """
        pass


class BarModelStimulusList(StimulusDescriptionList[BarModelStimulus]):
    """
    List wrapper for bar model stimuli.
    """
    root: List[BarModelStimulus] = Field(
        ...,
        description="A list of bar model stimuli to be displayed.",
        min_length=1,
        max_length=1,
    )


# Helper function to create common bar model patterns
def create_equal_parts_model(total: int, num_parts: int, 
                           context: str = "items",
                           highlight_part: Optional[int] = None) -> BarModelStimulus:
    """
    Helper to create a bar model with equal parts (like the candy example).
    
    Args:
        total: Total value
        num_parts: Number of equal parts to divide into
        context: What the model represents (e.g., "candies", "dollars")
        highlight_part: Optional index of part to highlight (0-based)
    
    Returns:
        BarModelStimulus with equal parts
    """
    part_value = total / num_parts
    sections = []
    
    for i in range(num_parts):
        sections.append(
            BarSection(
                value=part_value,
                highlighted=(i == highlight_part) if highlight_part is not None else False
            )
        )
    
    return BarModelStimulus(
        model_type="single_bar",
        single_bar=BarModel(
            total_value=total,
            total_label=f"{total} {context}",
            sections=sections,
            section_width_mode="proportional",
            problem_context=context
        )
    )


def create_part_whole_model(parts: List[Union[int, str]], 
                          total: Optional[Union[int, str]] = None,
                          context: str = "items") -> BarModelStimulus:
    """
    Helper to create a part-whole bar model.
    
    Args:
        parts: List of part values (can include unknowns like "?")
        total: Total value (optional)
        context: What the items represent
    
    Returns:
        BarModelStimulus for part-whole problem
    """
    # Different colors for each part
    category_colors = [
        "#FFE4E1",  # Light pink
        "#E0F6FF",  # Light sky blue  
        "#F0FFF0",  # Light mint
        "#FFF8DC",  # Light cream
        "#E6E6FA",  # Light lavender
        "#FFEFD5",  # Light peach
    ]
    
    sections = []
    for i, part in enumerate(parts):
        sections.append(BarSection(
            value=part,
            color=category_colors[i % len(category_colors)]
        ))
    
    bar_model = BarModel(
        total_value=total,
        total_label=f"{total} {context}" if total else None,
        sections=sections,
        section_width_mode="proportional" if all(isinstance(p, (int, float)) for p in parts) else "equal",
        problem_context=context
    )
    
    return BarModelStimulus(
        model_type="single_bar",
        single_bar=bar_model
    )


if __name__ == "__main__":
    # Generate the schema
    BarModelStimulusList.generate_assistant_function_schema("sat-math")
    
    # Example usage for the candy problem
    candy_model = create_equal_parts_model(
        total=9,
        num_parts=3,
        context="candies",
        highlight_part=0  # Highlight the part that gets eaten
    )