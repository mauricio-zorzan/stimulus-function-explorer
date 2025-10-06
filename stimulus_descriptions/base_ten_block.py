from typing import TYPE_CHECKING, List, Literal

from pydantic import BaseModel, Field, validator

from .stimulus_description import StimulusDescription

if TYPE_CHECKING:
    from content_generators.question_generator.schemas.context import (
        QuestionGeneratorContext,
    )


class BaseTenBlock(BaseModel):
    """
    Represents a set of base 10 blocks with a numerical value.
    """

    value: int = Field(
        ...,
        description="The numerical value represented by the base 10 blocks.",
        ge=1,
        le=1000,
    )
    display_as_decimal: bool = Field(
        default=False,
        description="If True, the value will be displayed as a decimal where ones=1.0, tens=0.1, hundreds=0.01",
    )


StimulusOperation = Literal["addition", "subtraction", "divide", "multiply"]


class BaseTenBlockStimulus(StimulusDescription):
    """
    One or Two sets of vertically stacked base 10 blocks with an optional operation between them.
    The hundreds are shown in cyan, the tens in green, and the ones in orange.

    By default, numerical values are displayed as labels below each block.
    The show_values field can be set to False to display only the visual blocks without labels.
    If the operation is used, the values are shown as labels separated by the symbol alongside the blocks.
    When display_as_decimal is True, the values are shown as decimals where hundreds=1.0, tens=0.1, ones=0.01.
    All blocks in a stimulus must use the same display mode (either all decimal or all non-decimal).
    """

    blocks: List[BaseTenBlock] = Field(
        ...,
        description="List of values that are represented in base 10 blocks.",
        max_length=2,
    )
    operation: StimulusOperation | None = Field(
        default=None,
        description="The operation being performed between the blocks. Represented by the appropriate symbol between the 2 values for the blocks. If used the values are shown as labels separated by the symbol.",
    )
    show_values: bool = Field(
        default=True,
        description="If True, numerical values will be displayed as labels below each set of blocks. If False, only the visual blocks are shown without numerical labels.",
    )

    @validator("blocks")
    def validate_consistent_display_mode(cls, blocks):
        if len(blocks) > 1:
            first_block_mode = blocks[0].display_as_decimal
            for block in blocks[1:]:
                if block.display_as_decimal != first_block_mode:
                    raise ValueError(
                        "All blocks must use the same display mode (either all decimal or all non-decimal)"
                    )
        return blocks

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        match pipeline_context.payload.placeholders.standard_id:
            case "CCSS.MATH.CONTENT.1.NBT.A.1+2":
                # Read and write numerals up to 120.
                assert len(self.blocks) == 1, "Only one block is allowed."
                assert (
                    self.blocks[0].value <= 120
                ), "The block's value must be less than or equal to 120."
            case "CCSS.MATH.CONTENT.1.NBT.C.4+1":
                # Within 100, add a two-digit number and a one-digit number using concrete models or drawings.
                assert self.operation == "addition", "The operation must be addition."
                assert all(
                    block.value < 100 for block in self.blocks
                ), "Each value must be smaller than 100."
                one_digit_count = sum(
                    1 for block in self.blocks if len(str(block.value)) == 1
                )
                two_digit_count = sum(
                    1 for block in self.blocks if len(str(block.value)) == 2
                )
                assert (
                    one_digit_count == 1 and two_digit_count == 1
                ), "One value must be 2 digits and the other value must be 1 digit."
            case "CCSS.MATH.CONTENT.1.NBT.C.6+1":
                # Subtract multiples of 10 in the range 10-90 from multiples of 10 in the
                # range 10-90 with positive or zero differences using concrete models or drawings.
                assert (
                    self.operation == "subtraction"
                ), "The operation must be subtraction."
                assert all(
                    block.value % 10 == 0 for block in self.blocks
                ), "All values must be multiples of 10."
                assert all(
                    10 <= block.value <= 90 for block in self.blocks
                ), "All values must be in the range 10-90."
                assert (
                    self.blocks[0].value - self.blocks[1].value >= 0
                ), "The difference between the values must be positive or zero."
            case "CCSS.MATH.CONTENT.2.NBT.B.7+1":
                # Add within 1000, using concrete models or drawings.
                assert self.operation == "addition", "The operation must be addition."
                assert all(
                    block.value <= 1000 for block in self.blocks
                ), "All values must be within 1000."
            case "CCSS.MATH.CONTENT.2.NBT.B.7+4":
                # Subtract within 1000, using concrete models or drawings.
                assert (
                    self.operation == "subtraction"
                ), "The operation must be subtraction."
                assert all(
                    block.value <= 1000 for block in self.blocks
                ), "All values must be within 1000."


class BaseTenBlockGridStimulus(StimulusDescription):
    """
    A grid of identical base 10 blocks arranged in 2 columns.
    Maximum of 6 figures (3 rows x 2 columns).
    The hundreds are shown in cyan, the tens in green, and the ones in orange.
    No value labels are displayed - only the visual block representations.

    This is useful for division problems where you want to show equal groups,
    for example 440÷4 would show 4 identical groups of 110 each.
    """

    block_value: int = Field(
        ...,
        description="The numerical value represented by each base 10 block in the grid.",
        ge=1,
        le=1000,
    )
    display_as_decimal: bool = Field(
        default=False,
        description="If True, the value will be displayed as a decimal where hundreds=1.0, tens=0.1, ones=0.01",
    )
    count: int = Field(
        ...,
        description="The number of identical base 10 blocks to display in the grid.",
        ge=1,
        le=6,
    )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        # Add any specific validation rules for grid stimuli if needed


class BaseTenBlockDivisionStimulus(StimulusDescription):
    """
    Base ten blocks arranged to show division layout.
    The dividend is represented by base ten blocks arranged in groups according to the divisor.
    The hundreds are shown in cyan, the tens in green, and the ones in orange.

    For example, 165 ÷ 11 would show the blocks representing 165 arranged in 11 groups,
    demonstrating the division process visually.
    """

    dividend: int = Field(
        ...,
        description="The number being divided (represented by base ten blocks).",
        ge=1,
        le=1000,
    )
    divisor: int = Field(
        ...,
        description="The number to divide by (determines the arrangement/grouping).",
        ge=1,
        le=20,
    )

    def pipeline_validate(self, pipeline_context: "QuestionGeneratorContext"):
        super().pipeline_validate(pipeline_context)
        # Ensure the division makes sense visually
        if self.dividend < self.divisor:
            raise ValueError(
                "Dividend must be greater than or equal to divisor for meaningful visual representation"
            )


if __name__ == "__main__":
    BaseTenBlockStimulus.generate_assistant_function_schema(type="mcq3")
    BaseTenBlockStimulus.generate_assistant_function_schema(type="mcq4")
    BaseTenBlockGridStimulus.generate_assistant_function_schema(type="mcq3")
    BaseTenBlockGridStimulus.generate_assistant_function_schema(type="mcq4")
