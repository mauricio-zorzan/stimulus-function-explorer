from __future__ import annotations

from enum import Enum
from typing import List, Optional

# Bring in your stimulus base(s)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.stimulus_description import (
    StimulusDescription,
)
from pydantic import BaseModel, Field, field_serializer, model_validator


class Position(Enum):
    LEFT = "left"
    RIGHT = "right"


ALLOWED_BIN_WIDTHS = {5, 10, 20}


class HistogramBin(BaseModel):
    """
    A single histogram bin with INCLUSIVE endpoints (e.g., 0–9 has width 10).
    """

    start: int = Field(..., ge=0, description="Inclusive start (e.g., 0).")
    end: int = Field(..., ge=0, description="Inclusive end (e.g., 9).")
    frequency: int = Field(..., ge=0, description="Non-negative integer count.")
    label: Optional[str] = Field(
        default=None,
        description="Optional label; if None, UI shows 'start–end'.",
    )

    @model_validator(mode="after")
    def _check_bin(self):
        if self.end < self.start:
            raise ValueError("Bin end must be ≥ start.")
        return self


class HistogramDescription(StimulusDescription):
    """
    Single histogram description (Grade-agnostic, but compatible with G6 constraints
    via validators below).
    """

    title: Optional[str] = Field(default=None, description="Figure title.")
    x_label: Optional[str] = Field(default=None, description="X-axis label.")
    # Must be a string (usually 'Frequency') — not None
    y_label: str = Field(default="Frequency", description="Y-axis label.")
    bins: List[HistogramBin] = Field(
        ...,
        min_length=5,
        max_length=10,
        description="Ordered, contiguous bins with equal width in {5,10,20}, range within 0–100.",
    )

    @model_validator(mode="after")
    def _validate_bins(self):
        bs = self.bins
        if not bs:
            raise ValueError("Bins cannot be empty.")

        # equal width (inclusive)
        widths = {b.end - b.start + 1 for b in bs}
        if len(widths) != 1:
            raise ValueError("All bins must have equal width.")
        w = next(iter(widths))
        if w not in ALLOWED_BIN_WIDTHS:
            raise ValueError(f"Bin width {w} not in {sorted(ALLOWED_BIN_WIDTHS)}.")

        # contiguous and in-range
        if bs[0].start < 0 or bs[-1].end > 100:
            raise ValueError("Histogram must be within 0–100.")
        for i in range(1, len(bs)):
            if bs[i].start != bs[i - 1].end + 1:
                raise ValueError("Bins must be contiguous (next.start = prev.end + 1).")

        return self

    # Convenience for renderer
    def effective_title(self) -> str:
        return self.title or "Interpreting a Histogram"

    def effective_x_label(self) -> str:
        return self.x_label or "Value Range"

    def tick_labels(self) -> List[str]:
        return [
            b.label if b.label is not None else f"{b.start}\u2013{b.end}"
            for b in self.bins
        ]


class MultiHistogramDescription(StimulusDescription):
    """
    Exactly two histograms to be rendered side-by-side for comparison.
    Enforces same bin edges across both panels for fair comparison.
    """

    title: Optional[str] = Field(default=None, description="Overall figure title.")
    y_label: str = Field(default="Frequency", description="Shared y-axis label.")
    histograms: List[HistogramDescription] = Field(
        ..., min_length=2, max_length=2, description="Exactly two histograms."
    )
    correct_histogram_position: Position = Field(
        ...,
        description="Which side should display the correct histogram (left or right).",
    )

    @model_validator(mode="after")
    def _validate_pair(self):
        hs = self.histograms
        a, b = hs[0], hs[1]

        # same width within each (already checked), and same width across both
        wa = a.bins[0].end - a.bins[0].start + 1
        wb = b.bins[0].end - b.bins[0].start + 1
        if wa != wb:
            raise ValueError("Both histograms must share the same bin width.")

        # same bin edges sequence (strict equality)
        edges_a = [(x.start, x.end) for x in a.bins]
        edges_b = [(x.start, x.end) for x in b.bins]
        if edges_a != edges_b:
            raise ValueError(
                "Both histograms must have IDENTICAL bins (same start/end sequence)."
            )

        return self

    def effective_title(self) -> str:
        return self.title or "Comparing Two Histograms"

    def tick_labels(self) -> List[str]:
        # Safe to take from either histogram (bins are identical by validation)
        return self.histograms[0].tick_labels()

    @field_serializer("correct_histogram_position")
    def serialize_position(self, value: Position) -> str:
        """Serialize Position enum to string for JSON compatibility."""
        return value.value

    @classmethod
    def with_random_position(cls, **kwargs) -> "MultiHistogramDescription":
        """Create a MultiHistogramDescription with random correct histogram position."""
        import random

        position = random.choice([Position.LEFT, Position.RIGHT])
        kwargs["correct_histogram_position"] = position
        return cls(**kwargs)


class HistogramWithDottedBinDescription(StimulusDescription):
    """
    Histogram description where one bin is shown as a dotted outline that needs to be completed.
    This is used for interactive questions where students need to determine the missing frequency.
    """

    title: Optional[str] = Field(default=None, description="Figure title.")
    x_label: Optional[str] = Field(default=None, description="X-axis label.")
    y_label: str = Field(default="Frequency", description="Y-axis label.")
    bins: List[HistogramBin] = Field(
        ...,
        min_length=5,
        max_length=10,
        description="Ordered, contiguous bins with equal width in {5,10,20}, range within 0–100.",
    )
    dotted_bin_index: int = Field(
        ...,
        ge=0,
        description="Index of the bin to show as dotted (0-based). This bin represents the missing frequency that needs to be determined.",
    )
    raw_data: List[int] = Field(
        ...,
        min_length=10,
        description="Raw data points used to calculate frequencies. This helps validate the dotted bin frequency.",
    )

    @model_validator(mode="after")
    def _validate_dotted_bin_and_data(self):
        # First run the standard histogram validation
        self._validate_bins()

        # Validate dotted_bin_index
        if self.dotted_bin_index >= len(self.bins):
            raise ValueError(
                f"dotted_bin_index {self.dotted_bin_index} is out of range for {len(self.bins)} bins"
            )

        # Validate that raw_data frequencies match the bins (except for the dotted bin)
        for i, bin_obj in enumerate(self.bins):
            if i == self.dotted_bin_index:
                continue  # Skip validation for dotted bin

            # Count data points in this bin range
            bin_count = sum(
                1 for value in self.raw_data if bin_obj.start <= value <= bin_obj.end
            )

            if bin_count != bin_obj.frequency:
                raise ValueError(
                    f"Bin {i} ({bin_obj.start}-{bin_obj.end}) frequency {bin_obj.frequency} "
                    f"does not match actual data count {bin_count}"
                )

        return self

    # Inherit validation from HistogramDescription
    def _validate_bins(self):
        bs = self.bins
        if not bs:
            raise ValueError("Bins cannot be empty.")

        # equal width (inclusive)
        widths = {b.end - b.start + 1 for b in bs}
        if len(widths) != 1:
            raise ValueError("All bins must have equal width.")
        w = next(iter(widths))
        if w not in ALLOWED_BIN_WIDTHS:
            raise ValueError(f"Bin width {w} not in {sorted(ALLOWED_BIN_WIDTHS)}.")

        # contiguous and in-range
        if bs[0].start < 0 or bs[-1].end > 100:
            raise ValueError("Histogram must be within 0–100.")
        for i in range(1, len(bs)):
            if bs[i].start != bs[i - 1].end + 1:
                raise ValueError("Bins must be contiguous (next.start = prev.end + 1).")

        return self

    # Convenience methods
    def effective_title(self) -> str:
        return self.title or "Complete the Histogram"

    def effective_x_label(self) -> str:
        return self.x_label or "Value Range"

    def tick_labels(self) -> List[str]:
        return [
            b.label if b.label is not None else f"{b.start}\u2013{b.end}"
            for b in self.bins
        ]

    def get_dotted_bin_correct_frequency(self) -> int:
        """Calculate the correct frequency for the dotted bin based on raw_data."""
        dotted_bin = self.bins[self.dotted_bin_index]
        return sum(
            1 for value in self.raw_data if dotted_bin.start <= value <= dotted_bin.end
        )


__all__ = [
    "Position",
    "HistogramBin",
    "HistogramDescription",
    "MultiHistogramDescription",
    "HistogramWithDottedBinDescription",
]

if __name__ == "__main__":
    HistogramDescription.generate_assistant_function_schema("mcq4")
    MultiHistogramDescription.generate_assistant_function_schema("mcq4")
    HistogramWithDottedBinDescription.generate_assistant_function_schema("mcq4")
