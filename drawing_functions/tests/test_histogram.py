import os

import pytest
from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
    draw_histogram,
    draw_histogram_pair,
    draw_histogram_with_dotted_bin,
)
from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
    HistogramBin,
    HistogramDescription,
    HistogramWithDottedBinDescription,
    MultiHistogramDescription,
    Position,
)
from pydantic import ValidationError


# ---------- helpers ----------
def make_bins(start=0, width=10, n=5, freqs=None):
    """Build n contiguous bins within 0–100 with inclusive width."""
    freqs = freqs or [1] * n
    bins = []
    s = start
    for i in range(n):
        e = s + width - 1
        bins.append(HistogramBin(start=s, end=e, frequency=freqs[i]))
        s = e + 1
    return bins


# ---------- rendering (happy paths) ----------
@pytest.mark.drawing_functions
def test_draw_single_histogram_saves_file(tmp_path):
    # 5 bins, width = 10, contiguous 0–49 (meets 0–100 & equal-width constraints)
    stim = HistogramDescription(
        title="Reading Time",
        x_label="Minutes",
        bins=make_bins(width=10, n=5, freqs=[2, 3, 4, 1, 0]),
    )
    out_path = draw_histogram(stim.model_dump_json())
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_saves_file(tmp_path):
    a = HistogramDescription(
        title="Histogram A",
        x_label="Minutes",
        bins=make_bins(width=10, n=5, freqs=[2, 3, 1, 0, 2]),
    )
    b = HistogramDescription(
        title="Histogram B",
        x_label="Minutes",
        bins=make_bins(width=10, n=5, freqs=[1, 2, 3, 1, 0]),
    )
    pair = MultiHistogramDescription(
        title="Comparing Distributions",
        histograms=[a, b],
        correct_histogram_position=Position.LEFT,
    )
    out_path = draw_histogram_pair(pair.model_dump_json())
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_with_distinct_ylabels_saves_file(tmp_path):
    a = HistogramDescription(
        title="Histogram A",
        x_label="Minutes",
        y_label="Count of Students",
        bins=make_bins(width=10, n=5, freqs=[2, 3, 1, 0, 2]),
    )
    b = HistogramDescription(
        title="Histogram B",
        x_label="Minutes",
        y_label="Number of Days",
        bins=make_bins(width=10, n=5, freqs=[1, 2, 3, 1, 0]),
    )
    pair = MultiHistogramDescription(
        title="Comparing Distributions",
        # top-level y_label can be present or blank; it will be ignored when per-panels differ
        y_label="Frequency",
        histograms=[a, b],
        correct_histogram_position=Position.LEFT,
    )
    out_path = draw_histogram_pair(pair.model_dump_json())
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


# ---------- model/validator tests ----------
def test_single_histogram_rejects_null_y_label():
    # Build dict payload so static typing doesn't block None; expect ValidationError
    bins_data = [b.model_dump() for b in make_bins(width=10, n=5)]
    with pytest.raises(ValidationError):
        HistogramDescription.model_validate(
            {
                "title": "T",
                "x_label": "X",
                "y_label": None,  # must be a string per spec
                "bins": bins_data,
            }
        )


def test_single_histogram_rejects_empty_bins():
    with pytest.raises(ValidationError):
        HistogramDescription(title="T", x_label="X", bins=[])


def test_single_histogram_rejects_noncontiguous():
    bins = make_bins(width=10, n=5)
    # Break contiguity: shift the third bin start by +2
    b0, b1, b2, b3, b4 = bins
    bad_third = HistogramBin(start=b1.end + 3, end=b1.end + 12, frequency=1)
    with pytest.raises(ValidationError):
        HistogramDescription(bins=[b0, b1, bad_third, b3, b4])


def test_single_histogram_rejects_wrong_width():
    # width 8 is not allowed (only 5, 10, 20)
    bins = make_bins(width=8, n=5)
    with pytest.raises(ValidationError):
        HistogramDescription(bins=bins)


def test_single_histogram_allows_widths_5_and_20():
    # Positive checks for allowed widths per standard
    stim5 = HistogramDescription(bins=make_bins(width=5, n=6))
    stim20 = HistogramDescription(bins=make_bins(width=20, n=5))
    assert len(stim5.bins) == 6 and len(stim20.bins) == 5


def test_single_histogram_rejects_out_of_range():
    # Push last bin end > 100 to violate 0–100 constraint
    bins = make_bins(start=70, width=10, n=4)  # 70–79, 80–89, 90–99, 100–109 -> invalid
    with pytest.raises(ValidationError):
        HistogramDescription(bins=bins)


def test_bins_count_bounds():
    # Fewer than 5 bins -> reject
    with pytest.raises(ValidationError):
        HistogramDescription(bins=make_bins(width=10, n=4))
    # More than 10 bins -> reject
    with pytest.raises(ValidationError):
        HistogramDescription(bins=make_bins(width=10, n=11))


def test_tick_labels_default_and_custom():
    # Build 5 contiguous bins, width = 10 (0–49), integer frequencies
    bins = make_bins(width=10, n=5, freqs=[1, 2, 3, 4, 5])

    # Default labels should be 'start–end' (en dash)
    stim = HistogramDescription(bins=bins)
    expected_default = [f"{b.start}\u2013{b.end}" for b in bins]
    assert stim.tick_labels() == expected_default

    # Custom label on the 2nd bin overrides default, others remain unchanged
    bins[1] = HistogramBin(start=10, end=19, frequency=2, label="10 to 19")
    stim2 = HistogramDescription(bins=[bins[0], bins[1], bins[2], bins[3], bins[4]])
    labels2 = stim2.tick_labels()
    assert labels2[1] == "10 to 19"
    assert labels2[0] == expected_default[0]
    assert labels2[2:] == expected_default[2:]


def test_multi_histogram_requires_exactly_two():
    a = HistogramDescription(bins=make_bins())
    with pytest.raises(ValidationError):
        MultiHistogramDescription(histograms=[a])  # only one
    with pytest.raises(ValidationError):
        MultiHistogramDescription(histograms=[a, a, a])  # three


def test_multi_histogram_rejects_mismatched_edges():
    a = HistogramDescription(bins=make_bins(width=10, n=5))
    b = HistogramDescription(bins=make_bins(start=5, width=10, n=5))  # shifted edges
    with pytest.raises(ValidationError):
        MultiHistogramDescription(histograms=[a, b])


def test_multi_histogram_accepts_same_edges_diff_freqs():
    a = HistogramDescription(bins=make_bins(width=10, n=5, freqs=[0, 1, 2, 3, 4]))
    b = HistogramDescription(bins=make_bins(width=10, n=5, freqs=[4, 3, 2, 1, 0]))
    pair = MultiHistogramDescription(
        histograms=[a, b], correct_histogram_position=Position.LEFT
    )
    assert len(pair.histograms) == 2


@pytest.mark.drawing_functions
def test_draw_histogram_with_dotted_bin_basic():
    """Test basic histogram with dotted bin functionality."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramWithDottedBinDescription,
    )

    # Create test data similar to the image example
    # Raw data: 22, 55, 45, 0, 78, 13, 91, 53, 21, 61, 16, 13, 35, 19, 41, 78, 92, 37, 55
    raw_data = [
        22,
        55,
        45,
        0,
        78,
        13,
        91,
        53,
        21,
        61,
        16,
        13,
        35,
        19,
        41,
        78,
        92,
        37,
        55,
    ]

    # Create bins with the first bin (0-19) as dotted
    # Count actual frequencies: 0-19: 5 (0,13,16,13,19), 20-39: 4 (22,21,35,37), 40-59: 5 (55,45,53,41,55), 60-79: 3 (78,61,78), 80-99: 2 (91,92)
    bins = [
        HistogramBin(start=0, end=19, frequency=5),  # This will be dotted
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=5),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    stim = HistogramWithDottedBinDescription(
        title="Roses per bush",
        x_label="Number of roses",
        y_label="Number of bushes",
        bins=bins,
        dotted_bin_index=0,  # First bin (0-19) is dotted
        raw_data=raw_data,
    )

    out_path = draw_histogram_with_dotted_bin(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_with_dotted_bin_middle_bin():
    """Test histogram with dotted bin in the middle."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramWithDottedBinDescription,
    )

    # Test data with middle bin missing - let's create data that matches our frequencies
    # 0-19: 5 values, 20-39: 5 values, 40-59: 5 values (dotted), 60-79: 3 values, 80-99: 2 values
    raw_data = [
        5,
        15,
        12,
        18,
        19,  # 0-19: 5 values
        25,
        35,
        22,
        28,
        32,  # 20-39: 5 values
        45,
        55,
        42,
        48,
        52,  # 40-59: 5 values (this will be dotted)
        65,
        75,
        68,  # 60-79: 3 values (fixed: 68 instead of 85)
        95,
        92,
    ]  # 80-99: 2 values

    bins = [
        HistogramBin(start=0, end=19, frequency=5),
        HistogramBin(start=20, end=39, frequency=5),
        HistogramBin(start=40, end=59, frequency=5),  # This will be dotted
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    stim = HistogramWithDottedBinDescription(
        title="Test Scores",
        x_label="Score Range",
        y_label="Number of Students",
        bins=bins,
        dotted_bin_index=2,  # Middle bin (40-59) is dotted
        raw_data=raw_data,
    )

    out_path = draw_histogram_with_dotted_bin(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_with_dotted_bin_last_bin():
    """Test histogram with dotted bin as the last bin."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramWithDottedBinDescription,
    )

    # Test data with last bin missing - create data that matches frequencies
    # 0-19: 3 values, 20-39: 4 values, 40-59: 4 values, 60-79: 4 values, 80-99: 5 values (dotted)
    raw_data = [
        10,
        15,
        12,  # 0-19: 3 values
        20,
        25,
        30,
        35,  # 20-39: 4 values
        40,
        45,
        50,
        55,  # 40-59: 4 values
        60,
        65,
        70,
        75,  # 60-79: 4 values
        80,
        85,
        90,
        95,
        92,
    ]  # 80-99: 5 values (this will be dotted)

    bins = [
        HistogramBin(start=0, end=19, frequency=3),
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=4),
        HistogramBin(start=60, end=79, frequency=4),
        HistogramBin(start=80, end=99, frequency=5),  # This will be dotted
    ]

    stim = HistogramWithDottedBinDescription(
        title="Daily Temperature (°C)",
        x_label="Temperature Range",
        y_label="Number of Days",
        bins=bins,
        dotted_bin_index=4,  # Last bin (80-99) is dotted
        raw_data=raw_data,
    )

    out_path = draw_histogram_with_dotted_bin(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_histogram_with_dotted_bin_validation():
    """Test validation of HistogramWithDottedBinDescription."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
    )
    from pydantic import ValidationError

    # Test invalid dotted_bin_index
    bins = make_bins(width=10, n=5)
    raw_data = [
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
        12,
        18,
        22,
        28,
        32,
        38,
        42,
        48,
        52,
        58,
    ]

    with pytest.raises(ValidationError):
        HistogramWithDottedBinDescription(
            bins=bins,
            dotted_bin_index=10,  # Invalid index
            raw_data=raw_data,
        )

    # Test mismatched frequency - create data that doesn't match the bin frequencies
    wrong_bins = [
        HistogramBin(start=0, end=19, frequency=10),  # Wrong frequency
        HistogramBin(start=20, end=39, frequency=5),
        HistogramBin(start=40, end=59, frequency=5),
        HistogramBin(start=60, end=79, frequency=5),
        HistogramBin(start=80, end=99, frequency=5),
    ]

    with pytest.raises(ValidationError):
        HistogramWithDottedBinDescription(
            bins=wrong_bins,
            dotted_bin_index=0,
            raw_data=[
                5,
                15,
                25,
                35,
                45,
                55,
                65,
                75,
                85,
                95,
            ],  # Only 10 values, but first bin expects 10
        )


def test_get_dotted_bin_correct_frequency():
    """Test the method to get correct frequency for dotted bin."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
    )

    raw_data = [
        22,
        55,
        45,
        0,
        78,
        13,
        91,
        53,
        21,
        61,
        16,
        13,
        35,
        19,
        41,
        78,
        92,
        37,
        55,
    ]

    bins = [
        HistogramBin(
            start=0, end=19, frequency=5
        ),  # Should have 5 values: 0, 13, 16, 13, 19
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=5),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    stim = HistogramWithDottedBinDescription(
        bins=bins,
        dotted_bin_index=0,
        raw_data=raw_data,
    )

    correct_frequency = stim.get_dotted_bin_correct_frequency()
    assert correct_frequency == 5  # 0, 13, 16, 13, 19


@pytest.mark.drawing_functions
def test_draw_histogram_with_dotted_bin_edge_case_first_bin_zero_frequency():
    """Test histogram with dotted bin having zero frequency."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_with_dotted_bin,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramWithDottedBinDescription,
    )

    # Test data where the dotted bin has zero frequency
    raw_data = [
        25,
        35,
        22,
        28,
        32,  # 20-39: 5 values
        45,
        55,
        42,
        48,
        52,  # 40-59: 5 values
        65,
        75,
        68,  # 60-79: 3 values
        95,
        92,
    ]  # 80-99: 2 values
    # Note: No values in 0-19 range

    bins = [
        HistogramBin(
            start=0, end=19, frequency=0
        ),  # This will be dotted with zero frequency
        HistogramBin(start=20, end=39, frequency=5),
        HistogramBin(start=40, end=59, frequency=5),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    stim = HistogramWithDottedBinDescription(
        title="Test Scores with Zero Frequency",
        x_label="Score Range",
        y_label="Number of Students",
        bins=bins,
        dotted_bin_index=0,  # First bin (0-19) is dotted with zero frequency
        raw_data=raw_data,
    )

    out_path = draw_histogram_with_dotted_bin(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_with_dotted_bin_edge_case_high_frequency():
    """Test histogram with very high frequency values."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_with_dotted_bin,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramWithDottedBinDescription,
    )

    # Test data with high frequencies
    raw_data = (
        [5] * 15 + [15] * 10 + [25] * 20 + [35] * 5 + [45] * 12
    )  # High frequency values

    bins = [
        HistogramBin(start=0, end=9, frequency=15),
        HistogramBin(start=10, end=19, frequency=10),
        HistogramBin(start=20, end=29, frequency=20),  # This will be dotted
        HistogramBin(start=30, end=39, frequency=5),
        HistogramBin(start=40, end=49, frequency=12),
    ]

    stim = HistogramWithDottedBinDescription(
        title="High Frequency Test",
        x_label="Value Range",
        y_label="Count",
        bins=bins,
        dotted_bin_index=2,  # Middle bin (20-29) is dotted
        raw_data=raw_data,
    )

    out_path = draw_histogram_with_dotted_bin(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_with_dotted_bin_edge_case_different_bin_widths():
    """Test histogram with different bin widths (5, 10, 20)."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_with_dotted_bin,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramWithDottedBinDescription,
    )

    # Test with bin width 5
    raw_data = [
        2,
        7,
        12,
        17,
        22,
        27,
        32,
        37,
        42,
        47,
        52,
        57,
        62,
        67,
        72,
        77,
        82,
        87,
        92,
        97,
    ]

    bins = [
        HistogramBin(start=0, end=4, frequency=1),
        HistogramBin(start=5, end=9, frequency=1),
        HistogramBin(start=10, end=14, frequency=1),
        HistogramBin(start=15, end=19, frequency=1),
        HistogramBin(start=20, end=24, frequency=1),
        HistogramBin(start=25, end=29, frequency=1),
        HistogramBin(start=30, end=34, frequency=1),
        HistogramBin(start=35, end=39, frequency=1),
        HistogramBin(start=40, end=44, frequency=1),
        HistogramBin(start=45, end=49, frequency=1),  # This will be dotted
    ]

    stim = HistogramWithDottedBinDescription(
        title="Bin Width 5 Test",
        x_label="Value Range",
        y_label="Frequency",
        bins=bins,
        dotted_bin_index=9,  # Last bin (45-49) is dotted
        raw_data=raw_data,
    )

    out_path = draw_histogram_with_dotted_bin(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_histogram_with_dotted_bin_edge_case_validation_errors():
    """Test various validation error cases."""
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramWithDottedBinDescription,
    )
    from pydantic import ValidationError

    # Test negative dotted_bin_index
    bins = make_bins(width=10, n=5)
    raw_data = [
        5,
        15,
        25,
        35,
        45,
        55,
        65,
        75,
        85,
        95,
        12,
        18,
        22,
        28,
        32,
        38,
        42,
        48,
        52,
        58,
    ]

    with pytest.raises(ValidationError):
        HistogramWithDottedBinDescription(
            bins=bins,
            dotted_bin_index=-1,  # Negative index
            raw_data=raw_data,
        )

    # Test dotted_bin_index equal to number of bins
    with pytest.raises(ValidationError):
        HistogramWithDottedBinDescription(
            bins=bins,
            dotted_bin_index=5,  # Equal to number of bins (should be < 5)
            raw_data=raw_data,
        )

    # Test insufficient raw_data
    with pytest.raises(ValidationError):
        HistogramWithDottedBinDescription(
            bins=bins,
            dotted_bin_index=0,
            raw_data=[1, 2, 3, 4, 5, 6, 7, 8, 9],  # Only 9 values, need at least 10
        )


@pytest.mark.drawing_functions
def test_draw_single_histogram_with_custom_colors():
    """Test single histogram with custom color pairs and larger font sizes."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
    )

    # Create test data with 6 bins
    bins = [
        HistogramBin(start=0, end=9, frequency=8),
        HistogramBin(start=10, end=19, frequency=12),
        HistogramBin(start=20, end=29, frequency=15),
        HistogramBin(start=30, end=39, frequency=10),
        HistogramBin(start=40, end=49, frequency=6),
        HistogramBin(start=50, end=59, frequency=4),
    ]

    stim = HistogramDescription(
        title="Student Test Scores Distribution",
        x_label="Score Range",
        y_label="Number of Students",
        bins=bins,
    )

    out_path = draw_histogram(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_single_histogram_edge_case_maximum_bins():
    """Test single histogram with maximum allowed bins (10 bins)."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
    )

    # Create test data with 10 bins (maximum allowed)
    bins = [
        HistogramBin(start=0, end=4, frequency=3),
        HistogramBin(start=5, end=9, frequency=5),
        HistogramBin(start=10, end=14, frequency=7),
        HistogramBin(start=15, end=19, frequency=9),
        HistogramBin(start=20, end=24, frequency=11),
        HistogramBin(start=25, end=29, frequency=13),
        HistogramBin(start=30, end=34, frequency=11),
        HistogramBin(start=35, end=39, frequency=9),
        HistogramBin(start=40, end=44, frequency=7),
        HistogramBin(start=45, end=49, frequency=5),
    ]

    stim = HistogramDescription(
        title="Maximum Bins Test",
        x_label="Value Range",
        y_label="Frequency",
        bins=bins,
    )

    out_path = draw_histogram(stim)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_with_high_frequencies():
    """Test histogram pair with high frequency values and custom colors."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
    )

    # Create two histograms with high frequencies
    bins_a = [
        HistogramBin(start=0, end=19, frequency=25),
        HistogramBin(start=20, end=39, frequency=30),
        HistogramBin(start=40, end=59, frequency=35),
        HistogramBin(start=60, end=79, frequency=20),
        HistogramBin(start=80, end=99, frequency=15),
    ]

    bins_b = [
        HistogramBin(start=0, end=19, frequency=15),
        HistogramBin(start=20, end=39, frequency=20),
        HistogramBin(start=40, end=59, frequency=25),
        HistogramBin(start=60, end=79, frequency=30),
        HistogramBin(start=80, end=99, frequency=35),
    ]

    hist_a = HistogramDescription(
        title="Group A Performance",
        x_label="Score Range",
        y_label="Number of Students",
        bins=bins_a,
    )

    hist_b = HistogramDescription(
        title="Group B Performance",
        x_label="Score Range",
        y_label="Number of Students",
        bins=bins_b,
    )

    pair = MultiHistogramDescription(
        title="Comparing Group Performance",
        histograms=[hist_a, hist_b],
        correct_histogram_position=Position.LEFT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_edge_case_zero_frequencies():
    """Test histogram pair with some zero frequency bins."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
    )

    # Create two histograms with some zero frequencies
    bins_a = [
        HistogramBin(start=0, end=19, frequency=0),  # Zero frequency
        HistogramBin(start=20, end=39, frequency=5),
        HistogramBin(start=40, end=59, frequency=0),  # Zero frequency
        HistogramBin(start=60, end=79, frequency=8),
        HistogramBin(start=80, end=99, frequency=3),
    ]

    bins_b = [
        HistogramBin(start=0, end=19, frequency=2),
        HistogramBin(start=20, end=39, frequency=0),  # Zero frequency
        HistogramBin(start=40, end=59, frequency=6),
        HistogramBin(start=60, end=79, frequency=0),  # Zero frequency
        HistogramBin(start=80, end=99, frequency=4),
    ]

    hist_a = HistogramDescription(
        title="Dataset A",
        x_label="Value Range",
        y_label="Count",
        bins=bins_a,
    )

    hist_b = HistogramDescription(
        title="Dataset B",
        x_label="Value Range",
        y_label="Count",
        bins=bins_b,
    )

    pair = MultiHistogramDescription(
        title="Comparing Datasets with Zero Frequencies",
        histograms=[hist_a, hist_b],
        correct_histogram_position=Position.LEFT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_with_correct_position_left():
    """Test histogram pair with correct histogram on the left (histograms[0])."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
        Position,
    )

    # Create two histograms
    bins_a = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=3),
        HistogramBin(start=60, end=79, frequency=1),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    bins_b = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    hist_a = HistogramDescription(
        title="Class A Test Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_a,
    )

    hist_b = HistogramDescription(
        title="Class B Test Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_b,
    )

    pair = MultiHistogramDescription(
        title="Quiz Score Distributions",
        histograms=[hist_a, hist_b],
        correct_histogram_position=Position.LEFT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_with_correct_position_right():
    """Test histogram pair with correct histogram on the right (histograms[1])."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
        Position,
    )

    # Create two histograms
    bins_a = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    bins_b = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=3),
        HistogramBin(start=60, end=79, frequency=1),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    hist_a = HistogramDescription(
        title="Class A Test Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_a,
    )

    hist_b = HistogramDescription(
        title="Class B Test Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_b,
    )

    pair = MultiHistogramDescription(
        title="Quiz Score Distributions",
        histograms=[hist_a, hist_b],
        correct_histogram_position=Position.RIGHT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_with_random_position():
    """Test histogram pair with random correct position."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
    )

    # Create two histograms
    bins_a = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=3),
        HistogramBin(start=60, end=79, frequency=1),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    bins_b = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    hist_a = HistogramDescription(
        title="Class A Test Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_a,
    )

    hist_b = HistogramDescription(
        title="Class B Test Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_b,
    )

    # Use the random position helper
    pair = MultiHistogramDescription.with_random_position(
        title="Quiz Score Distributions",
        histograms=[hist_a, hist_b],
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_same_data_left_correct():
    """Test histogram pair with SAME data but left histogram marked as correct."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
        Position,
    )

    # Create IDENTICAL histogram data
    bins_data = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    # Both histograms have the SAME data
    hist_left = HistogramDescription(
        title="Group A Quiz Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_data,
    )

    hist_right = HistogramDescription(
        title="Group B Quiz Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_data,  # SAME data
    )

    # Left histogram is correct
    pair = MultiHistogramDescription(
        title="Quiz Score Distributions - Left Correct",
        histograms=[hist_left, hist_right],
        correct_histogram_position=Position.LEFT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_same_data_right_correct():
    """Test histogram pair with SAME data but right histogram marked as correct."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
        Position,
    )

    # Create IDENTICAL histogram data
    bins_data = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    # Both histograms have the SAME data
    hist_left = HistogramDescription(
        title="Group A Quiz Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_data,
    )

    hist_right = HistogramDescription(
        title="Group B Quiz Scores",
        x_label="Score Range",
        y_label="Frequency",
        bins=bins_data,  # SAME data
    )

    # Right histogram is correct
    pair = MultiHistogramDescription(
        title="Quiz Score Distributions - Right Correct",
        histograms=[hist_left, hist_right],
        correct_histogram_position=Position.RIGHT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_different_data_left_correct():
    """Test histogram pair with DIFFERENT data and left histogram marked as correct."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
        Position,
    )

    # Create DIFFERENT histogram data
    bins_left = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=3),
        HistogramBin(start=60, end=79, frequency=1),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    bins_right = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    hist_left = HistogramDescription(
        title="Method A Results",
        x_label="Value Range",
        y_label="Count",
        bins=bins_left,
    )

    hist_right = HistogramDescription(
        title="Method B Results",
        x_label="Value Range",
        y_label="Count",
        bins=bins_right,
    )

    # Left histogram is correct
    pair = MultiHistogramDescription(
        title="Comparing Two Methods - Left Correct",
        histograms=[hist_left, hist_right],
        correct_histogram_position=Position.LEFT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


@pytest.mark.drawing_functions
def test_draw_histogram_pair_different_data_right_correct():
    """Test histogram pair with DIFFERENT data and right histogram marked as correct."""
    from content_generators.additional_content.stimulus_image.drawing_functions.histogram import (
        draw_histogram_pair,
    )
    from content_generators.additional_content.stimulus_image.stimulus_descriptions.histograms import (
        HistogramBin,
        HistogramDescription,
        MultiHistogramDescription,
        Position,
    )

    # Create DIFFERENT histogram data (same as previous test but flipped)
    bins_left = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=2),
        HistogramBin(start=40, end=59, frequency=2),
        HistogramBin(start=60, end=79, frequency=3),
        HistogramBin(start=80, end=99, frequency=1),
    ]

    bins_right = [
        HistogramBin(start=0, end=19, frequency=4),
        HistogramBin(start=20, end=39, frequency=4),
        HistogramBin(start=40, end=59, frequency=3),
        HistogramBin(start=60, end=79, frequency=1),
        HistogramBin(start=80, end=99, frequency=2),
    ]

    hist_left = HistogramDescription(
        title="Method A Results",
        x_label="Value Range",
        y_label="Count",
        bins=bins_left,
    )

    hist_right = HistogramDescription(
        title="Method B Results",
        x_label="Value Range",
        y_label="Count",
        bins=bins_right,
    )

    # Right histogram is correct
    pair = MultiHistogramDescription(
        title="Comparing Two Methods - Right Correct",
        histograms=[hist_left, hist_right],
        correct_histogram_position=Position.RIGHT,
    )

    out_path = draw_histogram_pair(pair)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
