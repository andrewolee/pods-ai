# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Tests for the README generator."""

import runpy
from pathlib import Path

import pytest

BUILD_README = runpy.run_path(
    Path(__file__).resolve().parents[1] / "src" / "build-readme.py"
)
extract_model_comparison = BUILD_README["extract_model_comparison"]
render_readme = BUILD_README["render_readme"]


def test_extract_model_comparison_skips_per_sample_output():
    comparison = (
        "Evaluating model: podsai\n"
        "  predicted='resident' -> correct\n"
        "\n"
        "====\n"
        "Model Comparison Summary\n"
        "====\n"
        "podsai 31.7%\n"
        "\n"
        "Confusion Matrix\n"
        "resident 2\n"
    )

    assert extract_model_comparison(comparison) == (
        "====\n"
        "Model Comparison Summary\n"
        "====\n"
        "podsai 31.7%\n"
        "\n"
        "Confusion Matrix\n"
        "resident 2"
    )


def test_render_readme_replaces_single_placeholder():
    template = "# README\n\n<!-- MODEL_COMPARISON -->\n"
    comparison = "====\nModel Comparison Summary\n====\n"

    assert render_readme(template, comparison) == (
        "# README\n\n```\n====\nModel Comparison Summary\n====\n```\n"
    )


@pytest.mark.parametrize(
    "template",
    ["No marker", "<!-- MODEL_COMPARISON -->\n<!-- MODEL_COMPARISON -->"],
)
def test_render_readme_requires_exactly_one_placeholder(template):
    with pytest.raises(ValueError, match="exactly one"):
        render_readme(template, "====\nModel Comparison Summary\n====")


def test_extract_model_comparison_requires_summary():
    with pytest.raises(ValueError, match="Model Comparison Summary"):
        extract_model_comparison("No summary")


def test_extract_model_comparison_requires_separator():
    with pytest.raises(ValueError, match="separator"):
        extract_model_comparison("Model Comparison Summary")
