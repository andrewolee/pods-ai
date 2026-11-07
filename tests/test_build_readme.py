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
combine_model_comparisons = BUILD_README["combine_model_comparisons"]


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


def model_comparison(model: str, skipped: bool = False) -> str:
    """Return representative single-model CLI output."""
    return (
        f"Evaluating model: {model}\n"
        "====================\n"
        "Model Comparison Summary\n"
        "====================\n"
        "Model      Accuracy\n"
        "--------------------\n"
        f"{model} 90.0%\n"
        + ("  (1 skipped due to missing WAV or inference error)\n" if skipped else "")
        + "====================\n"
        "\nDefinitions:\n"
        "  Accuracy = Correct / Evaluated\n"
        f"\nConfusion Matrix for {model} (rows=actual, cols=predicted):\n"
        "resident 2\n"
    )


def test_combine_model_comparisons_preserves_order_and_shared_sections():
    models = ["orcahello", "oldpodsai", "podsai"]
    combined = combine_model_comparisons([
        model_comparison(model, skipped=model == "oldpodsai")
        for model in models
    ])

    assert combined.count("Model Comparison Summary") == 1
    assert combined.count("Model      Accuracy") == 1
    assert combined.count("Definitions:") == 1
    assert "Evaluating model:" not in combined
    assert (
        "orcahello 90.0%\noldpodsai 90.0%\n"
        "  (1 skipped due to missing WAV or inference error)\n"
        "podsai 90.0%"
    ) in combined
    assert combined.count("resident 2") == 3
    assert [
        line for line in combined.splitlines()
        if line.startswith("Confusion Matrix for ")
    ] == [
        f"Confusion Matrix for {model} (rows=actual, cols=predicted):"
        for model in models
    ]


def test_combine_single_model_matches_extracted_summary():
    comparison = model_comparison("podsai")
    assert combine_model_comparisons([comparison]) == extract_model_comparison(comparison)


def test_combine_model_comparisons_allows_missing_confusion_matrix():
    comparison = model_comparison("orcahello", skipped=True).split(
        "\nConfusion Matrix for ", 1
    )[0]
    assert combine_model_comparisons([comparison]) == extract_model_comparison(comparison)
    combined = combine_model_comparisons([comparison, model_comparison("podsai")])
    assert "orcahello 90.0%" in combined
    assert combined.count("Confusion Matrix for ") == 1
    assert "Confusion Matrix for podsai" in combined


def test_combine_model_comparisons_requires_input():
    with pytest.raises(ValueError, match="At least one"):
        combine_model_comparisons([])


@pytest.mark.parametrize("replacement", ["Different header", "Different definitions"])
def test_combine_model_comparisons_rejects_inconsistent_format(replacement):
    comparison = model_comparison("podsai").replace(
        "Model      Accuracy" if replacement == "Different header"
        else "Accuracy = Correct / Evaluated",
        replacement,
    )
    with pytest.raises(ValueError, match="must match"):
        combine_model_comparisons([model_comparison("orcahello"), comparison])


def test_main_reads_all_model_files_and_writes_readme(tmp_path, monkeypatch):
    repository_root = tmp_path / "repository"
    source_path = repository_root / "src" / "build-readme.py"
    source_path.parent.mkdir(parents=True)
    template = "# README\n\n<!-- MODEL_COMPARISON -->\n"
    (repository_root / "README.md.template").write_text(template, encoding="utf-8")
    comparisons = []
    for model in BUILD_README["COMPARISON_MODELS"]:
        comparison = model_comparison(model)
        comparisons.append(comparison)
        (repository_root / f"model-comparison-{model}.txt").write_text(
            comparison, encoding="utf-8"
        )

    main = BUILD_README["main"]
    monkeypatch.setitem(main.__globals__, "__file__", str(source_path))
    main()
    assert (repository_root / "README.md").read_text(encoding="utf-8") == (
        render_readme(template, combine_model_comparisons(comparisons))
    )
    (repository_root / "model-comparison-podsai.txt").unlink()
    with pytest.raises(FileNotFoundError):
        main()
