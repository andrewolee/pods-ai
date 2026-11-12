#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Generate README.md from its template and the latest model comparison."""

from pathlib import Path

PLACEHOLDER = "<!-- MODEL_COMPARISON -->"
SUMMARY_TITLE = "Model Comparison Summary"
COMPARISON_MODELS = ("orcahello", "oldpodsai", "podsai")


def extract_model_comparison(comparison_text: str) -> str:
    """Return the comparison summary and confusion matrices from CLI output."""
    lines = comparison_text.splitlines()
    try:
        summary_index = lines.index(SUMMARY_TITLE)
    except ValueError as error:
        raise ValueError(
            f"Could not find {SUMMARY_TITLE!r} in model comparison output."
        ) from error

    section_start = summary_index - 1
    if (
        section_start < 0
        or not lines[section_start]
        or set(lines[section_start]) != {"="}
    ):
        raise ValueError("Could not find the comparison summary separator.")

    return "\n".join(lines[section_start:]).rstrip()


def combine_model_comparisons(comparison_texts: list[str]) -> str:
    """Combine per-model tables and matrices, keeping shared definitions once."""
    header = ""
    definitions = ""
    rows = []
    matrices = []
    for comparison_text in comparison_texts:
        section = extract_model_comparison(comparison_text)
        separator = section.splitlines()[0]
        table, matrix_marker, matrix = section.partition("\nConfusion Matrix for ")
        model_header, table_body = table.split(
            f"\n{'-' * len(separator)}\n", 1
        )
        model_rows, model_definitions = table_body.split(f"\n{separator}", 1)
        if not header:
            header = model_header
            definitions = model_definitions.rstrip()
        elif model_header != header or model_definitions.rstrip() != definitions:
            raise ValueError("Model comparison headers and definitions must match.")
        rows.append(model_rows)
        if matrix_marker:
            matrices.append(f"Confusion Matrix for {matrix}")

    if not rows:
        raise ValueError("At least one model comparison is required.")
    return (
        f"{header}\n{'-' * len(separator)}\n"
        + "\n".join(rows)
        + f"\n{separator}{definitions}"
        + ("\n\n" + "\n\n".join(matrices) if matrices else "")
    )


def render_readme(template_text: str, comparison_text: str) -> str:
    """Replace the template placeholder with the latest formatted comparison."""
    if template_text.count(PLACEHOLDER) != 1:
        raise ValueError(
            f"README.md.template must contain exactly one {PLACEHOLDER} marker."
        )

    comparison_block = f"```\n{extract_model_comparison(comparison_text)}\n```"
    return template_text.replace(PLACEHOLDER, comparison_block)


def main() -> None:
    """Build README.md in the repository root."""
    repository_root = Path(__file__).resolve().parents[1]
    template_path = repository_root / "README.md.template"
    readme_path = repository_root / "README.md"
    template_text = template_path.read_text(encoding="utf-8")
    comparison_text = combine_model_comparisons([
        (repository_root / f"model-comparison-{model}.txt").read_text(
            encoding="utf-8"
        )
        for model in COMPARISON_MODELS
    ])
    readme_path.write_text(
        render_readme(template_text, comparison_text),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
