#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
"""Generate README.md from its template and the latest model comparison."""

from pathlib import Path

PLACEHOLDER = "<!-- MODEL_COMPARISON -->"
SUMMARY_TITLE = "Model Comparison Summary"


def extract_model_comparison(comparison_text: str) -> str:
    """Return the comparison summary and confusion matrices from CLI output."""
    lines = comparison_text.splitlines()
    try:
        summary_index = lines.index(SUMMARY_TITLE)
    except ValueError as error:
        raise ValueError(
            f"Could not find {SUMMARY_TITLE!r} in model-comparison.txt."
        ) from error

    section_start = summary_index - 1
    if (
        section_start < 0
        or not lines[section_start]
        or set(lines[section_start]) != {"="}
    ):
        raise ValueError("Could not find the comparison summary separator.")

    return "\n".join(lines[section_start:]).rstrip()


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
    comparison_path = repository_root / "model-comparison.txt"
    readme_path = repository_root / "README.md"
    template_text = template_path.read_text(encoding="utf-8")
    comparison_text = comparison_path.read_text(encoding="utf-8")
    readme_path.write_text(
        render_readme(template_text, comparison_text),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
