# Contributing to PODS-AI

Thank you for your interest in contributing to PODS-AI!

## Code of Conduct

Please read and make sure you understand our [Code of Conduct](/CODE_OF_CONDUCT.md) before contributing to the project.

## Coding Conventions

### License Header

Every new source file must start with a copyright notice and SPDX license identifier. For Python scripts:

```python
#!/usr/bin/env python3
# Copyright (c) PODS-AI contributors
# SPDX-License-Identifier: MIT
```

### Language and Style

- **Language**: Python 3.12+
- **Spellings**: Use American English spellings (e.g., "color" not "colour", "recognize" not "recognise").
- **Typing**: Use `dataclasses`, built-in generic types (`list`, `tuple`), and `typing` utilities (`Optional`) with type annotations throughout.
- **Docstrings**: All public functions and classes must have Google-style or plain docstrings describing parameters and return values.
- **Constants**: Define module-level constants for magic values (e.g., `NEAR_MIN`, `MAX_DETECTION_PAGES`).

### Comments

- Comments must end in punctuation (typically a period).
- **DON'T** include commented-out code. Remove dead code rather than commenting it out.

### Naming

- **DON'T** use abbreviations unless they are already well-known terms (e.g., "app", "info"), or are already required for use by developers (e.g., "min", "max", "args").
  - Bad: `num_widgets` — use `widget_count` instead.
  - Bad: `opt_widgets` — use `option_widgets` or `optional_widgets` instead.

### Error Handling

- Catch exceptions at I/O boundaries (network, file), print a descriptive error message, and return an empty list or `None` as appropriate—do not let exceptions propagate silently.

### Project Metadata

- Keep `pods-ai.pyproj` updated so every `.py` file in the repository is listed there.
- Update `README.md` when behavior or usage changes, especially when adding a new script.
- Update `.github/copilot-instructions.md` when repository structure, workflows, or contributor guidance changes.
