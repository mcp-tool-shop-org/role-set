"""The kind-B puzzle generator: code builds and proves the puzzle, a model only words it (docs/generator-design.md).

Slice 1: `build` (generate-then-reduce over the solver's rule forms), `shortcuts` (the registry of named lazy readings,
shared with puzzle-check), `recipes` (T10, T11 and T12) and `render` (template wording that round-trips through
check_text)."""
