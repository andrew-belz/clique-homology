# Change Summary

## What Changed
- Colors are now treated as strings (`str`) across the stats engine.
- Optional `allowed_colors` palette support was added to:
  - `betti_numbers(...)`
  - `random_coloring(...)`
  - `null_distribution(...)`
- Palette matching is exact and case-sensitive when `allowed_colors` is provided.
- `DEFAULT_COLOR_PALETTE` is now exported from `clique_homology`.

## Why This Matters
- The color contract is now explicit and easier to enforce per graph/dataset.
- Invalid color inputs are caught early with clear runtime errors.

## Action Required for Collaborators
- Ensure all node colors are strings.
- Ensure `len(colors) == number_of_nodes`.
- If you want strict validation, pass `allowed_colors` and match case exactly.
- Import `DEFAULT_COLOR_PALETTE` if you want a starting palette to edit.

## Behavior Changes
- Non-string colors now raise `TypeError`.
- Colors outside `allowed_colors` now raise `ValueError` (when palette validation is used).
- `random_coloring(..., allowed_colors=...)` samples from `allowed_colors` in both proportional and non-proportional modes.
- `null_distribution(...)` returns the computed distribution list.

## API Updates
- `betti_numbers(G, colors, method="clique", allowed_colors=None)`
  - Valid methods are now `"clique"` and `"subgraph"`.
- `random_coloring(colors, proportional=False, allowed_colors=None)`
- `null_distribution(graph, coloring, iterations=100, allowed_colors=None)`
- `Color` is now a `str` type alias.
- `DEFAULT_COLOR_PALETTE` is exported from `clique_homology`.

## Files Changed
- `src/clique_homology/stats_engine/random_coloring.py`
- `src/clique_homology/stats_engine/betti_numbers.py`
- `src/clique_homology/stats_engine/null_distribution.py`
- `src/clique_homology/__init__.py`
- `tests/test_color_contract.py`
- `docs/CONTRACT.md`
