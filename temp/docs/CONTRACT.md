## Purpose

All input sources (Excel, CSV, and future connectors) must be normalized into one canonical graph data shape before running homology/statistics code. This ensures the stats engine only handles one internal format.

## Canonical schema

### CanonicalGraphData

```text
CanonicalGraphData:
  nodes: list[int]
  edges: list[tuple[int, int]]
  node_colors: list[str]
  node_attributes: dict[int, dict[str, Any]] | None
  id_map:
    original_to_canonical: dict[Any, int]
    canonical_to_original: dict[int, Any]
```

### Field definitions
- `nodes`: canonical integer node IDs exactly `0..n-1`.
- `edges`: undirected node pairs `(u, v)` using canonical IDs, with `u < v`.
- `node_colors`: list aligned by canonical node ID (`node_colors[i]` is color for node `i`).
- `node_attributes`: optional metadata keyed by canonical node ID.
- `id_map`: two-way mapping between original dataset IDs and canonical IDs for traceability.

## Current code assumptions (inventory)

This section reflects current implementation assumptions and should converge to the canonical schema above.

### Files reviewed
- `src/clique_homology/stats_engine/betti_numbers.py`
- `src/clique_homology/stats_engine/null_distribution.py`

### Node ID assumptions
- Node IDs are assumed to be dense integer indices `0..N-1`.
- `colors` is interpreted positionally: index `i` is the color for node `i` (`enumerate(node_colors)` in `betti_numbers.py`).
- Those indices are passed directly to NetworKit subgraph APIs, so they must be valid NetworKit node IDs.

### Color assumptions
- Colors are represented as `str` values that literally name the color (for example, `"RED"`, `"BLUE"`).
- `colors` is a node-aligned list: `colors[i]` is the color string for node `i`.
- Each graph/dataset can define its own allowed color palette (editable as needed for that graph).
- Every node color must be an exact, case-sensitive member of that graph's allowed palette when palette validation is enabled.
- `null_distribution.py` assumes `random_coloring(coloring, allowed_colors=...)` returns a list of color strings compatible with `betti_numbers`.

### Graph type assumptions
- Runtime graph type is NetworKit `Graph` (not generic graph interfaces).
- Code depends on NetworKit-only APIs (`nk.clique.MaximalCliques`, `nk.graphtools.subgraphFromNodes`).
- `null_distribution.py` type-hints `networkit.graph.Graph`.
- Note: `betti_numbers.py` docstring mentions possible `nx.Graph`, but implementation does not support NetworkX directly.

## Invariants

1. Canonical node IDs are contiguous integers `0..n-1`.
2. `len(node_colors) == n` where `n = len(nodes)`.
3. Every edge satisfies `0 <= u < v < n`.
4. Self-loops are not allowed.
5. Duplicate undirected edges are not allowed.
6. `node_attributes` keys (if present) must be a subset of canonical node IDs.
7. `id_map.original_to_canonical` and `id_map.canonical_to_original` must be exact inverses (bijection).
8. Color values are strings and case-sensitive; if palette validation is enabled, every color must be in the allowed palette.

## Validation errors

Use deterministic error code format:
- `ERROR_CODE: message (field=<field>, value=<value>)`

Required validation failures:
- `NON_CANONICAL_NODE_IDS`: nodes are not contiguous `0..n-1`.
- `COLOR_LENGTH_MISMATCH`: `len(node_colors)` does not equal `len(nodes)`.
- `EDGE_ENDPOINT_OUT_OF_RANGE`: edge endpoint is outside canonical node range.
- `SELF_LOOP_NOT_ALLOWED`: edge has `u == v`.
- `DUPLICATE_UNDIRECTED_EDGE`: duplicate edge exists after normalization.
- `NON_STRING_COLOR`: one or more color values are not `str`.
- `COLOR_NOT_IN_ALLOWED_PALETTE`: color is not in allowed palette (when palette validation is enabled).
- `INVALID_NODE_ATTRIBUTE_KEY`: attribute key is not a canonical node ID.
- `INVALID_ID_MAP`: `id_map` is missing entries or is not bijective.

### Acceptance scenarios
- Accept empty graph (`nodes=[]`, `edges=[]`, `node_colors=[]`, empty maps).
- Accept graph with external IDs after remapping to canonical IDs.
- Reject self-loop edges such as `(2, 2)`.
- Reject duplicate undirected edges such as `(1, 3)` and `(3, 1)` after normalization.
- Reject color length mismatch.
- Reject non-bijective `id_map`.
- Reject `node_attributes` keys outside canonical node IDs.

## Conversion to Networkit

1. Build NetworKit graph with `n = len(nodes)`.
2. Add each normalized, deduplicated edge `(u, v)` once.
3. Pass canonical `node_colors` list unchanged into stats functions.
4. Keep `id_map` outside the NetworKit graph object for traceability in outputs/logging.

## Versioning
