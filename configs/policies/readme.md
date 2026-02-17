# Policies (Chemistry + Geometry + Graph)

This folder defines **reproducible, versioned “data meaning”** for the QSAR pipeline.
A *policy* is anything that, if changed, should invalidate cached artifacts such as:
- standardized SMILES
- chosen protonation/tautomer form
- conformer bank entries
- 3D graph tensors

If you run preprocessing twice with the same input + same policies, you should get the same artifacts.

---

## Folder layout

---

## How to use policies in code

Typical flow:

1. Load policies from this directory
2. Compute a **policy hash** (or use `version.yaml` + full config hash)
3. Use that hash to name/cache artifacts:
   - `data/cache/conformers/<policy_hash>/...`
   - `data/cache/graphs_3d/<policy_hash>/...`

### Recommended cache key inputs
- canonical SMILES (after standardization)
- policy versions (from `version.yaml`)
- full policy content (the actual yaml values)
- software versions (RDKit version, model code version)

---

## Why `version.yaml` exists

`version.yaml` pins a semantic version for each policy group:

- `standardize`
- `protonation`
- `tautomer`
- `conformer`
- `graph`

Bump a version when you change a policy in a way that changes outputs. Examples:

### Bump `conformer` when
- changing ETKDG version
- changing `max_confs`, `prune_rms_thresh`
- changing minimizer type/iters
- changing which conformer you select/keep

### Bump `graph` when
- changing edge construction (radius vs knn)
- changing distance cutoffs
- adding/removing geometry features
- changing feature definitions

### Bump `standardize` when
- changing salt removal rules
- changing neutralization/canonicalization rules
- changing allow/deny SMARTS

---

## File descriptions

### `standardize.yaml`
Defines how raw SMILES are cleaned and canonicalized.
Goal: **consistent identity** for the same molecule.

### `protonation.yaml`
Defines how you choose protonation/protomer states (pH targets, method, variant limits).

### `tautomer.yaml`
Defines tautomer enumeration and how the “primary” tautomer is selected.

### `conformer.yaml`
Defines 3D conformer generation parameters (ETKDG settings, pruning, minimization, selection).

### `graph.yaml`
Defines 3D graph construction:
- how edges are built (radius/knn)
- what geometry features to include
- what atom/bond features are used

---

## Practical rules (so your caches don’t lie)

- Never silently change a policy without bumping its version.
- Store policy hash + versions inside every cache shard/manifest.
- Keep policies *small and explicit*; avoid hidden defaults in code.
- Treat policy changes as “data schema changes”.

---

## Suggested workflow

1. Start with conservative defaults (the provided YAMLs).
2. Run preprocessing to build conformer+graph caches.
3. Train models using those caches.
4. Only after baseline works, iterate on policy changes:
   - change a YAML
   - bump its version
   - regenerate only the affected caches
   - retrain/evaluate