# notebook_build — the merged BatSpot notebook is generated

`batspot-train-merged.ipynb` at the repo root is a **build artifact**.
**Never hand-edit it.** Your edits will be silently discarded the next time
`assemble.py` runs, and — worse — they will not appear in any diff, so a bug
fixed only in the artifact looks fixed right up until the next port lands.

## The source of truth is `cells/`

| Path | Role |
|---|---|
| `batspot-train.ipynb` | **Read-only base.** 24 cells. Copied verbatim except where `REPLACE` says otherwise. |
| `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb` | **Read-only parent.** A *source* for ports; `assemble.py` never reads it. |
| `notebook_build/cells/` | **Source of truth.** One file per cell that differs from the base. |
| `notebook_build/assemble.py` | The generator. The `REPLACE` / `APPEND_*` tables below it are the merge map. |
| `batspot-train-merged.ipynb` | Generated output. Commit it, but only as the product of a run. |

Neither parent notebook is modified. If a change seems to need a change there,
it does not — add a cell source under `cells/` and register it.

## Workflow

1. Edit or create a file in `cells/`. The **filename declares the cell type**:
   `.md` is markdown, anything else (conventionally `.py`) is code.
2. Register it in `assemble.py`:
   - `REPLACE[<base cell index>] = '<file>'` to replace a base cell.
   - `APPEND_MD += ['<file>.md']` / `APPEND_CODE += ['<file>.py']` to add cells
     after the base cells. Markdown appends first, then code.
3. Rebuild:

   ```bash
   venv/bin/python notebook_build/assemble.py
   ```

4. Run the tests:

   ```bash
   venv/bin/python notebook_build/tests/test_assemble.py
   ```

`build()` asserts loudly rather than failing quietly: a `REPLACE` index outside
the base, a `.md` file pointed at a code cell, a missing cell source, or a base
that is no longer 24 cells all stop the build. Every substitution is exact and
counted, so a silent no-op port is impossible.

## Why generated rather than hand-edited

The merge combines a verified 24-cell pipeline with 18 independently reviewed
fixes plus new measurement phases. Patching that by hand would make "what
changed relative to which parent" unanswerable. Because the artifact is built:

- **The diff is the review.** Base vs merged shows exactly which cells were
  replaced and by how much — `assemble.py` prints those counts on every run.
- **The empty build is a lossless pass-through.** With `REPLACE` and `APPEND_*`
  empty, the output equals the base cell for cell. That is what makes any later
  diff attributable to a port rather than to drift.
- **Runs are deterministic.** Building twice gives a byte-identical file.
- **No stale execution state.** Outputs and `execution_count` are cleared on every
  code cell, so the artifact never claims to have been run when it was not.

## Tests read the delivered notebook, never a copy

`notebook_build/tests/extract_cells.py` writes each cell of
`batspot-train-merged.ipynb` out to `src_NN.py` / `md_NN.md`, and every suite
`exec`s those extracted files. So a suite tests **the artifact that ships**. A
suite can never quietly pass against a stale scratch copy.

`redgreen.py` mutates one fix at a time in a scratch extraction and re-runs the
matching suite, pointing it at the mutant through the `NEW_CELLS` environment
variable (`NEW_CELLS=<dir> venv/bin/python notebook_build/tests/test_x.py`). A
suite that cannot be made to fail when its fix is reverted is not testing
anything.