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
| `notebook_build/assemble.py` | The generator. The `REPLACE` / `APPEND_*` tables below it are the merge map. |
| `notebook_build/cells/` | **Source of truth.** One file per cell that differs from the base. |
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
  empty, every cell's source equals the base's, cell for cell. The one thing
  that never carries over is *execution state*: `outputs` and `execution_count`
  are cleared on every code cell, so the artifact never claims to have been run.
  `test_base_carries_no_saved_execution_state` fails loudly if the base ever
  gains saved outputs, because at that point the pass-through is no longer
  lossless and this claim would quietly become false.
- **Runs are deterministic.** Building twice gives a byte-identical file.
- **No partial artifacts.** The output is written to `.tmp` and `os.replace`d,
  so an interrupted build cannot truncate a good artifact.
- **No silent no-ops.** Every guard rail raises `ValueError` rather than using
  `assert`, which `python -O` strips — a stripped guard is exactly the silent
  no-op this design exists to prevent.

## Reproducing the build from a clean checkout

`assemble.py` builds from the **working-tree** copy of `batspot-train.ipynb`,
and that file currently carries **uncommitted local modifications** (it is a
read-only parent and this build does not commit it). So a fresh clone will not
reproduce the committed `batspot-train-merged.ipynb` byte-for-byte: the clone's
base differs from the one the artifact was generated from. To reproduce it you
need those local modifications; otherwise rebuild, which is correct but yields
a different artifact until the base changes. This is recorded rather than papered
over — see the task-1 report, Finding 4.

## Tests read the delivered notebook, never a copy

*(Future work — Task 2 and Task 13. Neither file exists yet.)*

`extract_cells.py` will write each cell of `batspot-train-merged.ipynb` out to
`src_NN.py` / `md_NN.md`, and every suite will `exec` those extracted files, so a
suite tests **the artifact that ships** rather than a stale scratch copy.

`redgreen.py` will mutate one fix at a time in a scratch extraction and re-run
the matching suite, pointing it at the mutant through the `NEW_CELLS` environment
variable (`NEW_CELLS=<dir> venv/bin/python notebook_build/tests/test_x.py`). A
suite that cannot be made to fail when its fix is reverted is not testing
anything.
