#!/usr/bin/env python3
"""Pull cells out of a notebook so a suite can exec the cells that ship.

Every suite in this directory tests `batspot-train-merged.ipynb` by reading the
cells out of the delivered artifact:

    idx = merged_cells()                       # {cell index: path}
    exec(compile(open(idx[5]).read(), idx[5], 'exec'), ns)

Nothing here copies, reformats or repairs a cell. A suite therefore cannot
accidentally test a hand-edited or stale version of the code -- the only way to
change what a suite sees is to change the notebook, which is the point.

Extraction writes source only. Saved `outputs` are never written, so no suite
can pass by matching a printout of a previous run's code.

`NEW_CELLS=<dir>` points the suites at an already-extracted directory instead of
extracting the notebook. That is the red-green hook: `redgreen.py` extracts the
artifact once, reverts a single fix in a scratch copy of that directory, and
re-runs the matching suite against the mutant.

Cell indices are the notebook's own, so a suite names the cell it is testing
(`CELL_CONFIG`, `CELL_DATASET`, ...) rather than a filename that a reordering
would silently invalidate.
"""
import atexit
import json
import os
import re
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# The artifact under test, and the read-only parent it is built from. Tests may
# read the base parent for ATTRIBUTION ("the unported cell behaved like this"),
# never in place of the merged cell.
MERGED = os.path.join(REPO, 'batspot-train-merged.ipynb')
BASE = os.path.join(REPO, 'batspot-train.ipynb')

_EXTRACTED = re.compile(r'(src|md)_(\d+)')


def extract(notebook_path: str, out_dir: str) -> dict:
    """Write each cell of `notebook_path` to out_dir; return {index: written path}.

    Markdown cells become `md_NN.md`, code cells `src_NN.py`, with NN the cell's
    position in the notebook. Anything already matching those names in out_dir
    is removed first, so a directory reused across extractions can never hand back
    a cell the notebook no longer has.
    """
    with open(notebook_path, encoding='utf-8') as f:
        nb = json.load(f)
    if 'cells' not in nb:
        raise ValueError(f'{notebook_path} has no "cells"; not a notebook')

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        stem, ext = os.path.splitext(name)
        if ext in ('.py', '.md') and _EXTRACTED.fullmatch(stem):
            os.remove(os.path.join(out_dir, name))

    written = {}
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') not in ('code', 'markdown'):
            raise ValueError(f'cell {i} of {notebook_path} has cell_type '
                             f'{cell.get("cell_type")!r}; expected code or markdown')
        prefix = 'src' if cell['cell_type'] == 'code' else 'md'
        path = os.path.join(out_dir, f'{prefix}_{i:02d}{".py" if prefix == "src" else ".md"}')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(''.join(cell['source']))
        written[i] = path
    return written


def dir_map(out_dir: str) -> dict:
    """Map an already-extracted directory back to {index: path}, ignoring anything else."""
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f'NEW_CELLS={out_dir!r} is not a directory')
    found = {}
    for name in sorted(os.listdir(out_dir)):
        stem, ext = os.path.splitext(name)
        m = _EXTRACTED.fullmatch(stem)
        if m and ext in ('.py', '.md'):
            found[int(m.group(2))] = os.path.join(out_dir, name)
    if not found:
        raise FileNotFoundError(f'no src_NN.py / md_NN.md files in {out_dir}')
    return found


def _cells_of(notebook_path: str) -> dict:
    """Extract `notebook_path` into a scratch dir that is removed on exit."""
    d = tempfile.mkdtemp(prefix='batspot-cells-')
    atexit.register(shutil.rmtree, d, True)
    return extract(notebook_path, d)


def merged_cells() -> dict:
    """Cells of the artifact under test (or of NEW_CELLS, when redgreen.py sets it)."""
    override = os.environ.get('NEW_CELLS')
    if override:
        return dir_map(override)
    if not os.path.exists(MERGED):
        raise FileNotFoundError(f'{MERGED} does not exist; run notebook_build/assemble.py')
    return _cells_of(MERGED)


def base_cells() -> dict:
    """Cells of the read-only base parent, for attribution only."""
    return _cells_of(BASE)