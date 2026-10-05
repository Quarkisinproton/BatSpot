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

`cell_defs()` isolates individual definitions out of a cell for suites that
cannot exec the whole thing (the data-discovery cell globs paths and prints; the
training cell needs a stubbed `predict_proba`). It pulls in every module-level
binding the requested functions actually reference, by closure analysis, so it
does not depend on the names a cell happens to use today -- base cell 6 spells
its regex constant `_TAPE_RE`, Space Bunny's `_TS_RE` behind `import re as _re`,
and both must work.
"""
import ast
import atexit
import builtins
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


def dir_map(out_dir: str, caller: str = 'extracted cells') -> dict:
    """Map an already-extracted directory back to {index: path}, ignoring anything else.

    `caller` names who is asking, so the message stays accurate: `dir_map` is a general
    helper and `NEW_CELLS` is only one of its callers.
    """
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f'{caller}: {out_dir!r} is not a directory')
    found = {}
    for name in sorted(os.listdir(out_dir)):
        stem, ext = os.path.splitext(name)
        m = _EXTRACTED.fullmatch(stem)
        if m and ext in ('.py', '.md'):
            found[int(m.group(2))] = os.path.join(out_dir, name)
    if not found:
        raise FileNotFoundError(f'{caller}: no src_NN.py / md_NN.md files in {out_dir}')
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
        return dir_map(override, 'NEW_CELLS')
    if not os.path.exists(MERGED):
        raise FileNotFoundError(f'{MERGED} does not exist; run notebook_build/assemble.py')
    return _cells_of(MERGED)


def base_cells() -> dict:
    """Cells of the read-only base parent, for attribution only."""
    return _cells_of(BASE)


# --- isolating individual definitions out of a cell -----------------------------------------
#
# A suite that wants one function cannot always exec the whole cell: the data-discovery cell
# globs DATA_DIR and prints a split report, the training cell needs a stubbed `predict_proba`.
# These helpers pull out just the requested definitions -- and, by closure analysis, every
# module-level binding they reference.

def _bound_names(node: ast.AST) -> set:
    """Names a definition binds locally: params, assignments, nested defs, imports, loop vars.

    Correctly identifies *free* variables for the purpose of pulling in module-level
    dependencies. Deliberately over-approximates -- a name bound anywhere in the function is
    treated as local even if some loads precede the binding -- because under-approximating
    would make the analysis report a false dependency, which is the failure that bites.
    """
    bound = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(sub.name)
        elif isinstance(sub, ast.arg):
            bound.add(sub.arg)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                bound.add((alias.asname or alias.name).split('.')[0])
    return bound


def _free_names(node: ast.AST) -> set:
    """Names a definition reads but does not bind; builtins and locals excluded."""
    loaded = {n.id for n in ast.walk(node)
              if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loaded - _bound_names(node) - set(dir(builtins))


def _top_level_bindings(body) -> dict:
    """Map every name bound at a cell's top level to the statement that binds it."""
    binds = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            binds[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                binds.setdefault((alias.asname or alias.name).split('.')[0], node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        binds.setdefault(n.id, node)
    return binds


def cell_defs(path: str, names, ns: dict) -> dict:
    """Exec the named top-level definitions out of a cell, plus what they reference.

    `names` are the definitions the caller wants (e.g. `{'recording_id'}`). Everything they
    read at module level -- a regex constant, an `import x as y`, a helper function -- is
    pulled in transitively, so this keeps working when a port renames a private helper.
    Caller-supplied `ns` wins over the cell's own binding for any name present in both, which
    is how a suite substitutes a stub for something it cannot run.

    Returns `ns` with the requested definitions present. A name that cannot be found raises
    KeyError naming what was missing, rather than failing later as a NameError deep inside a
    call that happens to be the first thing the suite exercises.
    """
    body = ast.parse(open(path, encoding='utf-8').read()).body
    available = _top_level_bindings(body)
    wanted = list(names)
    for name in wanted:
        if name not in available:
            raise KeyError(f'{os.path.basename(path)} defines no top-level {name!r} '
                           f'(it has: {sorted(available)[:12]})')
    # Resolve a node's own dependencies BEFORE exec'ing it: `_TAPE_RE = re.compile(...)` needs
    # the cell's `import re` to have run first, or it raises NameError at the assignment.
    done, pulled = set(), []

    def pull(name):
        if name in done:
            return
        done.add(name)
        node = available.get(name)
        if node is None:
            return                      # supplied by the caller, or genuinely absent (a global)
        for dep in sorted(_free_names(node)):
            if dep not in ns:
                pull(dep)
        pulled.append(node)
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)

    for name in wanted:
        pull(name)

    # Everything a pulled definition reads must now be bound -- by the cell, or by the caller.
    # This walks every node that was actually exec'd, not only the requested names: a helper
    # pulled in transitively can read something neither the cell nor the caller supplies, and
    # that would otherwise surface as a NameError at the first call, part-way through a suite
    # that has already printed some checks. Naming it here turns that into one error at
    # extraction time.
    unresolved = set()
    for node in pulled:
        unresolved |= {dep for dep in _free_names(node) if dep not in ns}
    if unresolved:
        raise KeyError(f'{os.path.basename(path)}: pulling {sorted(wanted)} needs '
                       f'{sorted(unresolved)}, which the cell neither defines nor is given')
    return ns


def cell_method(path: str, cls_name: str, meth_name: str, ns: dict):
    """Exec one method out of a cell's class body; return the function.

    The method's free names are NOT chased: it is exec'd into the namespace the caller supplies
    (typically the one `cell_defs` built), so it sees the same imports and helpers the rest of
    the cell would have. A method reading something absent from `ns` raises NameError when it is
    called, which is why a suite should call this inside the same guard as `cell_defs`.
    """
    for node in ast.parse(open(path, encoding='utf-8').read()).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and sub.name == meth_name:
                    sub.decorator_list = []
                    exec(compile(ast.Module(body=[sub], type_ignores=[]), path, 'exec'), ns)
                    return ns[meth_name]
    raise KeyError(f'{os.path.basename(path)} has no {cls_name}.{meth_name}')
