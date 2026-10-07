#!/usr/bin/env python3
"""Scaffold tests for assemble.py.

Contracts the whole build rests on:

  1. to_source() round-trips text exactly, so a cell written to cells/ and read
     back out of the notebook is byte-identical to the source file.
  2. to_source() does not emit a trailing empty line, matching nbformat.
  3. The base notebook is still the 24-cell notebook the merge map is written
     against. If this fails, every REPLACE index in assemble.py is wrong.
  4. A replacement and an append actually land, and every cell that was not
     named is byte-identical to the base.
  5. Each bad REPLACE/APPEND entry is rejected loudly rather than silently
     producing a notebook with the wrong cell in the wrong place.
  6. Stale `outputs` / `execution_count` from a base that was run are stripped,
     so the artifact never claims to have been executed.
  7. The committed batspot-train-merged.ipynb matches a fresh build, so the
     artifact is never stale relative to assemble.py.
  8. Every name a code cell LOADS is bound by an earlier cell. Cells share one
     kernel namespace, so a cell reading a name nothing before it defines raises
     NameError when the notebook is run -- and nothing else here notices, because
     every suite that reads such a cell supplies the missing names in its own
     fixture. This check currently FAILS, at cell 14, by design: see the comment
     on test_every_loaded_name_is_bound_by_an_earlier_cell.

Run: venv/bin/python notebook_build/tests/test_assemble.py
"""
import ast
import builtins
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble
import extract_cells

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


# --- the three contracts the plan pins (verbatim) -----------------------------------------

def test_to_source_roundtrip():
    text = "a = 1\nb = 2"
    assert ''.join(assemble.to_source(text)) == text


def test_to_source_trailing_newline_dropped_on_last_line():
    assert assemble.to_source("x\n") == ["x\n"]


def test_base_notebook_has_expected_shape():
    nb = json.load(open(assemble.BASE))
    assert len(nb['cells']) == 24, f"expected the 24-cell base, got {len(nb['cells'])}"
    assert nb['cells'][0]['cell_type'] == 'markdown'
    assert nb['cells'][19]['cell_type'] == 'markdown', 'cell 19 is the inference intro'
    assert nb['cells'][23]['cell_type'] == 'code'


def test_base_carries_no_saved_execution_state():
    """Pins the 'lossless pass-through' claim made in README.md and assemble.py.

    build() clears outputs/execution_count on every code cell, so if the base ever
    gains saved outputs the pass-through is no longer lossless -- source-identical,
    but 22 of 24 cells would differ. Nothing else notices: the staleness check
    compares the committed artifact against a fresh, equally-stripped build. This
    check is what turns that silent gap into a loud failure."""
    nb = json.load(open(assemble.BASE))
    dirty = [i for i, c in enumerate(nb['cells'])
             if c['cell_type'] == 'code' and (c.get('outputs') or c.get('execution_count'))]
    assert not dirty, (
        f'base code cells {dirty} carry saved outputs/execution_count; build() strips '
        'them, so the empty build is no longer lossless. Commit the base without '
        'outputs (nbformat clears them on save), or drop the pass-through claim.')


# ---------------------------------------------------------------------------
# Why this check exists
#
# This is the check whose absence let an artifact through that cannot execute. Cell 14 (the
# export cell) reads `cls_best_member`, `cls_members` and `UNKNOWN_MODEL`, which no cell of the
# notebook defines -- the ensemble cell that will define them is Task 8. Nothing in the build
# noticed: build() checks that each replacement is the right cell TYPE and the right byte
# content, and every suite that reads cell 14 supplies the missing names itself in a fixture.
# The artifact only fails when a human runs it top to bottom.
#
# So this is checked at the artifact level, on every build. The contract is deliberately strict:
# cells share one kernel namespace top to bottom, so a name a cell loads and no earlier cell
# binds is a NameError waiting to happen. It FAILS today, at cell 14, and that is the intended
# signal -- not a bug in this suite. It goes green when Task 8 lands the ensemble cell.
# ---------------------------------------------------------------------------

_MAGIC = ('!', '%', '?')


def parse_cell(src):
    """Parse a cell's source, blanking IPython magics (`!pip`, `%cd`) that are not Python.

    Returns (tree, n_magics_blanked). One cell in the base is a `!pip install`; without this
    the analysis would abort on the first cell it reached and check nothing.
    """
    try:
        return ast.parse(src), 0
    except SyntaxError:
        pass
    lines = src.split('\n')
    n = sum(1 for ln in lines if ln.strip().startswith(_MAGIC))
    blanked = '\n'.join('' if ln.strip().startswith(_MAGIC) else ln for ln in lines)
    try:
        return ast.parse(blanked), n
    except SyntaxError as e:
        raise ValueError(f'cell does not parse even after blanking {n} magic line(s): {e}')


def bound_names(tree):
    """Every name the cell binds at any level: defs, classes, imports, stores, args, excepts."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split('.')[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out.update(n.names)
    return out


def unbound_loads(cells):
    """{cell index: sorted names that cell loads which no EARLIER cell binds}.

    Markdown cells bind nothing and are skipped. Cell 1 (the first code cell) is reported
    against the empty namespace, so a name it loads from nowhere shows up rather than being
    excused as "the first cell".
    """
    bound_so_far = set()
    out = {}
    for i, cell in enumerate(cells):
        if cell.get('cell_type') != 'code':
            continue
        tree, _n = parse_cell(''.join(cell['source']))
        own = bound_names(tree)
        loaded = {n.id for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        out[i] = sorted(loaded - own - set(dir(builtins)) - bound_so_far)
        bound_so_far |= own
    return out


def artifact_cells():
    """The artifact's cells, honouring NEW_CELLS.

    `_mutate.py` points the suites at an extracted, possibly-mutated copy through NEW_CELLS.
    Reading the committed artifact unconditionally would make this check blind to every
    mutation aimed at it -- a green run that proves nothing, the failure mode AGENTS.md 9.8
    records.
    """
    override = os.environ.get('NEW_CELLS')
    if not override:
        return json.load(open(assemble.OUT))['cells']
    return [{'cell_type': 'markdown' if p.endswith('.md') else 'code',
             'source': [open(p, encoding='utf-8').read()]}
            for _i, p in sorted(extract_cells.dir_map(override, 'NEW_CELLS').items())]


def test_every_loaded_name_is_bound_by_an_earlier_cell():
    """Cells share one namespace; a name a cell loads and nothing before it binds is a
    NameError at run time. Reported per offending cell, with the names, and FAILs if any
    cell has one -- including a partial failure, so the message names every affected cell."""
    nb_cells = artifact_cells()
    bad = {i: names for i, names in unbound_loads(nb_cells).items() if names}
    detail = '; '.join(f'cell {i}: {names}' for i, names in sorted(bad.items()))
    check('every code cell loads only names an earlier cell binds (the artifact executes '
          'top to bottom)', not bad,
          detail if bad else f'{len(nb_cells)} cells, no unbound loads')


def test_unbound_analysis_detects_a_missing_binding():
    """The oracle above must be able to fail. Two code cells where the second reads a name
    the first never binds; if the analysis returned {} for that, it would report 'no unbound
    names' for the real artifact no matter what was wrong with it."""
    cells = [
        {'cell_type': 'code', 'source': ['a = 1\n']},
        {'cell_type': 'code', 'source': ['b = a + 1\n', 'print(c)\n']},
    ]
    got = {i: n for i, n in unbound_loads(cells).items() if n}
    check('a name no earlier cell binds is reported, with its cell and name',
          got == {1: ['c']}, str(got))
    ok_cells = [
        {'cell_type': 'code', 'source': ['a = 1\n']},
        {'cell_type': 'code', 'source': ['b = a + 1\n', 'def f(x):\n', '    return x + b\n']},
    ]
    got_ok = {i: n for i, n in unbound_loads(ok_cells).items() if n}
    check('a properly chained pair of cells reports nothing', got_ok == {}, str(got_ok))
    check('a cell binding its own name is not reported against itself',
          unbound_loads([{'cell_type': 'code', 'source': ['x = 1\nprint(x)\n']}]) == {0: []})


def test_unbound_analysis_reaches_the_last_cell():
    """The check must be able to name a LATE cell, not just the first few.

    If the walk stopped early -- say at the first cell with a magic, or at some depth limit --
    it would silently under-report, and a reader would take its 'no unbound names' as a clean
    bill of health for the cells it never looked at. The artifact's real gap is at cell 14 of
    24, so a chain of 20 cells with the break in the last one must be reported.
    """
    cells = [{'cell_type': 'code', 'source': ['a = 1\n']}]
    for i in range(18):
        cells.append({'cell_type': 'code', 'source': [f'v{i} = a + {i}\n']})
    cells.append({'cell_type': 'code', 'source': ['w = v17 + 1\n', 'print(missing_name)\n']})
    got = {i: n for i, n in unbound_loads(cells).items() if n}
    check('a break in the LAST cell of a 20-cell chain is still reported',
          got == {19: ['missing_name']}, f'{len(cells)} cells, reported {got}')


def test_unbound_analysis_survives_an_ipython_magic_cell():
    """Cell 2 of the base is a bare `!pip install`, which is not Python. Without magic
    handling the walk raises SyntaxError on the second cell and the real check never runs --
    a green run that proved nothing, the failure mode AGENTS.md 9.8 records."""
    nb_cells = artifact_cells()
    got = unbound_loads(nb_cells)
    check('a cell that is only an IPython magic does not abort the walk',
          len(got) >= 20, f'{len(got)} code cells analysed of {len(nb_cells)} total')
    check('the magic cell itself is analysed (cell 2), not skipped',
          2 in got, f'cells analysed: {sorted(got)[:6]}...')
    check('the analysis reaches cell 14, where the real gap is',
          14 in got, f'cells analysed: {sorted(got)[:6]}...')




# --- the replacement machinery ------------------------------------------------------------

MD_BODY = '# replaced title\n\nreplaced body line\n'
CODE_BODY = "print('replaced cell 5')\n"
_BASE_NB = json.load(open(assemble.BASE))
BASE_SOURCES = [''.join(c['source']) for c in _BASE_NB['cells']]
BASE_TYPES = [c['cell_type'] for c in _BASE_NB['cells']]


_UNSET = object()  # means "use assemble.py's current tables", so the staleness
                  # check builds what is actually configured, not an empty notebook


def build_with(replace=_UNSET, append_md=_UNSET, append_code=_UNSET, tmp=None):
    """Run build(); returns the written notebook.

    An _UNSET table means "use assemble.py's current table". In that case CELLS
    must stay pointed at the real notebook_build/cells/ -- redirecting it to a
    scratch dir holding only stubs would make a build of the *real* tables read
    stub bodies, which is what the staleness check compares against.

    Passing an explicit table opts into a scratch CELLS dir seeded with the four
    sample sources, so a ported fix can be exercised without touching cells/.
    """
    explicit = replace is not _UNSET or append_md is not _UNSET or append_code is not _UNSET
    if explicit:
        cells_dir = os.path.join(tmp, 'cells')
        os.makedirs(cells_dir, exist_ok=True)
        for fname, body in [('md_00.md', MD_BODY), ('src_05.py', CODE_BODY),
                            ('md_extra.md', 'appended markdown\n'),
                            ('src_extra.py', "print('appended code')\n")]:
            with open(os.path.join(cells_dir, fname), 'w', encoding='utf-8') as f:
                f.write(body)
    else:
        cells_dir = assemble.CELLS  # the real cells/
    out = os.path.join(tmp, 'out.ipynb')

    saved = (assemble.CELLS, assemble.REPLACE, assemble.APPEND_MD, assemble.APPEND_CODE)
    assemble.CELLS = cells_dir
    assemble.REPLACE = assemble.REPLACE if replace is _UNSET else replace
    assemble.APPEND_MD = assemble.APPEND_MD if append_md is _UNSET else append_md
    assemble.APPEND_CODE = assemble.APPEND_CODE if append_code is _UNSET else append_code
    try:
        # build() logs a progress summary to stdout; keep the suite's output
        # to PASS/FAIL lines only.
        with contextlib.redirect_stdout(io.StringIO()):
            assemble.build(out)
    finally:
        (assemble.CELLS, assemble.REPLACE,
         assemble.APPEND_MD, assemble.APPEND_CODE) = saved
    return json.load(open(out))


def test_replacements_and_appends_land():
    with tempfile.TemporaryDirectory() as tmp:
        nb = build_with({0: 'md_00.md', 5: 'src_05.py'},
                        ['md_extra.md'], ['src_extra.py'], tmp)
    cells = nb['cells']
    check('a markdown and a code replacement both land',
          ''.join(cells[0]['source']) == MD_BODY and ''.join(cells[5]['source']) == CODE_BODY)
    check('a replacement joins back to its source file byte for byte',
          ''.join(cells[5]['source']) == CODE_BODY,
          repr(''.join(cells[5]['source'])))
    check('26 cells: 24 base + 1 markdown + 1 code',
          len(cells) == 26, f'{len(cells)}')
    check('appended markdown precedes appended code',
          cells[24]['cell_type'] == 'markdown' and cells[25]['cell_type'] == 'code')
    check('appended code cell carries the nbformat keys',
          list(cells[25]) == ['cell_type', 'execution_count', 'metadata', 'outputs', 'source'],
          str(list(cells[25])))
    check('appended markdown cell carries the nbformat keys',
          list(cells[24]) == ['cell_type', 'metadata', 'source'], str(list(cells[24])))
    changed = [i for i in range(24) if ''.join(cells[i]['source']) != BASE_SOURCES[i]]
    check('only the named cells changed', changed == [0, 5], f'changed={changed}')
    check('cell types are never altered', [c['cell_type'] for c in cells[:24]] == BASE_TYPES)
    check('no code cell carries stale outputs or an execution count',
          all(c['outputs'] == [] and c['execution_count'] is None
              for c in cells if c['cell_type'] == 'code'))


def test_bad_entries_are_rejected():
    """Guard rails must raise, never assert: `python -O` strips asserts, and a
    stripped guard is a silent no-op. `test_guards_survive_python_O` pins that."""
    # A missing file raises FileNotFoundError and the rest raise ValueError;
    # either way nothing may pass silently.
    cases = [
        ('a .md source pointed at a code cell',
         {5: 'md_extra.md'}, [], [], (ValueError,), 'supplies a markdown cell'),
        ('an out-of-range REPLACE index',
         {99: 'src_05.py'}, [], [], (ValueError,), 'outside the 24-cell base'),
        ('a negative REPLACE index',
         {-1: 'src_05.py'}, [], [], (ValueError,), 'outside the 24-cell base'),
        ('a missing cell source',
         {5: 'nope.py'}, [], [], (FileNotFoundError,), 'nope.py'),
        ('a .md file listed in APPEND_CODE',
         {}, [], ['md_extra.md'], (ValueError,), 'is a .md file'),
        ('a base that is no longer 24 cells',
         {}, [], [], (ValueError,), 'expected 24'),
    ]
    for name, replace, amd, acode, allowed, needle in cases:
        with tempfile.TemporaryDirectory() as tmp:
            if name.startswith('a base'):
                short = os.path.join(tmp, 'short.ipynb')
                json.dump({'cells': json.load(open(assemble.BASE))['cells'][:5],
                           'metadata': {}, 'nbformat': 4, 'nbformat_minor': 4},
                          open(short, 'w'))
                saved = assemble.BASE
                assemble.BASE = short
                try:
                    build_with(replace, amd, acode, tmp)
                except allowed as e:
                    check(f'{name} is rejected', needle in str(e), str(e))
                except Exception as e:
                    check(f'{name} is rejected', False, f'raised {type(e).__name__}: {e}')
                else:
                    check(f'{name} is rejected', False, 'no error raised')
                finally:
                    assemble.BASE = saved
                continue
            try:
                build_with(replace, amd, acode, tmp)
            except allowed as e:
                check(f'{name} is rejected', needle in str(e), str(e))
            except Exception as e:
                check(f'{name} is rejected', False, f'raised {type(e).__name__}: {e}')
            else:
                check(f'{name} is rejected', False, 'no error raised')


def test_guards_survive_python_O():
    """`python -O` strips every `assert`, so a guard written as one is not a guard.

    Runs a tiny script under -O that points BASE at a 5-cell notebook and calls
    build(). With ValueError-based guards it must raise; if anyone converts one
    back to `assert`, -O deletes it and the build silently succeeds."""
    probe = (
        'import contextlib, io, json, os, sys, tempfile\n'
        f'sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})\n'
        'import assemble\n'
        'd = tempfile.mkdtemp()\n'
        'bad = os.path.join(d, "bad.ipynb")\n'
        'base = json.load(open(assemble.BASE))\n'
        'json.dump({"cells": base["cells"][:5], "metadata": {},'
        ' "nbformat": 4, "nbformat_minor": 4}, open(bad, "w"))\n'
        'assemble.BASE = bad\n'
        'with contextlib.redirect_stdout(io.StringIO()):\n'
        '    assemble.build(os.path.join(d, "out.ipynb"))\n'
        'print("SUCCEEDED")\n'
    )
    r = subprocess.run([sys.executable, '-O', '-c', probe], capture_output=True, text=True)
    check('build() still rejects a bad base under python -O',
          r.returncode != 0 and 'SUCCEEDED' not in r.stdout,
          f'exit={r.returncode} stdout={r.stdout.strip()[:80]!r}')
    check('the -O rejection is a ValueError with the real diagnostic',
          'ValueError' in r.stderr and 'expected 24' in r.stderr,
          r.stderr.strip().splitlines()[-1][:120] if r.stderr.strip() else 'no stderr')


def test_committed_notebook_is_not_stale():
    """The artifact must equal a fresh build of the current tables. Compares the
    whole cell list, not just `source`, so any hand-edit is caught -- editing the
    generated notebook is the exact mistake this directory exists to prevent."""
    if not os.path.exists(assemble.OUT):
        check('the merged notebook exists', False, f'missing {assemble.OUT}')
        return
    with open(assemble.OUT, encoding='utf-8') as f:
        committed = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build_with(tmp=tmp)

    check('the merged notebook matches a fresh build of the current tables',
          committed['cells'] == fresh['cells'],
          're-run notebook_build/assemble.py')
    check('the merged notebook keeps the base metadata (Kaggle accelerator settings)',
          committed['metadata'] == json.load(open(assemble.BASE))['metadata'])
    base_nb = json.load(open(assemble.BASE))
    check('the merged notebook carries the base nbformat and top-level keys',
          (committed['nbformat'], committed['nbformat_minor'])
          == (base_nb['nbformat'], base_nb['nbformat_minor'])
          and set(committed) == set(base_nb),
          f"nbformat={committed['nbformat']}.{committed['nbformat_minor']}, "
          f"keys={sorted(committed)}")


def test_stale_execution_state_is_stripped():
    """The real base has no saved outputs, so this uses a synthetic base that has
    them. Without that, 'outputs are cleared' passes even if the clearing code
    is deleted -- a vacuous test."""
    cells = json.load(open(assemble.BASE))['cells']
    dirty = copy.deepcopy(cells)
    for i, c in enumerate(dirty):
        if c['cell_type'] == 'code':
            c['outputs'] = [{'output_type': 'stream', 'name': 'stdout',
                             'text': [f'stale output for cell {i}\n']}]
            c['execution_count'] = i + 1

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'dirty_base.ipynb')
        json.dump({'cells': dirty, 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 4},
                  open(base, 'w'))
        saved = assemble.BASE
        assemble.BASE = base
        try:
            nb = build_with(tmp=tmp)
            had_outputs = any(c['outputs'] for c in json.load(open(base))['cells']
                              if c['cell_type'] == 'code')
        finally:
            assemble.BASE = saved

    check('a base carrying saved outputs builds', had_outputs)
    check('no stale outputs survive into the artifact',
          all(c['outputs'] == [] for c in nb['cells'] if c['cell_type'] == 'code'),
          'a rebuilt notebook must not claim to have been run')
    check('no stale execution_count survives into the artifact',
          all(c['execution_count'] is None for c in nb['cells'] if c['cell_type'] == 'code'))


CHECKS = [
    ('to_source round-trips text', test_to_source_roundtrip),
    ('to_source drops the trailing newline on the last line',
     test_to_source_trailing_newline_dropped_on_last_line),
    ('base notebook has the expected 24-cell shape', test_base_notebook_has_expected_shape),
    ('base carries no saved execution state', test_base_carries_no_saved_execution_state),
    ('replacements and appends land, untouched cells do not', test_replacements_and_appends_land),
    ('malformed REPLACE/APPEND entries are rejected loudly', test_bad_entries_are_rejected),
    ('guard rails survive python -O', test_guards_survive_python_O),
    ('stale outputs and execution counts are stripped', test_stale_execution_state_is_stripped),
    ('the committed merged notebook is not stale', test_committed_notebook_is_not_stale),
    ('every loaded name is bound by an earlier cell',
     test_every_loaded_name_is_bound_by_an_earlier_cell),
    ('the unbound-name oracle can fail', test_unbound_analysis_detects_a_missing_binding),
    ('the unbound-name analysis reaches the last cell',
     test_unbound_analysis_reaches_the_last_cell),
    ('the unbound-name analysis survives an IPython magic cell',
     test_unbound_analysis_survives_an_ipython_magic_cell),
]


def main():
    for name, fn in CHECKS:
        before = len(fails)
        try:
            fn()
        except Exception as e:
            fails.append(name)
            print(f'  [FAIL] {name}  -- {type(e).__name__}: {e}')
        else:
            # A group that emitted its own sub-checks must not claim PASS when
            # one of them failed -- the sub-check line is the accurate report.
            if len(fails) == before:
                print(f'  [PASS] {name}')
    print('\n' + '=' * 62)
    print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
