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

Run: venv/bin/python notebook_build/tests/test_assemble.py
"""
import contextlib
import copy
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import assemble

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


# --- the replacement machinery ------------------------------------------------------------

MD_BODY = '# replaced title\n\nreplaced body line\n'
CODE_BODY = "print('replaced cell 5')\n"
_BASE_NB = json.load(open(assemble.BASE))
BASE_SOURCES = [''.join(c['source']) for c in _BASE_NB['cells']]
BASE_TYPES = [c['cell_type'] for c in _BASE_NB['cells']]


_UNSET = object()  # means "use assemble.py's current tables", so the staleness
                  # check builds what is actually configured, not an empty notebook


def build_with(replace=_UNSET, append_md=_UNSET, append_code=_UNSET, tmp=None):
    """Run build() against a scratch cells/ dir; returns the written notebook.

    The four sample sources are written into the scratch cells/ dir but are only
    used when a table is passed explicitly; the sentinel keeps assemble.py's own
    REPLACE/APPEND_* in force otherwise.
    """
    cells_dir = os.path.join(tmp, 'cells')
    os.makedirs(cells_dir, exist_ok=True)
    for fname, body in [('md_00.md', MD_BODY), ('src_05.py', CODE_BODY),
                        ('md_extra.md', 'appended markdown\n'),
                        ('src_extra.py', "print('appended code')\n")]:
        with open(os.path.join(cells_dir, fname), 'w', encoding='utf-8') as f:
            f.write(body)
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
    cases = [
        ('a .md source pointed at a code cell',
         {5: 'md_extra.md'}, [], [], AssertionError, 'supplies a markdown cell'),
        ('an out-of-range REPLACE index',
         {99: 'src_05.py'}, [], [], AssertionError, 'outside the 24-cell base'),
        ('a missing cell source',
         {5: 'nope.py'}, [], [], FileNotFoundError, 'nope.py'),
        ('a .md file listed in APPEND_CODE',
         {}, [], ['md_extra.md'], AssertionError, 'is a .md file'),
        ('a base that is no longer 24 cells',
         {}, [], [], AssertionError, 'expected 24'),
    ]
    for name, replace, amd, acode, exc, needle in cases:
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
                except exc as e:
                    check(f'{name} is rejected', needle in str(e), str(e))
                except Exception as e:  # wrong exception type
                    check(f'{name} is rejected', False, f'raised {type(e).__name__}: {e}')
                else:
                    check(f'{name} is rejected', False, 'no error raised')
                finally:
                    assemble.BASE = saved
                continue
            try:
                build_with(replace, amd, acode, tmp)
            except exc as e:
                check(f'{name} is rejected', needle in str(e), str(e))
            except Exception as e:  # wrong exception type
                check(f'{name} is rejected', False, f'raised {type(e).__name__}: {e}')
            else:
                check(f'{name} is rejected', False, 'no error raised')


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
    check('the merged notebook is nbformat 4',
          committed['nbformat'] == 4 and isinstance(committed['cells'], list))


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
    ('replacements and appends land, untouched cells do not', test_replacements_and_appends_land),
    ('malformed REPLACE/APPEND entries are rejected loudly', test_bad_entries_are_rejected),
    ('stale outputs and execution counts are stripped', test_stale_execution_state_is_stripped),
    ('the committed merged notebook is not stale', test_committed_notebook_is_not_stale),
]


def main():
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as e:
            fails.append(name)
            print(f'  [FAIL] {name}  -- {type(e).__name__}: {e}')
        else:
            print(f'  [PASS] {name}')
    print('\n' + '=' * 62)
    print(f'{len(fails)} failure(s)' + (': ' + ', '.join(fails) if fails else ''))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()