#!/usr/bin/env python3
"""Red-green sweep: revert ONE fix in a scratch copy of the artifact, run the matching suite.

This is the vacuity check. A test that passes both with and without the fix tests nothing
(AGENTS.md 9.8 records four such defects found by exactly this method). For each mutation
below it copies the delivered notebook's cells into a scratch directory, edits ONE cell,
points the suites at that directory with NEW_CELLS, and asserts the named suite FAILS.

Detection is a change in the SET of failing check names, not in the exit code: `test_bug1.py`
and `test_fixes.py` are red today, and for them a caught mutation is red->more-red or
red->green, both invisible to an exit-code comparison.

Usage: venv/bin/python notebook_build/tests/_mutate.py
"""
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile

import extract_cells

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'venv', 'bin', 'python')

# Scratch dirs are removed by their own `finally`; this is the belt-and-braces backstop for a
# crash between mkdtemp and its try, which would otherwise leave a full cell copy per mutation.
_SCRATCH = []


def _scratch(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    _SCRATCH.append(d)
    return d


atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _SCRATCH])

# (name, suite, {cell index: (old, new)}). The old text must occur exactly once in the cell.
MUTATIONS = [
    # --- cell 6: data discovery -----------------------------------------------------------
    ('c6 recording_id: hyphen-only regex',
     'test_recording_id.py', {6: [("re.compile(r'(\\d{8})[-_](\\d{6})')",
                                  "re.compile(r'^(\\d{8})-(\\d{6})$')")]}),
    ('c6 recording_id: positional field 4 (the pre-B4 parse)',
     'test_recording_id.py', {6: [(
        "    stem = os.path.basename(path)[:-4]\n    m = _TAPE_RE.search(stem)\n"
        "    return f'{m.group(1)}-{m.group(2)}' if m else stem",
        "    _p = os.path.basename(path)[:-4].split('_')\n"
        "    return _p[3] if len(_p) >= 6 else os.path.basename(path)[:-4]")]}),
    ('c6 recording_id: no regex at all (one group per clip)',
     'test_recording_id.py', {6: [(
        "    m = _TAPE_RE.search(stem)\n"
        "    return f'{m.group(1)}-{m.group(2)}' if m else stem",
        "    return stem")]}),
    ('c6 unusable-clip drop removed',
     'test_cell7.py', {6: [(
        "def _unusable_reason(path):", "def _unused_unusable_reason(path):")]}),
    ('c6 the unusable-clip threshold computed from a hard-coded frame count',
     'test_cell7.py', {6: [(
        "_MIN_CLIP_S = max(DET_CONFIG['n_fft'] / DET_CONFIG['sr'], CLS_CONFIG['n_fft'] / CLS_CONFIG['sr'])",
        "_MIN_CLIP_S = 1e-3")]}),
    ('c6 per-class recording counts removed',
     'test_cell7.py', {6: [(
        "    print(f'  {_c:<8} {_n:>3} recording(s)'", "    print(f'  {_c:<8} skipped')")]}),
    ('c6 the "single recording" warning text removed',
     'test_cell7.py', {6: [(
        "'   <-- single recording: cannot be validated on a new tape'", "'   <--'")]}),
    ('c6 tape-lookup baseline removed',
     'test_cell7.py', {6: [(
        "def _tape_lookup_baseline(test_files, label_of):",
        "def _unused_tape_lookup_baseline(test_files, label_of):")]}),
    ('c6 the tape-lookup baseline reports chance instead of the real number',
     'test_cell7.py', {6: [(
        "    bal = float(np.mean([np.mean([p == c for y, p in zip(ys, ps) if y == c]) for c in sorted(set(ys))]))",
        "    bal = 0.5")]}),
    ('c6 DET_CLASS_TO_IDX (the dead dict) reintroduced',
     'test_cell7.py', {6: [(
        "DET_CLASSES = ['noise', 'call']\n",
        "DET_CLASSES = ['noise', 'call']\nDET_CLASS_TO_IDX = {'noise': 0, 'call': 1}\n")]}),
    ('c6 a prior-correction global reintroduced',
     'test_cell7.py', {6: [(
        "print(f'\\nSplit: train=", "DET_EVAL_PRIOR = np.array([0.193, 0.807])\n"
        "print(f'\\nSplit: train=")]}),
    ('c6 the split no longer stratifies (test set shrinks)',
     'test_cell7.py', {6: [(
        "all_wavs, all_species_labels, test_size=0.15, random_state=42,\n"
        "        stratify=all_species_labels)",
        "all_wavs, all_species_labels, test_size=0.15, random_state=42)")]}),

    # --- cell 9: model staging -------------------------------------------------------------
    ('c9 staged under the shared basename (the ANIMAL-SPOT.pk collision)',
     'test_staging_export_eval.py', {9: [(
        "_dest = os.path.join(PRETRAINED_DIR, f'official_{_role}_{_mic}.pk')",
        "_dest = os.path.join(PRETRAINED_DIR, 'ANIMAL-SPOT.pk')")]}),
    ('c9 _official wiped before discovery (the 2.5 double-init bug)',
     'test_staging_export_eval.py', {9: [(
        "_manual_n = sum(len(v) for v in _official.values())",
        "_official = {}\n_manual_n = sum(len(v) for v in _official.values())")]}),
    ('c9 the config-vs-dataOpts check disabled',
     'test_staging_export_eval.py', {9: [(
        "        for _k in ('sr', 'n_fft', 'hop_length', 'fmin', 'fmax', 'n_freq_bins', 'freq_compression'):",
        "        for _k in ():")]}),
    ('c9 buzz/social detectors classified as call detectors',
     'test_staging_export_eval.py', {9: [(
        "        if _family == 'models_call_classifier':\n            _role = 'classifier'\n"
        "        elif _family == 'models_call_detector':\n            _role = 'detector'\n"
        "        else:\n            _skipped.append(f'{_family}/{_mic}')\n            continue",
        "        _role = 'classifier' if _family.endswith('classifier') else 'detector'")]}),
    ('c9 m09 no longer preferred as the starting point',
     'test_staging_export_eval.py', {9: [(
        "        _pick = 'm09' if 'm09' in _mics else sorted(_mics)[0]",
        "        _pick = sorted(_mics)[0]")]}),

    # --- cell 14: export ------------------------------------------------------------------
    ('c14 export_pk moves the LIVE model to the CPU (the 10.1 bug)',
     'test_staging_export_eval.py', {14: [(
        "    model = copy.deepcopy(model).cpu()", "    model = model.cpu()")]}),
    ('c14 the GUI model exported from members[0] instead of the best member',
     'test_staging_export_eval.py', {14: [(
        "export_pk(cls_best_member['model'], encoderOpts_cls", "export_pk(cls_members[0]['model'], encoderOpts_cls")]}),
    ('c14 ensemble members not exported',
     'test_staging_export_eval.py', {14: [(
        "if len(cls_members) > 1:", "if False:")]}),
    ('c14 the unknown sidecar not exported',
     'test_staging_export_eval.py', {14: [(
        "if UNKNOWN_MODEL is not None:", "if False:")]}),
    ('c14 the sidecar claims a single model instead of the ensemble',
     'test_staging_export_eval.py', {14: [(
        "save_unknown_model(UNKNOWN_MODEL, unknown_out_path, cls_member_paths or [cls_out_path])",
        "save_unknown_model(UNKNOWN_MODEL, unknown_out_path, [cls_out_path])")]}),
    ('c14 DataParallel no longer unwrapped',
     'test_staging_export_eval.py', {14: [(
        "    if isinstance(model, nn.DataParallel):\n        model = model.module\n", "")]}),

    # --- cell 16: evaluation ---------------------------------------------------------------
    ('c16 num_mels fallback removed',
     'test_staging_export_eval.py', {16: [(
        "'n_freq_bins': dataOpts.get('n_freq_bins') or dataOpts.get('num_mels') or 256,",
        "'n_freq_bins': dataOpts.get('n_freq_bins', 256),")]}),
    ('c16 class names sorted by dict value (breaks on non-contiguous indices)',
     'test_staging_export_eval.py', {16: [(
        "        names = [_inv.get(i, f'class{i}') for i in range(probs.shape[1])]",
        "        names = [k for k, _ in sorted(model_classes.items(), key=lambda kv: kv[1])]")]}),
    ('c16 the no_signal flag removed',
     'test_staging_export_eval.py', {16: [(
        "                'no_signal': out['auc'] is not None and abs(out['auc'] - 0.5) < AUC_NO_SIGNAL})",
        "                'no_signal': False})")]}),
    ('c16 the non-linear freq_compression warning removed',
     'test_staging_export_eval.py', {16: [(
        "    if _fc != 'linear':", "    if False:")]}),
    ('c16 the detector noise class read from a hard-coded index 1',
     'test_staging_export_eval.py', {16: [(
        "        p_call = probs[:, model_classes['target']]", "        p_call = probs[:, 1]")]}),

    # --- cell 17: summary -----------------------------------------------------------------
    ('c17 the doubled "official official_" label back',
     'test_b5_labels.py', {17: [(
        "    _lbl = _lbl.replace('_', ' ', 1) if _lbl.startswith('official_') else "
        "model_tags.get(r['path'], 'yours') + ' ' + _lbl",
        "    _lbl = 'official ' + _lbl")]}),
    ('c17 the reduced-denominator note removed',
     'test_b5_labels.py', {17: [(
        "        else r['kind'] + ('' if r['n_files'] == len(test_wavs) else f' (only {r[\"n_files\"]} clips)')",
        "        else r['kind']")]}),
    ('c17 the NO SIGNAL note replaced by the kind',
     'test_b5_labels.py', {17: [(
        "    _note = 'NO SIGNAL (AUC~0.5): plain accuracy just reflects the class mix' if r.get('no_signal') \\\n        else r['kind']",
        "    _note = r['kind']")]}),
    ('c17 the per-model NO-SIGNAL warning removed',
     'test_b5_labels.py', {17: [(
        "    if r.get('no_signal'):", "    if False:")]}),
    ('c17 the Files-evaluated denominator removed',
     'test_b5_labels.py', {17: [(
        "    print(f'  Files evaluated: {r[\"n_files\"]} of {len(test_wavs)} test clips'",
        "    print(f'  Files evaluated: {r[\"n_files\"]}'")]}),
    ('c17 the fine-tuned gain line removed',
     'test_b5_labels.py', {17: [(
        "print('\\nBalanced accuracy, official zero-shot -> fine-tuned (same test clips):')",
        "print('\\nNothing here')")]}),
    ('c17 the fine-tuned classifier row removed',
     'test_b5_labels.py', {17: [(
        "print(f'{\"fine-tuned classifier (noise?)\":<34}", "print(f'{\"untrained (noise?)\":<34}")]}),
    ('c17 the classifier row reads a hard-coded probability column instead of by name',
     'test_b5_labels.py', {17: [(
        "_cp = 1.0 - cls_metrics['probs'][:, CLS_CLASS_TO_IDX['noise']]",
        "_cp = cls_metrics['probs'][:, 1]")]}),
    # The reviewer's Important 1: an AUC-only defect must be caught. The earlier check read
    # split()[4], which is the ACCURACY column, so this mutation passed it.
    ('c17 the classifier row\'s AUC is computed with the labels inverted',
     'test_b5_labels.py', {17: [(
        "f'{(roc_auc_score(_cy, _cp) if len(set(_cy)) > 1 else float(\"nan\")):>7.4f}",
        "f'{(roc_auc_score(1 - _cy, _cp) if len(set(_cy)) > 1 else float(\"nan\")):>7.4f}")]}),
    ('c17 the classifier row\'s AUC is replaced by its accuracy (a plausible refactor)',
     'test_b5_labels.py', {17: [(
        "f'{(roc_auc_score(_cy, _cp) if len(set(_cy)) > 1 else float(\"nan\")):>7.4f}",
        "f'{np.mean((_cp >= 0.5) == _cy):>7.4f}")]}),
    ('c17 the classifier row\'s balanced accuracy column prints plain accuracy',
     'test_b5_labels.py', {17: [(
        "      f'{balanced_accuracy_score(_cy, (_cp >= 0.5).astype(int)):>8.4f} '",
        "      f'{np.mean((_cp >= 0.5) == _cy):>8.4f} '")]}),
    ('c17 the classifier row\'s accuracy is mislabelled: it prints the call fraction',
     'test_b5_labels.py', {17: [(
        "print(f'{\"fine-tuned classifier (noise?)\":<34} {len(_cy):>4} "
        "{np.mean((_cp >= 0.5) == _cy):>7.4f} '",
        "print(f'{\"fine-tuned classifier (noise?)\":<34} {len(_cy):>4} "
        "{np.mean(_cp >= 0.5):>7.4f} '")]}),
    ('c17 the gain line pairs the wrong rows (fine-tuned printed twice)',
     'test_b5_labels.py', {17: [(
        "        _o, _f = _off[0]['balanced_accuracy'], _dr['metrics']['balanced_accuracy']",
        "        _o, _f = _off[0]['balanced_accuracy'], _off[0]['balanced_accuracy']")]}),
    # Minor 11: the ordering claim. Printing the heading AFTER the gain lines must be caught,
    # which a bare "a line starts with balanced accuracy" would not.
    ('c17 the balanced-accuracy heading is printed AFTER the gain lines',
     'test_b5_labels.py', {17: [(
        "print('\\nBalanced accuracy, official zero-shot -> fine-tuned (same test clips):')\n"
        "for _mic, _dr in det_results.items():",
        "for _mic, _dr in det_results.items():")]}),
    # Minor 9: the column-shape checks. Each row must print n/acc/bal/auc then a note; a cell
    # that dropped the n column, or printed the total on every row, must fail.
    ('c17 the n column is dropped from the table',
     'test_b5_labels.py', {17: [(
        "    print(f'{_lbl[:34]:<34} {r[\"n_files\"]:>4} {r[\"accuracy\"]:>7.4f} '",
        "    print(f'{_lbl[:34]:<34} {len(test_wavs):>4} {r[\"accuracy\"]:>7.4f} '")]}),
    ('c17 the n column header is removed',
     'test_b5_labels.py', {17: [(
        "print(f'{\"Model\":<34} {\"n\":>4} {\"Acc\":>7} {\"Bal acc\":>8} {\"AUC\":>7}  Note')",
        "print(f'{\"Model\":<34} {\"Acc\":>7} {\"Bal acc\":>8} {\"AUC\":>7}  Note')")]}),
    # Minor 6: the per-model n_freq_bins checks must be able to fail on the model they name.
    ('c16 the mel model is fed 256 bins (the default) instead of its num_mels',
     'test_staging_export_eval.py', {16: [(
        "'n_freq_bins': dataOpts.get('n_freq_bins') or dataOpts.get('num_mels') or 256,",
        "'n_freq_bins': dataOpts.get('n_freq_bins', 256),")]}),
    # The bound-name analysis must be able to report a gap in a LATE cell, not just an early one.
    ('c6 cell 6 starts reading a name no earlier cell binds (a NEW unbound load)',
     'test_assemble.py', {6: [(
        "print(f'Found {len(all_wavs)} wav files')",
        "print(f'Found {len(all_wavs)} wav files {an_unbound_name_appears_here}')")]}),
    # A trailing-cell gap: the analysis has to walk the whole notebook, not stop early.
    ('c23 (near the end) starts reading a name nothing before it binds',
     'test_assemble.py', {23: [(
        "# Cell 24 (optional): score the detections against your own",
        "print('this cell reads a name nothing binds', trailing_unbound_name)\n"
        "# Cell 24 (optional): score the detections against your own")]}),
]


def failing_checks(output: str) -> set:
    """The checks a suite reported as FAIL, keyed by name AND detail.

    Detection is a change in this SET, not in the exit code. An exit code cannot tell
    "the mutation was caught" from "the suite was already failing for an unrelated reason",
    and it cannot see a mutation that turns an already-red suite GREEN -- which is why the
    first version of this harness discarded every red->red and red->more-red transition, i.e.
    the two normal cases for `test_bug1.py` and `test_fixes.py`, which are red today.

    The DETAIL is part of the key because one check can report several things at once: the
    artifact-wide unbound-name check fails today on cell 14, and a mutation that adds an
    unbound load to a DIFFERENT cell changes only its detail line. Keyed by name alone that
    would look unchanged, i.e. undetected.

    Every suite's harness renders a failing check as `  [FAIL] <name>  -- <detail>` (the
    f-string `[{"PASS" if cond else "FAIL"}]` in each check()). The leading `[FAIL]` marker is
    dropped, since it carries no information beyond presence in this set.
    """
    out = set()
    for ln in output.split('\n'):
        m = re.search(r'\[FAIL\]\s*(.+?)\s*$', ln.rstrip())
        if m:
            out.add(re.sub(r'\s+', ' ', m.group(1).strip()))
    return out


def outcome(suite, new_cells):
    """Run one suite against `new_cells`; return its observable outcome signature.

    The signature is `(exit code, sorted failing check names)`. BOTH parts matter:
      * the names are what distinguish a caught mutation from an already-failing suite -- an
        exit code alone cannot tell them apart, which is why the first version of this
        harness discarded every red->red and red->more-red transition;
      * the exit code is what catches a mutation that makes the suite CRASH before it prints
        any check. Four mutations do exactly that (renaming a function a later line calls),
        and with names alone they are indistinguishable from "unchanged".
    """
    env = dict(os.environ, NEW_CELLS=new_cells)
    p = subprocess.run([PY, os.path.join(HERE, suite)], env=env,
                       capture_output=True, text=True)
    out = p.stdout + p.stderr
    return p.returncode, tuple(sorted(failing_checks(out)))


def describe(cur, base):
    """Human-readable difference between two outcome signatures."""
    rc_c, names_c = cur
    rc_b, names_b = base
    bits = []
    if rc_c != rc_b:
        bits.append(f'exit {rc_b}->{rc_c}')
    new = sorted(set(names_c) - set(names_b))
    gone = sorted(set(names_b) - set(names_c))
    if new:
        bits.append(f'new={new[:2]}')
    if gone:
        bits.append(f'gone={gone[:2]}')
    return ', '.join(bits) or 'UNCHANGED'


def write_copy(cells, src, out_dir, edits=None):
    """Materialise the artifact's cells into `out_dir`, optionally after applying edits.

    Returns an error string if an anchor is not unique in its cell (which would silently
    patch the wrong site, or none), else None.
    """
    for i, path in cells.items():
        shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))
    for idx, pairs in (edits or {}).items():
        p = os.path.join(out_dir, os.path.basename(cells[idx]))
        txt = src[idx]
        for old, new in pairs:
            if txt.count(old) != 1:
                return f'ANCHOR NOT UNIQUE ({txt.count(old)}x) in cell {idx}: {old[:60]!r}'
            txt = txt.replace(old, new)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(txt)
    return None


def main():
    cells = extract_cells.merged_cells()
    src = {i: open(p, encoding='utf-8').read() for i, p in cells.items()}

    # Baseline per suite: its outcome signature on an unmutated copy. Every scratch directory is
    # removed by its own `finally`, so nothing leaks across the run.
    base = {}
    for suite in sorted({s for _n, s, _e in MUTATIONS}):
        d = _scratch('batspot-base-')
        try:
            write_copy(cells, src, d)
            base[suite] = outcome(suite, d)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    for suite in sorted(base):
        rc, names = base[suite]
        print(f'  baseline {suite:<30} exit={rc}, {len(names)} failing check(s)'
              + (f' already: {list(names)[:2]}' if names else ''))

    print(f'\n{"mutation":<62}{"suite":<30}  detected')
    print('-' * 110)
    undetected = []
    for name, suite, edits in MUTATIONS:
        d = _scratch('batspot-mut-')
        try:
            err = write_copy(cells, src, d, edits)
            if err:
                print(f'{name:<62}{suite:<30}  ----  {err}')
                undetected.append(name)
                continue
            cur = outcome(suite, d)
            diff = describe(cur, base[suite])
            ok = diff != 'UNCHANGED'
            print(f'{name:<62}{suite:<30}  {"YES" if ok else "NO":<4}{diff[:70]}')
            if not ok:
                undetected.append(name)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print('\n' + '=' * 110)
    print(f'{len(MUTATIONS) - len(undetected)}/{len(MUTATIONS)} mutations detected')
    for n in undetected:
        print(f'  UNDETECTED: {n}')
    return 1 if undetected else 0


if __name__ == '__main__':
    sys.exit(main())