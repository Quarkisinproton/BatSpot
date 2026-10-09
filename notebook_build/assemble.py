#!/usr/bin/env python3
"""Generate batspot-train-merged.ipynb from batspot-train.ipynb.

The merged notebook is a BUILD ARTIFACT. Never hand-edit it. Edit the cell
sources in notebook_build/cells/, register them in REPLACE / APPEND_CODE /
APPEND_MD below, and re-run this script:

    venv/bin/python notebook_build/assemble.py

`batspot-train.ipynb` and `batspot_train(claude_bug_fixes _by__SPACE_BUNNY_MODEL).ipynb`
are read-only parents; neither is modified by this script.

With REPLACE and APPEND_* empty, build() is a lossless pass-through: every cell's
source is the base's, cell for cell. The one thing that never carries over is
execution state -- `outputs` and `execution_count` are cleared on every code
cell, so the artifact never claims to have been run. That property is what makes
every later port auditable: a diff against the base shows exactly what was
ported.

Every guard rail raises ValueError rather than using `assert`, because `assert`
is stripped under `python -O` and this script's whole premise is that a silent
no-op is impossible.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, 'batspot-train.ipynb')
OUT = os.path.join(REPO, 'batspot-train-merged.ipynb')
CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cells')

# The merge map is written against a 24-cell base. If the base ever changes
# shape, these indices (and the plan's) stop meaning what they say.
N_BASE_CELLS = 24

# base cell index -> filename in cells/. A '.md' file supplies a markdown cell,
# any other filename a code cell; build() checks the pair agrees.
#
# 1 = config: the combined notebook's cell 2 (noise mixing, 3-seed ensemble, "unknown"),
# ported verbatim, plus a Nyquist guard the combined notebook does not have.
# 4 = architecture: the combined notebook's cell 5, ported verbatim. The ResNet's residual
# projection must stay spelled `shortcut` -- that is the on-disk key format of every
# official .pk -- so src_04.py is copied, never reformatted.
# 5 = transforms + dataset: the combined notebook's cell 6, ported verbatim. Carries the
# background-noise mixing, the _code_fingerprint cache tag and the open-set scorers, each
# with the measurements that chose its default. Not reformatted either: the comments record
# which constants are measured and why, which is the entire value of the port.
# 6 = data discovery: the combined notebook's cell 7. Unusable-clip pre-drop BEFORE the split
# (so detector and classifier always see the same clip list), the three split modes
# including per_species, the per-class recording counts and the tape-lookup baseline.
# 7 = clip extraction: the combined notebook's cell 8, the faithful port of export_clips.R.
# THIS IS THE EXTRACTION BUG FIX. The base cell took `'_'.join(name.split('_')[:2])` --
# `acsh_devon` for this repo's selection tables -- as the recording stamp, so the match
# against the audio file names never held, every table was skipped and it reported
# "Extracted 0 clips" on the dataset it was meant to rebuild. The port matches the Raven
# YYYYMMDD_HHMMSS stamp and writes the shipped clip layout, so cell 6's recording_id
# finds the tape. Copied verbatim: the comments recording the faithfulness to the R script
# and its three deliberate deviations are the point of the cell.
# 8 = training infrastructure: the combined notebook's cell 9. `train_model` plus the sampler /
# class-weight helpers. Copied verbatim because the invariants it carries are not stylistic:
#   - the loss is UNWEIGHTED CrossEntropyLoss while the batches are balanced by
#     WeightedRandomSampler. Passing class weights as well double-counts it; on the detector
#     that is a 4.18x noise penalty, which collapses the weakest-pretrained variant to all-noise.
#   - `effective_lr = config['base_lr']`, absolute. The paper's 1e-4 / 3e-4 must not be scaled
#     by the batch size (that is what diverged: 128x too large at batch 64 x accum 2).
#   - `no_improve += config['epochs_per_eval']`, so `early_stopping_patience_epochs` counts RAW
#     epochs. A validation-step counter silently doubled every patience in the notebook.
#   - the metric is selected by dict lookup, `{...}[SELECT_METRIC]`, so a typo raises KeyError
#     instead of degrading to plain accuracy on an 81/19 split. test_metric_guard.py reads this
#     line out of the delivered cell and executes it; test_bug1.py pins the other three.
#   - `config` stays a plain dict read by key, so a later cell can copy it per variant.
# It also adds what the merged cell was missing: `history['val_score']`, `history['topk_mean']`
# (selection optimism, made visible) and `history['best_score']`.
# 9 = model staging: the combined notebook's cell 10. The four official models are ALL
# basenamed ANIMAL-SPOT.pk, so staging by basename collides silently -- the classifier
# overwrites the detector and you fine-tune a "detector" from 15-class weights.
# 11 = detectors: the combined notebook's cell 12. Trains and evaluates ALL THREE variants
# (m03/m09/m11) from one loop, and defines `evaluate_model` (the clip-level scorer cells 12, 13,
# 16 and 17 all call). Copied verbatim so `det_results[mic]` keeps the key set the export and
# summary cells read: model / metrics / best_val_acc / history / encoderOpts / classifierOpts.
# 12 = classifier: the combined notebook's cell 13. One model per CLS_ENSEMBLE_SEEDS seed ->
# `cls_members` (dicts of seed/model/best_val/history/metrics), `cls_model` a `SoftmaxEnsemble`
# when there is more than one, `cls_best_member` the best single member (the BatSpot GUI / CLI
# loads one model), and `UNKNOWN_MODEL`, the fitted open-set sidecar. THIS CELL IS WHAT MAKES
# THE NOTEBOOK EXECUTABLE TOP TO BOTTOM: cell 14 reads all three of those names, and before this
# port nothing defined them. test_assemble.py's `every loaded name is bound by an earlier cell`
# check failed on exactly that gap and is now green.
# 13 = cascade: the combined notebook's cell 14. `run_combined_pipeline` tunes the detector
# threshold on VALIDATION (maximise call F1, grid 0.05-0.95) and then reports
# classifier-alone / hard-gate / soft-combine side by side, because the classifier already has
# a `noise` class and a hard gate can only add the detector's own errors. Prior correction is
# deliberately absent here (plan section 1 records it as a scope decision): with a tuned
# threshold, rescaling P(call) by a prior ratio is monotone and changes no decisions.
# 14 = export: the combined notebook's cell 15. export_pk deep-copies before .cpu(), so the
# LIVE models stay on the GPU; also writes every ensemble member and the unknown sidecar.
# 16 = evaluation: the combined notebook's cell 17. num_mels fallback, class names indexed by
# output width (not by sorted dict value), and the no_signal flag.
# 17 = summary: the combined notebook's cell 18. kind / n_files columns, NO SIGNAL markers and
# the reduced-denominator warning.
REPLACE: dict[int, str] = {
    1: 'src_01.py', 4: 'src_04.py', 5: 'src_05.py', 6: 'src_06.py', 7: 'src_07.py',
    8: 'src_08.py', 9: 'src_09.py', 11: 'src_11.py', 12: 'src_12.py', 13: 'src_13.py',
    14: 'src_14.py', 16: 'src_16.py', 17: 'src_17.py',
}

# Cells appended after the base cells, markdown first then code.
APPEND_MD: list[str] = []    # cells/, markdown
APPEND_CODE: list[str] = []  # cells/, code


def to_source(text: str) -> list[str]:
    """Notebook source as a list of lines, newline-terminated except the last."""
    lines = text.split('\n')
    return [ln + '\n' for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _cell_type_for(fname):
    """Cell type a cells/ filename supplies: '.md' is markdown, else code."""
    return 'markdown' if fname.endswith('.md') else 'code'


def _read_source(fname):
    """Read one cell source from cells/, naming the path if it is missing."""
    path = os.path.join(CELLS, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f'no such cell source: {path}')
    with open(path, encoding='utf-8') as f:
        return f.read()


def build(out_path: str) -> None:
    with open(BASE, encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    if len(cells) != N_BASE_CELLS:
        raise ValueError(f'base has {len(cells)} cells, expected {N_BASE_CELLS}')

    # --- replacements, in ascending index order so the log is deterministic ---
    for idx in sorted(REPLACE):
        fname = REPLACE[idx]
        if not 0 <= idx < len(cells):
            raise ValueError(
                f'REPLACE index {idx} is outside the {len(cells)}-cell base')
        want = _cell_type_for(fname)
        got = cells[idx]['cell_type']
        if got != want:
            raise ValueError(
                f'REPLACE[{idx}] = {fname} supplies a {want} cell but base cell {idx} is {got}')
        new = _read_source(fname)
        old_len = len(''.join(cells[idx]['source']))
        cells[idx]['source'] = to_source(new)
        print(f'  cell {idx:>2} <- {fname:<14} ({old_len} -> {len(new)} chars)')

    # --- appended cells -------------------------------------------------------------------
    appended = []
    for fname in APPEND_MD:
        if _cell_type_for(fname) != 'markdown':
            raise ValueError(f'APPEND_MD entry {fname} is not a .md file')
        appended.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': to_source(_read_source(fname)),
        })
    for fname in APPEND_CODE:
        if _cell_type_for(fname) == 'markdown':
            raise ValueError(f'APPEND_CODE entry {fname} is a .md file')
        appended.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': to_source(_read_source(fname)),
        })
    base_n = len(cells)
    cells.extend(appended)

    # The output is a fresh artifact, so it must not claim to have been run:
    # saved outputs belong to a previous build's code, not to this one's.
    for c in cells:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    nb['cells'] = cells
    # Write atomically: an interrupted build must not leave a half-written
    # artifact where a complete one used to be.
    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, out_path)

    print(f'  {len(REPLACE)} replaced, {len(appended)} appended '
          f'({base_n} base cells -> {len(cells)} total)')
    print(f'wrote {out_path}  ({os.path.getsize(out_path):,} bytes)')


if __name__ == '__main__':
    build(OUT)
