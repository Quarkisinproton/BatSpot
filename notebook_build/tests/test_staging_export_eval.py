#!/usr/bin/env python3
"""Cells 10 (staging), 15 (export) and 17 (evaluate) -- the three ported cells that
no earlier suite covers, tested by RUNNING them, not by reading them.

Every check here contrasts the cell's behaviour against a concrete alternative outcome,
so a cell that did nothing, or did the wrong thing, fails:

  * c09  the four official models are all basenamed `ANIMAL-SPOT.pk`. A basename copy
        collides SILENTLY: the classifier overwrites the detector and you fine-tune a
        "detector" from 15-class weights and never know (AGENTS.md 2.2). The check reads
        the staged files' contents, so a collision is visible, not just a name difference.
        It also pins the precedence rule (manual Cell 2 paths > auto-discovery) and the
        config-vs-dataOpts check, which must be able to FAIL.
  * c14  `export_pk` used to call `model.cpu()` on the LIVE model, moving every trained
        model off the GPU (AGENTS.md 10.1). The cell is exec'd with a module whose
        `.cpu()` records the call, so the check is device-independent and falsifiable on
        a CPU-only box. It also pins the three declared outputs: `classifier_250khz.pk`
        from the BEST member, one file per member, and the "unknown" sidecar.
  * c16  `num_mels` fallback, class names indexed by output width (not by sorted dict
        value), and the `no_signal` flag that the summary cell's NO SIGNAL row depends on.

Run: venv/bin/python notebook_build/tests/test_staging_export_eval.py
"""
import contextlib
import hashlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_STAGING, CELL_EXPORT, CELL_EVAL = 9, 14, 16

fails = []


def check(name, cond, detail=''):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


def run_cell(i, ns):
    """Exec merged cell i in `ns` with its stdout captured; return (path, output).

    A cell that cannot run at all (an unported cell asking for a name this fixture does
    not have) raises HERE. That is a legitimate RED, but it must not hide every later
    check, so the failure is recorded and the section continues.
    """
    path = extract_cells.merged_cells()[i]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(open(path, encoding='utf-8').read(), path, 'exec'), ns)
    except Exception as e:
        check(f'cell {i} runs at all', False,
              f'{type(e).__name__}: {e} -- the cell asks for a name the fixture does not '
              f'supply, i.e. it is the unported version or its interface changed')
    return path, buf.getvalue()


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


# =========================================================================================
# Cell 10 -- staging the four official .pk files under UNIQUE names
# =========================================================================================
print('=== Cell 10: .pk staging ===')

# The real layout: four files that all share the basename ANIMAL-SPOT.pk.
DET_DATA = {'sr': 192000, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
            'fmin': 1000, 'fmax': 95000, 'freq_compression': 'linear'}
CLS_DATA = {'sr': 250000, 'n_fft': 256, 'hop_length': 128, 'n_freq_bins': 256,
            'fmin': 10000, 'fmax': 125000, 'freq_compression': 'linear'}
# The official classifier's 15 European species codes, with `noise` at index 8 -- read from
# the real .pk, because a name-based lookup depends on it and a wrong index would silently
# score the wrong probability (AGENTS.md 9.1).
CLS_15 = {c: i for i, c in enumerate(
    ['barba', 'bles', 'brandt', 'dlug', 'dsch', 'epar', 'espe', 'evae', 'noise', 'myam',
     'myau', 'mybr', 'myca', 'mych', 'myli'])}
assert CLS_15['noise'] == 8, 'noise must sit at index 8, as in the official .pk'
DET_2 = {'noise': 0, 'call': 1}


def write_pk(path, classes, data, tag=1):
    """A .pk with the six real keys. `tag` makes each file's CONTENTS differ, the way the
    four real models differ (they are separately trained, not copies of one another)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'encoderOpts': {'mic_tag': tag},
                'classifierOpts': {'num_classes': len(classes)},
                'dataOpts': dict(data), 'encoderState': {'w': torch.tensor([float(tag)])},
                'classifierState': {}, 'classes': dict(classes)}, path)
    return path


def staging_tree(root, tag):
    """Four official models, all basenamed ANIMAL-SPOT.pk, in the real folder layout."""
    b = os.path.join(root, tag, 'batspot')
    return {
        'm03': write_pk(f'{b}/models_call_detector/m03/train/ANIMAL-SPOT.pk', DET_2,
                        DET_DATA, tag=3),
        'm09d': write_pk(f'{b}/models_call_detector/m09/train/ANIMAL-SPOT.pk', DET_2,
                         DET_DATA, tag=9),
        'm11': write_pk(f'{b}/models_call_detector/m11/train/ANIMAL-SPOT.pk', DET_2,
                        DET_DATA, tag=11),
        'cls': write_pk(f'{b}/models_call_classifier/m09/train/ANIMAL-SPOT.pk', CLS_15,
                        CLS_DATA, tag=90),
    }


tmp = tempfile.mkdtemp(prefix='batspot-staging-')
cwd = os.getcwd()
try:
    tree = staging_tree(tmp, 'official')
    out_dir = os.path.join(tmp, 'staged')
    base_cfg = dict(n_fft=256, hop_length=128, n_freq_bins=256, freq_compression='linear')

    def staging_ns(paths, det_cfg=None, cls_cfg=None):
        det = {'sr': 192000, 'fmin': 1000, 'fmax': 95000, **base_cfg}
        det.update(det_cfg or {})
        cls = {'sr': 250000, 'fmin': 10000, 'fmax': 125000, **base_cfg}
        cls.update(cls_cfg or {})
        return {
            'os': os, 'glob': __import__('glob'), 'shutil': shutil, 'torch': torch,
            'PRETRAINED_DIR': out_dir, 'USE_MANUAL_MODEL_PATHS': True,
            'DETECTOR_M03_PATH': paths['m03'], 'DETECTOR_M09_PATH': paths['m09d'],
            'DETECTOR_M11_PATH': paths['m11'], 'CLASSIFIER_M09_PATH': paths['cls'],
            'DET_CONFIG': det, 'CLS_CONFIG': cls, 'MODEL_OUTPUT_DIR': None,
        }

    ns = staging_ns(tree)
    p_stage, out_stage = run_cell(CELL_STAGING, ns)
    staged = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    check('four DISTINCT files are staged (a basename copy yields one)',
          len(staged) == 4, f'{staged}')
    digests = {n: md5(os.path.join(out_dir, n)) for n in staged}
    check('the four staged files have four different contents -- no silent overwrite',
          len(set(digests.values())) == 4,
          f'{len(set(digests.values()))} distinct md5 over {len(digests)} files')
    check('detector and classifier start points are different files',
          ns['detector_path'] != ns['classifier_path'],
          f'{os.path.basename(ns["detector_path"])} vs {os.path.basename(ns["classifier_path"])}')
    check('the classifier still holds its 15 classes (not overwritten by a detector)',
          len(ns['PRETRAINED_MODELS']['classifier']['m09']['classes']) == 15,
          f"{len(ns['PRETRAINED_MODELS']['classifier']['m09']['classes'])} classes")
    check('every detector variant is staged, with 2 classes each',
          sorted(ns['PRETRAINED_MODELS']['detector']) == ['m03', 'm09', 'm11']
          and all(len(v['classes']) == 2 for v in ns['PRETRAINED_MODELS']['detector'].values()),
          str(sorted(ns['PRETRAINED_MODELS']['detector'])))
    check('m09 is chosen as the paper\'s primary microphone for both roles',
          ns['detector_path'].endswith('official_detector_m09.pk')
          and ns['classifier_path'].endswith('official_classifier_m09.pk'))
    check('a matching spectrogram config prints OK for every model',
          out_stage.count('OK        official') == 4 and 'MISMATCH' not in out_stage,
          f"{out_stage.count('OK        official')} OK / {out_stage.count('MISMATCH')} MISMATCH")
    check('the staged sr really comes from each .pk, not from the config',
          ns['PRETRAINED_MODELS']['detector']['m09']['sr'] == 192000
          and ns['PRETRAINED_MODELS']['classifier']['m09']['sr'] == 250000)

    # A deliberately wrong config must be caught; a check that cannot fail proves nothing.
    ns_bad = staging_ns(tree, det_cfg={'fmax': 90000})
    _, out_bad = run_cell(CELL_STAGING, ns_bad)
    _m = [ln for ln in out_bad.split('\n') if 'MISMATCH' in ln]
    check('a config that disagrees with dataOpts is reported as MISMATCH',
          len(_m) == 3 and all('fmax: model=95000 config=90000' in ln for ln in _m),
          f'{len(_m)} MISMATCH line(s)')

    # Precedence: manual Cell 2 paths win, and discovery does not wipe them (AGENTS.md 2.5).
    manual = staging_tree(tmp, 'manual')
    ns_m = staging_ns(manual)
    os.makedirs(out_dir, exist_ok=True)
    for n in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, n))
    _, out_m = run_cell(CELL_STAGING, ns_m)
    sources = {ns_m['PRETRAINED_MODELS'][r][m]['source']
               for r in ns_m['PRETRAINED_MODELS'] for m in ns_m['PRETRAINED_MODELS'][r]}
    check('auto-discovery does not overwrite the Cell 2 paths',
          sources == {os.path.abspath(v) for v in manual.values()}
          and all(ns_m['PRETRAINED_MODELS'][r][m]['origin'] == 'manual'
                  for r in ns_m['PRETRAINED_MODELS'] for m in ns_m['PRETRAINED_MODELS'][r]),
          f'{len(sources)} source(s), origins '
          f'{sorted({ns_m["PRETRAINED_MODELS"][r][m]["origin"] for r in ns_m["PRETRAINED_MODELS"] for m in ns_m["PRETRAINED_MODELS"][r]})}')

    # Discovery branch, with no manual paths at all: chdir into a tree that holds them.
    # buzz/social detectors are INCLUDED on purpose: they use different band limits and
    # classifying them as call detectors would feed their weights into the wrong slot.
    os.makedirs(out_dir, exist_ok=True)
    for n in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, n))
    disc = os.path.join(tmp, 'BatSpot_article', 'batspot')
    for mic in ('m03', 'm09', 'm11'):
        write_pk(f'{disc}/models_call_detector/{mic}/train/ANIMAL-SPOT.pk', DET_2, DET_DATA,
                 tag=int(mic[1:]))
    write_pk(f'{disc}/models_call_classifier/m09/train/ANIMAL-SPOT.pk', CLS_15, CLS_DATA, tag=90)
    write_pk(f'{disc}/models_buzz_detector/m06/train/ANIMAL-SPOT.pk', DET_2,
             dict(DET_DATA, fmin=15000, fmax=45000), tag=6)
    write_pk(f'{disc}/models_social_detector/m05/train/ANIMAL-SPOT.pk', DET_2,
             dict(DET_DATA, fmin=20000, fmax=90000), tag=5)
    os.chdir(tmp)
    try:
        ns_d = {'os': os, 'glob': __import__('glob'), 'shutil': shutil, 'torch': torch,
                'PRETRAINED_DIR': out_dir, 'USE_MANUAL_MODEL_PATHS': False,
                'DETECTOR_M03_PATH': None, 'DETECTOR_M09_PATH': None,
                'DETECTOR_M11_PATH': None, 'CLASSIFIER_M09_PATH': None,
                'DET_CONFIG': dict(sr=192000, fmin=1000, fmax=95000, **base_cfg),
                'CLS_CONFIG': dict(sr=250000, fmin=10000, fmax=125000, **base_cfg),
                'MODEL_OUTPUT_DIR': None}
        _, out_d = run_cell(CELL_STAGING, ns_d)
        found = {r: sorted(v) for r, v in ns_d['PRETRAINED_MODELS'].items()}
        check('with no manual paths, auto-discovery finds all four models',
              found == {'classifier': ['m09'], 'detector': ['m03', 'm09', 'm11']}, str(found))
        check('discovered models are labelled as such',
              all(ns_d['PRETRAINED_MODELS'][r][m]['origin'] == 'discovered'
                  for r in ns_d['PRETRAINED_MODELS'] for m in ns_d['PRETRAINED_MODELS'][r]))
        # A cell that mapped family -> role by substring would stage m06/m05 as call
        # detectors. Two independent tells: m06/m05 absent from the inventory, and the
        # cell saying it skipped them.
        staged_names = sorted(os.listdir(out_dir))
        check('buzz/social detectors are not staged as call detectors',
              'official_detector_m06.pk' not in staged_names
              and 'official_detector_m05.pk' not in staged_names, str(staged_names))
        check('the cell reports which families it skipped',
              'buzz_detector' in out_d and 'social_detector' in out_d,
              [ln.strip() for ln in out_d.split('\n') if 'skipped' in ln][:1])
    finally:
        os.chdir(cwd)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# =========================================================================================
# Cell 15 -- export
# =========================================================================================
print('\n=== Cell 15: export ===')


class _CpuSpy(nn.Sequential):
    """A model that records whether `.cpu()` was called ON THIS OBJECT.

    `export_pk` used to call `model.cpu()` on the live model, which moved every trained
    model off the GPU in place (AGENTS.md 10.1). On a CPU-only box `model.cpu()` is a
    no-op and the bug is invisible; the spy makes it observable without a GPU.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cpu_called = False

    def cpu(self, *a, **k):
        self.cpu_called = True
        return super().cpu(*a, **k)


def make_model(n_out, tag):
    m = _CpuSpy(nn.Linear(4, 4), nn.Linear(4, n_out))
    with torch.no_grad():
        m[0].weight.fill_(tag)
        m[1].weight.fill_(tag)
        m[1].bias.fill_(tag)
    return m


tmp2 = tempfile.mkdtemp(prefix='batspot-export-')
try:
    members = [{'seed': s, 'model': make_model(3, float(s))} for s in (42, 43, 44)]
    best = members[1]                      # NOT the first member
    det = {'m09': {'model': make_model(2, 9.0), 'encoderOpts': {'e': 1},
                   'classifierOpts': {'num_classes': 2}}}
    sidecar = {}

    def save_unknown_model(model, path, members_in):
        sidecar['model'], sidecar['path'], sidecar['members'] = model, path, members_in
        with open(path, 'wb') as f:
            np.savez(f, method=model['method'])

    work = os.path.join(tmp2, 'work')
    os.makedirs(work)
    cls_cfg = dict(sr=250000, n_fft=256, hop_length=128, n_freq_bins=256, fmin=10000,
                   fmax=125000, freq_compression='linear')
    det_cfg = dict(sr=192000, n_fft=256, hop_length=128, n_freq_bins=256, fmin=1000,
                   fmax=95000, freq_compression='linear')
    ns14 = {'os': os, 'torch': torch, 'nn': nn, 'WORKING_DIR': work, 'DET_CONFIG': det_cfg,
            'CLS_CONFIG': cls_cfg, 'det_results': det, 'cls_members': members,
            'cls_best_member': best, 'UNKNOWN_MODEL': {'method': 'mahalanobis', 'mu': 0},
            'save_unknown_model': save_unknown_model, 'encoderOpts_cls': {'e': 1},
            'cls_classifierOpts': {'num_classes': 3},
            'CLS_CLASS_TO_IDX': {'acsh': 0, 'noise': 1, 'sasa': 2}}
    p_exp, out_exp = run_cell(CELL_EXPORT, ns14)
    written = sorted(os.listdir(work))

    check('one .pk per detector variant', 'detector_192khz_m09.pk' in written, str(written))
    check('one .pk per ensemble member, named by seed',
          all(f'classifier_250khz_seed{s}.pk' in written for s in (42, 43, 44)), str(written))
    check('the GUI-facing classifier_250khz.pk is written', 'classifier_250khz.pk' in written)
    check('the "unknown" sidecar is written',
          'classifier_250khz_unknown.npz' in written, str(written))

    def load_pk(name):
        p = os.path.join(work, name)
        return torch.load(p, map_location='cpu', weights_only=False) if os.path.isfile(p) else None

    cls_pk = load_pk('classifier_250khz.pk')
    if cls_pk is None:
        check('classifier_250khz.pk could be read back', False,
              'the cell did not write it, so its contents cannot be checked')
    else:
        best_state = best['model'][1].state_dict()
        check('classifier_250khz.pk holds the BEST member, not the first one',
              torch.equal(cls_pk['classifierState']['weight'], best_state['weight'])
              and not torch.equal(cls_pk['classifierState']['weight'],
                                  members[0]['model'][1].state_dict()['weight']),
              f"weight={cls_pk['classifierState']['weight'][0][0].item()} "
              f"(member seeds {[m['seed'] for m in members]}, best seed {best['seed']})")
        _six = sorted(['classifierOpts', 'classifierState', 'classes', 'dataOpts',
                       'encoderOpts', 'encoderState'])
        check('the exported dict has exactly the six .pk keys, no more and no fewer',
              sorted(cls_pk) == _six, f'got {sorted(cls_pk)}, want {_six}')
        check('exported dataOpts come from the config, including the 250 kHz rate',
              cls_pk['dataOpts']['sr'] == 250000 and cls_pk['dataOpts']['n_freq_bins'] == 256)
    per_seed_ok = all(
        (lambda pk, s: pk is not None and torch.equal(
            pk['classifierState']['weight'],
            next(m for m in members if m['seed'] == s)['model'][1].state_dict()['weight']))(
            load_pk(f'classifier_250khz_seed{s}.pk'), s)
        for s in (42, 43, 44))
    check('each member file holds its OWN member', per_seed_ok)
    check('the sidecar records WHICH member files it belongs to',
          bool(sidecar.get('members')) and [os.path.basename(p) for p in sidecar['members']]
          == ['classifier_250khz_seed42.pk', 'classifier_250khz_seed43.pk',
              'classifier_250khz_seed44.pk'],
          str([os.path.basename(p) for p in sidecar.get('members') or []]))

    moved = [f'cls{m["seed"]}' for m in members if m['model'].cpu_called] + \
            [f'det/{m}' for v in det.values() if v['model'].cpu_called]
    check('export_pk does NOT move the live models to the CPU in place',
          not moved, f'moved: {moved}' if moved else 'no model had .cpu() called on it')

    # DataParallel: train_model returns the WRAPPER, and its state_dict keys would all be
    # `module.`-prefixed, so an exported .pk would not load (AGENTS.md 2.9). export_pk must
    # unwrap first. On a CPU-only box `nn.DataParallel` still constructs and `.module` is the
    # wrapped Sequential, so this is testable here; without the unwrap, `model[0]` indexes the
    # wrapper (which has no __getitem__) and the cell raises.
    dp_work = os.path.join(tmp2, 'dp')
    os.makedirs(dp_work)
    inner = make_model(3, 7.0)
    wrapped = nn.DataParallel(inner)
    ns_dp = dict(ns14, WORKING_DIR=dp_work,
                 cls_members=[{'seed': 99, 'model': wrapped}],
                 cls_best_member={'seed': 99, 'model': wrapped},
                 UNKNOWN_MODEL=None)
    # Exec'd inline rather than through run_cell(), so the exception is available here as a
    # value instead of as a printed FAIL line the next check has to scrape back out.
    _dp_path = extract_cells.merged_cells()[CELL_EXPORT]
    dp_err, _dp_out = None, io.StringIO()
    try:
        with contextlib.redirect_stdout(_dp_out):
            exec(compile(open(_dp_path, encoding='utf-8').read(), _dp_path, 'exec'), ns_dp)
    except Exception as e:
        dp_err = f'{type(e).__name__}: {e}'
    check('a DataParallel-wrapped model exports without error',
          dp_err is None, dp_err or 'exported')
    if dp_err is None:
        dp_pk = torch.load(os.path.join(dp_work, 'classifier_250khz.pk'),
                           map_location='cpu', weights_only=False)
        check('the exported weights carry NO "module." prefix (they would not load otherwise)',
              all(not k.startswith('module.') for k in dp_pk['encoderState'])
              and all(not k.startswith('module.') for k in dp_pk['classifierState']),
              f"encoder keys e.g. {sorted(dp_pk['encoderState'])[:2]}")
        check('the exported weights are the wrapped model\'s own',
              torch.equal(dp_pk['classifierState']['weight'],
                          inner[1].state_dict()['weight']))
    check('the reuse hint names every member file for INFER_CLASSIFIER_PK',
          'classifier_250khz_seed42.pk' in out_exp and 'classifier_250khz_seed44.pk' in out_exp)
    check('the min_max_norm warning is still printed',
          'min_max_norm=true' in out_exp)
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

# =========================================================================================
# Cell 17 -- evaluating an existing model on the test split
# =========================================================================================
print('\n=== Cell 17: evaluate existing models ===')


class _Model:
    def __init__(self, classes):
        self.classes = classes

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def eval_ns(models, tag_map=None):
    """A namespace that runs the whole evaluation cell over `models` (path -> spec).

    `WINDOWED` records, per model, the dataset config the cell built for it. The model is
    identified by the `load_any_model` call that immediately precedes it, so the record is
    keyed by the .pk path rather than by the file list -- every model is scored on the SAME
    test clips, so the file list cannot tell them apart.
    """
    queue = list(models.values())

    def load_any_model(path, device):
        spec = models[path]
        _current.append(path)
        return _Model(spec['classes']), spec['classes'], spec['dataOpts']

    def windowed(files_, label_map, cfg, train=False):
        WINDOWED.append((_current[-1] if _current else None, list(files_),
                         dict(label_map), dict(cfg)))
        return list(files_)

    def predict_proba(model, ds, device):
        spec = queue.pop(0)
        return np.asarray(spec['probs']), np.asarray(spec['labels'])

    WINDOWED, _current = [], []
    return {
        'os': os, 'np': np, 'torch': torch,
        'roc_auc_score': roc_auc_score, 'accuracy_score': accuracy_score,
        'balanced_accuracy_score': balanced_accuracy_score, 'confusion_matrix': confusion_matrix,
        'load_any_model': load_any_model, 'WindowedBatDataset': windowed,
        'predict_proba': predict_proba, 'get_class_from_filename':
            lambda f: os.path.basename(f).split('-', 1)[0],
        'AUC_NO_SIGNAL': 0.05, 'DEVICE': 'cpu', 'model_inventory': list(models),
        'model_tags': tag_map or {p: 'official' for p in models},
        'test_wavs': FILES,
    }, WINDOWED


FILES = ['/d/acsh-bat_1.wav', '/d/sasa-bat_2.wav', '/d/noise-bat_3.wav', '/d/acsh-bat_4.wav',
         '/d/sasa-bat_5.wav', '/d/noise-bat_6.wav']
y2 = np.array([1, 1, 0, 1, 1, 0])                       # 4 calls, 2 noise

# The OFFICIAL DETECTOR class list is {noise, target}, not {noise, call}: `target` is the
# class the cell reads by NAME to get P(call). A fixture using 'call' would never enter the
# detector branch, and then a hard-coded `probs[:, 1]` would give the same number as the
# name-based lookup (the two columns are complementary here), hiding the bug.
DET_OFFICIAL = {'noise': 0, 'target': 1}

p_live = np.stack([1 - np.linspace(0.05, 0.95, 6), np.linspace(0.05, 0.95, 6)], 1)
# The signal-free 15-class classifier: P(noise) = 0.9 for EVERY clip, so it answers "call"
# for all of them -- exactly the 0.8069-accuracy/AUC-0.49 row AGENTS.md 9.2 describes. The
# AUC is at chance by construction of the fixture, not of the cell.
p_dead = np.full((len(FILES), 15), 0.1 / 14)
p_dead[:, CLS_15['noise']] = 0.9
# A model that stores num_mels and NOT n_freq_bins, which predict.py's fallback chain handles
# and a hard-coded 256-bin default would silently get wrong.
MEL_DATAOPTS = {'sr': 192000, 'n_fft': 256, 'hop_length': 128,
                'num_mels': 64, 'fmin': 1000, 'fmax': 95000}
p_multi = np.eye(6)[np.array([0, 5, 1, 0, 5, 1])]
ns16, w16 = eval_ns({
    '/m/detector.pk': {'classes': DET_OFFICIAL, 'dataOpts': dict(DET_DATA),
                       'probs': p_live, 'labels': y2},
    '/m/dead_classifier.pk': {'classes': CLS_15, 'dataOpts': dict(CLS_DATA),
                              'probs': p_dead, 'labels': y2},
    '/m/mel_model.pk': {'classes': DET_OFFICIAL, 'dataOpts': MEL_DATAOPTS,
                        'probs': p_live, 'labels': y2},
})
p16, out16 = run_cell(CELL_EVAL, ns16)
res16 = ns16.get('val_results', [])

check('every model is scored on the same test clips',
      len(res16) == 3 and all(r.get('n_files') == len(FILES) for r in res16),
      str([r.get('n_files') for r in res16]))
# The dataset configs are captured per model and keyed by the .pk they were built for, so each
# is asserted against the dataOpts of that model. An earlier version filtered
# `if c['n_freq_bins'] != 64`, which removed the mel model from its own check, so that check
# could not fail on it.
_bins = {path: cfg['n_freq_bins'] for path, _f, _l, cfg in w16}
check('a model storing num_mels is fed num_mels bins, not the 256-bin default',
      _bins.get('/m/mel_model.pk') == 64,
      f"mel model (num_mels=64) got {_bins.get('/m/mel_model.pk')}")
check('n_freq_bins from dataOpts is preferred when present (the detector, 256 bins)',
      _bins.get('/m/detector.pk') == 256,
      f"detector (n_freq_bins=256) got {_bins.get('/m/detector.pk')}")
check('each of the three models got a config of its own',
      len(_bins) == 3 and all(v is not None for v in _bins.values()), str(_bins))
check('the signal-free model is flagged no_signal; the informative one is not',
      len(res16) == 3 and res16[1].get('no_signal') is True
      and res16[0].get('no_signal') is False and res16[2].get('no_signal') is False,
      f"dead={res16[1].get('no_signal') if len(res16) > 1 else None}, "
      f"live={res16[0].get('no_signal') if res16 else None}")
check('the printed line carries the NO SIGNAL marker only for the dead model',
      out16.count('[NO SIGNAL: AUC ~ 0.5]') == 1, str(out16.count('[NO SIGNAL: AUC ~ 0.5]')))
check('the signal-free model really is at chance',
      len(res16) > 1 and res16[1].get('auc') is not None
      and abs(res16[1]['auc'] - 0.5) < 1e-9
      and abs(res16[1]['balanced_accuracy'] - 0.5) < 1e-9,
      f"auc={res16[1]['auc']:.4f}, bal={res16[1]['balanced_accuracy']:.4f}"
      if len(res16) > 1 and res16[1].get('auc') is not None else 'no result')
check('a model with no species overlap is scored as noise-vs-call, on all 6 clips',
      len(res16) > 1 and res16[1].get('kind', '').startswith('classifier')
      and res16[1].get('n_files') == len(FILES),
      f"kind={res16[1].get('kind') if len(res16) > 1 else None}")
# The detector branch must read P(call) from the class NAMED 'target'. The official class
# list is {noise: 0, target: 1}, so a hard-coded `probs[:, 1]` happens to agree with the
# name-based lookup on this fixture -- which is exactly why the check below needs a fixture
# where the two DIFFER: `target` at index 0 and `noise` at 1, which is a legal 2-output dict.
# Under the name-based lookup the cell reads column 0; under the hard-coded one, column 1.
# If the cell ever hard-codes an index, this is the row that catches it.
DET_SWAPPED = {'target': 0, 'noise': 1}
# Column 0 is P(call) under the name-based lookup. The values are chosen so the two candidate
# lookups give DIFFERENT accuracies: by-name (col 0) scores 4/6, the complement (col 1) 2/6.
# A symmetric ramp would score 3/6 either way and hide the bug.
_p_call = np.array([0.9, 0.9, 0.1, 0.9, 0.1, 0.9])
ns16s, _ = eval_ns({'/m/swapped.pk': {'classes': DET_SWAPPED,
                                       'dataOpts': dict(DET_DATA),
                                       'probs': np.stack([_p_call, 1 - _p_call], 1),
                                       'labels': y2}})
run_cell(CELL_EVAL, ns16s)
_sw = ns16s.get('val_results', [{}])[0]
check('P(call) is read by class NAME, not by a hard-coded column index',
      _sw.get('accuracy') is not None and abs(_sw['accuracy'] - 4 / 6) < 1e-9,
      f"accuracy={_sw.get('accuracy')}, preds={_sw.get('preds')}; "
      f"reading column 1 instead would give {2 / 6:.4f}")

# A classifier whose class indices are NON-CONTIGUOUS: names must come from the output width.
NS_NONCONTIG = {'noise': 0, 'acsh': 1, 'sasa': 5}
ns16b, _ = eval_ns({
    '/m/gappy.pk': {'classes': NS_NONCONTIG, 'dataOpts': dict(CLS_DATA),
                    'probs': p_multi, 'labels': np.array([0, 5, 1, 0, 5, 1])},
})
p16b, _ = run_cell(CELL_EVAL, ns16b)
g = ns16b['val_results'][0]
check('class names are indexed by output width, so a gap in the indices is kept',
      g.get('target_names') == ['noise', 'acsh', 'class2', 'class3', 'class4', 'sasa'],
      str(g.get('target_names')))
check('the confusion matrix is square over the model OUTPUT width',
      g.get('confusion_matrix') is not None and g['confusion_matrix'].shape == (6, 6),
      str(getattr(g.get('confusion_matrix'), 'shape', None)))

# A non-linear band compression must be flagged as not comparable.
ns16c, _ = eval_ns({
    '/m/melbank.pk': {'classes': DET_OFFICIAL,
                      'dataOpts': dict(DET_DATA, freq_compression='mel'),
                      'probs': p_live, 'labels': y2},
})
_, out16c = run_cell(CELL_EVAL, ns16c)
check('a non-linear freq_compression prints a not-comparable warning',
      'freq_compression' in out16c and 'not comparable' in out16c,
      [ln.strip() for ln in out16c.split('\n') if 'WARNING' in ln][:1])

print('\n' + '=' * 66)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)