#!/usr/bin/env python3
"""The four inference cells, on real audio, with real (deterministically weighted) models.

Cells 20-23 are the part of the notebook that runs on the operator's own recordings:

  cell 20  picks the detector + classifier (in memory, or exported .pk files, alone or as an
           ensemble) and refuses an 'unknown' sidecar fitted for different models;
  cell 21  scans each recording in 20 ms windows, merges the positive ones into selections,
           classifies them, measures Low/High/Peak frequency and the clock times;
  cell 22  lists a folder (or a .zip, or one file), survives a corrupt file, splits the
           classifier's 'noise' answers off and writes the detections file + Raven tables;
  cell 23  scores the output against the operator's own Raven tables.

The models are the architecture cell's real `Classifier` + `_assemble` with a deterministic
encoder and head, so `model[1].linear.out_features`, `.to(DEVICE).eval()`, `_layer_output`,
`SoftmaxEnsemble` and `forward_with_embedding` all behave as after training -- what changes is
that a high-frequency burst scores as a call and silence as noise, reproducibly, and that two
ensemble members disagree (see _inferfix). The audio is real files written by `soundfile`. The
only stubbed function is `load_model_from_pk` (see _inferfix), and `_model_from_pk` -- the thing
that consumes it -- is the subject of seven checks.

THE HARNESS, and why it is built this way
Every check is a plain function returning (cond, detail); `check` turns an exception inside one
into a FAIL carrying the exception rather than a traceback. Every SETUP (a scan, a cell exec, a
file read) goes through `V()`, which computes it once under a name and turns a raising setup into
one FAIL plus a reason that every check consuming it then reports. The unported base cells fail
by RAISING -- a list of .pk paths is not a path, `classify_selections` returns one array where this
notebook returns two -- so without this the suite would die on the third check and say nothing
about the other three cells.

Run: venv/bin/python notebook_build/tests/test_inference.py
"""
import csv
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _inferfix
import extract_cells

fails = []


class SetupFailed(RuntimeError):
    """A setup the checks depend on could not be built (see V())."""


_CACHE = {}


def check(name, thunk):
    """Call `thunk` -> (cond, detail). An exception in it is a FAIL, not a dead suite."""
    try:
        cond, detail = thunk()
    except Exception as e:                                        # noqa: BLE001
        cond, detail = False, f'{type(e).__name__}: {e}'
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)
    return cond


def V(name, factory):
    """Compute a setup once under a name; a raising setup becomes one FAIL plus a reason every
    dependent check then reports. Returns the value, or raises SetupFailed."""
    if name not in _CACHE:
        try:
            _CACHE[name] = (factory(), None)
        except Exception as e:                                    # noqa: BLE001
            _CACHE[name] = (None, f'{type(e).__name__}: {e}')
            fails.append(f'{name} (setup)')
            print(f'  [FAIL] {name} (setup)  -- {type(e).__name__}: {e}')
    value, err = _CACHE[name]
    if err is not None:
        raise SetupFailed(f'{name} could not be built: {err}')
    return value


def raises(fn, exc):
    """(raised, message): True when `fn` raised `exc`, so a MISSING guard is a failing check
    rather than a crash that loses the rest of the suite."""
    try:
        fn()
    except exc as e:
        return True, str(e)
    except Exception as e:                                        # noqa: BLE001
        return False, f'raised {type(e).__name__} instead: {e}'
    return False, 'did not raise'


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


def clock_seconds(text):
    """'20:40:00.4000' -> seconds since midnight, so a clock time can be compared to a time."""
    h, m, s = text.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def first_line_with(text, needle):
    return next((ln.strip() for ln in text.split('\n') if needle in ln), '(not reported)')


def synth(sr, dur_s, parts, amp=0.35):
    """Silence with each (t0, t1, f_lo, f_hi) part an FM sweep under a Hann envelope."""
    x = np.zeros(int(round(sr * dur_s)), np.float32)
    for t0, t1, f_lo, f_hi in parts:
        i0, i1 = int(round(t0 * sr)), int(round(t1 * sr))
        m = i1 - i0
        freq = np.linspace(f_lo, f_hi, m)
        x[i0:i1] += amp * np.hanning(m) * np.sin(2 * np.pi * np.cumsum(freq) / sr).astype(np.float32)
    return x


idx = extract_cells.merged_cells()
print(f'=== under test: {os.environ.get("NEW_CELLS") or extract_cells.MERGED} '
      f'(cells {[os.path.basename(idx[i]) for i in (20, 21, 22, 23)]}) ===')
arch = _inferfix.arch_ns(idx)
work = _inferfix.working_dir()

# =============================================================================================
print('\n=== cell 20: which models it picks, and what it refuses ===')
# DET_CLASSES is {'call': 0, 'noise': 1} -- the read-only base notebook's official detector
# layout, i.e. NOT in name order. A cell reading probs[:, 1] instead of looking the name up
# would score 'noise' as 'call' on every window, so the call_idx check below is not cosmetic.
PKS = {
    'detector_192khz_m09.pk': (2, dict(_inferfix.DET_CLASSES), _inferfix.DET_DATA_OPTS),
    'classifier_250khz_seed42.pk': (3, {'rhro': 0, 'noise': 1, 'rhle': 2}, _inferfix.CLS_DATA_OPTS),
    'classifier_250khz_seed43.pk': (3, {'rhro': 0, 'noise': 1, 'rhle': 2}, _inferfix.CLS_DATA_OPTS),
    'classifier_other_classes.pk': (3, {'rhbe': 0, 'noise': 1, 'rhle': 2}, _inferfix.CLS_DATA_OPTS),
    'detector_384khz.pk': (2, dict(_inferfix.DET_CLASSES),
                           dict(_inferfix.DET_DATA_OPTS, sr=384000)),
}
pk = {name: os.path.join(work, 'pks', name) for name in PKS}
for p in pk.values():
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').close()             # the stub loader keys on the basename only
loader = _inferfix.stub_loader(PKS, arch)
DET_PK, CLS1, CLS2 = pk['detector_192khz_m09.pk'], pk['classifier_250khz_seed42.pk'], \
    pk['classifier_250khz_seed43.pk']
ONE_PK = ('cell20: one exported .pk', lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=DET_PK, INFER_CLASSIFIER_PK=CLS1))
ENS_PK = ('cell20: two .pk as an ensemble', lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=DET_PK, INFER_CLASSIFIER_PK=[CLS1, CLS2]))
side_ok = _inferfix.write_unknown_sidecar(os.path.join(work, 'side_ok.npz'), [CLS1, CLS2])
side_one = _inferfix.write_unknown_sidecar(os.path.join(work, 'side_one.npz'), [CLS1])
SIDE_PK = ('cell20: sidecar for this ensemble', lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=DET_PK, INFER_CLASSIFIER_PK=[CLS1, CLS2],
    INFER_UNKNOWN_FILE=side_ok))


def one_model_is_not_an_ensemble():
    model = V(*ONE_PK)[0]['INFER_CLS']['model']
    return not isinstance(model, arch['SoftmaxEnsemble']), \
        f"classes {V(*ONE_PK)[0]['INFER_CLS']['names']}, {type(model).__name__}"


def call_class_by_name():
    got = V(*ONE_PK)[0]['INFER_DET']['call_idx']
    return got == _inferfix.DET_CLASSES['call'], \
        (f"classes {_inferfix.DET_CLASSES} -> call_idx={got}, where the hard-coded index 1 "
         f"is 'noise'")


def list_becomes_ensemble():
    ns = V(*ENS_PK)[0]
    model = ns['INFER_CLS']['model']
    # `ns['SoftmaxEnsemble']` is the class THIS cell exec'd: every call to arch_ns() exec's the
    # architecture cell again, so isinstance against another exec's class object is always False.
    return (isinstance(model, ns['SoftmaxEnsemble']) and len(list(model.members)) == 2,
            f'{type(model).__name__} with {len(getattr(model, "members", []))} member(s)')


def ensemble_averages_its_members():
    """softmax of a SoftmaxEnsemble must equal the MEAN of its members' softmax, and the two
    members must actually DISAGREE -- otherwise this check would also pass for an ensemble that
    just returns member 0, which is the bug it exists to catch."""
    ns = V(*ENS_PK)[0]
    ens = ns['INFER_CLS']['model']
    if not isinstance(ens, ns['SoftmaxEnsemble']):
        return False, f'{type(ens).__name__}: a list of .pk did not become an ensemble'
    members = list(ens.members)
    x = torch.zeros(1, 1, 30, 256)
    with torch.no_grad():      # the stub models' parameters require grad; numpy() refuses them
        per_member = [torch.softmax(m(x).float(), 1)[0].numpy() for m in members]
        got = torch.softmax(ens(x).float(), 1)[0].numpy()
    want = np.mean(per_member, axis=0)
    spread = float(np.abs(per_member[0] - per_member[1]).max()) if len(per_member) > 1 else 0.0
    return (spread > 1e-3 and float(np.abs(got - want).max()) < 1e-6), \
        (f'{len(members)} members, their widest disagreement {spread:.3e} (so averaging is '
         f'observable), max|ensemble - mean(members)| = {float(np.abs(got - want).max()):.3e}')


check('a single exported .pk becomes a plain model, not an ensemble', one_model_is_not_an_ensemble)
check("the detector's call class is found BY NAME, not by a hard-coded column", call_class_by_name)
check('a LIST of .pk files becomes a SoftmaxEnsemble holding both members', list_becomes_ensemble)
check("   ...and its output IS the mean of those members' probabilities", ensemble_averages_its_members)


def refused(*args):
    got, why = raises(*args)
    return got, why[:88]


got_, why = refused(lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=DET_PK,
    INFER_CLASSIFIER_PK=[CLS1, pk['classifier_other_classes.pk']]), ValueError)
check('an ensemble member with DIFFERENT classes is refused', lambda: (got_, why))

got_, why = refused(lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=[DET_PK, pk['detector_384khz.pk']],
    INFER_CLASSIFIER_PK=CLS1), ValueError)
check('an ensemble member with DIFFERENT data options (a 384 kHz detector) is refused',
      lambda: (got_, why))

got_, why = refused(lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=CLS1, INFER_CLASSIFIER_PK=CLS1),
    AssertionError)
check('a 3-class classifier .pk offered as the detector is refused', lambda: (got_, why))


def sidecar_loaded():
    um, out = V(*SIDE_PK)
    return (um['INFER_UNKNOWN'] is not None and um['INFER_UNKNOWN']['method'] == 'msp'
            and 'unknown   : "unknown"' in out), \
        f"method={um['INFER_UNKNOWN'] and um['INFER_UNKNOWN']['method']}"


check("the 'unknown' sidecar is loaded when it was fitted for exactly these models", sidecar_loaded)

got_, why = refused(lambda: _inferfix.config_ns(
    idx, load_model_from_pk=loader, INFER_DETECTOR_PK=DET_PK, INFER_CLASSIFIER_PK=[CLS1, CLS2],
    INFER_UNKNOWN_FILE=side_one), ValueError)
check('a sidecar fitted for a DIFFERENT set of models is refused (its threshold would be '
      'meaningless for these)', lambda: (got_, why))

# --- the in-memory path: 'auto' selects on VALIDATION, never on the test score -----------------
det_models = {m: _inferfix.detector_model(arch) for m in ('m03', 'm09', 'm11')}
det_results = {
    # m03 has by far the best TEST accuracy and must still not be chosen for it.
    'm03': _inferfix.det_result(det_models['m03'], 0.900, test_acc=0.10),
    'm09': _inferfix.det_result(det_models['m09'], 0.880, test_acc=0.99),
    'm11': _inferfix.det_result(det_models['m11'], 0.900, test_acc=0.50),
}
mem = dict(det_results=det_results, cls_model=_inferfix.classifier_model(arch),
           cls_members=[{'seed': 42}], cls_best_acc=0.87, UNKNOWN_MODEL=None)
MEM_MODELS = ('cell20: in-memory models', lambda: _inferfix.config_ns(
    idx, INFER_DETECTOR_PK=None, INFER_CLASSIFIER_PK=None, **mem))
# m09 and m11 with the SAME validation score and different test scores, so choosing m09 can only
# come from the tie-break. (det_results above has m09 at val 0.880, so it cannot be reused here.)
tie_results = {'m09': _inferfix.det_result(det_models['m09'], 0.900, test_acc=0.10),
               'm11': _inferfix.det_result(det_models['m11'], 0.900, test_acc=0.99)}
TIE_MODELS = ('cell20: an m09/m11 tie', lambda: _inferfix.config_ns(
    idx, INFER_DETECTOR_PK=None, INFER_CLASSIFIER_PK=None,
    **dict(mem, det_results=tie_results)))


def auto_uses_validation():
    label = V(*MEM_MODELS)[0]['INFER_DET']['label']
    return label.startswith('fine-tuned detector m03'), label


def tie_does_not_fall_to_m09():
    label = V(*MEM_MODELS)[0]['INFER_DET']['label']
    return 'm09' not in label, f'm03 and m11 both have val 0.9000; chose {label}'


def tie_with_m09_goes_to_m09():
    label = V(*TIE_MODELS)[0]['INFER_DET']['label']
    return label.startswith('fine-tuned detector m09'), label


def noise_index_by_name():
    got = V(*MEM_MODELS)[0]['INFER_CLS']['noise_idx']
    names = V(*MEM_MODELS)[0]['INFER_CLS']['names']
    return got == _inferfix.CLS_CLASSES.index('noise'), f'classes {names} -> noise_idx={got}'


def combo_is_printed():
    out = V(*MEM_MODELS)[1]
    return ('rhro' in out and '10 ms hop' in out and 'threshold 0.5' in out
            and 'unknown   : off' in out), first_line_with(out, 'scan:')


check("INFER_DETECTOR='auto' follows the VALIDATION score, not the best test accuracy",
      auto_uses_validation)
check('   ...so a tie between m03 and m11 cannot fall to m09', tie_does_not_fall_to_m09)
check("   ...but a tie m09 IS part of goes to m09, the paper's primary microphone",
      tie_with_m09_goes_to_m09)
check("the classifier's 'noise' index is found by NAME", noise_index_by_name)
check('the combo prints the classes, the scan settings and the unknown threshold it will use',
      combo_is_printed)

# =============================================================================================
print('\n=== cell 21: one recording end to end (real scan, real selections, real measurements) ===')
# A 4 s 384 kHz recording with three things in it:
#   0.30-0.42 s  a 40-50 kHz sweep   -- inside the detector's band, below the stub's 60 kHz edge
#   1.50-1.90 s  a 70-85 kHz sweep   -- the only thing that may be detected
#   3.00-3.01 s  a 400 Hz tone       -- below the detector's fmin (1 kHz), cropped away entirely
x = synth(384000, 4.0, [(0.30, 0.42, 40000, 50000), (1.50, 1.90, 70000, 85000)])
x[int(3.00 * 384000):int(3.01 * 384000)] += (
    0.3 * np.hanning(int(0.01 * 384000))
    * np.sin(2 * np.pi * 400 * np.arange(int(0.01 * 384000)) / 384000)).astype(np.float32)
REC = os.path.join(work, 'rec_20260521_204000.wav')
sf.write(REC, x, 384000, subtype='FLOAT')

DET = {'model': _inferfix.detector_model(arch), 'cfg': _inferfix.DET_CONFIG, 'call_idx': 1}
CLS = {'model': _inferfix.classifier_model(arch), 'cfg': _inferfix.CLS_CONFIG,
       'names': list(_inferfix.CLS_CLASSES), 'noise_idx': 0}
FNS = ('cell21: the functions cell', lambda: _inferfix.funcs_ns(idx))
SCAN = ('cell21: one pass over the recording', lambda: V(*FNS)['scan_detector'](REC, DET))
SELS = ('cell21: selections at 0.5', lambda: V(*FNS)['build_selections'](*V(*SCAN), 0.5, 0.1, 2))
CLASSIFIED = ('cell21: classification of the selection',
              lambda: V(*FNS)['classify_selections'](REC, V(*SELS), CLS))
# The stub classifier saturates at confidence 1.0, so a threshold BELOW 1.0 would never fire and
# the check would pass for the wrong reason. 1.01 rejects every known-score; the check below and
# the one after it (same selection, INFER_UNKNOWN=None) then differ only by the flag.
UNKNOWN_ON = ('cell21: the unknown answer on', lambda: _inferfix.funcs_ns(
    idx, INFER_UNKNOWN={'method': 'msp', 'threshold': 1.01, 'noise_idx': 0}
)['process_recording'](REC, DET, CLS))
ONE_PASS = ('cell21: scan in one chunk', lambda: _inferfix.funcs_ns(idx, INFER_CHUNK_S=10 ** 6)
            ['scan_detector'](REC, DET))
CHUNKED = ('cell21: scan in 0.35 s chunks', lambda: _inferfix.funcs_ns(idx, INFER_CHUNK_S=0.35)
           ['scan_detector'](REC, DET))
EXPANDED = ('cell21: 10x time-expanded copy',
            lambda: _inferfix.write_time_expanded(os.path.join(work, 'te_20260521_204000.wav'),
                                                  REC, 10))


def scan_grid():
    t0, t1, p = V(*SCAN)
    # The last window starts 30 ms before the end (hop 10 ms, window 20.5 ms), so t1[-1] lands
    # within one hop of 4.0 s rather than exactly on it.
    ok = (len(t0) == len(t1) == len(p) and t0[0] == 0.0
          and abs(float(t0[1] - t0[0]) - 0.010) < 1e-9
          and abs(float(t1[-1]) - 4.0) < 0.015 and 395 <= len(p) <= 400)
    return ok, (f'{len(p)} windows, t0[0]={t0[0]:.4f}, hop={float(t0[1] - t0[0]):.4f} s, '
                f't1[-1]={t1[-1]:.4f} for a 4 s recording')


def only_the_high_sweep():
    t0, t1, p = V(*SCAN)
    loud = [i for i in range(len(p)) if p[i] >= 0.5]
    if not loud:
        return False, 'no positive window'
    # Every positive window must lie inside the 70-85 kHz sweep. A detection on the 40-50 kHz
    # sweep (0.30-0.42 s) or the 400 Hz tone (3.00-3.01 s) is a false alarm this would catch.
    ok = (len(loud) > 20 and min(float(t0[i]) for i in loud) >= 1.49
          and max(float(t1[i]) for i in loud) <= 1.92 and max(p[i] for i in loud) > 0.99)
    return ok, (f'{len(loud)} positive window(s), all inside '
                f'[{min(float(t0[i]) for i in loud):.3f}, '
                f'{max(float(t1[i]) for i in loud):.3f}] s = the 1.50-1.90 s sweep; the '
                f'40-50 kHz sweep and the 400 Hz tone are not among them. max P = '
                f'{max(p[i] for i in loud):.4f}, silent-window P = {p[5]:.3f}')


def one_selection():
    t0, t1, p = V(*SCAN)
    sels = V(*SELS)
    if not sels:
        return False, 'no selection'
    ok = (len(sels) == 1 and 1.49 <= sels[0]['begin'] <= 1.52
          and 1.88 <= sels[0]['end'] <= 1.92
          and sels[0]['n'] == len([i for i in range(len(p)) if p[i] >= 0.5]))
    return ok, (f"[{sels[0]['begin']:.3f}, {sels[0]['end']:.3f}) over {sels[0]['n']} positive "
                f"windows (one selection)")


def min_windows_drops():
    n_loud = len([i for i in V(*SCAN)[2] if i >= 0.5])
    got = V('cell21: min_windows above the loud count',
            lambda: V(*FNS)['build_selections'](*V(*SCAN), 0.5, 0.1, n_loud + 1))
    return got == [], f'asked for {n_loud + 1} windows, kept {len(got)} selection(s)'


def impossible_threshold():
    got = V('cell21: threshold 1.01', lambda: V(*FNS)['build_selections'](*V(*SCAN), 1.01, 0.1, 2))
    return got == [], f'kept {len(got)} selection(s)'


def classify_shapes():
    probs, emb = V(*CLASSIFIED)
    ok = (probs.shape == (len(V(*SELS)), 3) and emb is not None
          and emb.shape[0] == len(V(*SELS)) and emb.shape[1] == 512)
    return ok, f'probs {probs.shape}, emb {None if emb is None else emb.shape}'


def classify_is_rhro():
    probs = V(*CLASSIFIED)[0]
    return (_inferfix.CLS_CLASSES[int(probs[0].argmax())] == 'rhro' and probs[0, 1] > 0.5), \
        str(dict(zip(_inferfix.CLS_CLASSES, probs[0].round(4))))


def classify_nothing():
    probs, emb = V('cell21: classify no selection',
                   lambda: V(*FNS)['classify_selections'](REC, [], CLS))
    return (probs.shape == (0, 3) and emb is None), f'{probs.shape}, embedding={emb}'


def unknown_replaces_the_species():
    sels = V(*UNKNOWN_ON)[0]
    if not sels:
        return False, 'no selections'
    s = sels[0]
    return (len(sels) == 1 and s['species'] == 'unknown' and s['best_guess'] == 'rhro'
            and s['is_noise'] is False and s['confidence'] > 0.5), \
        f"species={s['species']}, best_guess={s['best_guess']}, confidence={s['confidence']:.4f}"


def unknown_off_gives_argmax():
    sels = V('cell21: the unknown answer off', lambda: _inferfix.funcs_ns(
        idx, INFER_UNKNOWN=None)['process_recording'](REC, DET, CLS))[0]
    if not sels:
        return False, 'no selections'
    return (len(sels) == 1 and sels[0]['species'] == 'rhro'
            and sels[0]['best_guess'] == 'rhro'), f"species={sels[0]['species']}"


FREQ_CONSTS = V('cell20: INFER_* constants', lambda: _inferfix.infer_constants(idx)[0])
CLICK = np.zeros(int(0.3 * 384000), np.float32)
CLICK[int(0.15 * 384000)] = 1.0
CLICK_FILE = os.path.join(work, 'click_20260521_210000.wav')
sf.write(CLICK_FILE, CLICK, 384000, subtype='FLOAT')
FREQS = ('cell21: Low/High/Peak of the selection', lambda: V(*FNS)['measure_frequencies'](
    REC, V(*UNKNOWN_ON)[0][0]['begin'], V(*UNKNOWN_ON)[0][0]['end']))
CLICK_FREQ = ('cell21: Low/High/Peak of a 180 kHz click', lambda: V(*FNS)['measure_frequencies'](
    CLICK_FILE, 0.149, 0.151))


def frequencies_on_the_sweep():
    low, high, peak = V(*FREQS)
    ceil_hz, floor_hz = FREQ_CONSTS['INFER_FREQ_CEIL_HZ'], FREQ_CONSTS['INFER_FREQ_FLOOR_HZ']
    ok = (70000 - 375 <= peak <= 85000 + 375 and low <= peak <= high
          and low >= floor_hz and high <= ceil_hz)
    return ok, (f'low={low:.0f}, high={high:.0f}, peak={peak:.0f} for a 70-85 kHz sweep '
                f'(floor {floor_hz}, ceiling {ceil_hz})')


def click_stays_under_the_ceiling():
    lhp = V(*CLICK_FREQ)
    ceil_hz = FREQ_CONSTS['INFER_FREQ_CEIL_HZ']
    return (not (lhp[2] > ceil_hz) and all(np.isfinite(v) for v in lhp)), \
        f'low={lhp[0]:.0f}, high={lhp[1]:.0f}, peak={lhp[2]:.0f}, ceiling={ceil_hz}'


def start_from_the_file_name():
    start, src = V('cell21: recording start', lambda: V(*FNS)['recording_start'](REC))
    return (start is not None and src == 'file name'
            and start.strftime('%H:%M:%S') == '20:40:00'), f'{start} ({src})'


def clock_times_are_start_plus_time():
    s = V(*UNKNOWN_ON)[0][0]
    base = 20 * 3600 + 40 * 60
    return (abs(clock_seconds(s['begin_clock']) - (base + s['begin'])) < 5e-4
            and abs(clock_seconds(s['end_clock']) - (base + s['end'])) < 5e-4), \
        (f"{s['begin_clock']} .. {s['end_clock']} for [{s['begin']:.4f}, {s['end']:.4f}]")


def clock_from_zero_wraps():
    ct = V('cell21: clock_time', lambda: V(*FNS)['clock_time'])
    return (ct(None, 12.3456) == '00:00:12.3456' and ct(None, 86400.5) == '00:00:00.5000'), \
        f'{ct(None, 12.3456)}, {ct(None, 86400.5)}'


def chunked_scan_matches_one_pass():
    one, many = V(*ONE_PASS), V(*CHUNKED)
    d = float(np.abs(one[2] - many[2]).max()) if len(one[2]) == len(many[2]) else float('inf')
    return (len(one[2]) == len(many[2]) and np.array_equal(one[0], many[0]) and d <= 1e-6), \
        f'{len(many[2])} windows in 0.35 s chunks, max|dP| = {d:.2e}'


def time_expansion_is_a_setting():
    path = V(*EXPANDED)
    plain = V('cell21: header rate', lambda: V(*FNS)['_audio_info'](path)[0])
    expanded = V('cell21: expanded rate', lambda: _inferfix.funcs_ns(
        idx, INFER_TIME_EXPANSION=10)['_audio_info'](path)[0])
    return (plain == 38400 and expanded == 384000), \
        f'{plain} Hz at factor 1, {expanded} Hz at INFER_TIME_EXPANSION=10'


def time_expanded_copy_agrees():
    te_sels = V('cell21: selections in the time-expanded copy', lambda: (lambda ns: (
        ns['build_selections'](*ns['scan_detector'](V(*EXPANDED), DET), 0.5, 0.1, 2)
    ))(_inferfix.funcs_ns(idx, INFER_TIME_EXPANSION=10)))
    if not te_sels:
        return False, 'no selections'
    ref = V(*SELS)[0]
    return (len(te_sels) == 1 and abs(te_sels[0]['begin'] - ref['begin']) < 0.02
            and abs(te_sels[0]['end'] - ref['end']) < 0.02), \
        (f"original [{ref['begin']:.3f}, {ref['end']:.3f}] vs the 10x copy "
         f"[{te_sels[0]['begin']:.3f}, {te_sels[0]['end']:.3f})")


check('the scan starts at 0, hops by exactly 10 ms and reaches the end of the recording', scan_grid)
check('only the 70-85 kHz sweep is detected: not the 40-50 kHz one, not the sub-1 kHz rumble',
      only_the_high_sweep)
check('the positive windows merge into ONE selection bracketing the sweep', one_selection)
check('an isolated single positive window is dropped (INFER_MIN_WINDOWS)', min_windows_drops)
check('a threshold above every window yields no selections at all', impossible_threshold)
check('classify_selections returns BOTH the probabilities and the mean embeddings', classify_shapes)
check("   ...and the sweep is labelled rhro with a confident probability", classify_is_rhro)
check('no selections -> an empty (0, n_classes) array and no embedding, not a crash',
      classify_nothing)
check('process_recording answers UNKNOWN for an unlike selection and keeps the best guess',
      unknown_replaces_the_species)
check("with the 'unknown' answer off, the species is the classifier's argmax", unknown_off_gives_argmax)
check('Low/High/Peak land on the 70-85 kHz sweep, within one 375 Hz bin at the peak',
      frequencies_on_the_sweep)
check('a click at 180 kHz is NOT reported at 180 kHz (the 150 kHz ceiling guard)',
      click_stays_under_the_ceiling)
check('the recording start comes from the YYYYMMDD_HHMMSS in the file name', start_from_the_file_name)
check('each clock time is the recording start plus the selection time (to the millisecond)',
      clock_times_are_start_plus_time)
check('with no start time the clock counts from 00:00:00, and wraps at midnight',
      clock_from_zero_wraps)
check('scanning in 0.35 s chunks gives the same windows and P(call) as one pass '
      '(this is what the phase-aligned excerpt is for)', chunked_scan_matches_one_pass)
check('a 10x time-expanded file reports the header rate unless INFER_TIME_EXPANSION says otherwise',
      time_expansion_is_a_setting)
check('the time-expanded copy yields the SAME selections, in seconds, as the original',
      time_expanded_copy_agrees)

# =============================================================================================
print('\n=== cell 22: listing a folder of recordings and writing the output ===')
IN = os.path.join(work, 'recordings')
for sub in ('rhro', 'rhle'):
    os.makedirs(os.path.join(IN, sub), exist_ok=True)
sf.write(os.path.join(IN, 'rhro', '20260521_204000.WAV'),
         synth(384000, 2.0, [(0.40, 0.80, 70000, 85000)]), 384000, subtype='FLOAT')
sf.write(os.path.join(IN, 'rhle', '20260522_101500.flac'),
         synth(384000, 2.0, [(0.60, 0.90, 85000, 95000)]), 384000, format='FLAC', subtype='PCM_16')
sf.write(os.path.join(IN, 'rhro', 'low_rate_20260523_080000.wav'),
         synth(44100, 2.0, [(0.30, 0.60, 70000, 85000)]), 44100, subtype='FLOAT')
_inferfix.write_corrupt(os.path.join(IN, 'broken_20260524_090000.wav'))
_inferfix.touch(os.path.join(IN, 'rhro', '._20260521_204000.WAV'))     # macOS resource fork
_inferfix.touch(os.path.join(IN, '__MACOSX', 'junk.WAV'))
_inferfix.touch(os.path.join(IN, 'rhle', 'more.zip'), b'PK\x05\x06' + b'\0' * 18)
_inferfix.touch(os.path.join(IN, 'rhro', 'notes.txt'), b'not audio')

# Every run below gets its OWN WORKING_DIR: the cells are computed lazily, in whatever order the
# checks ask for them, and they all write to `<WORKING_DIR>/batspot_detections.txt`,
# `<WORKING_DIR>/raven_tables/` and `<WORKING_DIR>/batspot_detections_per_folder/`. Sharing one
# directory would let whichever run executed last decide what the other runs' checks read.
_RUN_DIRS = {}


def run_dir(tag):
    if tag not in _RUN_DIRS:
        _RUN_DIRS[tag] = os.path.join(work, 'runs', tag)
        os.makedirs(_RUN_DIRS[tag], exist_ok=True)
    return _RUN_DIRS[tag]


def run_over(tag, **over):
    return _inferfix.run_ns(idx, WORKING_DIR=run_dir(tag),
                            INFER_OUTPUT_FILE=os.path.join(run_dir(tag),
                                                           'batspot_detections.txt'), **over)


MAIN_OUT = os.path.join(run_dir('main'), 'batspot_detections.txt')
RAVEN = os.path.join(run_dir('main'), 'raven_tables', 'rhro', '20260521_204000.BatSpot.selections.txt')
PER_FOLDER = os.path.join(run_dir('main'), 'batspot_detections_per_folder')
RUN = ('cell22: run over the folder', lambda: run_over(
    'main', INFER_INPUT_DIR=IN, INFER_DET=DET, INFER_CLS=CLS, INFER_UNKNOWN=None))
MAIN_ROWS = ('cell22: the detections file of that run', lambda: (V(*RUN), read_csv(MAIN_OUT))[1])
NO_PER_FOLDER = ('cell22: per-folder files switched off', lambda: run_over(
    'nofolder', INFER_INPUT_DIR=IN, INFER_DET=DET, INFER_CLS=CLS, INFER_UNKNOWN=None,
    INFER_PER_FOLDER_FILES=False))
NOISE_RUN = ('cell22: every window is a selection', lambda: run_over(
    'noise', INFER_INPUT_DIR=IN, INFER_DET=DET, INFER_CLS=CLS, INFER_UNKNOWN=None,
    INFER_DROP_NOISE=True, INFER_DET_THRESHOLD=0.0))
NOISE_OUT = os.path.join(run_dir('noise'), 'batspot_detections.txt')
rej = NOISE_OUT.replace('.txt', '_rejected_noise.txt')
NOISE_ROWS = ('cell22: detections after the noise split',
              lambda: (V(*NOISE_RUN), read_csv(NOISE_OUT))[1])


def listing_is_exactly_the_recordings():
    got = sorted(os.path.relpath(p, IN) for p in V(*RUN)[0]['infer_wavs'])
    want = sorted([os.path.join('rhle', '20260522_101500.flac'),
                   os.path.join('rhro', '20260521_204000.WAV'),
                   os.path.join('rhro', 'low_rate_20260523_080000.wav'),
                   'broken_20260524_090000.wav'])
    return got == want, f'{got}'


def inner_zip_is_reported():
    out = V(*RUN)[1]
    return ('are NOT read' in out and 'more.zip' in out), first_line_with(out, 'NOT read')


def low_rate_is_flagged():
    out = V(*RUN)[1]
    return ('sampled below 100 kHz' in out and 'INFER_TIME_EXPANSION' in out), \
        first_line_with(out, 'sampled below 100 kHz')


def corrupt_is_listed_as_unreadable():
    out = V(*RUN)[1]
    return 'UNREADABLE (skipped): broken_20260524_090000.wav' in out, \
        first_line_with(out, 'UNREADABLE')


def the_run_continues_past_it():
    out = V(*RUN)[1]
    return 'Processed 3/4 recordings' in out, first_line_with(out, 'Processed')


def corrupt_is_recorded_as_failed():
    got = [rel for rel, _ in V(*RUN)[0]['infer_failed']]
    return got == ['broken_20260524_090000.wav'], str(V(*RUN)[0]['infer_failed'])


def low_rate_yields_nothing():
    files = V(*RUN)[0]['infer_files']
    return ([f['rel'] for f in files if f['n_kept'] == 0]
            == [os.path.join('rhro', 'low_rate_20260523_080000.wav')]
            and all(f['n_kept'] == 1 for f in files if 'low_rate' not in f['rel'])), \
        str([(f['rel'], f['n_kept'], f['n_noise']) for f in files])


def header_is_the_requested_one():
    return V(*NOISE_ROWS)[0] == [
        'Selection', 'name_of_file', 'Channel', 'Begin Time (s)', 'End Time (s)',
        'Begin Clock Time', 'End Clock Time', 'Low Freq (Hz)', 'High Freq (Hz)', 'Peak Freq (Hz)',
        'Delta Time (s)', 'Species detected', 'Confidence'], str(V(*NOISE_ROWS)[0])


def rows_named(name):
    """The one detections row for `name_of_file` (the rows come out in sorted folder order, which
    is not the order the fixture folder was built in, so they are looked up by name)."""
    return next((r for r in V(*MAIN_ROWS)[1:] if r[1] == name), None)


def one_row_per_kept_selection():
    rows = V(*MAIN_ROWS)
    names = sorted(r[1] for r in rows[1:])
    ok = (len(rows) == 3 and [r[0] for r in rows[1:]] == ['1', '2']
          and names == ['rhle/20260522_101500.flac', 'rhro/20260521_204000.WAV'])
    return ok, f'{[r[:2] for r in rows[1:]]}'


def rows_match_the_synthesised_calls():
    rhro, rhle = rows_named('rhro/20260521_204000.WAV'), rows_named('rhle/20260522_101500.flac')
    if rhro is None or rhle is None:
        return False, f'{(rhro or rhle) is None}: {[r[1] for r in V(*MAIN_ROWS)[1:]]}'
    return (0.38 <= float(rhro[3]) <= 0.43 and 0.79 <= float(rhro[4]) <= 0.84
            and 0.58 <= float(rhle[3]) <= 0.63 and 0.89 <= float(rhle[4]) <= 0.94), \
        (f'rhro {rhro[3]}..{rhro[4]} s (synthesised 0.40-0.80), '
         f'rhle {rhle[3]}..{rhle[4]} s (synthesised 0.60-0.90)')


def delta_and_band_are_consistent():
    rows = V(*MAIN_ROWS)
    return (all(abs(float(r[10]) - (float(r[4]) - float(r[3]))) < 5e-4 for r in rows[1:])
            and all(float(r[7]) <= float(r[8]) and float(r[7]) <= float(r[9]) <= float(r[8])
                    for r in rows[1:])), str([(r[10], r[7], r[8], r[9]) for r in rows[1:]])


def row_clock_times():
    rhro, rhle = rows_named('rhro/20260521_204000.WAV'), rows_named('rhle/20260522_101500.flac')
    if rhro is None or rhle is None:
        return False, f'{(rhro or rhle) is None}: {[r[1] for r in V(*MAIN_ROWS)[1:]]}'
    return (abs(clock_seconds(rhro[5]) - (20 * 3600 + 40 * 60 + float(rhro[3]))) < 5e-4
            and abs(clock_seconds(rhle[5]) - (10 * 3600 + 15 * 60 + float(rhle[3]))) < 5e-4), \
        f'{rhro[5]} for a 20:40:00 file, {rhle[5]} for a 10:15:00 file'


def every_row_has_species_and_confidence():
    rows = V(*MAIN_ROWS)
    return all(r[11] in _inferfix.CLS_CLASSES and 0.0 <= float(r[12]) <= 1.0 for r in rows[1:]), \
        str([(r[11], r[12]) for r in rows[1:]])


def raven_tables_are_written():
    V(*RUN)
    if not os.path.exists(RAVEN):
        return False, f'no table at {RAVEN}'
    head = open(RAVEN).readline().rstrip('\n').split('\t')
    rhle_dir = os.path.join(run_dir('main'), 'raven_tables', 'rhle')
    return (sorted(os.listdir(rhle_dir)) == ['20260522_101500.BatSpot.selections.txt']
            and head[-2:] == ['Confidence', 'Best guess']
            and len(open(RAVEN).read().rstrip('\n').split('\n')) == 2), '\t'.join(head)


def noise_answers_are_split_off():
    V(*NOISE_RUN)
    main_rows = V(*NOISE_ROWS)
    if not os.path.exists(rej):
        return False, f'no {os.path.basename(rej)}'
    rej_rows = read_csv(rej)
    # Threshold 0.0 turns every window into a selection, so most of them are silent ones the
    # classifier calls 'noise': those must be in the rejected file and ONLY those.
    return (rej_rows[0] == main_rows[0]
            and all(r[11] == 'noise' for r in rej_rows[1:])
            and all(r[11] != 'noise' for r in main_rows[1:])
            and len(main_rows) - 1 + len(rej_rows) - 1 > 1), \
        (f'{len(main_rows) - 1} kept row(s) {sorted({r[11] for r in main_rows[1:]})}, '
         f'{len(rej_rows) - 1} rejected row(s) {sorted({r[11] for r in rej_rows[1:]})}')


def per_folder_files_add_up():
    """One detections file per input folder: same header, Selection renumbered from 1 inside each
    file, and the counts summing to the combined file -- nothing dropped and nothing duplicated."""
    V(*RUN)
    rows = V(*MAIN_ROWS)
    if not os.path.isdir(PER_FOLDER):
        return False, f'no {PER_FOLDER}'
    names = sorted(os.listdir(PER_FOLDER))
    per = {n: read_csv(os.path.join(PER_FOLDER, n)) for n in names}
    total = sum(len(r) - 1 for r in per.values())
    same_header = all(r[0] == rows[0] for r in per.values())
    renumbered = all([x[0] for x in r[1:]] == [str(i + 1) for i in range(len(r) - 1)]
                     for r in per.values())
    return (names == ['batspot_detections_rhle.txt', 'batspot_detections_rhro.txt']
            and total == len(rows) - 1 and same_header and renumbered), \
        (f'{names}: ' + ', '.join(f'{n}={len(r) - 1}' for n, r in per.items())
         + f' vs {len(rows) - 1} in the combined file')


def per_folder_files_are_reported():
    out = V(*RUN)[1]
    return ('Per-folder detection files (2)' in out and 'batspot_detections_rhro.txt' in out), \
        first_line_with(out, 'Per-folder detection files')


def per_folder_files_can_be_switched_off():
    V(*NO_PER_FOLDER)
    d = os.path.join(run_dir('nofolder'), 'batspot_detections_per_folder')
    out = V(*NO_PER_FOLDER)[1]
    return (not os.path.exists(d) and 'Per-folder detection files' not in out), \
        f'with INFER_PER_FOLDER_FILES=False: {d} exists = {os.path.exists(d)}'


check('the listing finds .WAV / .flac / .wav in any case, at any depth, and nothing else '
      '(no ._ forks, no __MACOSX, no .txt)', listing_is_exactly_the_recordings)
check('a .zip inside the folder is reported instead of being silently ignored',
      inner_zip_is_reported)
check('a file sampled below 100 kHz is flagged: calls cannot be in it unless it is '
      'time-expanded', low_rate_is_flagged)
check('the corrupt file is reported as UNREADABLE in the LISTING, not left to crash later',
      corrupt_is_listed_as_unreadable)
check('   ...and the run still processes every other recording', the_run_continues_past_it)
check('   ...recording the corrupt one in infer_failed with its error', corrupt_is_recorded_as_failed)
check('the low-rate file produced NO selection (its calls are above its Nyquist frequency)',
      low_rate_yields_nothing)
check('the detections file header is exactly the requested 13 columns, in order', header_is_the_requested_one)
check('one row per kept selection, numbered 1..N, name_of_file = <folder>/<file>',
      one_row_per_kept_selection)
check('the rows describe the windows that were synthesised (0.40-0.80 s and 0.60-0.90 s)',
      rows_match_the_synthesised_calls)
check('Delta Time (s) is End - Begin, Low <= High and the peak is inside the band',
      delta_and_band_are_consistent)
check('the clock times are the file-name stamps plus the selection times', row_clock_times)
check('every row carries a species from the classifier and a confidence in [0, 1]',
      every_row_has_species_and_confidence)
check('one Raven table per recording, with the Best guess column the port adds',
      raven_tables_are_written)
check("every selection the classifier calls 'noise' goes to *_rejected_noise.txt",
      noise_answers_are_split_off)
check('one detections file per input folder: same columns, renumbered, and the counts add up',
      per_folder_files_add_up)
check('   ...and the report names them', per_folder_files_are_reported)
check('   ...and INFER_PER_FOLDER_FILES=False writes none', per_folder_files_can_be_switched_off)

# =============================================================================================
print("\n=== cell 23: scoring against the operator's own Raven tables ===")
TRUTH = os.path.join(work, 'truth')
os.makedirs(os.path.join(TRUTH, 'rhro'), exist_ok=True)
# Cell 23 keeps a row when its View contains 'spectrogram' and drops it otherwise, so the
# duplicate is a WAVEFORM-view row -- 'Spectrogram 2' would be kept and counted twice.
with open(os.path.join(TRUTH, 'rhro', '20260521_204000.Table.1.selections.txt'), 'w') as f:
    f.write('\t'.join(['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
                       'Low Freq (Hz)', 'High Freq (Hz)', 'Species']) + '\n')
    f.write('1\tSpectrogram 1\t1\t0.42\t0.78\t70000\t85000\tmyro\n')     # found, predicted rhro,
                                                                            # and myro is a species
                                                                            # the classifier never saw
    f.write('2\tWaveform 1\t1\t0.42\t0.78\t70000\t85000\tmyro\n')      # waveform view: ignored
    f.write('3\tSpectrogram 1\t1\t1.60\t1.70\t90000\t95000\trhle\n')     # in the file, not detected
    f.write('4\tSpectrogram 1\t1\t1.90\t2.00\t0\t1000\tnoise\n')         # noise box
os.makedirs(os.path.join(TRUTH, 'rhle'), exist_ok=True)
# The table name must CONTAIN the recording's file stem: that is how cell 23 pairs them.
with open(os.path.join(TRUTH, 'rhle', '20260522_101500_labelled_rhro.txt'), 'w') as f:
    f.write('\t'.join(['Begin Time (s)', 'End Time (s)', 'Species']) + '\n')
    f.write('0.62\t0.88\trhro\n')

TRUTH_RUN = ('cell22: run with truth tables', lambda: run_over(
    'truth', INFER_INPUT_DIR=IN, INFER_DET=DET, INFER_CLS=CLS, INFER_UNKNOWN=None,
    INFER_TRUTH_DIR=TRUTH))
SCORED = ('cell23: scored', lambda: _inferfix.truth_ns(idx, V(*TRUTH_RUN)[0],
                                                      INFER_TRUTH_DIR=TRUTH)[1])
NO_TRUTH = ('cell23: INFER_TRUTH_DIR unset', lambda: _inferfix.truth_ns(
    idx, V(*TRUTH_RUN)[0], INFER_TRUTH_DIR=None)[1])
NO_MATCH = ('cell23: tables matching no recording', lambda: _inferfix.truth_ns(
    idx, V(*TRUTH_RUN)[0], INFER_TRUTH_DIR=os.path.join(work, 'no_such_tables'))[1])


def two_recordings_scored():
    return 'Scored 2 recording(s)' in V(*SCORED), first_line_with(V(*SCORED), 'Scored')


def waveform_view_rows_ignored():
    # 3 bat boxes over the 2 scored recordings (rhro: 0.42-0.78 myro, 1.60-1.70 rhle; rhle:
    # 0.62-0.88 rhro) plus 1 noise box. The waveform-view duplicate must NOT be counted: with it,
    # the rhro table alone would report 4 bat boxes and the total 5.
    out = V(*SCORED)
    return ('bat boxes found (any overlap with a kept selection): 2/3' in out
            and 'noise boxes hit by a kept selection (false alarms): 0/1' in out), \
        ' | '.join(ln.strip() for ln in out.split('\n') if 'boxes' in ln)


def species_of_found_boxes_scored():
    # myro is the operator's label on the box that WAS found, and the classifier answered rhro:
    # a fixture where every found box is labelled correctly would not test this line at all.
    return 'species correct on found boxes: 1/2 = 0.500' in V(*SCORED), \
        first_line_with(V(*SCORED), 'species correct')


def unseen_species_named():
    out = V(*SCORED)
    if 'never trained on' not in out:
        return False, f'not reported; the box labels were {{rhro, myro}} against classes ' \
                      f'{_inferfix.CLS_CLASSES}'
    return 'myro' in out.split('never trained on')[1][:30], \
        out.split('never trained on')[1][:60].strip()


def confusion_lists_only_what_occurred():
    out = V(*SCORED)
    tail = out.split('confusion on found boxes')[1] if 'confusion on found boxes' in out else ''
    # rhro and myro occur; rhle and noise do not and must not be listed.
    return ('confusion on found boxes' in out and '\n  rhro' in tail and '\n  myro' in tail
            and '\n  rhle' not in tail and '\n  noise' not in tail
            and 'rhle' not in tail.split(chr(10))[0]), \
        ' | '.join(ln.strip() for ln in out.split('\n')
                   if ln.strip().startswith(('rhro', 'rhle', 'myro', 'noise')))[-80:]


def truth_unset_is_skipped():
    return 'INFER_TRUTH_DIR not set' in V(*NO_TRUTH), V(*NO_TRUTH).strip()


def no_matching_tables_is_reported():
    return 'No table matched any recording' in V(*NO_MATCH), \
        V(*NO_MATCH).strip().split('\n')[0]


check('only the recordings whose file stem a table contains are scored', two_recordings_scored)
check('the waveform-view row is ignored, so 3 bat boxes + 1 noise box are counted, not 4+1',
      waveform_view_rows_ignored)
check("the species of each FOUND box is scored against the operator's label",
      species_of_found_boxes_scored)
check('a species the classifier was never trained on is called out by name', unseen_species_named)
check('the confusion matrix lists the species that occur and not the full class list',
      confusion_lists_only_what_occurred)
check('with INFER_TRUTH_DIR unset the comparison is skipped, not failed', truth_unset_is_skipped)
check('truth tables matching no recording are reported, not silently scored as 0/0',
      no_matching_tables_is_reported)

print('\n' + '=' * 70)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)
