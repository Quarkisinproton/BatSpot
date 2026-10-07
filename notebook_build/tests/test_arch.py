#!/usr/bin/env python3
"""Cell 4 (architecture): the residual port's `shortcut` naming, and the ensemble / embedding API.

Two separate things are under test.

1. `shortcut`, not `downsample`. The residual projection parameter is *serialised* as
   `shortcut.*` -- `animal_spot/models/residual_base.py:32` assigns `self.shortcut =
   downsample`. So the key names are not an implementation detail: they are the on-disk
   format of every official `.pk`. A renamed submodule makes `load_state_dict` raise
   `Missing key(s): "layer2.0.downsample.0.weight"` for the whole encoder, and a cell that
   cannot load the official weights defeats the entire fine-tuning premise. The reference
   key set is built from the vendored official code at run time, so this check cannot pass
   by having been written from the ported cell's own output.

2. The ensemble API. `SoftmaxEnsemble` returns LOG-probabilities on purpose: every caller
   that does `softmax(model(x))` -- `predict_proba`, the cascade, the inference cells --
   must keep working when a single model is swapped for an ensemble, so the returned value
   has to re-softmax to the members' AVERAGE probability. `forward_with_embedding` is what
   the 'unknown' answer consumes, and its embedding must be the per-member concatenation:
   each ensemble member has its own 512-d space, so averaging them would be meaningless.

The `hidden_layer_1` key is pinned by name, not just by shape: Task 5's `predict_proba` and
Task 12's export both read `Classifier._layer_output['hidden_layer_1']` through
`forward_with_embedding`.

The cell is exec'd from the delivered notebook, so this is the cell that ships.

Run: venv/bin/python notebook_build/tests/test_arch.py
"""
import contextlib
import io
import os
import sys
import types
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_cells

CELL_ARCH = 4   # Cell 5: Model Architecture & Loading
REPO = extract_cells.REPO

fails = []
# Set once, from the constructions below. Every check that needs the ensemble API reads it
# through check(..., guarded=True), so a missing definition is a reportable FAIL naming the
# cause rather than a traceback part-way through the section.
ENS_ERROR = None


def check(name, cond, detail='', guarded=False):
    """One assertion. `cond` and `detail` may be callables, evaluated lazily.

    `guarded=True` marks a check that needs the ensemble API. When that API could not be
    constructed these checks report FAIL with the reason, and the callables are never
    called -- which is what keeps a renamed definition from turning into a traceback.
    """
    if guarded and ENS_ERROR:
        cond, detail = False, ENS_ERROR
    if callable(cond):
        cond = cond()
    if callable(detail):
        detail = detail()
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


# --- the cell under test, read out of the artifact that ships -----------------------------
# Cell 3 ("Imports") is what supplies torch / nn / OrderedDict to this cell; it is
# reproduced here so the cell can be exec'd on its own. Nothing else is stubbed.
idx = extract_cells.merged_cells()
UNDER_TEST = os.environ.get('NEW_CELLS') or extract_cells.MERGED
print(f'=== under test: {UNDER_TEST} cell {CELL_ARCH} ===')

ARCH_PATH = idx[CELL_ARCH]
with open(ARCH_PATH, encoding='utf-8') as f:
    ARCH_SRC = f.read()
check('extracted cell 4 really is the architecture cell',
      'ResidualEncoder' in ARCH_SRC and 'def build_model' in ARCH_SRC,
      os.path.basename(ARCH_PATH))

_ns = {'torch': torch, 'nn': nn, 'OrderedDict': OrderedDict}
with contextlib.redirect_stdout(io.StringIO()):        # the cell prints a banner
    # ONE namespace for globals and locals. With two, exec'd top-level names are STORE_NAME'd
    # into the locals mapping while the cell's functions resolve them as LOAD_GLOBAL, so
    # ResidualEncoder cannot see get_block_sizes -- a harness artefact, not a cell defect.
    exec(compile(ARCH_SRC, ARCH_PATH, 'exec'), _ns)
mod = types.SimpleNamespace(**_ns)

# The names the plan names as this cell's interface. A missing one is a mismatch like any
# other, not a traceback three checks later.
INTERFACE = ['ResidualEncoder', 'Classifier', 'build_model', 'load_model_from_pk',
             'load_any_model', 'unwrap_model', 'SoftmaxEnsemble', 'model_members',
             'forward_with_embedding']
missing_iface = [n for n in INTERFACE if n not in _ns]
check('the cell defines every name the plan names as its interface', not missing_iface,
      ', '.join(missing_iface) if missing_iface else f'{len(INTERFACE)} names')

# --- 1. shortcut, not downsample ------------------------------------------------------------
sys.path.insert(0, REPO)
from animal_spot.models.residual_encoder import ResidualEncoder as OfficialEncoder  # noqa: E402
from animal_spot.models.classifier import Classifier as OfficialClassifier  # noqa: E402

OFFICIAL_OPTS = {'input_channels': 1, 'conv_kernel_size': 7, 'max_pool': 1, 'resnet_size': 18}
HEAD_OPTS = {'input_channels': 512, 'pooling': 'avg', 'num_classes': 2}

official_keys = set(OfficialEncoder(OFFICIAL_OPTS).state_dict())
torch.manual_seed(0)
ported_keys = set(mod.ResidualEncoder(OFFICIAL_OPTS).state_dict())

check('the ported encoder has exactly the official key set',
      ported_keys == official_keys,
      f'{len(ported_keys)} keys' if ported_keys == official_keys else
      f'missing {sorted(official_keys - ported_keys)[:4]}, '
      f'extra {sorted(ported_keys - official_keys)[:4]}')
off_sc = sorted(k for k in official_keys if 'shortcut' in k)
port_sc = sorted(k for k in ported_keys if 'shortcut' in k)
check('the residual projection is serialised as shortcut.* in both (18 keys)',
      off_sc == port_sc and len(off_sc) == 18,
      f'{len(off_sc)} official vs {len(port_sc)} ported')
check('NO key uses the downsample.* spelling',
      not [k for k in ported_keys if 'downsample' in k],
      str([k for k in ported_keys if 'downsample' in k][:3]))

# The classifier head is the other half of a .pk.
off_cls = set(OfficialClassifier(HEAD_OPTS).state_dict())
port_cls = set(mod.Classifier(HEAD_OPTS).state_dict())
check('the ported classifier head has exactly the official key set',
      off_cls == port_cls == {'linear.weight', 'linear.bias'}, f'{sorted(port_cls)}')

# --- 2. the ensemble API --------------------------------------------------------------------
torch.manual_seed(0)


def fresh(n_classes=2):
    """A freshly built single model, exactly as build_model assembles one."""
    return mod._assemble(mod.ResidualEncoder(OFFICIAL_OPTS),
                         mod.Classifier({**HEAD_OPTS, 'num_classes': n_classes}))


# Everything from here on is inference-shaped: the sharp members below are given fixed head
# weights, and several checks read a scalar out of a graph tensor -- both warn unless grad
# is off. Set before the constructions so the in-place head edit is legal.
torch.set_grad_enabled(False)

try:
    # The detector's window geometry: 256 freq bins x 30 frames (0.02 s @ 192 kHz, hop 128).
    # 30 frames is what makes the time axis collapse to 1 after four stride-2 stages, which
    # is the only way the pooled head vector is 512-d -- as in every official .pk.
    x = torch.randn(4, 1, 256, 30)
    m1, m2, m3 = fresh(), fresh(), fresh()
    e2 = mod.SoftmaxEnsemble([m1, m2])
    e3 = mod.SoftmaxEnsemble([m1, m2, m3])
    dp = nn.DataParallel(m1)
    # Two members that are CONFIDENT and DISAGREE. With a freshly initialised head the logits
    # are near zero, softmax is nearly linear there, and mean-of-probabilities coincides with
    # softmax-of-mean-logits to ~2e-6 -- so the contrast below would be vacuous on m1/m2.
    # These two heads emit logits chosen so the two formulas differ by ~1e-2.
    sh1 = fresh()
    sh1[1].linear.weight.zero_()
    sh1[1].linear.bias.copy_(torch.tensor([6.0, -6.0]))
    sh2 = fresh()
    sh2[1].linear.weight.zero_()
    sh2[1].linear.bias.copy_(torch.tensor([6.0, 0.0]))
    sharp = mod.SoftmaxEnsemble([sh1, sh2])
    ENS = {'x': x, 'm1': m1, 'm2': m2, 'm3': m3, 'e2': e2, 'e3': e3, 'dp': dp,
           'sh1': sh1, 'sh2': sh2, 'sharp': sharp}
except Exception as e:       # any construction failure is a reportable FAIL, not a traceback
    ENS = {}
    ENS_ERROR = f'{type(e).__name__}: {e}'


def _ens(name):
    return ENS.get(name)


# SoftmaxEnsemble returns LOG-probabilities so a `softmax(model(x))` caller is unaffected.
check('the ensemble returns LOG-probabilities that re-softmax to the members\' MEAN probability',
      lambda: torch.allclose(_ens('e2')(x).exp(),
                             torch.stack([m1(x), m2(x)]).softmax(-1).mean(0), atol=1e-6),
      lambda: f'max dev {float((_ens("e2")(x).exp() - torch.stack([m1(x), m2(x)]).softmax(-1).mean(0)).abs().max()):.2e}',
      guarded=True)
check('   ...so it is NOT softmax-of-mean-logits (what a naive ensemble would return)',
      lambda: not torch.allclose(_ens('sharp')(x).exp(),
                                 torch.stack([_ens('sh1')(x), _ens('sh2')(x)]).mean(0)
                                 .log_softmax(-1).exp(), atol=1e-4),
      lambda: 'averaged probs '
              f'{[round(v, 4) for v in _ens("sharp")(x).exp()[0].tolist()]} vs '
              f'softmax-of-mean-logits '
              f'{[round(v, 4) for v in torch.stack([_ens("sh1")(x), _ens("sh2")(x)]).mean(0).log_softmax(-1).exp()[0].tolist()]}'
              + ' -- averaging logits lets one over-confident member dominate',
      guarded=True)
check('the returned values are <= 0 (they really are log-probabilities)',
      lambda: bool((_ens('e2')(x) <= 0).all()),
      lambda: f'max {float(_ens("e2")(x).max()):.3e}', guarded=True)
check('the ensemble is a no-op with one member',
      lambda: torch.allclose(mod.SoftmaxEnsemble([m1])(x).exp(), m1(x).softmax(-1), atol=1e-6),
      guarded=True)
check('a member CAN change the ensemble answer -- it averages probabilities, it does not vote',
      lambda: not torch.allclose(_ens('e2')(x), m2(x), atol=1e-3), guarded=True)

# forward_with_embedding: averaged probabilities, CONCATENATED embeddings.
check('forward_with_embedding concatenates one 512-d embedding per member',
      lambda: tuple(mod.forward_with_embedding(_ens('e3'), x)[1].shape) == (4, 3 * 512),
      lambda: str(tuple(mod.forward_with_embedding(_ens('e3'), x)[1].shape)), guarded=True)
check('the embeddings are the members\' OWN pooled vectors, in member order',
      lambda: (torch.allclose(mod.forward_with_embedding(_ens('e3'), x)[1][:, :512],
                              mod.forward_with_embedding(_ens('m1'), x)[1], atol=1e-6)
               and torch.allclose(mod.forward_with_embedding(_ens('e3'), x)[1][:, 1024:],
                                  mod.forward_with_embedding(_ens('m3'), x)[1], atol=1e-6)),
      'a mean over members would average embeddings from different 512-d spaces', guarded=True)
check('the probabilities are the members\' mean, same as the ensemble forward',
      lambda: torch.allclose(mod.forward_with_embedding(_ens('e3'), x)[0],
                             _ens('e3')(x).exp(), atol=1e-6),
      lambda: f'max dev {float((mod.forward_with_embedding(_ens("e3"), x)[0] - _ens("e3")(x).exp()).abs().max()):.2e}',
      guarded=True)
check('a single model returns probabilities that already sum to 1',
      lambda: torch.allclose(mod.forward_with_embedding(_ens('m1'), x)[0].sum(1),
                             torch.ones(4), atol=1e-5),
      lambda: f'max dev {float((mod.forward_with_embedding(_ens("m1"), x)[0].sum(1) - 1).abs().max()):.2e}',
      guarded=True)
check('a single model returns a 512-d embedding',
      lambda: tuple(mod.forward_with_embedding(_ens('m1'), x)[1].shape) == (4, 512),
      lambda: str(tuple(mod.forward_with_embedding(_ens('m1'), x)[1].shape)), guarded=True)

# The name Task 5 (predict_proba) and Task 12 (export) read. Pinned, not inferred.
def _pooled():
    m1(x)                       # the pooled vector is only stored during a forward pass
    return m1[1]._layer_output['hidden_layer_1']


check("Classifier stores the pooled vector as _layer_output['hidden_layer_1']",
      lambda: tuple(_pooled().shape) == (4, 512), lambda: str(tuple(_pooled().shape)),
      guarded=True)
check('that stored vector IS the embedding forward_with_embedding returns',
      lambda: torch.allclose(_pooled(), mod.forward_with_embedding(_ens('m1'), x)[1], atol=1e-6),
      guarded=True)
check("the stored vector is exactly the linear head's input",
      lambda: torch.allclose(m1[1].linear(_pooled()), m1(x), atol=1e-6),
      'if this drifts, the embedding stops describing the decision', guarded=True)

# model_members / unwrap_model.
check('unwrap_model unwraps a DataParallel wrapper',
      lambda: mod.unwrap_model(_ens('dp')) is _ens('m1'), guarded=True)
check('unwrap_model is a no-op on a plain model',
      lambda: mod.unwrap_model(_ens('m1')) is _ens('m1'), guarded=True)
check('model_members returns the single model itself when there is no ensemble',
      lambda: mod.model_members(_ens('m1')) == [_ens('m1')]
      and mod.model_members(_ens('dp')) == [_ens('m1')], guarded=True)
check('model_members returns the members of an ensemble',
      lambda: mod.model_members(_ens('e3')) == [_ens('m1'), _ens('m2'), _ens('m3')], guarded=True)
check('SoftmaxEnsemble accepts DataParallel members (train_model returns the wrapper)',
      lambda: (len(mod.SoftmaxEnsemble([_ens('dp'), _ens('m2')]).members) == 2
               and mod.SoftmaxEnsemble([_ens('dp'), _ens('m2')]).members[0] is _ens('m1')),
      guarded=True)
check('forward_with_embedding accepts an ensemble of unwrapped members',
      lambda: tuple(mod.forward_with_embedding(
          mod.SoftmaxEnsemble([_ens('dp'), _ens('dp')]), x)[1].shape) == (4, 2 * 512),
      guarded=True)

# --- 3. the official .pk still loads --------------------------------------------------------
# The real consequence of the key naming: the official models must load through the ported
# cell with no missing/unexpected key. BatSpot_article/ is untracked (AGENTS.md), so its
# absence is reported as a FAIL naming the path rather than a silent skip -- a green run
# that proved nothing is the failure mode this repo has been bitten by before.
DET_PK = os.path.join(REPO, 'BatSpot_article/batspot/models_call_detector/m09/train/ANIMAL-SPOT.pk')
CLS_PK = os.path.join(REPO, 'BatSpot_article/batspot/models_call_classifier/m09/train/ANIMAL-SPOT.pk')
check(f'the official detector .pk is present, else the load checks below prove nothing -- {DET_PK}',
      os.path.exists(DET_PK))
check(f'the official classifier .pk is present -- {CLS_PK}', os.path.exists(CLS_PK))

if os.path.exists(DET_PK) and os.path.exists(CLS_PK):
    loaded = {}
    for label, path, sr in (('detector', DET_PK, 192000), ('classifier', CLS_PK, 250000)):
        seq = int(0.02 * sr / 128)
        try:
            model, classes, data_opts = mod.load_model_from_pk(path, torch.device('cpu'))
            out = model(torch.zeros(1, 1, 256, seq))
            err = None
        except Exception as exc:
            model = classes = data_opts = out = None
            err = f'{type(exc).__name__}: {exc}'
        loaded[label] = (classes, data_opts, err)
        check(f'the official {label} .pk loads through the ported cell and runs a forward pass',
              err is None and tuple(out.shape) == (1, len(classes)),
              err if err else f'{len(classes)} classes, logits {tuple(out.shape)}')
        check(f'   ...with its own dataOpts (sr {sr} kHz)',
              err is None and data_opts.get('sr') == sr,
              f"sr={data_opts.get('sr')}" if err is None else 'not loaded')

    # load_any_model dispatches on the file's contents and refuses an unknown one.
    try:
        r = mod.load_any_model(DET_PK, torch.device('cpu'))
        disp_err = None
    except Exception as exc:
        r = disp_err = exc
    n_det = len(loaded['detector'][0]) if loaded['detector'][2] is None else -1
    check('load_any_model routes a .pk to load_model_from_pk',
          disp_err is None and len(r[1]) == n_det,
          f'{disp_err}' if disp_err else f'{len(r[1])} classes')
    junk = os.path.join(os.environ.get('TMPDIR', '/tmp'), '_arch_junk_model.bin')
    torch.save({'nonsense': 1}, junk)
    try:
        mod.load_any_model(junk, torch.device('cpu'))
        junk_err = None
    except Exception as exc:
        junk_err = exc
    os.remove(junk)
    check('load_any_model rejects an unknown format with a ValueError',
          isinstance(junk_err, ValueError),
          f'got {type(junk_err).__name__}' if junk_err else 'no exception -- silent bad model')

print('\n' + '=' * 62)
print(f'{len(fails)} failure(s)' + ((': ' + ', '.join(fails)) if fails else ''))
sys.exit(1 if fails else 0)