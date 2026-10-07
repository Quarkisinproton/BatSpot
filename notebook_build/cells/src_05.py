# Cell 6: Transforms & Dataset
#
# WHY THIS CELL MATTERS (see AGENTS.md "Windowing fix"):
# The clips are 15 ms - 2.6 s long (median ~375 ms) but every model sees a 20 ms window. An old
# version kept spec[:seq_len] -- the FIRST 20 ms of every clip, always the same one -- so ~95 % of
# each clip was never used and the network memorised 674 fixed inputs. Training now takes a fresh
# random 20 ms crop from the loudest 20 % of each clip's windows every epoch (a stricter variant of
# the official random crop: it avoids the silence between pulses), and validation / test / inference
# average the softmax of the top-5 loudest windows. Measured: classifier 0.754 -> 0.853, detector m09
# 0.866 -> 0.933 (AGENTS.md 8.5). WINDOW_MODE = 'first' reproduces the old window choice.
#
# Front end = official animal_spot/data/transforms.py, step for step (verified bit-exact against the
# vendored code, max|diff| = 0):
#   mono mean -> resample (kaiser_best) -> pre-emphasis 0.98 ->
#   torch.stft(center=False) / sqrt(sum(w^2)) -> power -> keep fmin..fmax bins
#   -> NEAREST interpolation to n_freq_bins -> 10*log10 floored at -100 dB
#   -> per-window min-max normalisation (the official models were trained with
#      min_max_norm=true, and predict.py normalises each window on its own).
#
# Spectrogram cache: the full-clip dB spectrogram is computed once per clip and memory-mapped. An entry
# is reused only if its source wav still has the same size and mtime and the front-end code is
# unchanged (the cache tag is derived from clip_to_db_spectrogram's source); entries are written
# atomically; a clip that cannot be read is reported and dropped instead of killing the run.

import random as _random
import hashlib

import json as _json


def get_class_from_filename(filename):
    """Extract class label from BatSpot filename (before first '-')."""
    return os.path.basename(filename).split('-', 1)[0]


def load_audio_file(path, target_sr):
    """Load audio (mono mean, like the official loader) and resample to target_sr."""
    audio, native_sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if native_sr != target_sr:
        audio = resampy.resample(audio, native_sr, target_sr, filter='kaiser_best')
    return audio.astype(np.float32), target_sr


def clip_to_db_spectrogram(audio, cfg, preemphasis=0.98, min_level_db=-100):
    """Official front end up to (and incl.) amplitude->dB. Returns (frames, n_freq_bins) float32.
    NOT normalised: min-max is applied per 20 ms window later."""
    y = torch.from_numpy(audio)[None]
    y = torch.cat((y[:, :1], y[:, 1:] - preemphasis * y[:, :-1]), dim=-1)
    win = torch.hann_window(cfg['n_fft'])
    S = torch.stft(y, n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], window=win,
                   center=False, onesided=True, return_complex=True)
    S = torch.view_as_real(S).transpose(1, 2)            # (1, frames, bins, 2)
    S = S / win.pow(2).sum().sqrt()
    S = S.pow(2).sum(-1)                                  # power, (1, frames, n_fft/2+1)
    n_fft = (S.size(2) - 1) * 2
    lo = int(max(0, np.floor(n_fft * cfg['fmin'] / cfg['sr'])))
    hi = int(min(n_fft - 1, np.ceil(n_fft * cfg['fmax'] / cfg['sr'])))
    S = S[:, :, lo:hi]
    S = torch.nn.functional.interpolate(S[None], size=(S.size(1), cfg['n_freq_bins']),
                                        mode='nearest')[0]
    S = torch.clamp(S, min=float(np.exp(min_level_db / 10 * np.log(10))))
    return (10 * torch.log10(S))[0].numpy().astype(np.float32)


def minmax_normalize(spec):
    """Min-max normalise to [0, 1] (official MinMaxNormalize)."""
    spec = spec - spec.min()
    m = spec.max()
    return spec / m if m > 0 else spec


def pad_window(win, seq_len):
    """Centre-pad a short window with zeros to seq_len frames (official PaddedSubsequenceSampler pads
    at the centre when not training; it pads at a random offset when training -- only reachable for
    clips shorter than one window, 1 of 964 here, so the deviation is documented, not randomised)."""
    n = win.shape[0]
    if n >= seq_len:
        return win
    out = np.zeros((seq_len, win.shape[1]), dtype=np.float32)
    p = (seq_len - n) // 2
    out[p:p + n] = win
    return out


def window_scores(db, seq_len, stride):
    """Candidate window starts + a loudness score (mean of the per-frame peak dB)."""
    n = db.shape[0]
    if n <= seq_len:
        return np.array([0]), np.array([0.0])
    peak = db.max(axis=1)
    cs = np.concatenate([[0.0], np.cumsum(peak)])
    starts = np.arange(0, n - seq_len + 1, stride)
    return starts, (cs[starts + seq_len] - cs[starts]) / seq_len


def train_window_starts(starts, scores, mode, top_frac):
    """Windows a training clip may be cropped at: the loudest `top_frac` (>= 3).
    Selected by RANK, not by a percentile threshold: with tied scores (flat / saturated clips)
    a threshold keeps every tied window, i.e. a uniform crop over the whole clip."""
    if mode == 'first':
        return starts[:1]
    n_keep = min(len(starts), max(3, int(np.ceil(top_frac * len(starts)))))
    return starts[np.argsort(-scores, kind='stable')[:n_keep]]


def eval_window_starts(starts, scores, mode, k, seq_len):
    """Top-k loudest windows, greedy NMS: chosen windows start >= seq_len/2 frames apart."""
    if mode == 'first':
        return starts[:1]
    chosen = []
    for i in np.argsort(-scores, kind='stable'):
        if all(abs(starts[i] - starts[j]) >= seq_len // 2 for j in chosen):
            chosen.append(i)
        if len(chosen) >= k:
            break
    return starts[chosen]


def _augment_spec(spec):
    """Optional on-the-fly spectrogram augmentation (training only, USE_AUGMENTATION).
    Input shape (seq_len, n_freq_bins). Time shift, Gaussian noise, SpecAugment masks.
    NOTE: random window cropping already provides most of the useful augmentation."""
    seq_len, n_freq = spec.shape
    spec = spec.copy()
    shift = int(np.random.uniform(-0.20, 0.20) * seq_len)
    if shift > 0:
        spec = np.concatenate([np.zeros((shift, n_freq)), spec[:-shift]], axis=0)
    elif shift < 0:
        spec = np.concatenate([spec[-shift:], np.zeros((-shift, n_freq))], axis=0)
    spec = np.clip(spec + np.random.normal(0, 0.01, spec.shape), 0.0, 1.0)
    f_mask = int(np.random.uniform(0, 0.20) * n_freq)
    if f_mask > 0:
        f0 = np.random.randint(0, max(1, n_freq - f_mask))
        spec[:, f0:f0 + f_mask] = 0.0
    t_mask = int(np.random.uniform(0, 0.20) * seq_len)
    if t_mask > 0:
        t0 = np.random.randint(0, max(1, seq_len - t_mask))
        spec[t0:t0 + t_mask, :] = 0.0
    return spec.astype(np.float32)


def mix_background_db(win_db, noise_db, snr_db, min_level_db=-100):
    """Add a noise window to a window in the POWER domain (both are un-normalised dB spectrograms of
    the same shape). The noise is scaled so that the window's mean power is `snr_db` above the added
    noise's mean power. Training-only augmentation: the label stays the window's own label, and the
    new background comes from another recording, which breaks the "recognise the tape" shortcut."""
    ps = np.power(10.0, win_db.astype(np.float64) / 10.0)
    pn = np.power(10.0, noise_db.astype(np.float64) / 10.0)
    g = ps.mean() / max(pn.mean(), 1e-30) * 10.0 ** (-snr_db / 10.0)
    floor = 10.0 ** (min_level_db / 10.0)
    return (10.0 * np.log10(np.maximum(ps + g * pn, floor))).astype(np.float32)


def _code_fingerprint(*fns):
    """Hash of the compiled code (bytecode, names, constants) of `fns`. Unlike inspect.getsource it
    needs no source file -- getsource on exec'd code silently reads lines of an unrelated file -- and
    comment edits do not invalidate the cache, while any change to the computation (incl. default
    argument values such as the pre-emphasis) does."""
    def walk(co):
        parts = [co.co_code, repr(co.co_names).encode(), repr(co.co_varnames).encode()]
        parts += [walk(k) if hasattr(k, 'co_code') else repr(k).encode() for k in co.co_consts]
        return b'|'.join(parts)
    return hashlib.md5(b'#'.join(walk(f.__code__) + repr((f.__defaults__, f.__kwdefaults__)).encode()
                                 for f in fns)).hexdigest()[:6]


def _cache_version():
    """Cache tag: changes whenever the audio loading / resampling or the front end changes, so stale
    spectrograms can never be reused silently."""
    return 'v3-' + _code_fingerprint(load_audio_file, clip_to_db_spectrogram)


class WindowedBatDataset(Dataset):
    """Clip-level dataset served as 20 ms windows.

    train=True : __getitem__(i) -> (window, label); a RANDOM window from the loudest TRAIN_TOP_FRAC
                 of clip i (new crop every epoch), optionally mixed with a random window of a 'noise'
                 clip of the same dataset (noise_mix_prob) and augmented (augment).
    train=False: flat list of the TEST_TOPK loudest windows of every clip;
                 __getitem__ -> (window, label, clip_index). predict_proba() averages them per clip.
    """
    _CACHE_ROOT = os.path.join(WORKING_DIR, 'spec_cache')
    _CACHE_VERSION = _cache_version()

    def __init__(self, file_names, class_to_idx, cfg, train=False, augment=False,
                 mode=None, topk=None, noise_mix_prob=0.0, noise_mix_snr_db=(0.0, 20.0)):
        self.file_names = list(file_names)
        self.class_to_idx = class_to_idx
        self.cfg = cfg
        self.train = train
        self.augment = augment and train
        self.mode = WINDOW_MODE if mode is None else mode
        if self.mode not in ('energy_crop', 'first'):
            raise ValueError(f"WINDOW_MODE must be 'energy_crop' or 'first', got {self.mode!r}")
        self.topk = TEST_TOPK if topk is None else int(topk)
        self.noise_mix_prob = float(noise_mix_prob) if train else 0.0
        self.noise_mix_snr_db = tuple(noise_mix_snr_db)
        self.seq_len = int(float(cfg['sequence_len']) / 1000 * cfg['sr'] / cfg['hop_length'])
        key = (f"{self._CACHE_VERSION}_{cfg['sr']}_{cfg['n_fft']}_{cfg['hop_length']}_"
               f"{cfg['n_freq_bins']}_{cfg['fmin']}_{cfg['fmax']}")
        self.cache_dir = os.path.join(self._CACHE_ROOT, hashlib.md5(key.encode()).hexdigest()[:8])
        os.makedirs(self.cache_dir, exist_ok=True)
        self._build_cache()
        usable = [f for f in self.file_names if self._cache_is_valid(f)]
        if len(usable) != len(self.file_names):
            print(f'[WindowedBatDataset] WARNING: dropped {len(self.file_names) - len(usable)} '
                  f'unusable clip(s) of {len(self.file_names)}')
        self.file_names = usable
        if not self.file_names:
            raise RuntimeError('WindowedBatDataset: no usable clips')
        self.labels = np.array([class_to_idx[get_class_from_filename(f)] for f in self.file_names])
        # window bookkeeping
        self.train_starts, self.items = [], []
        n_short = 0
        for i, f in enumerate(self.file_names):
            db = self._mmap(i)
            n_short += db.shape[0] < self.seq_len
            starts, scores = window_scores(db, self.seq_len, WINDOW_STRIDE)
            if train:
                self.train_starts.append(train_window_starts(starts, scores, self.mode, TRAIN_TOP_FRAC))
            else:
                self.items += [(i, int(s)) for s in
                               eval_window_starts(starts, scores, self.mode, self.topk, self.seq_len)]
        if n_short:
            print(f'[WindowedBatDataset] note: {n_short}/{len(self.file_names)} clip(s) are shorter than '
                  f'one {self.seq_len}-frame window; they are zero-padded after min-max.')
        self._noise_ids = ([i for i, f in enumerate(self.file_names) if get_class_from_filename(f) == 'noise']
                           if self.noise_mix_prob > 0 else [])
        if self.noise_mix_prob > 0 and not self._noise_ids:
            print('[WindowedBatDataset] noise mixing requested but there are no noise clips -> off')

    # ---- cache -------------------------------------------------------------------------------
    def _spec_path(self, f):
        tag = hashlib.md5(os.path.abspath(f).encode()).hexdigest()[:8]    # same basename, other folder
        return os.path.join(self.cache_dir, f'{tag}_{os.path.basename(f)}.npy')

    def _meta_path(self, f):
        return self._spec_path(f) + '.meta.json'

    def _cache_is_valid(self, f):
        p, m = self._spec_path(f), self._meta_path(f)
        if not (os.path.exists(p) and os.path.exists(m)):
            return False
        try:
            with open(m) as fh:
                rec = _json.load(fh)
            st = os.stat(f)
        except Exception:
            return False
        return (rec.get('version') == self._CACHE_VERSION and rec.get('size') == st.st_size
                and rec.get('mtime_ns') == st.st_mtime_ns)

    def _build_cache(self):
        missing = [f for f in self.file_names if not self._cache_is_valid(f)]
        if not missing:
            return
        print(f'[WindowedBatDataset] caching {len(missing)} clips -> {self.cache_dir}')
        failed = []
        for f in tqdm(missing, desc='cache', leave=False):
            try:
                audio, _ = load_audio_file(f, self.cfg['sr'])
                spec = clip_to_db_spectrogram(audio, self.cfg)
                spec = np.nan_to_num(spec, nan=-100.0, posinf=0.0, neginf=-100.0)
                tmp = f'{self._spec_path(f)}.{os.getpid()}.tmp.npy'     # atomic: never a truncated entry
                np.save(tmp, spec)
                os.replace(tmp, self._spec_path(f))
                st = os.stat(f)
                with open(self._meta_path(f), 'w') as fh:
                    _json.dump({'version': self._CACHE_VERSION, 'size': st.st_size,
                                'mtime_ns': st.st_mtime_ns, 'frames': int(spec.shape[0])}, fh)
            except Exception as e:                          # report, do not abort the run
                failed.append((os.path.basename(f), repr(e)))
        if failed:
            print(f'[WindowedBatDataset] WARNING: {len(failed)} clip(s) failed and are SKIPPED:')
            for nm, err in failed[:10]:
                print(f'    {nm}: {err}')

    def _mmap(self, i):
        """Per-instance memory maps (created lazily, so each DataLoader worker opens its own)."""
        c = self.__dict__.setdefault('_mmap_cache', {})
        if i not in c:
            if len(c) > 64:                                  # bounded open files
                c.clear()
            c[i] = np.load(self._spec_path(self.file_names[i]), mmap_mode='r')
        return c[i]

    def __getstate__(self):                                  # never pickle open memory maps
        d = dict(self.__dict__)
        d.pop('_mmap_cache', None)
        return d

    # ---- windows ------------------------------------------------------------------------------
    def _raw_window(self, i, start):
        return np.array(self._mmap(i)[start:start + self.seq_len], dtype=np.float32)

    def _finish(self, raw):
        # normalise FIRST, pad second (official order): padding a dB array with 0.0 before min-max
        # makes the padding the array maximum and squashes the real signal.
        return pad_window(minmax_normalize(raw), self.seq_len).astype(np.float32)

    def _window(self, i, start):
        return self._finish(self._raw_window(i, start))

    def _mix_noise(self, raw, idx):
        cand = self._noise_ids if len(self._noise_ids) == 1 else [j for j in self._noise_ids if j != idx]
        nd = self._mmap(_random.choice(cand))
        n = raw.shape[0]
        if nd.shape[0] < n:
            return raw
        s = _random.randint(0, nd.shape[0] - n)
        return mix_background_db(raw, np.asarray(nd[s:s + n], dtype=np.float32),
                                 _random.uniform(*self.noise_mix_snr_db))

    def __len__(self):
        return len(self.file_names) if self.train else len(self.items)

    def __getitem__(self, idx):
        if self.train:
            raw = self._raw_window(idx, int(_random.choice(self.train_starts[idx])))
            if self._noise_ids and _random.random() < self.noise_mix_prob:
                raw = self._mix_noise(raw, idx)
            win = self._finish(raw)
            if self.augment:
                win = _augment_spec(win)
            return torch.from_numpy(win).unsqueeze(0), int(self.labels[idx])
        i, start = self.items[idx]
        return torch.from_numpy(self._window(i, start)).unsqueeze(0), int(self.labels[i]), i


def predict_proba(model, dataset, device, batch_size=256, return_embedding=False):
    """Clip-level probabilities: softmax averaged over each clip's windows.
    `dataset` must be a train=False WindowedBatDataset. Returns (probs[n_clips, C], labels[n_clips]),
    plus the clip-mean embedding [n_clips, D] when return_embedding (used by the 'unknown' answer)."""
    assert not dataset.train, 'predict_proba needs an evaluation (train=False) dataset'
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums, esums, counts = None, None, np.zeros(len(dataset.file_names))
    with torch.no_grad():
        for inputs, _, clip_idx in loader:
            x = inputs.to(device)
            if return_embedding:
                pt, et = forward_with_embedding(model, x)
                e = et.cpu().numpy()
            else:
                pt = torch.softmax(model(x).float(), dim=1)
            p = pt.cpu().numpy()
            if sums is None:
                sums = np.zeros((len(dataset.file_names), p.shape[1]))
                if return_embedding:
                    esums = np.zeros((len(dataset.file_names), e.shape[1]))
            ci = clip_idx.numpy()
            np.add.at(sums, ci, p)
            if return_embedding:
                np.add.at(esums, ci, e)
            np.add.at(counts, ci, 1)
    if not (counts > 0).all():   # a clip without windows would silently score as class 0
        _bad = [os.path.basename(dataset.file_names[i]) for i in np.flatnonzero(counts == 0)][:5]
        raise RuntimeError(f'{int((counts == 0).sum())} clip(s) produced no windows, e.g. {_bad}')
    out = (sums / counts[:, None], dataset.labels.copy())
    return out + (esums / counts[:, None],) if return_embedding else out


# ---- "unknown" answer (open set) -------------------------------------------------------------------
# A classifier can only name the classes it was trained on. These functions give every clip / selection
# a "known-ness" score; below a threshold the answer becomes UNKNOWN_LABEL instead of a species name.
#   'msp'   : the highest class probability (classic baseline)
#   'knnK'  : mean cosine similarity of the clip's embedding to its K nearest TRAINING clips
#   'maha'  : minus the smallest Mahalanobis distance to a class centre (shared, shrunk covariance),
#             fitted per ensemble member on its own 512-d embedding and averaged over the members
# Measured by hiding each species from training in turn (AGENTS.md section 11): 'maha' separates a
# never-seen species from the known ones best (AUROC 0.84 vs 0.70 for 'msp'); 'msp' is unreliable --
# the model is often MORE confident on a species it never saw (rhro looks like rhle).
# The method and the threshold are fitted in Cell 13 and exported next to the classifier (Cell 15).

def _l2n(e):
    e = np.asarray(e, dtype=np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


def known_scores(probs, emb, um):
    """Known-ness score per row (higher = more like the training data) for the unknown model `um`."""
    m = um['method']
    if m == 'msp':
        return np.asarray(probs).max(axis=1)
    if m.startswith('knn'):
        sim = _l2n(emb) @ um['ref'].T
        k = min(int(m[3:] or 5), sim.shape[1])
        return np.sort(sim, axis=1)[:, -k:].mean(axis=1)
    if m == 'maha':
        parts = np.split(np.asarray(emb, dtype=np.float64), len(um['mu']), axis=1)   # one per member
        return np.mean([-np.stack([np.einsum('ij,jk,ik->i', e - mu, P, e - mu) for mu in MU], axis=1).min(axis=1)
                        for e, MU, P in zip(parts, um['mu'], um['prec'])], axis=0)
    raise ValueError(f'unknown method {m!r}')


def fit_unknown_model(method, val_probs, val_labels, val_emb, train_emb, train_labels, keep, noise_idx,
                      n_members=1):
    """Threshold = the (1-keep) quantile of the scores of the CORRECT, non-noise validation answers, so
    `keep` of them keep their species name. `n_members`: the embedding is the concatenation of that many
    ensemble members' embeddings. Returns the dict used by known_scores / apply_unknown."""
    um = {'method': method, 'keep': float(keep), 'noise_idx': noise_idx}
    if method.startswith('knn'):
        um['ref'] = _l2n(train_emb)
    elif method == 'maha':
        ty = np.asarray(train_labels)
        cls = np.unique(ty)
        mus, precs = [], []
        for te in np.split(np.asarray(train_emb, np.float64), n_members, axis=1):
            mu = np.stack([te[ty == c].mean(axis=0) for c in cls])
            X = te - mu[np.searchsorted(cls, ty)]
            cov = X.T @ X / len(X)
            cov = 0.9 * cov + 0.1 * np.trace(cov) / len(cov) * np.eye(len(cov))   # shrinkage
            mus.append(mu); precs.append(np.linalg.inv(cov))
        um.update({'mu': np.stack(mus), 'prec': np.stack(precs)})
    pred = np.asarray(val_probs).argmax(axis=1)
    ok = (pred == np.asarray(val_labels)) & (pred != (-1 if noise_idx is None else noise_idx))
    s = known_scores(val_probs, val_emb, um)
    um['threshold'] = float(np.quantile(s[ok], 1.0 - keep)) if ok.any() else float('-inf')
    return um


def apply_unknown(probs, emb, um):
    """Boolean mask: True where the answer should be 'unknown' (never for a 'noise' answer)."""
    pred = np.asarray(probs).argmax(axis=1)
    unk = known_scores(probs, emb, um) < um['threshold']
    return unk & (pred != (-1 if um['noise_idx'] is None else um['noise_idx']))


def save_unknown_model(um, path, members):
    np.savez(path, **{k: v for k, v in um.items() if v is not None},
             members=np.array([os.path.basename(m) for m in members]))


def load_unknown_model(path):
    d = np.load(path, allow_pickle=False)
    um = {k: d[k] for k in d.files}
    for k in ('method',):
        um[k] = str(um[k])
    for k in ('keep', 'threshold'):
        um[k] = float(um[k])
    um['noise_idx'] = int(um['noise_idx']) if 'noise_idx' in um else None
    um['members'] = [str(m) for m in um.get('members', [])]
    return um


print('Windowed transforms + WindowedBatDataset defined '
      f'(WINDOW_MODE={WINDOW_MODE!r}, train top {TRAIN_TOP_FRAC:.0%} loudest windows, test top-{TEST_TOPK}); '
      f'cache tag {WindowedBatDataset._CACHE_VERSION}')
