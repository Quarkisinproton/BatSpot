# Cell 22: Inference functions -- detector scan, selections, classification, measurements, tables
import re as _re_inf
import csv as _csv
import time as _time
import datetime as _dt
import scipy.signal as _ss
import resampy.filters as _rf
from fractions import Fraction

# ---- resampling, identical to training (resampy 'kaiser_best') ------------------------------
# Resampling a 5-min file with resampy takes ~40 s on the CPU (about an hour for 84 such files). For a
# RATIONAL ratio sr_out/sr_in = L/M (384->192 kHz is 1/2, 250->192 is 96/125, 500->192 is 48/125,
# 256->192 is 3/4, 384->250 is 125/192, 441->192 kHz is 64/147) resampy's band-limited interpolation
# reads its filter table at only L distinct fractional positions, so it is exactly a bank of L FIR
# filters -- one strided conv1d on the GPU (or CPU). The taps are computed with resampy's own table and
# its linear interpolation between table entries, and a self-check against resampy runs once per ratio;
# on a mismatch (or > 512 phases) resampy itself is used.
_POLY, _POLY_OK = {}, {}


def _kaiser_best_polyphase(sr_in, sr_out, max_phases=512):
    """(L, M, lo, W) with W[p, j] the weight of input sample q*M + lo + j for output q*L + p."""
    key = (int(sr_in), int(sr_out))
    if key not in _POLY:
        fr = Fraction(int(sr_out), int(sr_in))
        L, M = fr.numerator, fr.denominator
        if L > max_phases:
            _POLY[key] = None
            return None
        win, prec, _ = _rf.get_filter('kaiser_best')
        ratio = sr_out / sr_in
        if ratio < 1:
            win = ratio * win
        delta = np.diff(win, append=win[-1])
        scale = min(1.0, ratio)
        step = int(scale * prec)

        def wing(frac):
            index_frac = frac * prec
            offset = int(index_frac)
            eta = index_frac - offset
            idx = offset + np.arange((len(win) - offset) // step) * step
            return win[idx] + eta * delta[idx]

        taps = []
        for p in range(L):
            n_p = (p * M) // L
            frac = scale * (p * M - n_p * L) / L
            taps.append((n_p, wing(frac), wing(scale - frac)))   # left: x[n-i], right: x[n+1+k]
        lo = min(n - len(lw) + 1 for n, lw, _ in taps)
        hi = max(n + len(rw) for n, _, rw in taps)
        W = np.zeros((L, hi - lo + 1))
        for p, (n, lw, rw) in enumerate(taps):
            W[p, n - lo - np.arange(len(lw))] = lw
            W[p, n + 1 - lo + np.arange(len(rw))] = rw
        _POLY[key] = (L, M, lo, W.astype(np.float32))
    return _POLY[key]


def _polyphase_resample(x, poly, n_out):
    L, M, lo, W = poly
    K = W.shape[1]
    Q = -(-n_out // L)
    need = (Q - 1) * M + K
    xp = np.zeros(max(need, len(x) - lo), dtype=np.float32)
    xp[-lo:-lo + len(x)] = x
    xt = torch.from_numpy(xp).to(DEVICE)[None, None]
    wt = torch.from_numpy(W).to(DEVICE)[:, None, :]
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False     # TF32 would cost ~3 digits; keep full float32
    try:
        y = torch.nn.functional.conv1d(xt, wt, stride=M)[0, :, :Q]
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    return y.T.reshape(-1)[:n_out].cpu().numpy()


def resample_like_training(x, sr_in, sr_out):
    """= resampy.resample(x, sr_in, sr_out, filter='kaiser_best') (float32), fast for rational ratios."""
    x = np.asarray(x, dtype=np.float32)
    sr_in, sr_out = int(round(sr_in)), int(round(sr_out))
    if sr_in == sr_out:
        return x
    n_out = int(len(x) * float(sr_out) / float(sr_in))
    poly = _kaiser_best_polyphase(sr_in, sr_out)
    if poly is not None and (sr_in, sr_out) not in _POLY_OK:
        probe = np.random.default_rng(0).standard_normal(max(sr_in // 4, 4096)).astype(np.float32) * 0.1
        ref = resampy.resample(probe, sr_in, sr_out, filter='kaiser_best')
        fast = _polyphase_resample(probe, poly, len(ref))
        # For some ratios (250->192, 500->192 kHz) resampy's float64 time grid t*(sr_in/sr_out) rounds
        # just below an integer at ~0.02 % of the outputs, where it then drops one far-tail filter tap;
        # there the two differ by <= 8e-5 of full scale (an artefact of resampy, which is not even
        # chunk-consistent at that level). Everywhere else they agree to float32 rounding (~5e-7).
        dev = float(np.abs(ref - fast).max() / max(np.abs(ref).max(), 1e-12)) if len(ref) == len(fast) else np.inf
        _POLY_OK[(sr_in, sr_out)] = dev <= 2e-4
        print(f'[resample] {sr_in}->{sr_out} Hz: polyphase path ({poly[0]} phases, {DEVICE.type}), max deviation '
              f'from resampy {dev:.1e} of full scale -> '
              f'{"used" if _POLY_OK[(sr_in, sr_out)] else "MISMATCH, using resampy"}')
    if poly is not None and _POLY_OK.get((sr_in, sr_out)) and n_out > 0:
        return _polyphase_resample(x, poly, n_out)
    return resampy.resample(x, sr_in, sr_out, filter='kaiser_best').astype(np.float32)


def _audio_info(path):
    """(true sample rate, number of frames). The rate is the header rate x INFER_TIME_EXPANSION."""
    info = sf.info(path)
    return info.samplerate * INFER_TIME_EXPANSION, info.frames


def _read_mono(path, n_in, start, stop):
    """Samples [start, stop) of a file as mono (channel mean, like training), zeros outside the file."""
    s, e = max(0, start), min(n_in, stop)
    if e > s:
        x, _ = sf.read(path, start=s, stop=e, dtype='float32', always_2d=True)
        x = x.mean(axis=1)
    else:
        x = np.zeros(0, np.float32)
    if start < s or stop > e:
        x = np.concatenate([np.zeros(s - start, np.float32), x,
                            np.zeros(max(0, stop - max(e, s)), np.float32)])
    return x.astype(np.float32)


def _resampled(path, n_in, sr_in, sr, r0, r1, margin=4096):
    """Samples [r0, r1) of the WHOLE recording resampled to `sr`, computed from a margin-padded
    excerpt. The resampling filters reach < 200 input samples, so the result equals resampling the
    whole file at once -- which lets long files be scanned in chunks. The excerpt starts on an input
    sample that falls exactly on the output grid (output index q0 = a multiple of L for the ratio
    L/M), so its output samples ARE full-file output samples, not ones shifted by a fraction."""
    fr = Fraction(int(round(sr)), int(round(sr_in)))
    L, M = fr.numerator, fr.denominator
    q0 = ((r0 - margin) // L) * L
    s = q0 * M // L
    e = int(np.ceil((r1 + margin) * sr_in / sr))
    y = resample_like_training(_read_mono(path, n_in, s, e), sr_in, sr)
    i0 = r0 - q0
    out = y[i0:i0 + (r1 - r0)]
    return out if len(out) == r1 - r0 else np.pad(out, (0, r1 - r0 - len(out)))


def _minmax_batch(w):
    """Per-window min-max to [0, 1] (same rule as minmax_normalize), w: (B, frames, bins) tensor."""
    w = w - w.amin(dim=(1, 2), keepdim=True)
    mx = w.amax(dim=(1, 2), keepdim=True)
    return torch.where(mx > 0, w / mx.clamp_min(1e-30), w)


@torch.no_grad()
def scan_detector(path, det, batch=1024):
    """P(call) for every 20 ms window (hop INFER_HOP_S) of a recording.
    Returns (window_begin_s, window_end_s, p_call) as numpy arrays."""
    cfg, model = det['cfg'], det['model']
    sr, hl, nfft = cfg['sr'], cfg['hop_length'], cfg['n_fft']
    seq = int(float(cfg['sequence_len']) / 1000 * sr / hl)
    step = max(1, int(round(INFER_HOP_S * sr / hl)))
    sr_in, n_in = _audio_info(path)
    n_res = int(np.ceil(n_in * sr / sr_in))
    n_frames = (n_res - nfft) // hl + 1 if n_res >= nfft else 0

    def _p(windows):                            # windows: (B, seq, bins) float tensor on DEVICE
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=INFER_AMP and DEVICE.type == 'cuda'):
            out = model(_minmax_batch(windows).unsqueeze(1))
        return torch.softmax(out.float(), dim=1)[:, det['call_idx']].cpu().numpy()

    if n_frames < seq:                          # shorter than one window: one zero-padded window
        y = _resampled(path, n_in, sr_in, sr, 0, max(n_res, nfft))
        db = clip_to_db_spectrogram(y, cfg)
        win = pad_window(minmax_normalize(db), seq)
        p = _p(torch.from_numpy(win[None]).to(DEVICE))
        return np.array([0.0]), np.array([n_in / sr_in]), p

    starts = np.arange(0, n_frames - seq + 1, step)
    per_chunk = max(1, int(INFER_CHUNK_S * sr / hl) // step)
    lead = 16                                   # left context frames: pre-emphasis edge falls outside
    probs = []
    for c in range(0, len(starts), per_chunk):
        ws = starts[c:c + per_chunk]
        f0, f1 = int(ws[0]), int(ws[-1]) + seq  # frames needed: [f0, f1)
        g0 = max(0, f0 - lead)
        db = clip_to_db_spectrogram(_resampled(path, n_in, sr_in, sr, g0 * hl, (f1 - 1) * hl + nfft), cfg)
        x = torch.from_numpy(np.ascontiguousarray(db[f0 - g0:])).to(DEVICE)
        W = x.unfold(0, seq, step).permute(0, 2, 1)          # (n_windows, seq, bins), a view
        assert W.shape[0] == len(ws), (W.shape, len(ws))
        for b in range(0, len(ws), batch):
            probs.append(_p(W[b:b + batch]))
    p = np.concatenate(probs)
    t0 = starts * hl / sr
    t1 = ((starts + seq - 1) * hl + nfft) / sr
    return t0, t1, p


def build_selections(t0, t1, p, threshold, merge_gap, min_windows, max_len=None):
    """Windows with p >= threshold -> selections. Windows at most `merge_gap` apart join one selection;
    a selection longer than `max_len` (continuous activity) is split at its widest internal silence,
    so long passes come out as bout-sized rows like the training selection tables."""
    max_len = INFER_MAX_SELECTION_S if max_len is None else max_len
    pos = np.flatnonzero(p >= threshold)
    if not len(pos):
        return []
    todo = np.split(pos, np.flatnonzero(t0[pos[1:]] - t1[pos[:-1]] > merge_gap) + 1)[::-1]
    groups = []
    while todo:                                  # explicit stack: long runs cannot hit recursion limits
        g = todo.pop()
        if len(g) < 2 or not max_len or t1[g[-1]] - t0[g[0]] <= max_len:
            groups.append(g)
            continue
        gaps = t0[g[1:]] - t1[g[:-1]]            # silence between consecutive positive windows
        ok = ((t1[g[:-1]] - t0[g[0]] >= max_len / 4) & (t1[g[-1]] - t0[g[1:]] >= max_len / 4))
        mid = np.abs(np.arange(len(gaps)) - (len(gaps) - 1) / 2) / len(gaps)
        # widest silence first, then the least confident boundary, then the most central one
        score = np.where(ok, gaps, -np.inf) - 1e-6 * (p[g[1:]] + p[g[:-1]]) - 1e-9 * mid
        if not np.isfinite(score).any():
            score = gaps - 1e-9 * mid
        k = int(np.argmax(score))
        todo += [g[k + 1:], g[:k + 1]]           # left piece is processed first -> time order kept
    return [{'begin': float(t0[g[0]]), 'end': float(t1[g[-1]]), 'n': len(g), 'p_max': float(p[g].max())}
            for g in groups if len(g) >= min_windows]


@torch.no_grad()
def classify_selections(path, sels, cls, batch=512):
    """Classifier probabilities per selection, using the test-clip protocol (top-k loudest
    windows, softmax averaged). Returns ((n_selections, n_classes) probabilities, (n_selections, D)
    mean embeddings) -- the embedding feeds the 'unknown' answer."""
    cfg, model = cls['cfg'], cls['model']
    sr, hl, nfft = cfg['sr'], cfg['hop_length'], cfg['n_fft']
    seq = int(float(cfg['sequence_len']) / 1000 * sr / hl)
    min_dur = ((seq - 1) * hl + nfft) / sr      # one full 20 ms window
    sr_in, n_in = _audio_info(path)
    wins, owner = [], []
    for j, s in enumerate(sels):
        b, e = s['begin'], s['end']
        if e - b < min_dur:                     # too short for one window: centre-extend
            c = 0.5 * (b + e); b, e = c - min_dur / 2, c + min_dur / 2
        audio = resample_like_training(
            _read_mono(path, n_in, int(round(b * sr_in)), int(round(e * sr_in))), sr_in, sr)
        db = clip_to_db_spectrogram(audio, cfg)
        st, sc = window_scores(db, seq, WINDOW_STRIDE)
        for w0 in eval_window_starts(st, sc, WINDOW_MODE, TEST_TOPK, seq):
            wins.append(pad_window(minmax_normalize(np.array(db[w0:w0 + seq])), seq))
            owner.append(j)
    n_cls = len(cls['names'])
    if not wins:
        return np.zeros((0, n_cls)), None
    X = torch.from_numpy(np.stack(wins).astype(np.float32)).unsqueeze(1)
    Ps, Es = [], []
    for b in range(0, len(X), batch):        # same computation as predict_proba (incl. the embedding)
        pt, et = forward_with_embedding(model, X[b:b + batch].to(DEVICE))
        Ps.append(pt.cpu().numpy()); Es.append(et.cpu().numpy())
    P, E = np.concatenate(Ps), np.concatenate(Es)
    owner = np.array(owner)
    cnt = np.bincount(owner, minlength=len(sels))[:, None]
    out, emb = np.zeros((len(sels), n_cls)), np.zeros((len(sels), E.shape[1]))
    np.add.at(out, owner, P)
    np.add.at(emb, owner, E)
    return out / cnt, emb / cnt


def measure_frequencies(path, begin, end, floor_hz=None, snr_db=None, context_s=1.0, level_db=25):
    """(Low, High, Peak) frequency in Hz of a selection.
    Spectrogram at the native rate: Hann, ~375 Hz bins (1024 points at 384 kHz, as in the training
    Raven tables), 50 % overlap. Background = per-frequency median over the selection +- context_s
    (a wide context matters for the high-duty-cycle CF bats rhle/rhro, whose calls fill most frames
    of their own selection). Low/High = the contiguous band around the bin that stands highest above
    background, keeping bins >= snr_db above background AND within level_db of that bin's level;
    Peak = frequency of the loudest cell inside the band (Raven's Peak Freq rule). If no bin clears
    snr_db, Low = High = Peak. Checked on 400 annotated boxes (median error, bats): Peak 0.0 kHz
    (within one bin 65 %), Low 2.4 kHz, High 3.2 kHz. Weak spot: Low for rhle/rhro is often 40-55 kHz
    too low (faint CF calls; the band bridges into noise)."""
    floor_hz = INFER_FREQ_FLOOR_HZ if floor_hz is None else floor_hz
    snr_db = INFER_FREQ_SNR_DB if snr_db is None else snr_db
    sr_in, n_in = _audio_info(path)
    nfft = int(2 ** np.ceil(np.log2(sr_in / 375.0)))
    c, half = 0.5 * (begin + end), max(end - begin, 0.1) / 2
    a = max(0.0, c - half - context_s)
    x = _read_mono(path, n_in, int(round(a * sr_in)), int(round(min(n_in / sr_in, c + half + context_s) * sr_in)))
    if len(x) < 2 * nfft:
        x = np.pad(x, (0, 2 * nfft - len(x)))
    f, t, S = _ss.spectrogram(x, fs=sr_in, window='hann', nperseg=nfft, noverlap=nfft // 2, mode='psd')
    # Ceiling: next to Nyquist the recorder's anti-alias filter leaves an almost-zero background,
    # so any click there shows a huge "SNR" (a first version reported 192000 Hz for such clicks).
    keep = (f >= floor_hz) & (f <= min(INFER_FREQ_CEIL_HZ, 0.95 * sr_in / 2))
    if not keep.any():
        return float('nan'), float('nan'), float('nan')
    f, D = f[keep], 10 * np.log10(S[keep] + 1e-20)
    t = t + a
    inside = (t >= c - half) & (t <= c + half)
    Din = D[:, inside] if inside.any() else D
    snr = (Din - np.median(D, axis=1, keepdims=True)).max(axis=1)
    level = Din.max(axis=1)
    ip = int(np.argmax(snr))
    ok = (snr >= snr_db) & (level >= level[ip] - level_db)
    lo = hi = ip
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    peak = f[lo + int(np.argmax(level[lo:hi + 1]))]
    return float(f[lo]), float(f[hi]), float(peak)


# ---- clock time ------------------------------------------------------------------------------
_STAMP_RE = _re_inf.compile(r'(\d{8})[-_](\d{6})')
_CLIP_OFFSET_RE = _re_inf.compile(r'\d{8}[-_]\d{6}_(\d+)_(\d+)$')     # BatSpot clip: ..._<start ms>_<end ms>
_AUDIOMOTH_RE = _re_inf.compile(rb'Recorded at (\d{2}):(\d{2}):(\d{2}) (\d{2})/(\d{2})/(\d{4})')


def recording_start(path):
    """(datetime of the first sample, source). File name YYYYMMDD_HHMMSS (or -) first -- that is what
    Raven used for the training tables -- then the AudioMoth header comment, else (None, ...)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _STAMP_RE.search(stem)
    if m:
        try:
            t = _dt.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            off = _CLIP_OFFSET_RE.search(stem)
            if off:
                return t + _dt.timedelta(milliseconds=int(off.group(1))), 'file name + clip offset'
            return t, 'file name'
        except ValueError:
            pass
    try:
        with open(path, 'rb') as fh:
            m = _AUDIOMOTH_RE.search(fh.read(8192))
        if m:
            hh, mi, ss, dd, mo, yy = map(int, m.groups())
            return _dt.datetime(yy, mo, dd, hh, mi, ss), 'AudioMoth header'
    except (OSError, ValueError):
        pass
    return None, 'unknown -> clock times count from 00:00:00'


def clock_time(start, seconds):
    """Raven-style HH:MM:SS.ffff of `seconds` after `start` (wraps at midnight)."""
    base = 0.0 if start is None else start.hour * 3600 + start.minute * 60 + start.second + start.microsecond / 1e6
    t = int(round(((base + seconds) % 86400) * 1e4)) % (86400 * 10000)
    h, rem = divmod(t, 36000000)
    m, rem = divmod(rem, 600000)
    return f'{h:02d}:{m:02d}:{rem // 10000:02d}.{rem % 10000:04d}'


def process_recording(path, det, cls):
    """Detect, classify and measure every selection in one recording."""
    t_start, t_src = recording_start(path)
    t0, t1, p = scan_detector(path, det)
    sels = build_selections(t0, t1, p, INFER_DET_THRESHOLD, INFER_MERGE_GAP_S, INFER_MIN_WINDOWS)
    probs, emb = classify_selections(path, sels, cls)
    unk = (apply_unknown(probs, emb, INFER_UNKNOWN) if INFER_UNKNOWN is not None and len(sels)
           else np.zeros(len(sels), bool))
    for s, pr, u in zip(sels, probs, unk):
        k = int(pr.argmax())
        s['best_guess'], s['confidence'] = cls['names'][k], float(pr[k])
        s['species'] = UNKNOWN_LABEL if u else s['best_guess']
        s['is_noise'] = k == cls['noise_idx']
        s['low'], s['high'], s['peak'] = measure_frequencies(path, s['begin'], s['end'])
        s['begin_clock'], s['end_clock'] = clock_time(t_start, s['begin']), clock_time(t_start, s['end'])
    sr_in, n_in = _audio_info(path)
    return sels, {'duration': n_in / sr_in, 'sr': sr_in,
                  'n_windows': len(p), 'p_max': float(p.max()) if len(p) else float('nan'),
                  'clock_source': t_src}


INFER_HEADER = ['Selection', 'name_of_file', 'Channel', 'Begin Time (s)', 'End Time (s)',
                'Begin Clock Time', 'End Clock Time', 'Low Freq (Hz)', 'High Freq (Hz)',
                'Peak Freq (Hz)', 'Delta Time (s)', 'Species detected', 'Confidence']


def _fmt_row(i, s):
    return [i, s['name'], 1, f'{s["begin"]:.4f}', f'{s["end"]:.4f}', s['begin_clock'], s['end_clock'],
            f'{s["low"]:.1f}', f'{s["high"]:.1f}', f'{s["peak"]:.1f}', f'{s["end"] - s["begin"]:.4f}',
            s['species'], f'{s["confidence"]:.4f}']


def write_detections(path, sels):
    """The requested comma-separated text file; Selection numbers run 1..N over the whole file."""
    with open(path, 'w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(INFER_HEADER)
        for i, s in enumerate(sels, 1):
            w.writerow(_fmt_row(i, s))


def write_raven_table(path, sels):
    """Tab-separated Raven selection table for one recording (File -> Open Selection Table)."""
    cols = ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)',
            'High Freq (Hz)', 'Begin Clock Time', 'End Clock Time', 'Peak Freq (Hz)', 'Delta Time (s)',
            'Species', 'Confidence', 'Best guess']
    with open(path, 'w', newline='') as fh:
        fh.write('\t'.join(cols) + '\n')
        for i, s in enumerate(sels, 1):
            r = _fmt_row(i, s)
            fh.write('\t'.join(map(str, [i, 'Spectrogram 1', 1, r[3], r[4], r[7], r[8], r[5], r[6],
                                         r[9], r[10], r[11], r[12], s.get('best_guess', r[11])])) + '\n')


print('Inference functions defined')