# Cell 24 (optional): score the detections against your own Raven selection tables
#
# Set INFER_TRUTH_DIR (Cell 21) to a folder of Raven selection tables (.txt, tab-separated, with
# 'Begin Time (s)', 'End Time (s)' and a 'Species' column). A table belongs to a recording when its
# file name contains the recording's file stem -- e.g. '20260521_204000.Table.1.selections.txt' or
# 'acsh_devon_20260521_204000.txt' for '20260521_204000.WAV'; a table in the same sub-folder wins.
# Matching is by time overlap (any overlap), because a detected selection and a hand-drawn box rarely
# share exact edges. Precision is only meaningful if your tables mark EVERY call in the file.

def _read_truth_table(path):
    rows = []
    with open(path, newline='', encoding='utf-8', errors='replace') as fh:
        for r in _csv.DictReader(fh, delimiter='\t'):
            r = {(k or '').strip().lower(): (v or '').strip() for k, v in r.items()}
            if 'view' in r and 'spectrogram' not in r['view'].lower():
                continue                          # Raven repeats each box for the waveform view
            sp = (r.get('species') or r.get('species detected') or '').lower()
            try:
                rows.append((float(r['begin time (s)']), float(r['end time (s)']), sp))
            except (KeyError, ValueError):
                continue
    return rows


if INFER_TRUTH_DIR:
    _tables = sorted(glob.glob(os.path.join(INFER_TRUTH_DIR, '**', '*.txt'), recursive=True))
    print(f'{len(_tables)} truth table(s) under {INFER_TRUTH_DIR}')
    _tp_sp, _n_bat, _n_found, _n_noise, _n_noise_hit = [], 0, 0, 0, 0
    _n_det, _n_det_on_bat, _n_files = 0, 0, 0
    _per_folder = defaultdict(lambda: [0, 0, 0])      # bat boxes, found, species correct
    for _fi in infer_files:
        _stem = os.path.splitext(os.path.basename(_fi['rel']))[0].lower()
        _cand = [t for t in _tables if _stem in os.path.basename(t).lower()]
        _same = [t for t in _cand if os.path.relpath(os.path.dirname(t), INFER_TRUTH_DIR) == _fi['folder']]
        _cand = _same or _cand
        if not _cand:
            continue
        _n_files += 1
        _truth = [row for t in _cand for row in _read_truth_table(t)]
        _det = [s for s in infer_selections if os.path.relpath(s['path'], _src) == _fi['rel']]
        _ov = lambda s, b, e: min(s['end'], e) - max(s['begin'], b)
        for b, e, sp in _truth:
            _hits = [s for s in _det if _ov(s, b, e) > 0]
            if sp == 'noise':
                _n_noise += 1; _n_noise_hit += bool(_hits)
                continue
            _n_bat += 1; _per_folder[_fi['folder']][0] += 1
            if _hits:
                _n_found += 1; _per_folder[_fi['folder']][1] += 1
                _best = max(_hits, key=lambda s: _ov(s, b, e))
                _tp_sp.append((sp, _best['species']))
                _per_folder[_fi['folder']][2] += _best['species'] == sp
        _n_det += len(_det)
        _n_det_on_bat += sum(any(_ov(s, b, e) > 0 and sp != 'noise' for b, e, sp in _truth) for s in _det)

    if not _n_files:
        print('No table matched any recording -- table names must contain the recording file stem.')
    else:
        print(f'Scored {_n_files} recording(s) that have a truth table.')
        if _n_bat:
            print(f'  bat boxes found (any overlap with a kept selection): {_n_found}/{_n_bat} = {_n_found/_n_bat:.3f}')
        if _tp_sp:
            print(f'  species correct on found boxes: {sum(a == b for a, b in _tp_sp)}/{len(_tp_sp)} = '
                  f'{np.mean([a == b for a, b in _tp_sp]):.3f}')
            _n_unk = sum(b == UNKNOWN_LABEL for _, b in _tp_sp)
            if _n_unk:
                _ans = [(a, b) for a, b in _tp_sp if b != UNKNOWN_LABEL]
                print(f'  answered "{UNKNOWN_LABEL}" on {_n_unk} found boxes; species correct on the other '
                      f'{len(_ans)}: {np.mean([a == b for a, b in _ans]):.3f}' if _ans else '')
            _new = sorted({a for a, _ in _tp_sp} - set(INFER_CLS['names']))
            if _new:
                print(f'  your tables contain species the classifier was never trained on: {_new} -- the best '
                      f'it can answer for them is "{UNKNOWN_LABEL}"')
        if _n_noise:
            print(f'  noise boxes hit by a kept selection (false alarms): {_n_noise_hit}/{_n_noise}')
        if _n_det:
            print(f'  kept selections overlapping a bat box: {_n_det_on_bat}/{_n_det} = {_n_det_on_bat/_n_det:.3f}'
                  '   (a precision only if the tables are exhaustive)')
        print(f'\n  {"folder":<24}{"bat boxes":>10}{"found":>8}{"species ok":>12}')
        for _f, (_a, _b, _c) in sorted(_per_folder.items()):
            print(f'  {_f[:23]:<24}{_a:>10}{_b:>8}{_c:>12}')
        if _tp_sp:
            _labs = sorted({a for a, _ in _tp_sp} | {b for _, b in _tp_sp})
            _cm = confusion_matrix([a for a, _ in _tp_sp], [b for _, b in _tp_sp], labels=_labs)
            _rows = {a for a, _ in _tp_sp}
            print('\n  confusion on found boxes (rows = your label, columns = predicted):')
            print('  ' + ' ' * 8 + ''.join(f'{l[:6]:>7}' for l in _labs))
            for _l, _row in [(l, r) for l, r in zip(_labs, _cm) if l in _rows]:
                print(f'  {_l[:7]:<8}' + ''.join(f'{v:>7}' for v in _row))
else:
    print('INFER_TRUTH_DIR not set (Cell 21) -- skipping the comparison with your own selection tables.')