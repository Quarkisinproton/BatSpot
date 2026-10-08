# Cell 23: Run the detector -> classifier combo over every recording and write the detections file
import zipfile as _zipfile

_src = INFER_INPUT_DIR
assert os.path.exists(_src), (f'INFER_INPUT_DIR not found: {INFER_INPUT_DIR!r} -- add the test data '
                              '(Add Data) and paste its path into Cell 21')


_single = None
if os.path.isfile(_src) and _src.lower().endswith('.zip'):
    _dst = os.path.join(WORKING_DIR, 'infer_input', os.path.splitext(os.path.basename(_src))[0])
    with _zipfile.ZipFile(_src) as _z:
        _z.extractall(_dst)
    _src = _dst
elif os.path.isfile(_src):                       # a single recording
    _single, _src = _src, os.path.dirname(os.path.abspath(_src))
if _single:
    infer_wavs = [_single]
else:
    infer_wavs = sorted(p for p in glob.glob(os.path.join(_src, '**', '*'), recursive=True)
                        if p.lower().endswith(tuple(INFER_EXTENSIONS)) and os.path.isfile(p)
                        and '__MACOSX' not in p and not os.path.basename(p).startswith('._'))
    _zips = glob.glob(os.path.join(_src, '**', '*.zip'), recursive=True)
    if _zips:
        print(f'NOTE: {len(_zips)} .zip file(s) inside {_src} are NOT read; unpack them or point '
              f'INFER_INPUT_DIR at the .zip itself, e.g. {_zips[0]}')
assert infer_wavs, f'no audio files ({", ".join(INFER_EXTENSIONS)}) under {_src}'

_by_folder = defaultdict(list)
for _p in infer_wavs:
    _by_folder[os.path.dirname(os.path.relpath(_p, _src)) or '.'].append(_p)
print(f'{len(infer_wavs)} recordings in {len(_by_folder)} folder(s) under {_src}:')
_unreadable = []
for _f, _ps in sorted(_by_folder.items()):
    _infos = []
    for p in _ps:
        try:
            _i = sf.info(p)
            _infos.append((p, _i.samplerate * INFER_TIME_EXPANSION, _i.frames / (_i.samplerate * INFER_TIME_EXPANSION)))
        except Exception as _e:                  # listed now, reported as FAILED below
            _unreadable.append((os.path.relpath(p, _src), repr(_e)))
    print(f'  {_f:<30} {len(_ps):3d} files, {sum(d for _, _, d in _infos)/60:6.1f} min, '
          f'sample rates {sorted({int(r) for _, r, _ in _infos})}')
    _low = [os.path.basename(p) for p, r, _ in _infos if r < INFER_CLS['cfg']['sr']]
    if _low:
        print(f'    NOTE: {len(_low)} file(s) below {INFER_CLS["cfg"]["sr"]//1000} kHz -- calls above '
              f'their Nyquist frequency cannot be seen, e.g. {_low[:3]}')
    if any(r < 100000 for _, r, _ in _infos):
        print('    !! some files are sampled below 100 kHz: bat calls (20-120 kHz) cannot be in them. If they '
              'are TIME-EXPANDED recordings, set INFER_TIME_EXPANSION (e.g. 10) in Cell 21.')
for _rel, _err in _unreadable:
    print(f'  UNREADABLE (skipped): {_rel}: {_err}')

infer_selections, infer_rejected, infer_failed, infer_files = [], [], [], []
_t = _time.time()
for _p in tqdm(infer_wavs, desc='recordings'):
    _rel = os.path.relpath(_p, _src)
    try:
        _sels, _info = process_recording(_p, INFER_DET, INFER_CLS)
    except Exception as _e:                      # one bad file must not stop the other 83
        infer_failed.append((_rel, repr(_e)))
        continue
    _folder = os.path.dirname(_rel) or '.'
    for _s in _sels:
        _s['name'] = _rel.replace(os.sep, '/') if INFER_NAME_WITH_FOLDER else os.path.basename(_p)
        _s['folder'], _s['path'] = _folder, _p
        (infer_rejected if (INFER_DROP_NOISE and _s['is_noise']) else infer_selections).append(_s)
    infer_files.append({'rel': _rel, 'folder': _folder, **_info,
                        'n_kept': sum(not (INFER_DROP_NOISE and s['is_noise']) for s in _sels),
                        'n_noise': sum(bool(s['is_noise']) for s in _sels)})
_elapsed = _time.time() - _t

write_detections(INFER_OUTPUT_FILE, infer_selections)
_rej_path = os.path.splitext(INFER_OUTPUT_FILE)[0] + '_rejected_noise.txt'
if INFER_DROP_NOISE:
    write_detections(_rej_path, infer_rejected)
if INFER_WRITE_RAVEN_TABLES:
    _rdir = os.path.join(WORKING_DIR, 'raven_tables')
    for _fi in infer_files:
        _out = os.path.join(_rdir, os.path.splitext(_fi['rel'])[0] + '.BatSpot.selections.txt')
        os.makedirs(os.path.dirname(_out), exist_ok=True)
        write_raven_table(_out, [s for s in infer_selections
                                 if os.path.relpath(s['path'], _src) == _fi['rel']])
_pf_dir = os.path.join(WORKING_DIR, 'batspot_detections_per_folder')
_pf_files = []
if INFER_PER_FOLDER_FILES:
    os.makedirs(_pf_dir, exist_ok=True)
    for _f in sorted({fi['folder'] for fi in infer_files}):    # every folder gets a file, even without detections
        _tag = 'top_level' if _f == '.' else _f.replace(os.sep, '_').replace('/', '_')
        _out = os.path.join(_pf_dir, f'batspot_detections_{_tag}.txt')
        _sel = [s for s in infer_selections if s['folder'] == _f]
        write_detections(_out, _sel)
        _pf_files.append((_out, len(_sel)))

# ---- report ---------------------------------------------------------------------------------
_audio_min = sum(f['duration'] for f in infer_files) / 60
print(f'\nProcessed {len(infer_files)}/{len(infer_wavs)} recordings ({_audio_min:.1f} min of audio) '
      f'in {_elapsed:.0f} s on {DEVICE}.')
print(f'Selections: {len(infer_selections)} kept'
      + (f', {len(infer_rejected)} labelled noise by the classifier (-> {os.path.basename(_rej_path)})'
         if INFER_DROP_NOISE else ''))
print(f'Detections file: {INFER_OUTPUT_FILE}')
if INFER_WRITE_RAVEN_TABLES:
    print(f'Raven tables:    {os.path.join(WORKING_DIR, "raven_tables")}/<folder>/<file>.BatSpot.selections.txt')
if _pf_files:
    print(f'Per-folder detection files ({len(_pf_files)}) in {_pf_dir}:')
    for _out, _n in _pf_files:
        print(f'  {os.path.basename(_out):<44} {_n:6d} selections')
for _rel, _err in infer_failed:
    print(f'  FAILED {_rel}: {_err}')
_clock = Counter(f['clock_source'] for f in infer_files)
print('Clock-time source: ' + ', '.join(f'{k}: {v} file(s)' for k, v in _clock.items()))

_species = [n for n in INFER_CLS['names'] if n != 'noise'] + ([UNKNOWN_LABEL] if INFER_UNKNOWN is not None else [])
print(f'\nSelections per folder and species detected:')
print(f'{"folder":<24}{"files":>6}{"w/ det":>7}' + ''.join(f'{s:>7}' for s in _species) + f'{"noise*":>8}')
for _f in sorted({fi['folder'] for fi in infer_files}):
    _fs = [fi for fi in infer_files if fi['folder'] == _f]
    _c = Counter(s['species'] for s in infer_selections if s['folder'] == _f)
    print(f'{_f[:23]:<24}{len(_fs):>6}{sum(fi["n_kept"] > 0 for fi in _fs):>7}'
          + ''.join(f'{_c.get(s, 0):>7}' for s in _species) + f'{sum(fi["n_noise"] for fi in _fs):>8}')
print('  (* noise = selections the classifier labelled noise; not in the main file when INFER_DROP_NOISE)')

# If the folders are named after species, show how often the folder's species was predicted.
_named = [f for f in sorted({fi['folder'] for fi in infer_files})
          if os.path.basename(f).strip().lower() in _species]
if _named:
    print('\nFolders named after a species -- share of kept selections labelled with that species:')
    for _f in _named:
        _sp = os.path.basename(_f).strip().lower()
        _sel = [s for s in infer_selections if s['folder'] == _f]
        if _sel:
            _hit = sum(s['species'] == _sp for s in _sel)
            print(f'  {_f:<24} {_hit:4d}/{len(_sel):<4d} = {_hit/len(_sel):.2f}   '
                  f'(median confidence {np.median([s["confidence"] for s in _sel]):.2f})')
        else:
            print(f'  {_f:<24} no kept selections')
    print('  This is a sanity check, not an accuracy: a folder can contain other species and noise.')

_none = [fi['rel'] for fi in infer_files if fi['n_kept'] == 0]
if _none:
    print(f'\n{len(_none)} recording(s) with no kept selection, e.g. {_none[:5]}')
    print('  If they contain calls, lower INFER_DET_THRESHOLD (Cell 21) or check them in Raven.')

print('\nFirst lines of the detections file:')
with open(INFER_OUTPUT_FILE) as _fh:
    for _i, _line in enumerate(_fh):
        if _i > 8:
            break
        print('  ' + _line.rstrip())