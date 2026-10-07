# Cell 8: Clip Extraction (Python port of Data/scripts/export_clips.R)
#
# Faithful to the R script: one class per sub-folder of SELECTIONS_DIR; each Raven table is matched to
# its recording by the YYYYMMDD_HHMMSS stamp in the table's file name (e.g. acsh_devon_20260521_204000.txt
# -> 20260521_204000.WAV); only 'Spectrogram' view rows; clips > 3 ms; written to DATA_DIR/<class>/ as
#   <class>-bat_<ID>_<YEAR>_<YYYYMMDD-HHMMSS>_<start ms>_<end ms>.wav
# i.e. exactly the layout of the shipped dataset, so recording_id() (Cell 7) finds the tape.
# (The previous port took the first two '_' fields of the table name -- 'acsh_devon' here -- as the
# recording stamp, so it matched no audio file and extracted nothing on this repo's selections.)
# Differences from R: the ID is a deterministic hash of table + row (R draws a random number), the YEAR
# field is the recording's year (R writes a constant 2026), and only the needed samples are read.
import csv as _csv8
import zlib as _zlib

_SEL_STAMP_RE = re.compile(r'(\d{8})_(\d{6})')


def extract_clips_from_selections(raw_audio_dir, selections_dir, output_dir, min_clip_s=0.003):
    """Cut the selections of every Raven table into BatSpot-named clips. Returns the number written."""
    audio_files = sorted(p for p in glob.glob(os.path.join(raw_audio_dir, '**', '*'), recursive=True)
                         if p.lower().endswith('.wav'))
    extracted, problems = 0, []
    for species in sorted(os.listdir(selections_dir)):
        species_dir = os.path.join(selections_dir, species)
        if not os.path.isdir(species_dir):
            continue
        tables = sorted(glob.glob(os.path.join(species_dir, '**', '*.txt'), recursive=True))
        out_dir = os.path.join(output_dir, species)
        os.makedirs(out_dir, exist_ok=True)
        for sel_file in tables:
            m = _SEL_STAMP_RE.search(os.path.basename(sel_file))
            if not m:
                problems.append(f'{species}/{os.path.basename(sel_file)}: no YYYYMMDD_HHMMSS in the name')
                continue
            stamp = f'{m.group(1)}_{m.group(2)}'
            audio = [a for a in audio_files if stamp.lower() in os.path.basename(a).lower()]
            if not audio:
                problems.append(f'{species}/{os.path.basename(sel_file)}: no recording matching {stamp}')
                continue
            info = sf.info(audio[0])
            sr = info.samplerate
            with open(sel_file, newline='', encoding='utf-8', errors='replace') as fh:
                rows = [{(k or '').strip().lower(): (v or '').strip() for k, v in r.items()}
                        for r in _csv8.DictReader(fh, delimiter='\t')]
            for n_row, r in enumerate(rows, 1):
                if 'view' in r and 'spectrogram' not in r['view'].lower():
                    continue
                try:
                    begin, end = float(r['begin time (s)']), float(r['end time (s)'])   # any capitalisation
                except (KeyError, ValueError):
                    continue
                start, stop = int(round(begin * sr)), min(int(round(end * sr)), info.frames)
                if stop <= start or (stop - start) / sr <= min_clip_s:
                    continue
                clip, _ = sf.read(audio[0], start=start, stop=stop, dtype='float32', always_2d=True)
                if not np.isfinite(clip).all():
                    continue
                clip_id = _zlib.crc32(f'{os.path.basename(sel_file)}:{n_row}'.encode()) % 10_000_000
                name = (f'{species}-bat_{clip_id}_{m.group(1)[:4]}_{m.group(1)}-{m.group(2)}_'
                        f'{int(round(start / sr * 1000))}_{int(round(stop / sr * 1000))}.wav')
                sf.write(os.path.join(out_dir, name), clip if clip.shape[1] > 1 else clip[:, 0], sr,
                         subtype=info.subtype if info.subtype in sf.available_subtypes('WAV') else None)
                extracted += 1
    print(f'Extracted {extracted} clips to {output_dir}')
    for p in problems[:20]:
        print(f'  skipped {p}')
    return extracted

# Extraction is opt-in, and refuses to clobber an existing dataset.
if not RUN_CLIP_EXTRACTION:
    print('Clip extraction SKIPPED (RUN_CLIP_EXTRACTION = False in Cell 2).')
    print(f'Using the clips already present in: {DATA_DIR}')
elif not (RAW_AUDIO_DIR and SELECTIONS_DIR):
    print('Clip extraction skipped: set RAW_AUDIO_DIR and SELECTIONS_DIR in Cell 2.')
else:
    _existing = glob.glob(os.path.join(DATA_DIR, '**', '*.wav'), recursive=True)
    if _existing and not OVERWRITE_EXISTING_CLIPS:
        print(f'Clip extraction ABORTED: {len(_existing)} wav files already exist in')
        print(f'  {DATA_DIR}')
        print('Re-extracting would overwrite them. Set OVERWRITE_EXISTING_CLIPS = True')
        print('in Cell 2 to force it, or point DATA_DIR at an empty folder.')
    else:
        if _existing:
            print(f'WARNING: overwriting {len(_existing)} existing clips in {DATA_DIR}')
        extract_clips_from_selections(RAW_AUDIO_DIR, SELECTIONS_DIR, DATA_DIR)