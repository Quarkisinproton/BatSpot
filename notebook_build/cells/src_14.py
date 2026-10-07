# Cell 15: Export Models as .pk
#
# Exports all fine-tuned detector variants + the single classifier.
import copy

def export_pk(model, encoderOpts, classifierOpts, dataOpts, classes, path):
    """Export model as .pk file compatible with BatSpot GUI/CLI."""
    if isinstance(model, nn.DataParallel):
        model = model.module
    # Export a CPU COPY. model.cpu() moved the live models off the GPU in place, so any later
    # cell that ran them on DEVICE (e.g. the inference cells) failed with a device mismatch.
    model = copy.deepcopy(model).cpu()
    encoder = model[0]
    classifier = model[1]
    torch.save({
        'encoderOpts':     encoderOpts,
        'classifierOpts':  classifierOpts,
        'dataOpts':        dataOpts,
        'encoderState':    encoder.state_dict(),
        'classifierState': classifier.state_dict(),
        'classes':         classes,
    }, path)
    print(f'Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)')

det_dataOpts = {
    'sr': DET_CONFIG['sr'], 'n_fft': DET_CONFIG['n_fft'],
    'hop_length': DET_CONFIG['hop_length'], 'n_freq_bins': DET_CONFIG['n_freq_bins'],
    'fmin': DET_CONFIG['fmin'], 'fmax': DET_CONFIG['fmax'],
    'freq_compression': DET_CONFIG['freq_compression'],
    'min_level_db': -100, 'ref_level_db': 20, 'preemphases': 0.98,
}
det_classes = {'noise': 0, 'call': 1}

# Export each detector variant
det_out_paths = {}
for _mic, _dr in det_results.items():
    _out = os.path.join(WORKING_DIR, f'detector_192khz_{_mic}.pk')
    export_pk(_dr['model'], _dr['encoderOpts'], _dr['classifierOpts'],
              det_dataOpts, det_classes, _out)
    det_out_paths[_mic] = _out

# Export classifier (250kHz, multi-species)
cls_dataOpts = {
    'sr': CLS_CONFIG['sr'], 'n_fft': CLS_CONFIG['n_fft'],
    'hop_length': CLS_CONFIG['hop_length'], 'n_freq_bins': CLS_CONFIG['n_freq_bins'],
    'fmin': CLS_CONFIG['fmin'], 'fmax': CLS_CONFIG['fmax'],
    'freq_compression': CLS_CONFIG['freq_compression'],
    'min_level_db': -100, 'ref_level_db': 20, 'preemphases': 0.98,
}
# The GUI / CLI load ONE model: classifier_250khz.pk = the member with the best validation score.
# Every member is exported too; the inference cells (21-24) can load them as an ensemble.
cls_out_path = os.path.join(WORKING_DIR, 'classifier_250khz.pk')
export_pk(cls_best_member['model'], encoderOpts_cls, cls_classifierOpts, cls_dataOpts, CLS_CLASS_TO_IDX,
          cls_out_path)
cls_member_paths = []
if len(cls_members) > 1:
    for _c in cls_members:
        _p = os.path.join(WORKING_DIR, f'classifier_250khz_seed{_c["seed"]}.pk')
        export_pk(_c['model'], encoderOpts_cls, cls_classifierOpts, cls_dataOpts, CLS_CLASS_TO_IDX, _p)
        cls_member_paths.append(_p)
    print(f'  best single member: seed {cls_best_member["seed"]} -> {os.path.basename(cls_out_path)}')
# Settings of the "unknown" answer for exactly this (ensemble of) classifier(s). The scores of the
# ensemble are not those of a single member, so the file records which member files it belongs to.
unknown_out_path = None
if UNKNOWN_MODEL is not None:
    unknown_out_path = os.path.join(WORKING_DIR, 'classifier_250khz_unknown.npz')
    save_unknown_model(UNKNOWN_MODEL, unknown_out_path, cls_member_paths or [cls_out_path])
    print(f'Saved: {unknown_out_path} (method {UNKNOWN_MODEL["method"]}, for '
          f'{len(cls_member_paths) or 1} classifier file(s))')

# Verify exports
print('\nVerifying exports...')
for _mic, p in det_out_paths.items():
    obj = torch.load(p, map_location='cpu', weights_only=False)
    print(f'  detector_{_mic}: enc={len(obj["encoderState"])} params, '
          f'cls={len(obj["classifierState"])} params, classes={obj["classes"]}')
obj = torch.load(cls_out_path, map_location='cpu', weights_only=False)
print(f'  classifier:    enc={len(obj["encoderState"])} params, '
      f'cls={len(obj["classifierState"])} params, classes={list(obj["classes"].keys())}')

print('\nTo reuse the ensemble later (skip training): INFER_CLASSIFIER_PK = '
      + (repr([os.path.basename(p) for p in cls_member_paths]) if cls_member_paths else repr('classifier_250khz.pk'))
      + ' (full paths), INFER_UNKNOWN_FILE = classifier_250khz_unknown.npz')
print('\nNOTE: these models were trained with per-window MIN-MAX normalisation on 20 ms windows,')
print('exactly like the official BatSpot models. When predicting with PREDICTION/start_prediction.py')
print('(or the GUI) enable min-max normalisation (min_max_norm=true), otherwise inputs are scaled differently.')