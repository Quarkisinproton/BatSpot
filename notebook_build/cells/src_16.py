# Cell 17: Load & Evaluate Each Existing Model on the TEST split
#
# WHY THIS CELL CHANGED: the previous version only scored files whose species name appeared
# in the model's own class list. The official detectors know {'noise','target'} and the official
# classifier 15 European species, so only the 186 'noise' files matched -- the reported
# "accuracy" (e.g. m03 = 0.984) was NOISE RECALL on all clips, i.e. "it says noise to almost
# everything". That number is not comparable to a fine-tuned model's accuracy on calls + noise.
#
# Now every model is scored on the SAME held-out test clips with the SAME windowed protocol:
#   * detector {noise,target}         : every bat species counts as 'target' -> call-vs-noise
#   * classifier w/o species overlap  : only "does it say noise?" is meaningful -> scored as a
#                                       noise-vs-call detector (species accuracy is N/A)
#   * model whose class names overlap : exact-name multi-class on the overlapping classes
from sklearn.metrics import roc_auc_score


def run_model_evaluation(model_path, wav_files, device):
    """Evaluate one existing model on `wav_files` (pass test_wavs)."""
    model_name = os.path.basename(model_path)
    try:
        model, model_classes, dataOpts = load_any_model(model_path, device)
    except Exception as e:
        return {'name': model_name, 'path': model_path, 'error': str(e)}
    model.to(device).eval()

    data_species = {get_class_from_filename(f) for f in wav_files} - {'noise'}
    # predict.py's fallback chain: a model that stores num_mels instead of n_freq_bins must not be fed
    # a 256-bin spectrogram silently
    _fc = dataOpts.get('freq_compression', 'linear')
    if _fc != 'linear':
        print(f'  WARNING: {model_name} uses freq_compression={_fc!r}; this notebook only implements linear '
              f'band cropping, so its score below is not comparable to its training distribution.')
    cfg = {'sr': dataOpts.get('sr', 384000), 'n_fft': dataOpts.get('n_fft', 256),
           'hop_length': dataOpts.get('hop_length', 128),
           'n_freq_bins': dataOpts.get('n_freq_bins') or dataOpts.get('num_mels') or 256,
           'fmin': dataOpts.get('fmin', 18000), 'fmax': dataOpts.get('fmax', 90000),
           'sequence_len': 20}

    if set(model_classes) == {'noise', 'target'}:
        kind = 'detector (call vs noise)'
        files = list(wav_files)
        label_map = {c: 0 for c in ['noise']}
        label_map.update({c: 1 for c in data_species})
        names = ['noise', 'call']
    elif 'noise' in model_classes and not (set(model_classes) & data_species):
        kind = 'classifier scored as noise-vs-call (no species overlap)'
        files = list(wav_files)
        label_map = {c: 0 for c in ['noise']}
        label_map.update({c: 1 for c in data_species})
        names = ['noise', 'call']
    else:
        kind = 'multi-class (overlapping classes only)'
        files = [f for f in wav_files if get_class_from_filename(f) in model_classes]
        label_map = {get_class_from_filename(f): model_classes[get_class_from_filename(f)] for f in files}
        names = None          # one name per model OUTPUT, built from the probability width below
    if not files:
        return {'name': model_name, 'path': model_path, 'unsupported': True,
                'error': f'no overlapping files: model knows {sorted(model_classes)}, '
                         f'data has {sorted(data_species | {"noise"})}'}

    ds = WindowedBatDataset(files, label_map, cfg, train=False)
    probs, labels = predict_proba(model, ds, device)
    if names is None:   # sorting the class dict by value breaks on non-contiguous indices
        _inv = {v: k for k, v in model_classes.items()}
        names = [_inv.get(i, f'class{i}') for i in range(probs.shape[1])]
    out = {'name': model_name, 'path': model_path, 'kind': kind, 'n_files': len(files),
           'target_names': names, 'labels': labels, 'files': files, 'auc': None}
    if kind.startswith('detector'):
        p_call = probs[:, model_classes['target']]
    elif kind.startswith('classifier'):
        p_call = 1.0 - probs[:, model_classes['noise']]
    else:
        p_call = None
    if p_call is not None:
        preds = (p_call >= 0.5).astype(int)
        out['auc'] = float(roc_auc_score(labels, p_call)) if len(set(labels)) > 1 else None
    else:
        preds = probs.argmax(axis=1)
    out.update({'preds': preds, 'accuracy': accuracy_score(labels, preds),
                'balanced_accuracy': balanced_accuracy_score(labels, preds),
                'confusion_matrix': confusion_matrix(labels, preds, labels=list(range(len(names)))),
                'model_classes': model_classes,
                'no_signal': out['auc'] is not None and abs(out['auc'] - 0.5) < AUC_NO_SIGNAL})
    return out


# Evaluate all models on the held-out TEST split
val_results = []
for m_path in model_inventory:
    result = run_model_evaluation(m_path, test_wavs, DEVICE)
    val_results.append(result)
    tag = model_tags.get(m_path, '?')
    if 'error' in result:
        status = 'UNSUPPORTED' if result.get('unsupported') else 'ERROR'
        print(f'[{status}] {tag} / {result["name"]}: {result["error"]}')
    else:
        _auc = f', auc={result["auc"]:.4f}' if result['auc'] is not None else ''
        print(f'[OK] {tag} / {result["name"]}: {result["kind"]}: acc={result["accuracy"]:.4f}, '
              f'bal_acc={result["balanced_accuracy"]:.4f}{_auc}, files={result["n_files"]}'
              + ('  [NO SIGNAL: AUC ~ 0.5]' if result.get('no_signal') else ''))