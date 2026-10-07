# Cell 14: Combined Pipeline — each detector variant (m03/m09/m11) x the single classifier.
#
# All probabilities are clip-level (softmax averaged over the clip's top-k windows).
# Three ways of combining detector + classifier are reported side by side, because on this
# dataset the classifier already has a 'noise' class and a hard detector gate can only ADD errors:
#   classifier alone : argmax of the 8-class classifier (no detector)
#   hard gate        : detector says noise (P(call) < tuned threshold) -> 'noise', else classifier
#   soft combine     : average the detector's and classifier's noise evidence, then argmax
# The detector threshold is tuned on the VALIDATION set (maximise call-class F1).

def run_combined_pipeline(det_model, mic_label, cls_model, det_val, det_test, cls_test, device):
    """Tune threshold on val, then evaluate classifier-alone / hard gate / soft combine on test."""
    assert [os.path.basename(f) for f in det_test.file_names] == \
           [os.path.basename(f) for f in cls_test.file_names], 'detector/classifier test order differs'
    noise_idx = CLS_CLASS_TO_IDX['noise']

    val_probs, val_true = predict_proba(det_model, det_val, device)
    val_call = val_probs[:, 1]
    best_t, best_f1 = 0.05, -1.0  # first grid value; m11 sat on the old 0.10 grid edge
    for t in np.arange(0.05, 0.96, 0.05):
        _, _, f1, _ = precision_recall_fscore_support(
            val_true, (val_call >= t).astype(int), average='binary', pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    print(f'  [{mic_label}] threshold={best_t:.2f}  val-F1={best_f1:.4f}')

    det_p, _ = predict_proba(det_model, det_test, device)
    p_call = det_p[:, 1]
    cls_p, clabels = predict_proba(cls_model, cls_test, device)

    alone = cls_p.argmax(axis=1)
    gate = np.where(p_call < best_t, noise_idx, alone)
    p_noise = 0.5 * cls_p[:, noise_idx] + 0.5 * (1 - p_call)
    rest = cls_p.copy(); rest[:, noise_idx] = 0
    rest = rest / np.maximum(rest.sum(axis=1, keepdims=True), 1e-9) * (1 - p_noise)[:, None]
    rest[:, noise_idx] = p_noise
    soft = rest.argmax(axis=1)

    out = {'threshold': best_t, 'variants': {}}
    for name, preds in (('classifier alone', alone), ('hard gate', gate), ('soft combine', soft)):
        prec, rec, f1, _ = precision_recall_fscore_support(
            clabels, preds, average='macro', zero_division=0)
        out['variants'][name] = {'acc': accuracy_score(clabels, preds), 'prec': prec, 'rec': rec,
                                 'f1': f1, 'cm': confusion_matrix(clabels, preds,
                                                                  labels=list(range(len(CLS_CLASSES)))),
                                 'report': classification_report(clabels, preds,
                                                                 labels=list(range(len(CLS_CLASSES))),
                                                                 target_names=CLS_CLASSES, zero_division=0)}
    # keep the original keys (hard gate) so older downstream code keeps working
    g = out['variants']['hard gate']
    out.update({'acc': g['acc'], 'prec': g['prec'], 'rec': g['rec'], 'f1': g['f1'],
                'cm': g['cm'], 'report': g['report']})
    return out


pipeline_results = {}
for _mic, _dr in det_results.items():
    print(f'\n{"="*60}')
    print(f'COMBINED PIPELINE: Detector {_mic} -> Classifier')
    print(f'{"="*60}')
    _res = run_combined_pipeline(_dr['model'], _mic, cls_model, det_val, det_test, cls_test, DEVICE)
    pipeline_results[_mic] = _res

    print(f'\n=== Combined {_mic} (hard gate): Accuracy={_res["acc"]:.4f}  F1={_res["f1"]:.4f} ===')
    print(_res['report'])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(_res['cm'], interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    n_cls = len(CLS_CLASSES)
    ax.set(xticks=range(n_cls), yticks=range(n_cls),
           xticklabels=CLS_CLASSES, yticklabels=CLS_CLASSES,
           xlabel='Predicted', ylabel='True',
           title=f'Combined {_mic} (hard gate) Confusion Matrix (Test)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(n_cls):
        for j in range(n_cls):
            v = _res['cm'][i, j]
            ax.text(j, i, str(v), ha='center', va='center',
                    color='white' if v > _res['cm'].max()/2 else 'black')
    plt.tight_layout(); plt.show()

# Final comparison table
print(f'\n{"="*72}')
print('COMBINED PIPELINE COMPARISON SUMMARY (test accuracy, 8 classes)')
print(f'{"="*72}')
print(f'{"Detector":<10} {"Thresh":>7} {"Clf alone":>10} {"Hard gate":>10} {"Soft comb.":>11} {"Gate F1":>8}')
print('-'*72)
for _mic, _r in pipeline_results.items():
    v = _r['variants']
    print(f'{_mic:<10} {_r["threshold"]:>7.2f} {v["classifier alone"]["acc"]:>10.4f} '
          f'{v["hard gate"]["acc"]:>10.4f} {v["soft combine"]["acc"]:>11.4f} {_r["f1"]:>8.4f}')