# Cell 18: Results Summary & Confusion Matrices

print('\n' + '='*80)
print('EXISTING MODEL EVALUATION SUMMARY (held-out test split)')
print('='*80)

for r in val_results:
    if 'error' in r:
        print(f'\n{r["name"]}: ERROR - {r["error"]}')
        continue
    print(f'\n--- {r["name"]} ---')
    print(f'  Mode: {r["kind"]}')
    print(f'  Files evaluated: {r["n_files"]} of {len(test_wavs)} test clips'
          + ('   <-- reduced denominator, NOT comparable to full-split rows' if r['n_files'] != len(test_wavs) else ''))
    if r.get('no_signal'):
        print(f'  *** AUC within {AUC_NO_SIGNAL} of 0.5: NO usable signal on this task -- its accuracy only '
              'reflects the class mix; read balanced accuracy.')
    print(f'  Accuracy: {r["accuracy"]:.4f}   Balanced accuracy: {r["balanced_accuracy"]:.4f}'
          + (f'   AUC: {r["auc"]:.4f}' if r['auc'] is not None else ''))
    print(classification_report(r['labels'], r['preds'], labels=list(range(len(r['target_names']))),
                                target_names=r['target_names'], zero_division=0))

    cm = r['confusion_matrix']
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    names = r['target_names']
    n = len(names)
    ax.set(xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names,
           xlabel='Predicted', ylabel='True', title=f'{r["name"][:40]}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.show()

# ---- Official (zero-shot) vs fine-tuned, same test clips, call-vs-noise -------------------
print('\n' + '='*80)
print('OFFICIAL (zero-shot) vs FINE-TUNED -- call vs noise on the same test clips')
print('='*80)
print(f'{"Model":<34} {"n":>4} {"Acc":>7} {"Bal acc":>8} {"AUC":>7}  Note')
print('-'*92)
for r in val_results:
    if 'error' in r or r['auc'] is None:
        continue
    _note = 'NO SIGNAL (AUC~0.5): plain accuracy just reflects the class mix' if r.get('no_signal') \
        else r['kind'] + ('' if r['n_files'] == len(test_wavs) else f' (only {r["n_files"]} clips)')
    _lbl = r['name'].replace('.pk', '')
    _lbl = _lbl.replace('_', ' ', 1) if _lbl.startswith('official_') else model_tags.get(r['path'], 'yours') + ' ' + _lbl
    print(f'{_lbl[:34]:<34} {r["n_files"]:>4} {r["accuracy"]:>7.4f} '
          f'{r["balanced_accuracy"]:>8.4f} {r["auc"]:>7.4f}  {_note}')
for _mic, _dr in det_results.items():
    _m = _dr['metrics']
    _auc = roc_auc_score(_m['labels'], _m['probs'][:, 1]) if len(set(_m['labels'])) > 1 else float('nan')
    print(f'{"fine-tuned detector " + _mic:<34} {len(_m["labels"]):>4} {_m["accuracy"]:>7.4f} '
          f'{_m["balanced_accuracy"]:>8.4f} {_auc:>7.4f}  fine-tuned call-vs-noise')
_cy = (cls_metrics['labels'] != CLS_CLASS_TO_IDX['noise']).astype(int)
_cp = 1.0 - cls_metrics['probs'][:, CLS_CLASS_TO_IDX['noise']]
print(f'{"fine-tuned classifier (noise?)":<34} {len(_cy):>4} {np.mean((_cp >= 0.5) == _cy):>7.4f} '
      f'{balanced_accuracy_score(_cy, (_cp >= 0.5).astype(int)):>8.4f} '
      f'{(roc_auc_score(_cy, _cp) if len(set(_cy)) > 1 else float("nan")):>7.4f}  classifier as noise-vs-call')

# Balanced accuracy is the fair comparison: on an 81/19 call/noise split plain accuracy rewards
# a model that answers "call" for everything.
print('\nBalanced accuracy, official zero-shot -> fine-tuned (same test clips):')
for _mic, _dr in det_results.items():
    _off = [r for r in val_results if 'error' not in r and r['name'] == f'official_detector_{_mic}.pk']
    if _off:
        _o, _f = _off[0]['balanced_accuracy'], _dr['metrics']['balanced_accuracy']
        print(f'  {_mic}: {_o:.3f} -> {_f:.3f}  ({_f - _o:+.3f})')