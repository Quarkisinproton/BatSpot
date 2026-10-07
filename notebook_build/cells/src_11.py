# Cell 12: Train ALL 3 detector variants (m03, m09, m11) + evaluate each individually.
#
# PRETRAINED_MODELS['detector'] = {'m03': {...}, 'm09': {...}, 'm11': {...}}
# For each mic: load its pretrained encoder, fine-tune, evaluate on det_test.
# Results stored in det_results[mic].

def evaluate_model(model, dataset, class_names, device, model_name='Model', verbose=True):
    """Evaluate on a windowed eval dataset (clip-level, windows averaged). Returns metrics dict.
    verbose=False prints one summary line instead of the full per-class report."""
    all_probs, all_labels = predict_proba(model, dataset, device)
    all_preds = all_probs.argmax(axis=1)
    eval_labels = list(range(len(class_names)))
    acc  = accuracy_score(all_labels, all_preds)
    bal  = balanced_accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    report = classification_report(all_labels, all_preds, labels=eval_labels,
                                   target_names=class_names, zero_division=0,
                                   output_dict=True)
    cm = confusion_matrix(all_labels, all_preds, labels=eval_labels)
    if verbose:
        print(f'\n=== {model_name} ===')
        print(f'Accuracy: {acc:.4f}  Balanced acc: {bal:.4f}  Precision: {prec:.4f}  '
              f'Recall: {rec:.4f}  F1: {f1:.4f}')
        print(classification_report(all_labels, all_preds, labels=eval_labels,
                                    target_names=class_names, zero_division=0))
    else:
        print(f'  {model_name}: accuracy {acc:.4f}  balanced {bal:.4f}  macro-F1 {f1:.4f}')
    return {'accuracy': acc, 'balanced_accuracy': bal, 'precision': prec, 'recall': rec, 'f1': f1,
            'predictions': all_preds, 'labels': all_labels, 'probs': all_probs,
            'confusion_matrix': cm, 'report': report, 'class_names': class_names}


encoderOpts_det = {'input_channels': 1, 'conv_kernel_size': 7, 'max_pool': 2, 'resnet_size': 18}
_det_mics = sorted(PRETRAINED_MODELS.get('detector', {}).keys())
if not _det_mics:
    print('No pretrained detector variants found -> training one from scratch.')
    _det_mics = ['scratch']

# det_results[mic] = {'model', 'metrics', 'best_val_acc', 'encoderOpts', 'classifierOpts'}
det_results = {}

for _mic in _det_mics:
    print(f'\n{"="*60}')
    print(f'DETECTOR VARIANT: {_mic}')
    print(f'{"="*60}')

    set_seed(SEED)   # same init + sampling order for every variant (Cell 2)
    model, encoder, _cls_head, classifierOpts = build_model(
        encoderOpts_det, DET_CONFIG['num_classes'], DEVICE)

    _pk = PRETRAINED_MODELS.get('detector', {}).get(_mic, {}).get('path')
    if _pk:
        try:
            _pre, _, _pre_opts = load_model_from_pk(_pk, DEVICE)
            encoder.load_state_dict(dict(list(_pre.named_children())[0][1].state_dict()))
            print(f'Encoder loaded from {os.path.basename(_pk)} ({_pre_opts.get("sr", 0)//1000}kHz)')
        except Exception as _e:
            print(f'Could not load encoder for {_mic}: {_e}')
    else:
        print(f'No pretrained path for {_mic} -> random init.')

    model = model.to(DEVICE)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    model, _hist, best_acc = train_model(
        model, det_train, det_val, DET_CONFIG, DEVICE,
        save_path=os.path.join(WORKING_DIR, f'detector_{_mic}_curves.png'),
    )
    model = unwrap_model(model)
    print(f'Best val {SELECT_METRIC} ({_mic}): {best_acc:.4f}')

    metrics = evaluate_model(model, det_test, DET_CLASSES, DEVICE,
                             f'Detector {_mic} (Test Set)')

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(2), yticks=range(2),
           xticklabels=DET_CLASSES, yticklabels=DET_CLASSES,
           xlabel='Predicted', ylabel='True',
           title=f'Detector {_mic} Confusion Matrix (Test)')
    for i in range(2):
        for j in range(2):
            v = metrics['confusion_matrix'][i, j]
            ax.text(j, i, str(v), ha='center', va='center',
                    color='white' if v > metrics['confusion_matrix'].max()/2 else 'black')
    plt.tight_layout(); plt.show()

    det_results[_mic] = {'model': model, 'metrics': metrics,
                         'best_val_acc': best_acc, 'history': _hist,
                         'encoderOpts': encoderOpts_det,
                         'classifierOpts': classifierOpts}

# Comparison table
print(f'\n{"="*60}')
print('DETECTOR COMPARISON SUMMARY')
print(f'{"="*60}')
print(f'{"Variant":<10} {"Val score":>9} {"Test Acc":>10} {"Bal Acc":>8} {"Prec":>8} {"Rec":>8} {"F1":>8}')
print('-'*70)
for _mic, _r in det_results.items():
    m = _r['metrics']
    print(f'{_mic:<10} {_r["best_val_acc"]:>9.4f} {m["accuracy"]:>10.4f} {m["balanced_accuracy"]:>8.4f} '
          f'{m["precision"]:>8.4f} {m["recall"]:>8.4f} {m["f1"]:>8.4f}')