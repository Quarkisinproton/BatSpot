# Cell 13: Build Classifier Dataset + Train the Classifier (one model per seed -> ensemble)
#
# Every seed in CLS_ENSEMBLE_SEEDS trains one classifier on the SAME split (same data, different
# initialisation and sampling order). Printed per member and as mean +- sd -- that sd is the run-to-run
# noise any comparison on this split has to beat. The ENSEMBLE averages the members' softmax; it is
# what the cascade (Cell 14) and the inference cells (21-24) use. Cell 15 exports every member plus the
# best single member as classifier_250khz.pk (the BatSpot GUI / CLI loads one model).
#
# Then the "unknown" answer is fitted (UNKNOWN_DETECTION, Cell 2): a known-ness score per clip and a
# threshold set on the VALIDATION clips so that UNKNOWN_KEEP_KNOWN of the correct species answers keep
# their name. Below the threshold the inference cells answer UNKNOWN_LABEL instead of a species.

cls_seq_len = int(float(CLS_CONFIG['sequence_len']) / 1000 * CLS_CONFIG['sr'] / CLS_CONFIG['hop_length'])
print(f'Classifier: sr={CLS_CONFIG["sr"]}, window={cls_seq_len} frames (20 ms), WINDOW_MODE={WINDOW_MODE!r}')

cls_train = WindowedBatDataset(train_wavs, CLS_CLASS_TO_IDX, CLS_CONFIG, train=True,
                               augment=USE_AUGMENTATION,
                               noise_mix_prob=CLS_CONFIG.get('noise_mix_prob', 0.0),
                               noise_mix_snr_db=CLS_CONFIG.get('noise_mix_snr_db', (0.0, 20.0)))
cls_val   = WindowedBatDataset(val_wavs,   CLS_CLASS_TO_IDX, CLS_CONFIG, train=False)
cls_test  = WindowedBatDataset(test_wavs,  CLS_CLASS_TO_IDX, CLS_CONFIG, train=False)

print(f'Classifier train: {Counter([CLS_CLASS_TO_IDX[get_class_from_filename(f)] for f in train_wavs])}')
print(f'Classifier val:   {Counter([CLS_CLASS_TO_IDX[get_class_from_filename(f)] for f in val_wavs])}')
print(f'Classifier test:  {Counter([CLS_CLASS_TO_IDX[get_class_from_filename(f)] for f in test_wavs])}')

encoderOpts_cls = {'input_channels': 1, 'conv_kernel_size': 7, 'max_pool': 2, 'resnet_size': 18}
_cls_seeds = list(CLS_ENSEMBLE_SEEDS) if CLS_ENSEMBLE_SEEDS else [SEED]
cls_members = []
for _s in _cls_seeds:
    print(f'\n{"=" * 60}\nCLASSIFIER seed {_s}  ({len(cls_members) + 1}/{len(_cls_seeds)})\n{"=" * 60}')
    set_seed(_s)
    _m, _enc, _head, cls_classifierOpts = build_model(encoderOpts_cls, CLS_CONFIG['num_classes'], DEVICE)
    if classifier_path:
        try:
            _pre, _, _pre_opts = load_model_from_pk(classifier_path, DEVICE)
            _enc.load_state_dict(dict(list(_pre.named_children())[0][1].state_dict()))
            print(f'Encoder loaded from {os.path.basename(classifier_path)} ({_pre_opts.get("sr", 0)//1000}kHz)')
        except Exception as _e:
            print(f'Could not load classifier pretrained: {_e}')
    else:
        print('No pre-trained classifier. Training from scratch.')
    _m = _m.to(DEVICE)
    _m, _hist, _best = train_model(_m, cls_train, cls_val, CLS_CONFIG, DEVICE,
                                   save_path=os.path.join(WORKING_DIR, f'classifier_seed{_s}_curves.png'))
    _m = unwrap_model(_m).eval()
    _met = evaluate_model(_m, cls_test, CLS_CLASSES, DEVICE, f'classifier seed {_s} (test)', verbose=False)
    cls_members.append({'seed': _s, 'model': _m, 'best_val': _best, 'history': _hist, 'metrics': _met})

if len(cls_members) > 1:
    cls_model = SoftmaxEnsemble([c['model'] for c in cls_members]).to(DEVICE).eval()
    _vp, _vy = predict_proba(cls_model, cls_val, DEVICE)
    cls_best_acc = {'balanced_accuracy': balanced_accuracy_score, 'accuracy': accuracy_score}[SELECT_METRIC](
        _vy, _vp.argmax(axis=1))
else:
    cls_model, cls_best_acc = cls_members[0]['model'], cls_members[0]['best_val']
cls_history = cls_members[0]['history']
cls_best_member = max(cls_members, key=lambda c: c['best_val'])
print(f'\nClassifier training complete. Validation {SELECT_METRIC}: '
      + ('ensemble ' if len(cls_members) > 1 else '') + f'{cls_best_acc:.4f}')

cls_metrics = evaluate_model(cls_model, cls_test, CLS_CLASSES, DEVICE,
                             'Classifier ' + ('ENSEMBLE of %d seeds' % len(cls_members)
                                              if len(cls_members) > 1 else '') + ' (Test Set)')

print(f'{"member":<10} {"best val":>9} {"@epoch":>7} {"top-k mean":>11} {"test acc":>9} {"test bal":>9} {"macro-F1":>9}')
for c in cls_members:
    _t = c['history'].get('topk_mean')
    print(f'seed {c["seed"]:<5} {c["best_val"]:>9.4f} {str(c["history"]["best_epoch"]):>7} '
          f'{(f"{_t:.4f}" if _t is not None else "-"):>11} {c["metrics"]["accuracy"]:>9.4f} '
          f'{c["metrics"]["balanced_accuracy"]:>9.4f} {c["metrics"]["f1"]:>9.4f}')
if len(cls_members) > 1:
    _a = np.array([[c['metrics']['accuracy'], c['metrics']['balanced_accuracy'], c['metrics']['f1']]
                   for c in cls_members])
    print(f'{"mean":<10} {"":>9} {"":>7} {"":>11} {_a[:, 0].mean():>9.4f} {_a[:, 1].mean():>9.4f} {_a[:, 2].mean():>9.4f}')
    print(f'{"sd":<10} {"":>9} {"":>7} {"":>11} {_a[:, 0].std(ddof=1):>9.4f} {_a[:, 1].std(ddof=1):>9.4f} '
          f'{_a[:, 2].std(ddof=1):>9.4f}')
    print(f'{"ENSEMBLE":<10} {cls_best_acc:>9.4f} {"":>7} {"":>11} {cls_metrics["accuracy"]:>9.4f} '
          f'{cls_metrics["balanced_accuracy"]:>9.4f} {cls_metrics["f1"]:>9.4f}')
    print('  (differences between members are run-to-run noise on this split; one test clip = '
          f'{1 / len(test_wavs):.4f} accuracy)')

# ---- "unknown" answer ------------------------------------------------------------------------------
UNKNOWN_MODEL = None
if UNKNOWN_DETECTION:
    _noise_idx = CLS_CLASS_TO_IDX.get('noise')
    _need_ref = UNKNOWN_METHOD != 'msp'
    _vp, _vy, _ve = predict_proba(cls_model, cls_val, DEVICE, return_embedding=True)
    _te, _ty = None, None
    if _need_ref:
        cls_train_eval = WindowedBatDataset(train_wavs, CLS_CLASS_TO_IDX, CLS_CONFIG, train=False)
        _, _ty, _te = predict_proba(cls_model, cls_train_eval, DEVICE, return_embedding=True)
    UNKNOWN_MODEL = fit_unknown_model(UNKNOWN_METHOD, _vp, _vy, _ve, _te, _ty, UNKNOWN_KEEP_KNOWN, _noise_idx,
                                      n_members=len(model_members(cls_model)))
    _tp, _ty2, _tem = predict_proba(cls_model, cls_test, DEVICE, return_embedding=True)
    _unk = apply_unknown(_tp, _tem, UNKNOWN_MODEL)
    _correct = _tp.argmax(axis=1) == _ty2
    print(f'\n"UNKNOWN" ANSWER: method {UNKNOWN_METHOD!r}, threshold {UNKNOWN_MODEL["threshold"]:.4f} '
          f'(keeps {UNKNOWN_KEEP_KNOWN:.0%} of the correct validation answers)')
    print(f'  test clips of KNOWN species answered "{UNKNOWN_LABEL}": {_unk.sum()}/{len(_unk)} '
          f'({_unk.mean():.1%}); of them {(_unk & _correct).sum()} would have been correct and '
          f'{(_unk & ~_correct).sum()} wrong')
    print(f'  test accuracy counting "{UNKNOWN_LABEL}" as an error: {np.mean(_correct & ~_unk):.4f} '
          f'(without the unknown answer {np.mean(_correct):.4f})')
    print('  How often a species the model has NEVER seen gets "unknown" cannot be measured here (all test '
          'species are known); it was measured by hiding each species in turn -- see AGENTS.md section 11.')

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cls_metrics['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
n_cls = len(CLS_CLASSES)
ax.set(xticks=range(n_cls), yticks=range(n_cls),
       xticklabels=CLS_CLASSES, yticklabels=CLS_CLASSES,
       xlabel='Predicted', ylabel='True',
       title='Classifier ' + ('ensemble ' if len(cls_members) > 1 else '') + 'Confusion Matrix (Test)')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
for i in range(n_cls):
    for j in range(n_cls):
        v = cls_metrics['confusion_matrix'][i, j]
        ax.text(j, i, str(v), ha='center', va='center',
                color='white' if v > cls_metrics['confusion_matrix'].max()/2 else 'black')
plt.tight_layout(); plt.show()
