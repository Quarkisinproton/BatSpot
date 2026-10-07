# Cell 9: Training Infrastructure

def set_seed(seed):
    """Seed python / numpy / torch (incl. CUDA); DataLoader workers derive their seeds from torch."""
    if seed is None:
        return
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compute_class_weights(file_names, class_to_idx, num_labels=None):
    """Inverse-frequency class weights, one per MODEL OUTPUT class.

    num_labels must be the classifier head size (config['num_classes']), not
    len(class_to_idx): the detector folds 8 bat species + 'noise' into 2
    labels, so keying the vector off the dict length yields a 9-element
    tensor for a 2-output head and CrossEntropyLoss rejects it.
    """
    labels = [class_to_idx[get_class_from_filename(f)] for f in file_names]
    counts = Counter(labels)
    total = len(labels)
    n = num_labels if num_labels else len(class_to_idx)
    weights = torch.zeros(n)
    n_present = len(counts)
    for idx, count in counts.items():
        weights[idx] = total / (n_present * count) if count else 0.0
    return weights

def make_weighted_sampler(file_names, class_to_idx, num_labels=None):
    """Create WeightedRandomSampler for class balancing.
    Every class is then drawn with probability 1/n_classes, so the model's outputs reflect a UNIFORM
    class prior (the detector trains at 50/50 noise/call while the data is ~19/81) -- that is why a
    detector threshold tuned on the test mix does not transfer to recordings that are mostly noise."""
    labels = [class_to_idx[get_class_from_filename(f)] for f in file_names]
    weights = compute_class_weights(file_names, class_to_idx, num_labels)
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def train_model(model, train_dataset, val_dataset, config, device, save_path=None):
    """
    Fine-tune `model` on windowed clips. AMP, optional gradient accumulation, DataParallel.

    train_dataset : WindowedBatDataset(train=True)  -> a new random 20 ms crop per clip per epoch
    val_dataset   : WindowedBatDataset(train=False) -> clip-level score = softmax averaged over
                    that clip's top-k windows (same protocol used for the test set)
    Model selection / LR schedule / early stopping all use SELECT_METRIC on the validation
    set (balanced accuracy by default: plain accuracy cannot tell a real detector from an
    "always predict call" one on an 80/20 split).
    save_path : if given, the training curves (train loss, validation accuracy / balanced
    accuracy, learning rate) are written there as a PNG. It never affects which weights are
    returned (the best-validation weights are always restored).
    Returns (model, history, best_score).
    """
    # Weighted sampler for balanced batches (50/50 per class).
    # NOTE: We do NOT also pass class_weights to CrossEntropyLoss.
    # WeightedRandomSampler already balances; adding loss weights on top
    # creates a 4x noise penalty that collapses m03 (weak pretrain) to all-noise.
    all_file_names = train_dataset.file_names
    class_to_idx = train_dataset.class_to_idx
    sampler = make_weighted_sampler(all_file_names, class_to_idx,
                                   config.get('num_classes'))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              sampler=sampler, num_workers=4, pin_memory=True,
                              persistent_workers=True)

    # Loss, optimizer, scheduler
    # unweighted (the sampler already balances batches); optional label smoothing from the config
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(config.get('label_smoothing', 0.0)))
    effective_lr = config['base_lr']  # paper LR is absolute; do NOT scale by batch size
    optimizer = optim.Adam(model.parameters(), lr=effective_lr, betas=(config.get('beta1', 0.5), 0.999))
    patience_lr = int(max(1, ceil(config.get('lr_patience_epochs', 8) / config['epochs_per_eval'])))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=patience_lr, factor=config.get('lr_decay_factor', 0.5), threshold=1e-3, threshold_mode='abs')
    patience_early = int(config.get('early_stopping_patience_epochs', 20))  # RAW epochs, read from config
    grad_accum_steps = config.get('grad_accum_steps', 1)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Multi-GPU
    is_dp = config.get('use_multi_gpu', False) and torch.cuda.device_count() > 1
    if is_dp:
        model = nn.DataParallel(model)
        print(f'DataParallel across {torch.cuda.device_count()} GPUs')

    print(f'Grad accum: {grad_accum_steps}, AMP: {use_amp}, DP: {is_dp}, '
          f'batch: {config["batch_size"]} ({len(train_loader)} steps/epoch)')
    print(f'Effective LR: {effective_lr:.2e}   selection metric: {SELECT_METRIC}')
    print(f'Patience LR: {patience_lr} validations, Patience ES: {patience_early} raw epochs')

    history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': [], 'val_score': [], 'val_epoch': [],
               'lr': [], 'best_epoch': None}
    best_score = float('-inf')   # so a first validation score of exactly 0.0 is still captured
    best_state = None
    no_improve = 0

    for epoch in range(config['n_epochs']):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["n_epochs"]}', leave=False)
        for step, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(inputs)
                loss = loss_fn(output, labels) / grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history['train_loss'].append(avg_loss)

        # Validate every epochs_per_eval
        if (epoch + 1) % config['epochs_per_eval'] == 0 or epoch == config['n_epochs'] - 1:
            val_probs, val_labels = predict_proba(model, val_dataset, device)
            val_pred = val_probs.argmax(axis=1)
            val_acc = accuracy_score(val_labels, val_pred)
            val_bal = balanced_accuracy_score(val_labels, val_pred)
            val_score = {'balanced_accuracy': val_bal, 'accuracy': val_acc}[SELECT_METRIC]  # KeyError on a typo
            history['val_acc'].append(val_acc)
            history['val_bal_acc'].append(val_bal)
            history['val_score'].append(val_score)
            history['val_epoch'].append(epoch + 1)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            scheduler.step(val_score)

            if val_score > best_score:
                best_score = val_score
                history['best_epoch'] = epoch + 1
                no_improve = 0
                # Always snapshot bare keys (strip DataParallel's "module." prefix)
                _sd = model.module.state_dict() if is_dp else model.state_dict()
                best_state = {k: v.detach().clone() for k, v in _sd.items()}
            else:
                no_improve += config['epochs_per_eval']  # RAW epochs elapsed, not validation steps

            print(f'  Epoch {epoch+1}: loss={avg_loss:.4f} val_acc={val_acc:.4f} val_bal={val_bal:.4f} '
                  f'best={best_score:.4f} lr={optimizer.param_groups[0]["lr"]:.2e}')

            if no_improve >= patience_early:
                print(f'Early stopping at epoch {epoch+1}.')
                break

    # Restore best weights (best_state always has bare keys) -- unconditionally
    if best_state is not None:
        if is_dp:
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
    else:
        print('  WARNING: no validation score was recorded; returning the last-epoch weights.')

    # Selection optimism, made visible: the best of many validations of a discrete metric on ~145
    # clips sits above the run's typical level. Print the top-k mean beside the argmax.
    history['best_score'] = best_score
    history['topk_mean'] = None
    if REPORT_TOPK_MEAN and history['val_score']:
        _k = min(int(REPORT_TOPK_MEAN), len(history['val_score']))
        history['topk_mean'] = float(np.mean(sorted(history['val_score'])[-_k:]))
        _vl = np.asarray(getattr(val_dataset, 'labels', []))
        _cnt = np.unique(_vl, return_counts=True)[1] if len(_vl) else []
        # score change from flipping one validation clip (averaged over the classes)
        _clip = ((float(np.mean(1.0 / _cnt)) / len(_cnt) if SELECT_METRIC == 'balanced_accuracy'
                  else 1.0 / len(_vl)) if len(_cnt) > 1 else None)
        _gap = best_score - history['topk_mean']
        print(f'  selection: best {best_score:.4f} @ epoch {history["best_epoch"]} | mean of top-{_k} '
              f'{history["topk_mean"]:.4f} | mean of all {np.mean(history["val_score"]):.4f}'
              + (f' | best - top-{_k} = {_gap:+.4f} (~{_gap / _clip:.1f} val clips)' if _clip else ''))

    # Training curves (the *_curves.png paths passed by Cells 12-13 were never written before)
    if save_path and history['val_epoch']:
        try:
            fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
            ax[0].plot(range(1, len(history['train_loss']) + 1), history['train_loss'])
            ax[0].set(title='train loss', xlabel='epoch')
            ax[1].plot(history['val_epoch'], history['val_acc'], label='accuracy')
            ax[1].plot(history['val_epoch'], history['val_bal_acc'], label='balanced accuracy')
            if history['best_epoch']:
                ax[1].axvline(history['best_epoch'], ls=':', c='k', label=f'best (epoch {history["best_epoch"]})')
            if history['topk_mean'] is not None:
                ax[1].axhline(history['topk_mean'], ls='--', c='grey', lw=0.8,
                              label=f'mean of top-{REPORT_TOPK_MEAN}')
            ax[1].set(title=f'validation (selection: {SELECT_METRIC})', xlabel='epoch'); ax[1].legend(fontsize=8)
            ax[2].semilogy(history['val_epoch'], history['lr']); ax[2].set(title='learning rate', xlabel='epoch')
            fig.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)
            print(f'  training curves -> {save_path}')
        except Exception as e:   # never fail a training run over a plot
            print(f'  (could not write training curves to {save_path}: {e!r})')

    return model, history, best_score

print('Training infrastructure defined')