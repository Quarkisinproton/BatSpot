# Cell 5: Model Architecture & Loading
# --- Faithful port of animal_spot/models/ (GPL-3.0, Bergler & Schroeter) ---
# Module/attribute names MUST match the original, or saved .pk state_dicts
# will not load. In particular the residual projection is `shortcut`
# (ResidualBase assigns self.shortcut = downsample), NOT `downsample`.

def get_padding(kernel_size):
    """Return `same` padding for a given kernel size."""
    if isinstance(kernel_size, int):
        return kernel_size // 2
    return tuple(s // 2 for s in kernel_size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, upsample=None, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        if downsample is not None and upsample is not None:
            raise ValueError("Either downsample or upsample has to be None")

        if upsample is None:
            self.shortcut = downsample
            self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride,
                                   padding=get_padding(3), bias=False)
        else:
            self.shortcut = upsample
            self.conv1 = nn.ConvTranspose2d(in_ch, mid_ch, kernel_size=3, stride=stride,
                                            padding=get_padding(3),
                                            output_padding=get_padding(stride), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1,
                               padding=get_padding(3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None, upsample=None, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        if downsample is not None and upsample is not None:
            raise ValueError("Either downsample or upsample has to be None")
        self.shortcut = None
        if upsample is not None or downsample is not None:
            self.shortcut = downsample if downsample is not None else upsample

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        if upsample is not None:
            self.conv2 = nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=3, stride=stride,
                                            padding=get_padding(3),
                                            output_padding=get_padding(stride), bias=False)
        else:
            self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride,
                                   padding=get_padding(3), bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_ch, out_ch * self.expansion, kernel_size=1,
                               stride=1, padding=get_padding(1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out += residual
        out = self.relu3(out)
        return out


def get_block_sizes(resnet_size):
    if resnet_size == 18:
        return [2, 2, 2, 2]
    elif resnet_size == 34:
        return [3, 4, 6, 3]
    elif resnet_size in (50, 101):
        return [3, 4, 23, 3]
    elif resnet_size == 152:
        return [3, 8, 36, 3]
    raise ValueError("Unsuported resnet size: ", resnet_size)


def get_block_type(resnet_size):
    if resnet_size in (18, 34):
        return BasicBlock
    elif resnet_size in (50, 101, 152):
        return Bottleneck
    raise ValueError("Unsuported resnet size ", resnet_size)


class ResidualBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.cur_in_ch = 64

    def make_layer(self, block, out_ch, size, stride=1, shortcut="downsample"):
        layers = []
        stride_mean = stride
        if isinstance(stride, tuple):
            stride_mean = sum(stride) / len(stride)

        if shortcut == "upsample" and (stride_mean > 1
                                       or self.cur_in_ch != out_ch * block.expansion):
            shortcut = nn.ConvTranspose2d(self.cur_in_ch, out_ch * block.expansion,
                                          kernel_size=3, stride=stride,
                                          padding=get_padding(3),
                                          output_padding=get_padding(stride), bias=False)
            layers.append(block(self.cur_in_ch, out_ch, stride=stride,
                                mid_ch=self.cur_in_ch // block.expansion,
                                upsample=shortcut))
        elif shortcut == "downsample" and (stride_mean > 1
                                           or self.cur_in_ch != out_ch * block.expansion):
            shortcut = nn.Sequential(
                nn.Conv2d(self.cur_in_ch, out_ch * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )
            layers.append(block(self.cur_in_ch, out_ch, stride, downsample=shortcut))
        else:
            layers.append(block(self.cur_in_ch, out_ch))
        self.cur_in_ch = out_ch * block.expansion

        for _ in range(1, size):
            layers.append(block(self.cur_in_ch, out_ch))
        return nn.Sequential(*layers)


class ResidualEncoder(ResidualBase):
    def __init__(self, opts=None):
        super().__init__()
        if opts is None:
            opts = {"input_channels": 1, "conv_kernel_size": 7,
                    "max_pool": 1, "resnet_size": 18}
        self._opts = opts
        self.cur_in_ch = 64
        self.block_sizes = get_block_sizes(opts["resnet_size"])
        self.block_type = get_block_type(opts["resnet_size"])

        self.conv1 = nn.Conv2d(opts["input_channels"], out_channels=64,
                               kernel_size=opts["conv_kernel_size"], stride=(2, 2),
                               padding=get_padding(opts["conv_kernel_size"]), bias=False)
        self.bn1 = nn.BatchNorm2d(self.cur_in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        if opts["max_pool"] == 1:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=get_padding(3))
            stride1 = (1, 1)
        elif opts["max_pool"] == 0:
            self.max_pool = None
            stride1 = (2, 2)
        elif opts["max_pool"] == 2:
            self.max_pool = None
            stride1 = (1, 1)
        else:
            raise ValueError("Unknown max_pool option")

        self.layer1 = self.make_layer(self.block_type, 64, self.block_sizes[0], stride1)
        self.layer2 = self.make_layer(self.block_type, 128, self.block_sizes[1], (2, 2))
        self.layer3 = self.make_layer(self.block_type, 256, self.block_sizes[2], (2, 2))
        self.layer4 = self.make_layer(self.block_type, 512, self.block_sizes[3], (2, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def model_opts(self):
        return self._opts


class Classifier(nn.Module):
    def __init__(self, opts=None):
        super().__init__()
        if opts is None:
            opts = {"input_channels": 512, "pooling": "avg", "num_classes": 2}
        self._opts = opts
        self._layer_output = dict()
        if opts["pooling"] == "avg":
            self.pooling = lambda x: torch.mean(x, dim=-1)
        elif opts["pooling"] == "max":
            self.pooling = lambda x: torch.max(x, dim=-1)[0]
        else:
            raise ValueError("Unkown pooling option")
        self.linear = nn.Linear(opts["input_channels"], opts["num_classes"])

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        hidden_layer = self.pooling(x)
        hidden_layer = hidden_layer.view(hidden_layer.size(0), -1)
        self._layer_output["hidden_layer_1"] = hidden_layer
        output_layer = self.linear(hidden_layer)
        self._layer_output["output_layer"] = output_layer
        return output_layer

    def model_opts(self):
        return self._opts

def _assemble(encoder, classifier):
    model = nn.Sequential(OrderedDict([('encoder', encoder), ('classifier', classifier)]))
    model.eval()
    return model


def unwrap_model(m):
    """The bare model inside a DataParallel wrapper (train_model returns the wrapper on 2 GPUs)."""
    return m.module if isinstance(m, nn.DataParallel) else m


class SoftmaxEnsemble(nn.Module):
    """Average of the members' softmax outputs, returned as LOG-probabilities: every caller that does
    softmax(model(x)) -- predict_proba, the cascade, the inference cells -- gets the averaged
    probabilities back unchanged, so an ensemble can be used anywhere a single model is."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList([unwrap_model(m) for m in members])

    def forward(self, x):
        p = torch.stack([torch.softmax(m(x).float(), dim=1) for m in self.members]).mean(0)
        return torch.log(p.clamp_min(1e-12))


def model_members(model):
    """The list of single models behind `model` (one element unless it is a SoftmaxEnsemble)."""
    model = unwrap_model(model)
    return list(model.members) if isinstance(model, SoftmaxEnsemble) else [model]


def forward_with_embedding(model, x):
    """(probabilities, embedding) for a batch. The embedding is the 512-d pooled encoder output that
    feeds the linear head (Classifier stores it as 'hidden_layer_1'); for an ensemble the members'
    probabilities are averaged and their embeddings concatenated."""
    probs, embs = [], []
    for m in model_members(model):
        logits = m(x)
        probs.append(torch.softmax(logits.float(), dim=1))
        embs.append(m[1]._layer_output['hidden_layer_1'].float())
    return torch.stack(probs).mean(0), torch.cat(embs, dim=1)


def build_model(encoderOpts, num_classes, device):
    """Build a fresh model (random init) with a num_classes-wide head."""
    classifierOpts = {'input_channels': 512, 'pooling': 'avg', 'num_classes': num_classes}
    encoder = ResidualEncoder(encoderOpts)
    classifier = Classifier(classifierOpts)
    return _assemble(encoder, classifier), encoder, classifier, classifierOpts


def load_model_from_pk(path, device):
    """Load a .pk file. Returns (model, classes_dict, dataOpts)."""
    obj = torch.load(path, map_location=device, weights_only=False)
    encoder = ResidualEncoder(obj['encoderOpts'])
    encoder.load_state_dict(obj['encoderState'])
    classifier = Classifier(obj['classifierOpts'])
    classifier.load_state_dict(obj['classifierState'])
    return _assemble(encoder, classifier), obj.get('classes', {}), obj.get('dataOpts', {})


# .checkpoint files (Data/model_output/*) store no opts, so these are supplied
# from that training run's config. Verify against your own TRAIN.log if in doubt.
CHECKPOINT_ENCODER_OPTS = {'input_channels': 1, 'conv_kernel_size': 7,
                           'max_pool': 2, 'resnet_size': 18}
CHECKPOINT_DATA_OPTS = {'sr': 384000, 'preemphases': 0.98, 'n_fft': 256,
                        'hop_length': 128, 'n_freq_bins': 256, 'fmin': 18000,
                        'fmax': 90000, 'freq_compression': 'linear',
                        'min_level_db': -100, 'ref_level_db': 20}


def load_any_model(path, device):
    """Load either a .pk or a .checkpoint file."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if 'encoderOpts' in obj:
        return load_model_from_pk(path, device)
    if 'modelState' in obj:
        classes = obj.get('classes', {})
        classifierOpts = {'input_channels': 512, 'pooling': 'avg',
                          'num_classes': len(classes)}
        encoder = ResidualEncoder(CHECKPOINT_ENCODER_OPTS)
        classifier = Classifier(classifierOpts)
        ms = obj['modelState']
        enc_state = {k[len('encoder.'):]: v for k, v in ms.items() if k.startswith('encoder.')}
        cls_state = {k[len('classifier.'):]: v for k, v in ms.items() if k.startswith('classifier.')}
        encoder.load_state_dict(enc_state)
        classifier.load_state_dict(cls_state)
        return _assemble(encoder, classifier), classes, dict(CHECKPOINT_DATA_OPTS)
    raise ValueError(f'Unknown model format: {list(obj.keys())}')


print('Architecture + loading functions defined')
