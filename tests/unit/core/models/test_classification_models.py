from fedot_ind.core.models.nn.network_impl.resnet import resnet18_one_channel, ResNet, resnet34_one_channel, \
    resnet50_one_channel, resnet101_one_channel, resnet152_one_channel, CLF_MODELS_ONE_CHANNEL, CLF_MODELS


def test_resnet18_one_channel():
    model = resnet18_one_channel()
    assert isinstance(model, ResNet)


def test_resnet34_one_channel():
    model = resnet34_one_channel()
    assert isinstance(model, ResNet)


def test_resnet50_one_channel():
    model = resnet50_one_channel()
    assert isinstance(model, ResNet)


def test_resnet101_one_channel():
    model = resnet101_one_channel()
    assert isinstance(model, ResNet)


def test_resnet152_one_channel():
    model = resnet152_one_channel()
    assert isinstance(model, ResNet)


def test_CLF_MODELS():
    models = CLF_MODELS
    assert isinstance(models, dict)


def test_CLF_MODELS_ONE_CHANNEL():
    models = CLF_MODELS_ONE_CHANNEL
    assert isinstance(models, dict)