from fedot_ind.core.models.cnn.classification_models import *


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