import os

from core.architecture.experiment.nn_experimenter import ClassificationExperimenter, \
    FasterRCNNExperimenter
from core.operation.optimization.structure_optimization import SVDOptimization, \
    SFPOptimization
from tests.unit.architecture.experiment.test_nn_experimenter import \
    classification_predict, detection_predict

SVD_PARAMS = {'energy_thresholds': [0.9]}
SFP_PARAMS = {'pruning_ratio': 0.5}


def test_sfp_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SFP_P-0.50/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SFP_P-0.50/')
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(models.joinpath('pruned.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    assert os.path.exists(summary.joinpath('size.csv'))
    assert os.path.exists(summary.joinpath('pruned/train.csv'))
    assert os.path.exists(summary.joinpath('pruned/val.csv'))
    classification_predict(experimenter)


def test_svd_channel_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SVD_channel_O-100.0_H-0.001000/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SVD_channel_O-100.0_H-0.001000/')
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(models.joinpath('trained.model.pt'))
    assert os.path.exists(models.joinpath('e_0.9.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    assert os.path.exists(summary.joinpath('size.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/train.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/val.csv'))
    classification_predict(experimenter)


def test_svd_spatial_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='spatial', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SVD_spatial_O-100.0_H-0.001000/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SVD_spatial_O-100.0_H-0.001000/')
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(models.joinpath('trained.model.pt'))
    assert os.path.exists(models.joinpath('e_0.9.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    assert os.path.exists(summary.joinpath('size.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/train.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/val.csv'))
    classification_predict(experimenter)


def test_sfp_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/ALET10/FasterRCNN_SFP_P-0.50/')
    summary = tmp_path.joinpath('summary/ALET10/FasterRCNN_SFP_P-0.50/')
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    detection_predict(experimenter)


def test_svd_channel_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/ALET10/FasterRCNN_SVD_channel_O-100.0_H-0.001000/')
    summary = tmp_path.joinpath('summary/ALET10/FasterRCNN_SVD_channel_O-100.0_H-0.001000/')
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(models.joinpath('trained.model.pt'))
    assert os.path.exists(models.joinpath('e_0.9.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    assert os.path.exists(summary.joinpath('size.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/train.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/val.csv'))
    detection_predict(experimenter)


def test_svd_spatial_fasterrcnn_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = FasterRCNNExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='spatial', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/ALET10/FasterRCNN_SVD_spatial_O-100.0_H-0.001000/')
    summary = tmp_path.joinpath('summary/ALET10/FasterRCNN_SVD_spatial_O-100.0_H-0.001000/')
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(models.joinpath('trained.model.pt'))
    assert os.path.exists(models.joinpath('e_0.9.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    assert os.path.exists(summary.joinpath('size.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/train.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/val.csv'))
    detection_predict(experimenter)
