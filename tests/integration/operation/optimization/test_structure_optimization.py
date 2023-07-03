import os

from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, \
    ObjectDetectionExperimenter
from fedot_ind.core.operation.optimization.structure_optimization import SVDOptimization, \
    SFPOptimization
from tests.integration.experiment.test_nn_experimenter import \
    classification_predict, detection_predict, prepare_detection, prepare_classification

SVD_PARAMS = {'energy_thresholds': [0.9]}
SFP_PERCENTAGE_PARAMS = {'zeroing_mode': 'percentage', 'zeroing_mode_params': {'pruning_ratio': 0.5}}
SFP_ENERGY_PARAMS = {'zeroing_mode': 'energy', 'zeroing_mode_params': {'energy_threshold': 0.9}}


def check_sfp_paths(models, summary, classification=True):
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))

    if classification:
        assert os.path.exists(models.joinpath('pruned.sd.pt'))
        assert os.path.exists(summary.joinpath('size.csv'))
        assert os.path.exists(summary.joinpath('pruned/train.csv'))
        assert os.path.exists(summary.joinpath('pruned/val.csv'))


def check_svd_paths(models, summary, classification=True):
    assert os.path.exists(models.joinpath('train.sd.pt'))
    assert os.path.exists(models.joinpath('trained.model.pt'))
    assert os.path.exists(models.joinpath('e_0.9.sd.pt'))
    assert os.path.exists(summary.joinpath('train/train.csv'))
    assert os.path.exists(summary.joinpath('train/val.csv'))
    assert os.path.exists(summary.joinpath('size.csv'))
    assert os.path.exists(summary.joinpath('pruning.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/train.csv'))
    assert os.path.exists(summary.joinpath('e_0.9/val.csv'))

def test_sfp_percentage_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_PERCENTAGE_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SFP_pruning_ratio-0.5/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SFP_pruning_ratio-0.5/')
    check_sfp_paths(models, summary)
    classification_predict(experimenter)


def test_sfp_energy_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_ENERGY_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SFP_energy_threshold-0.9/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SFP_energy_threshold-0.9/')
    check_sfp_paths(models, summary)
    classification_predict(experimenter)


def test_svd_channel_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SVD_channel_O-10_H-0.1/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SVD_channel_O-10_H-0.1/')
    check_svd_paths(models, summary)
    classification_predict(experimenter)


def test_svd_channel_classification_experimenter_two_layers(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', forward_mode='two_layers', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SVD_channel_O-10_H-0.1/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SVD_channel_O-10_H-0.1/')
    check_svd_paths(models, summary)
    classification_predict(experimenter)


def test_svd_channel_classification_experimenter_three_layers(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', forward_mode='three_layers', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SVD_channel_O-10_H-0.1/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SVD_channel_O-10_H-0.1/')
    check_svd_paths(models, summary)
    classification_predict(experimenter)


def test_svd_spatial_classification_experimenter(prepare_classification):
    exp_params, fit_params, tmp_path = prepare_classification
    experimenter = ClassificationExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='spatial', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/Agricultural/ResNet_SVD_spatial_O-10_H-0.1/')
    summary = tmp_path.joinpath('summary/Agricultural/ResNet_SVD_spatial_O-10_H-0.1/')
    check_svd_paths(models, summary)
    classification_predict(experimenter)


def test_sfp_percentage_objectdetection_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = ObjectDetectionExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_PERCENTAGE_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/ALET10/SSD_SFP_pruning_ratio-0.5/')
    summary = tmp_path.joinpath('summary/ALET10/SSD_SFP_pruning_ratio-0.5/')
    check_sfp_paths(models, summary, classification=False)
    detection_predict(experimenter)


def test_sfp_enegry_objectdetection_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = ObjectDetectionExperimenter(**exp_params)
    optimization = SFPOptimization(**SFP_ENERGY_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params)
    models = tmp_path.joinpath('models/ALET10/SSD_SFP_energy_threshold-0.9/')
    summary = tmp_path.joinpath('summary/ALET10/SSD_SFP_energy_threshold-0.9/')
    check_sfp_paths(models, summary, classification=False)
    detection_predict(experimenter)


def test_svd_channel_objectdetection_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = ObjectDetectionExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='channel', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/ALET10/SSD_SVD_channel_O-10_H-0.1/')
    summary = tmp_path.joinpath('summary/ALET10/SSD_SVD_channel_O-10_H-0.1/')
    check_svd_paths(models, summary)
    detection_predict(experimenter)


def test_svd_spatial_objectdetection_experimenter(prepare_detection):
    exp_params, fit_params, tmp_path = prepare_detection
    experimenter = ObjectDetectionExperimenter(**exp_params)
    optimization = SVDOptimization(decomposing_mode='spatial', **SVD_PARAMS)
    optimization.fit(exp=experimenter, params=fit_params, ft_params=fit_params)
    models = tmp_path.joinpath('models/ALET10/SSD_SVD_spatial_O-10_H-0.1/')
    summary = tmp_path.joinpath('summary/ALET10/SSD_SVD_spatial_O-10_H-0.1/')
    check_svd_paths(models, summary)
    detection_predict(experimenter)
