import pathlib

import fedot.core.data.data_split as fedot_data_split
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.data.merge.data_merger import ImageDataMerger, TSDataMerger
from fedot.core.operations.evaluation.operation_implementations.data_operations.topological.fast_topological_extractor \
    import TopologicalFeaturesImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    LaggedImplementation, TsSmoothingImplementation
from fedot.core.operations.operation import Operation
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.pipelines.verification import class_rules
from fedot.core.repository.operation_types_repository import OperationTypesRepository

from fedot_ind.api.utils.path_lib import PROJECT_PATH
from fedot_ind.core.repository.industrial_implementations.abstract import preprocess_industrial_predicts, \
    transform_lagged_for_fit_industrial, transform_smoothing_industrial, transform_lagged_industrial, \
    merge_industrial_predicts, merge_industrial_targets, build_industrial, postprocess_industrial_predicts, \
    split_any_industrial, split_time_series_industrial, predict_operation_industrial, predict_industrial, \
    predict_for_fit_industrial, update_column_types_industrial, _check_and_correct_window_size_industrial, \
    fit_topo_extractor_industrial, transform_topo_extractor_industrial
from fedot_ind.core.repository.industrial_implementations.optimisation import _get_default_industrial_mutations
from fedot_ind.core.repository.industrial_implementations.optimisation import \
    has_no_data_flow_conflicts_in_industrial_pipeline
from fedot_ind.core.tuning.search_space import get_industrial_search_space

FEDOT_METHOD_TO_REPLACE = [(PipelineSearchSpace, "get_parameters_dict"),
                           (ApiParamsRepository, "_get_default_mutations"),
                           (ImageDataMerger, "preprocess_predicts"),
                           (ImageDataMerger, "merge_predicts"),
                           (TSDataMerger, "merge_predicts"),
                           (TSDataMerger, "merge_targets"),
                           (TSDataMerger, 'postprocess_predicts'),
                           (DataSourceSplitter, "build"),
                           (fedot_data_split, "_split_any"),
                           (fedot_data_split, "_split_time_series"),
                           (Operation, "_predict"),
                           (Operation, "predict"),
                           (Operation, "predict_for_fit"),
                           (LaggedImplementation, '_update_column_types'),
                           (LaggedImplementation, 'transform'),
                           (TopologicalFeaturesImplementation, 'fit'),
                           (TopologicalFeaturesImplementation, 'transform'),
                           (LaggedImplementation, 'transform_for_fit'),
                           (LaggedImplementation, '_check_and_correct_window_size'),
                           (TsSmoothingImplementation, 'transform')]
INDUSTRIAL_REPLACE_METHODS = [get_industrial_search_space,
                              _get_default_industrial_mutations,
                              preprocess_industrial_predicts,
                              merge_industrial_predicts,
                              merge_industrial_predicts,
                              merge_industrial_targets,
                              postprocess_industrial_predicts,
                              build_industrial,
                              split_any_industrial,
                              split_time_series_industrial,
                              predict_operation_industrial,
                              predict_industrial,
                              predict_for_fit_industrial,
                              update_column_types_industrial,
                              transform_lagged_industrial,
                              fit_topo_extractor_industrial,
                              transform_topo_extractor_industrial,
                              transform_lagged_for_fit_industrial,
                              _check_and_correct_window_size_industrial,
                              transform_smoothing_industrial]
DEFAULT_METHODS = [getattr(class_impl[0], class_impl[1])
                   for class_impl in FEDOT_METHOD_TO_REPLACE]


class IndustrialModels:
    def __init__(self):

        self.industrial_data_operation_path = pathlib.Path(
            PROJECT_PATH,
            'fedot_ind',
            'core',
            'repository',
            'data',
            'industrial_data_operation_repository.json')

        self.base_data_operation_path = pathlib.Path(
            'data_operation_repository.json')


        self.industrial_model_path = pathlib.Path(
            PROJECT_PATH,
            'fedot_ind',
            'core',
            'repository',
            'data',
            'industrial_model_repository.json')

        self.base_model_path = pathlib.Path('model_repository.json')

    def _replace_operation(self, to_industrial=True):
        if to_industrial:
            method = INDUSTRIAL_REPLACE_METHODS
        else:
            method = DEFAULT_METHODS
        for class_impl, method_to_replace in zip(FEDOT_METHOD_TO_REPLACE, method):
            setattr(class_impl[0], class_impl[1], method_to_replace)

    def setup_repository(self):
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)
        # replace mutations
        self._replace_operation(to_industrial=True)

        class_rules.append(has_no_data_flow_conflicts_in_industrial_pipeline)
        return OperationTypesRepository

    def setup_default_repository(self):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.base_data_operation_path,
                                'initialized_repo': None,
                                'default_tags': [
                                    OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo(
            'data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
        self._replace_operation(to_industrial=False)
        return OperationTypesRepository

    def __enter__(self):
        """
        Switching to industrial models
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.industrial_data_operation_path,
                                'initialized_repo': True,
                                'default_tags': []}})

        OperationTypesRepository.assign_repo(
            'data_operation', self.industrial_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.industrial_model_path,
                       'initialized_repo': True,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo(
            'model', self.industrial_model_path)

        setattr(PipelineSearchSpace, "get_parameters_dict",
                get_industrial_search_space)
        setattr(ApiComposer, "_get_default_mutations",
                _get_default_industrial_mutations)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Switching to fedot models.
        """
        OperationTypesRepository.__repository_dict__.update(
            {'data_operation': {'file': self.base_data_operation_path,
                                'initialized_repo': None,
                                'default_tags': [
                                    OperationTypesRepository.DEFAULT_DATA_OPERATION_TAGS]}})
        OperationTypesRepository.assign_repo(
            'data_operation', self.base_data_operation_path)

        OperationTypesRepository.__repository_dict__.update(
            {'model': {'file': self.base_model_path,
                       'initialized_repo': None,
                       'default_tags': []}})
        OperationTypesRepository.assign_repo('model', self.base_model_path)
