import pandas as pd
from pymonad.list import ListMonad
from pymonad.either import Right
from core.architecture.experiment.TimeSeriesClassifier import TimeSeriesClassifier
from core.architecture.pipelines.abstract_pipeline import AbstractPipelines


class AnomalyDetectionPipelines(AbstractPipelines):

    def __call__(self, pipeline_type: str = 'SpecifiedFeatureGeneratorTSC'):
        pipeline_dict = {'SST': self.__singular_transformation_pipeline,
                         'FunctionalPCA': self.__,
                         'Kalman': self.__kalman_filter_pipeline,
                         'VectorAngle': self.__specified_fg_pipeline,
                         }
        return pipeline_dict[pipeline_type]

    def __singular_transformation_pipeline(self):
        pass

    def __functional_pca_pipeline(self):
        pass

    def __kalman_filter_pipeline(self):
        pass

    def __vector_based_pipeline(self):
        pass
