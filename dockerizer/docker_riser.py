import docker
from core.operation.utils.utils import PROJECT_PATH
from core.operation.utils.LoggerSingleton import Logger
import os
from joblib import load, dump


class DockerMaker:
    folder_path = os.path.join(PROJECT_PATH, 'dockerizer', 'dockerfiles')

    def __init__(self, fedot_pipeline,
                 docker_name,
                 feature_generator,
                 ecm_models=None):

        if fedot_pipeline is None:
            raise ValueError('No fedot model provided')

        self.fedot_model = fedot_pipeline
        self.feature_generator = feature_generator
        self.ecm_models = ecm_models
        self.docker_name = docker_name
        self.logger = Logger().get_logger()

    def make_docker(self):
        self.logger.info('Start docker making')
        os.makedirs(self.folder_path, exist_ok=True)

        self.dump_models()
        # client = docker.from_env()
        # client.images.build(path=self.folder_path, tag=self.docker_name)
        # self.logger.info('Docker image was built')

        return self.docker_name

    def dump_models(self):
        self.logger.info('Dumping FEDOT model')
        dump(self.fedot_model, os.path.join(self.folder_path, 'fedot_model.joblib'))

        # if self.ecm_models is not None:
        #     self.logger.info('Dumping ecm models')
        #     dump(self.ecm_models, os.path.join(self.folder_path, 'ecm_models.joblib'))

        self.logger.info('Start feature generator dumping')
        dump(self.feature_generator, os.path.join(self.folder_path, 'feature_generator.joblib'))

        self.logger.info('Models were dumped')

    def load_models(self):
        self.logger.info('Loading FEDOT model')
        fedot_model = load(os.path.join(self.folder_path, 'fedot_model.joblib'))

        self.logger.info('Start feature generator loading')
        feature_generator = load(os.path.join(self.folder_path, 'feature_generator.joblib'))

        self.logger.info('Models were loaded')

        return fedot_model, feature_generator
