import os
import shutil

from fedot.api.main import Fedot
from joblib import dump, load

from core.operation.utils.LoggerSingleton import Logger
from core.operation.utils.utils import PROJECT_PATH

DOCKER_FILE = '''
FROM python:3.8

RUN pip install joblib
RUN pip install fedot

COPY fedot_pipeline.joblib /fedot_pipeline.joblib
COPY feature_generator.joblib /feature_generator.joblib

COPY predictor.py /predictor.py

'''


class DockerMaker:
    """Class responsible for building docker image with Fedot model and fedture generator.

    Args:
        docker_name: name of the docker image

    Attributes:
        logger (Logger): logger instance

    """
    folder_path = os.path.join(PROJECT_PATH, 'dockerfiles')

    def __init__(self, docker_name):

        self.docker_name = docker_name
        self.logger = Logger().get_logger()
        os.makedirs(self.folder_path, exist_ok=True)

    def make_docker(self, fedot_pipeline, feature_generator):
        self.logger.info('Start docker making')
        if not isinstance(fedot_pipeline, Fedot):
            raise ValueError('No fedot model is provided')

        self.dump_models(fedot_pipeline, feature_generator)
        self.create_docker_file()
        self.copy_predictor()
        try:
            os.system(f'docker build -t dockerfiles {self.folder_path}')
            return True
        except Exception as ex:
            self.logger.error(f'Failed to build docker: {ex}')
            return False

    def dump_models(self, fedot_pipeline, feature_generator: object):
        self.logger.info('Dumping FEDOT model')
        dump(fedot_pipeline, os.path.join(self.folder_path, 'fedot_pipeline.joblib'))

        self.logger.info(f'Dumping {feature_generator.__class__.__name__} feature generator')
        dump(feature_generator, os.path.join(self.folder_path, 'feature_generator.joblib'))

    def create_docker_file(self):
        self.logger.info('Start docker file creation')
        with open(os.path.join(self.folder_path, 'Dockerfile'), 'w') as f:
            f.write(DOCKER_FILE)

    def load_models(self):
        self.logger.info('Loading FEDOT model')
        fedot_model = load(os.path.join(self.folder_path, 'fedot_model.joblib'))

        self.logger.info('Start feature generator loading')
        feature_generator = load(os.path.join(self.folder_path, 'feature_generator.joblib'))

        return fedot_model, feature_generator

    def copy_predictor(self):
        scr = os.path.join(PROJECT_PATH, 'dockerizer', 'predictor.py')
        dst = os.path.join(self.folder_path, 'predictor.py')

        shutil.copy(scr, dst)
