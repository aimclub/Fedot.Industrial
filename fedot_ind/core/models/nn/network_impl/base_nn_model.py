from typing import Optional
import numpy as np
import torch
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup

from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from torch import nn, Tensor, optim
import torch.nn.functional as F
from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array, \
    convert_to_torch_tensor, \
    fedot_data_type, convert_inputdata_to_torch_dataset
from fedot_ind.core.architecture.settings.computational import default_device


class BaseNeuralModel:
    """Class responsible for InceptionTime model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot_ind.tools.loader import DataLoader
            from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('minirocket_features').add_node(
                    'rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = {}):
        self.num_classes = params.get('num_classes', 1)
        self.epochs = params.get('epochs', 10)
        self.batch_size = params.get('batch_size', 20)

    @convert_inputdata_to_torch_dataset
    def _create_dataset(self, ts: InputData):
        return ts

    def _init_model(self, ts):
        self.model = None
        return

    def _convert_predict(self, pred):
        pred = F.softmax(pred, dim=1)
        if self.num_classes == 2:
            pred = torch.argmax(pred, dim=1)
        y_pred = pred.cpu().detach().numpy()
        predict = OutputData(
            idx=np.arange(len(y_pred)),
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)
        return predict

    def _prepare_data(self, ts, split_data: bool = True):
        if split_data:
            train_data, val_data = train_test_data_setup(ts, shuffle_flag=True, split_ratio=0.7)
        train_dataset = self._create_dataset(train_data)
        val_dataset = self._create_dataset(val_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader

    def _train_loop(self, train_loader, val_loader, loss_fn, optimizer):
        for epoch in range(1, self.epochs + 1):
            training_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                output = self.model(inputs)
                loss = loss_fn(output, targets.float())
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)

            self.model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                output = self.model(inputs)
                loss = loss_fn(output, targets.float())
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                                   torch.max(targets, dim=1)[1])

                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)

            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch,
                                                                                                        training_loss,
                                                                                                        valid_loss,
                                                                                                        num_correct / num_examples))

    @convert_to_3d_torch_array
    def _fit_model(self, ts: InputData, split_data: bool = True):
        self._train_loop(*self._prepare_data(ts, split_data), *self._init_model(ts))

    @convert_to_3d_torch_array
    def _predict_model(self, x_test):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device())
        pred = self.model(x_test)
        return self._convert_predict(pred)

    def fit(self,
            input_data: InputData):
        """
        Method for feature generation for all series
        """
        self.num_classes = input_data.num_classes
        self.target = input_data.target
        self.task_type = input_data.task
        self._fit_model(input_data)

    @fedot_data_type
    def predict(self,
                input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data)

    @fedot_data_type
    def predict_for_fit(self,
                        input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        return self._predict_model(input_data)
