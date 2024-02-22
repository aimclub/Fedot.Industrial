from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import optim

from fedot_ind.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot_ind.core.architecture.settings.computational import default_device
from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot_ind.core.models.nn.network_modules.activation import get_activation_fn
from fedot_ind.core.models.nn.network_modules.layers.special import ParameterizedLayer
from fedot_ind.core.models.nn.network_modules.other import *
from fedot_ind.core.repository.constanst_repository import CROSS_ENTROPY, MULTI_CLASS_CROSS_ENTROPY


class OmniScaleCNN(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seq_len,
                 layers=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
                 few_shot=False,
                 activation='ReLU'):

        receptive_field_shape = max(seq_len // 4, 1)
        layer_parameter_list = self.generate_layer_parameter_list(1,
                                                                  receptive_field_shape,
                                                                  layers,
                                                                  input_dim=input_dim)
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        for i in range(len(layer_parameter_list)):
            layer = ParameterizedLayer(layer_parameter_list[i])
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)
        self.gap = GAP1d(1)
        out_put_channel_number = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_number = out_put_channel_number + \
                final_layer_parameters[1]
        self.hidden = nn.Linear(out_put_channel_number, output_dim)
        self.activation = get_activation_fn(activation)

    def get_prime_number_in_a_range(self, start, end):
        Prime_list = []
        for val in range(start, end + 1):
            prime_or_not = True
            for n in range(2, val):
                if (val % n) == 0:
                    prime_or_not = False
                    break
            if prime_or_not:
                Prime_list.append(val)
        return Prime_list

    def get_out_channel_number(self,
                               paramenter_layer,
                               in_channel,
                               prime_list):
        out_channel_expect = max(
            1, int(paramenter_layer / (in_channel * sum(prime_list))))
        return out_channel_expect

    def generate_layer_parameter_list(self,
                                      start,
                                      end,
                                      layers,
                                      input_dim=1):
        prime_list = self.get_prime_number_in_a_range(start, end)

        layer_parameter_list = []
        for parameter_number_of_layer in layers:
            out_channel = self.get_out_channel_number(
                parameter_number_of_layer, input_dim, prime_list)

            tuples_in_layer = []
            for prime in prime_list:
                tuples_in_layer.append((input_dim, out_channel, prime))
            input_dim = len(prime_list) * out_channel

            layer_parameter_list.append(tuples_in_layer)

        tuples_in_layer_last = []
        first_out_channel = len(
            prime_list) * self.get_out_channel_number(layers[0], 1, prime_list)
        tuples_in_layer_last.append((input_dim, first_out_channel, 1))
        tuples_in_layer_last.append((input_dim, first_out_channel, 2))
        layer_parameter_list.append(tuples_in_layer_last)
        return layer_parameter_list

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x)
        if not self.few_shot:
            x = self.activation(self.hidden(x))
        return x


class OmniScaleModel(BaseNeuralModel):
    """Class responsible for OmniScale model implementation.

       Attributes:
           self.num_features: int, the number of features.

       Example:
           To use this operation you can create pipeline as follows::
               from fedot.core.pipelines.pipeline_builder import PipelineBuilder
               from examples.fedot.fedot_ex import init_input_data
               from fedot_ind.tools.loader import DataLoader
               from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels
               train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
               input_data = init_input_data(train_data[0], train_data[1])
               val_data = init_input_data(test_data[0], test_data[1])
               with IndustrialModels():
                   pipeline = PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                    'batch_size': 10}).build()
                   pipeline.fit(input_data)
                   target = pipeline.predict(val_data).predict
                   metric = evaluate_metric(target=test_data[1], prediction=target)

       References:
            @article{tang2020omni,
            title={Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification},
            author={Tang, Wensi and Long, Guodong and Liu, Lu and Zhou, Tianyi and Blumenstein, Michael and Jiang, Jing},
            journal={arXiv preprint arXiv:2002.10061},
            year={2020}
            }
            Original paper: https://arxiv.org/pdf/2002.10061.pdf
       """

    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__(params)
        self.num_classes = params.get('num_classes', 1)

    def __repr__(self):
        return "OmniNN"

    def _init_model(self, ts):
        self.model = OmniScaleCNN(input_dim=ts.features.shape[1],
                                  output_dim=self.num_classes,
                                  seq_len=ts.features.shape[2],
                                  activation=self.activation).to(default_device())
        self.model_for_inference = OmniScaleCNN(input_dim=ts.features.shape[1],
                                                output_dim=self.num_classes,
                                                seq_len=ts.features.shape[2],
                                                activation=self.activation)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        if ts.num_classes == 2:
            loss_fn = CROSS_ENTROPY()
        else:
            loss_fn = MULTI_CLASS_CROSS_ENTROPY()
        return loss_fn, optimizer

    def _fit_model(self, ts: InputData, split_data: bool = False):
        loss_fn, optimizer = self._init_model(ts)
        train_loader, val_loader = self._prepare_data(ts, split_data)

        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

    @convert_to_3d_torch_array
    def _predict_model(self, x_test, output_mode: str = 'default'):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device('cpu'))
        pred = self.model(x_test)
        return self._convert_predict(pred, output_mode)
