from core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from typing import Optional, Callable, Any, List, Union
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData
from fedot_ind.core.repository.constanst_repository import CROSS_ENTROPY
import torch.optim as optim
from torch.nn import LSTM, GRU, Linear, Module, RNN
import torch
from fedot_ind.core.models.nn.network_modules.layers.special import RevIN
from fedot_ind.core.models.nn.network_modules.losses import NormalDistributionLoss

class DeepARModule(Module):
    _loss_fns = {
        'normal': NormalDistributionLoss
    }

    def __init__(self, cell_type, input_size, hidden_size, rnn_layers, dropout, distribution):
        super().__init__()
        self.rnn = {'LSTM': LSTM, 'GRU': GRU, 'RNN': RNN}[cell_type](
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = rnn_layers,
        batch_first = True,
        dropout = dropout if rnn_layers > 1 else 0.
        )
        self.scaler = RevIN(
            affine=False,
            input_dim=input_size, 
            dim=-1, # -1 in case series-wise normalization, 0 for batch-wise
        )
        self.distribution = self._loss_fns[distribution]
        if distribution is not None:
            self.projector = Linear(self.hidden_size, len(self.distribution.distribution_arguments))
        else:
            self.projector = Linear(self.hidden_size, 2)
            

    def encode(self, ts: torch.Tensor):
        """
        Encode sequence into hidden state
        ts.size = (length, hidden)
        """
        _, hidden_state = self.rnn(
            ts, enforce_sorted=False
        )  
        # self.distribution_projector = self.distribution_projector()
        return hidden_state
    
    def _decode_whole_seq(self, ts: torch.Tensor, hidden_state: torch.Tensor):
        output, hidden_state = self.rnn(
            ts, hidden_state
            #   enforce_sorted=False
        )
        output = self.projector(output)
        return output, hidden_state
    

    def forward(self, x: torch.Tensor, n_samples: int = None):
        """
        Forward pass
        x.size == (nseries, length)
        """
        x = self.scaler(x, mode=True)
        hidden_state = self.encode(x)
        # decode
        
        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output, hidden_state = self.decode(
            x,
            hidden_state=hidden_state,
        )
        # return relevant part
        return output, hidden_state
    
    def to_quantiles(self, params: torch.Tensor, quantiles=None):
        if quantiles is None:
            quantiles = self.quantiles
        distr = self.distribution.map_x_to_distribution(params)
        return distr.icdf(quantiles)
    
    def to_samples(self, params: torch.Tensor, n_samples=100):
        distr = self.distribution.map_x_to_distribution(params)
        return distr.sample((n_samples,)).T # distr_n x n_samples

    def to_predictions(self, params: torch.Tensor):
        distr = self.distribution.map_x_to_distribution(params)
        return distr.sample((1,)).T.squeeze() # distr_n x 1
    
    def _transform_params(self, distr_params):
        mode = self.forecast_mode
        if mode == 'raw':
            transformed = distr_params
        if mode == 'quantiles':
            transformed = self.to_quantiles(distr_params)
        elif mode == 'predictions':
            transformed = self.to_predictions(distr_params)
        elif mode == 'samples':
            transformed = self.to_samples(distr_params)
        else:
            raise ValueError('Unexpected forecast mode!')
        # transformed = self.scaler
        return transformed
        
    def predict(self, x):
        self.eval()
        distr_params, _ = self(x)
        output = self._transform_params(distr_params)
        return output
    
    def decode(self, x, n_samples=0):
        if not n_samples:
                output, _ = self._decode_whole_seq(x, hidden_state)
                output = self._transform_params(output,)
        else:
                # run in eval, i.e. simulation mode
                # target_pos = self.target_positions
                # lagged_target_positions = self.lagged_target_positions
                # repeat for n_samples
                x = x.repeat_interleave(n_samples, 0)
                hidden_state = self.rnn.repeat_interleave(hidden_state, n_samples)

                # make predictions which are fed into next step
                output = self.decode_autoregressive(
                    first_target=x[:, 0],
                    first_hidden_state=hidden_state,
                    # target_scale=target_scale,
                    n_decoder_steps=x.size(1),
                    n_samples=n_samples,
                )
                # reshape predictions for n_samples:
                # from n_samples * batch_size x time steps to batch_size x time steps x n_samples
                output = (output, lambda x:: x.reshape(-1, n_samples, input_vector.size(1)).permute(0, 2, 1))
        return output
    

    def _decode_one(self, x,
                idx,
                # lagged_targets,
                hidden_state,
                ):
        x = x[:, [idx], ...]
                # x[:, 0, target_pos] = lagged_targets[-1]
                # for lag, lag_positions in lagged_target_positions.items():
                #     if idx > lag:
                #         x[:, 0, lag_positions] = lagged_targets[-lag]
        prediction, hidden_state = self._decode_whole_seq(x, hidden_state)
        prediction = prediction[:, 0]  # select first time step fo this index
        return prediction, hidden_state

    def decode_autoregressive(
        self,
        hidden_state: Any,
        # target_scale: Union[List[torch.Tensor], torch.Tensor],
        first_target: Union[List[torch.Tensor], torch.Tensor],
        n_decoder_steps: int,
        n_samples: int = 1,
        **kwargs,
    ) -> Union[List[torch.Tensor], torch.Tensor]:

        # make predictions which are fed into next step
        output = []
        current_target = first_target
        current_hidden_state = hidden_state

        normalized_output = [first_target]

        for idx in range(n_decoder_steps):
            # get lagged targets
            current_target, current_hidden_state = self._decode_one(
                idx, 
                # lagged_targets=normalized_output, 
                hidden_state=current_hidden_state, **kwargs
            )

            # get prediction and its normalized version for the next step
            prediction, current_target = self.output_to_prediction(
                current_target, 
                # target_scale=target_scale,
                n_samples=n_samples
            )
            # save normalized output for lagged targets
            normalized_output.append(current_target)
            # set output to unnormalized samples, append each target as n_batch_samples x n_random_samples

            output.append(prediction)
        output = torch.stack(output, dim=1)
        return output

    
            





    

class DeepAR(BaseNeuralModel):
    """No exogenous variable support
    Variational Inference + Probable Anomaly detection"""


    def __init__(self, params: Optional[OperationParameters] = {}):
        super().__init__()
        self.num_classes = 2 #params.get('num_classes', 2)

        #INSIDE
        # self.epochs = params.get('epochs', 100)
        # self.batch_size = params.get('batch_size', 16)
        # self.activation = params.get('activation', 'ReLU')
        # self.learning_rate = 0.001

        # self.label_encoder = None
        # self.model = None
        # self.model_for_inference = None
        # self.target = None
        # self.task_type = None
        
        self.cell_type = params.get('cell_type', 'LSTM')
        self.hidden_size = params.get('hidden_size', 10)
        self.rnn_layers = params.get('rnn_layers', 2)
        self.dropout = params.get('dropout', 0.1)
        self.horizon = params.get('forecast_length', None)
        self.expected_distribution = params.get('expected_distribution', 'normal')

        ###
        self.preprocess_to_lagged = False
        self.forecast_mode = params.get('forecast_mode', 'raw')
        self.quantiles = params.get('quantiles', None)

    


    def _init_model(self, ts) -> tuple:
        self.loss_fn = DeepARModule._loss_fns[self.expected_distribution]()
        self.model = DeepARModule(input_size=ts.features.shape[1],
                                   hidden_size=self.hidden_size,
                                   cell_type=self.cell_type,
                                   dropout=self.dropout,
                                   rnn_layer=self.rnn_layers,
                                   distribution_params_n=len(self.loss_fn.distribution_arguments)).to(default_device())
        self.model_for_inference = DeepARModule(input_size=ts.features.shape[1],
                                   hidden_size=self.hidden_size,
                                   cell_type=self.cell_type,
                                   dropout=self.dropout,
                                   rnn_layer=self.rnn_layers,
                                   distribution_params_n=len(self.loss_fn.distribution_arguments))
        self._evaluate_num_of_epochs(ts)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return self.loss_fn, self.optimizer
    
    