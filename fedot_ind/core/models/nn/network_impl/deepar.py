from fedot_ind.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from typing import Optional, Callable, Any, List, Union
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.data.data import InputData, OutputData
from fedot_ind.core.repository.constanst_repository import CROSS_ENTROPY
import torch.optim as optim
from torch.nn import LSTM, GRU, Linear, Module, RNN
import torch
from fedot_ind.core.models.nn.network_modules.layers.special import RevIN
from fedot_ind.core.models.nn.network_modules.losses import NormalDistributionLoss
from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.architecture.abstraction.decorators import convert_inputdata_to_torch_time_series_dataset
from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector
import pandas as pd
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot_ind.core.models.nn.network_modules.layers.special import adjust_learning_rate, EarlyStopping
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
import torch.utils.data as data
from fedot_ind.core.architecture.settings.computational import default_device
import torch.optim.lr_scheduler as lr_scheduler 
from fedot.core.data.data_split import train_test_data_setup



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
        self.hidden_size = hidden_size
        self.scaler = RevIN(
            affine=False,
            input_dim=input_size, 
            dim=-1, # -1 in case series-wise normalization, 0 for batch-wise, RNN needs series_wise
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
            ts,
        )  
        return hidden_state
    
    def _decode_whole_seq(self, ts: torch.Tensor, hidden_state: torch.Tensor):
        output, hidden_state = self.rnn(
            ts, hidden_state
        )
        output = self.projector(output)
        return output, hidden_state
    

    def forward(self, x: torch.Tensor, n_samples: int = None, mode='raw'):
        """
        Forward pass
        x.size == (nseries, length)
        """
        x = self.scaler(x, mode=True)
        hidden_state = self.encode(x)
        # decode
        
        if self.training:
            assert n_samples is None, "cannot sample from decoder when training"
        output = self.decode(
            x,
            hidden_state, n_samples=0, mode=mode
        )        
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
    
    def _transform_params(self, distr_params, mode='raw'):
        if mode == 'raw':
            transformed = distr_params
        elif mode == 'quantiles':
            transformed = self.to_quantiles(distr_params)
        elif mode == 'predictions':
            transformed = self.to_predictions(distr_params)
        elif mode == 'samples':
            transformed = self.to_samples(distr_params)
        else:
            raise ValueError('Unexpected forecast mode!')
        transformed = self.scaler(transformed, False)
        return transformed
        
    def predict(self, test_x: torch.Tensor, mode=None):
        self.eval()
        distr_params, _ = self(test_x)
        return self._transform_params(distr_params, mode)

    def decode(self, x, hidden_state=None, n_samples=0, mode='raw'):
        if hidden_state is None:
            hidden_state = torch.zeros((self.hidden_size,)).float()
        if not n_samples:
                output, _ = self._decode_whole_seq(x, hidden_state)
                output = self._transform_params(output, mode=mode)
        else:
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

        return output
    

    def _decode_one(self, x,
                idx,
                hidden_state,
                ):
        x = x[:, [idx], ...]
        prediction, hidden_state = self._decode_whole_seq(x, hidden_state)
        prediction = prediction[:, 0]  # select first time step fo this index
        return prediction, hidden_state

    def decode_autoregressive(
        self,
        hidden_state: Any,
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

        #INSIDE
        # self.epochs = params.get('epochs', 100)
        # self.batch_size = params.get('batch_size', 16)
        # self.activation = params.get('activation', 'ReLU')
        # self.learning_rate = 0.001
        
        self.cell_type = params.get('cell_type', 'LSTM')
        self.hidden_size = params.get('hidden_size', 10)
        self.rnn_layers = params.get('rnn_layers', 2)
        self.dropout = params.get('dropout', 0.1)
        self.horizon = params.get('horizon', 1)
        self.forecast_length = self.horizon
        self.expected_distribution = params.get('expected_distribution', 'normal')

        ###
        self.preprocess_to_lagged = False
        self.patch_len = params.get('patch_len', None)
        self.forecast_mode = params.get('forecast_mode', 'raw')
        self.quantiles = params.get('quantiles', None)

    
    def _init_model(self, ts) -> tuple:
        self.loss_fn = DeepARModule._loss_fns[self.expected_distribution]()
        input_size = self.patch_len or ts.features.shape[-1]
        self.model = DeepARModule(input_size=input_size,
                                   hidden_size=self.hidden_size,
                                   cell_type=self.cell_type,
                                   dropout=self.dropout,
                                   rnn_layers=self.rnn_layers,
                                   distribution=self.expected_distribution).to(default_device())
        self.model_for_inference = DeepARModule(input_size=input_size,
                                   hidden_size=self.hidden_size,
                                   cell_type=self.cell_type,
                                   dropout=self.dropout,
                                   rnn_layers=self.rnn_layers,
                                   distribution=self.expected_distribution)
        self._evaluate_num_of_epochs(ts)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        return self.loss_fn, self.optimizer

    def fit(self, input_data: InputData):
        self._fit_model(input_data)
    
    def _fit_model(self, input_data: InputData, split_data: bool = False):
        val_loader = None
        if self.preprocess_to_lagged:
            self.patch_len = input_data.features.shape[-1]
            train_loader = self.__create_torch_loader(input_data)
        else:
            if self.patch_len is None:
                dominant_window_size = WindowSizeSelector(
                    method='dff').get_window_size(input_data.features)
                self.patch_len = 2 * dominant_window_size
            train_loader, val_loader = self._prepare_data(
                    input_data.features, self.patch_len, split_data)

        self.test_patch_len = self.patch_len
        loss_fn, optimizer = self._init_model(input_data)
        return self._train_loop(model=self.model, 
                        train_loader=train_loader,
                        loss_fn=loss_fn, 
                        optimizer=optimizer, 
                        val_loader=val_loader,
                        )

    def _predict(self, test_loader, output_mode):
        model = self.model # or model for inference? 
        output = model.predict(test_loader, output_mode)

        y_pred = output #
        forecast_idx_predict = np.arange(start=test_data.idx[-self.horizon],
                                         stop=test_data.idx[-self.horizon] +
                                         self.horizon,
                                         step=1)
        predict = OutputData(
            idx=forecast_idx_predict,
            task=self.task_type,
            predict=y_pred.reshape(1, -1),
            target=self.target,
            data_type=DataTypesEnum.table)
        return predict

    def predict_for_fit(self,
                        test_data,
                        output_mode: str='samples'):
        y_pred = []
        true_mode = self.forecast_mode

        self.forecast_mode = output_mode
        model = self.model
        ##########

        y_pred = np.array(y_pred)
        y_pred = y_pred.squeeze()
        forecast_idx_predict = test_data.idx
        predict = OutputData(
            idx=forecast_idx_predict,
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)
        self.forecast_mode = true_mode
        return predict

    def _train_loop(self, model,
                    train_loader,
                    loss_fn,
                    val_loader,
                    optimizer):
        train_steps = len(train_loader)
        early_stopping = EarlyStopping()
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            epochs=self.epochs,
                                            max_lr=self.learning_rate)
        kwargs = {'lradj': 'type3'}

        best_model = None
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            iter_count = 0
            train_loss = []
            model.train()
            training_loss = 0.0
            valid_loss = 0.0

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()
                batch_x = batch_x.float().to(default_device())
                batch_y = batch_y.float().to(default_device())
                outputs, *hidden_state = model(batch_x)
                # return batch_x, outputs, batch_y

                loss = loss_fn(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                optimizer.step()

                # adjust_learning_rate(optimizer, scheduler,
                #                      epoch, learning_rate=, printout=False, **kwargs)
                scheduler.step()
            if val_loader is not None and epoch % val_interval == 0:
                model.eval()
                total = 0
                for batch in val_loader:
                    inputs, targets = batch
                    output = model(inputs)

                    loss = loss_fn(output, targets.float())

                    valid_loss += loss.data.item() * inputs.size(0)
                    total += inputs.size(0)
                valid_loss /= total
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_model = copy.deepcopy(model)

            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Updating learning rate to {}'.format(
                scheduler.get_last_lr()[0]))
        return best_model

    def _predict_loop(self, test_loader):
        outputs = []
        with torch.no_grad():
            for x_test in test_loader:
                outputs.append(self.model.predict(x_test))
        output = torch.stack(outputs, dim=0)
        return output


    @convert_inputdata_to_torch_time_series_dataset
    def _create_dataset(self,
                        ts: InputData,
                        flag: str = 'test',
                        batch_size: int = 16,
                        freq: int = 1):
        return ts
    
    def _prepare_data(self,
                      ts,
                      patch_len=None,
                      split_data: bool = True,
                      validation_blocks: int = None, 
                      unsqueeze_0=True):
        if patch_len is None:
            patch_len = self.horizon
        train_data = self.__ts_to_input_data(ts)
        if split_data:
            raise NotImplementedError('Problem with lagged_data splitting')
            train_data, val_data = train_test_data_setup(
                train_data, validation_blocks=validation_blocks)
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(train_data,
                                                                                                  self.horizon,
                                                                                                  patch_len)
            _, val_data.features, val_data.target = transform_features_and_target_into_lagged(val_data,
                                                                                              self.horizon,
                                                                                              patch_len)
            val_loader = self.__create_torch_loader(val_data)
            train_loader = self.__create_torch_loader(train_data)
            return train_loader, val_loader
        else:
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(train_data,
                                                                                                  self.horizon,
                                                                                                  patch_len)
        train_loader = self.__create_torch_loader(train_data)
        return train_loader, None


    def __ts_to_input_data(self, input_data: Union[InputData, pd.DataFrame]):
        if isinstance(input_data, InputData):
            return input_data
        
        if not isinstance(input_data, pd.DataFrame):
            time_series = pd.DataFrame(input_data)
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.horizon))
        if 'datetime' in time_series.columns:
            idx = pd.to_datetime(time_series['datetime'].values)
        else:
            idx = np.arange(len(time_series.values.flatten()))

        time_series = time_series.values

        train_input = InputData(idx=idx,
                                features=time_series.flatten(),
                                target=time_series.flatten(),
                                task=task,
                                data_type=DataTypesEnum.ts)
        
        return train_input

    def __create_torch_loader(self, train_data):
        train_dataset = self._create_dataset(train_data)
        train_loader = torch.utils.data.DataLoader(
            data.TensorDataset(train_dataset.x, train_dataset.y),
            batch_size=self.batch_size, shuffle=False)
        return train_loader
    
    