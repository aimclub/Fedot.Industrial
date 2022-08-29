![](doc/fedot-industrial.png)

Instead of using complex and resource-demanding deep learning techniques, which could be considered state-of-the-art
solutions, we propose using a combination of feature extractors with an ensemble of lightweight models obtained by the
algorithmic kernel of [**AutoML framework FEDOT.**](https://github.com/nccr-itmo/FEDOT)

Application field of the framework is the following:

### Classification (time series or image)

For this purpose we introduce four feature
generators:

![](doc/all-generators.png)

After feature generation process apply evolutionary
algorithm of FEDOT to find the best model for classification task.

### Anomaly detection (time series or image)

*--work in progress--*

### Change point detection (only time series)

*--work in progress--*

### Object detection (only image)

*--work in progress--*

## Usage

FEDOT.Industrial provides a high-level API that allows you
to use its capabilities in a simple way.

#### Classification

To conduct time series classification you need to first
set experiment configuration using file `cases/config/Config_Classification.yaml` and then run the following command:

    from cases.API import Industrial

    config_name = 'Config_Classification.yaml'
    ExperimentHelper = Industrial()
    ExperimentHelper.run_experiment(config_name)

Possible feature generators which could be specified in configuration are
`window_quantile`, `quantile`, `spectral_window`, `spectral`,
`wavelet` and `topological`.

There is also a possibility to ensemble several
feature generators. It could be done by the following instruction in `feature_generator`
field of `Config_Classification.yaml` file:

    'ensemble: topological wavelet window_quantile quantile spectral spectral_window'

Results of experiment are stored in `results_of_experiments/{feature_generator name}` directory.
Logs of experiment are stored in `log` directory.

#### Anomaly detection

*--work in progress--*

#### Change point detection

*--work in progress--*

#### Object detection

## Examples & Tutorials

## Publications about FEDOT.Industrial

Our plan for publication activity is to publish papers related to
framework's usability and its applications.

First article `AUTOMATED MACHINE LEARNING APPROACH FOR TIME SERIES
CLASSIFICATION PIPELINES USING EVOLUTIONARY OPTIMISATION` by Ilya E. Revin,
Vadim A. Potemkin, Nikita R. Balabanov, Nikolay O. Nikitin is under review.

Stay tuned!

## Project structure

## Current R&D and future plans

## Documentation

*--work in progress--*

## Supported by

- `<https://sai.itmo.ru/>`

## Citation


