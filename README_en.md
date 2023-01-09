![](/docs/img/fedot-industrial.png)

[![GitHub issues](https://img.shields.io/github/issues/ITMO-NSS-team/Fedot.Industrial?style=for-the-badge)](https://github.com/ITMO-NSS-team/Fedot.Industrial/issues) 
[![GitHub stars](https://img.shields.io/github/stars/ITMO-NSS-team/Fedot.Industrial?style=for-the-badge)](https://github.com/ITMO-NSS-team/Fedot.Industrial/stargazers) 
![](https://img.shields.io/badge/python-3.8-green?style=for-the-badge&logo=python)
[![GitHub license](https://img.shields.io/github/license/ITMO-NSS-team/Fedot.Industrial?style=for-the-badge)](https://github.com/ITMO-NSS-team/Fedot.Industrial/blob/main/LICENSE.md)
[![lang](https://img.shields.io/badge/lang-ru-yellow.svg?style=for-the-badge)](/README.md)

Instead of using complex and resource-demanding deep learning techniques, which could be considered state-of-the-art
solutions, we propose using a combination of feature extractors with an ensemble of lightweight models obtained by the
algorithmic kernel of the [**AutoML framework FEDOT.**](https://github.com/nccr-itmo/FEDOT)

The application fields of the framework are the following:

- **Classification (time series or image)**

For this purpose we introduce four feature
generators:

![](/docs/img/all-generators.png)

Once the feature generation process is complete FEDOT's evolutionary algorithm is applied 
to find the best model for the classification task.

- **Anomaly detection (time series or image)**

*--work in progress--*

- **Change point detection (only time series)**

*--work in progress--*

- **Object detection (only image)**

*--work in progress--*

## Usage

FEDOT.Industrial provides a high-level API that allows you
to use its capabilities in a simple way.

#### Classification

To conduct time series classification you need to set the experiment configuration via a dictionary, 
then create an instance of the ``Industrial`` class, and call its ``run_experiment`` method:

```python
from core.api.API import Industrial

config = {'feature_generator': ['spectral', 'wavelet'],
          'datasets_list': ['UMD', 'Lightning7'],
          'use_cache': True,
          'error_correction': False,
          'launches': 3,
          'timeout': 15}

ExperimentHelper = Industrial()
ExperimentHelper.run_experiment(config=config)
```

The config contains the following parameters:

- `feature_generator` - list of feature generators to use in the experiment
- `datasets_list` - list of datasets to use in the experiment
- `launches` - number of launches for each dataset
- `error_correction` - flag for applying the error correction model in the experiment
- `n_ecm_cycles` - number of cycles for the error correction model
- `use_cache` - flag for using cache for feature generation
- `timeout` - timeout for classification pipeline composition


Datasets for classification should be stored in the `data` directory and
divided into `train` and `test` sets with the `.tsv` extension. So the folder name
in the `data` directory should be set to the name of dataset that you want
to use in the experiment. In case there is no data in the local folder, the `DataLoader` 
class will try to load data from the [**UCR archive**](https://www.cs.ucr.edu/~eamonn/time_series_data/)
from **http://www.timeseriesclassification.com/dataset.php**.

Possible feature generators which could be specified in configuration are
`window_quantile`, `quantile`, `spectral_window`, `spectral`,
`wavelet`, `recurrence` and `topological`.

It is also possible to ensemble several feature generators.
It could be done by setting the `feature_generator` field of the config, where
you need to specify the list of feature generators, to the following value:

    'ensemble: topological wavelet window_quantile quantile spectral spectral_window'

The experiment results which include generated features, predicted classes, metrics and
pipelines are stored in the `results_of_experiments/{feature_generator name}` directory.
The experiment logs are stored in the `log` directory.

#### Error correction model

It is up to you to decide whether to use the error correction model or not. To apply it, the `error_correction`
flag in the config should be set to `True`. The number of cycles can be adjusted in the `n_ecm_cycles` field 
using the advanced version of the config. In this case after each launch of the FEDOT algorithmic kernel the error 
correction model will be trained on the produced error.

![](/docs/img/error_corr_model.png)

The error correction model is a linear regression model consisting of
three stages: at every next stage the model learns the error of
prediction. This type of ensemble model for error correction is dependent
on a number of classes:
- For `binary classification` the ensemble is also
linear regression, trained on predictions of correction stages.
- For `multiclass classification` the ensemble is a sum of previous predictions.

#### Feature caching 
To speed up the experiment, you can cache the features produced by the feature generators.
If the `use_cache` bool flag in the config is `True`, then every feature space generated during the experiment is
cached into the corresponding folder. To do so a hash from the `get_features` function arguments and the generator attributes
is obtained. Then the resulting feature space is dumped via the `pickle` library.

The next time when the same feature space is requested, the hash is calculated again and the corresponding
feature space is loaded from the cache which is much faster than generating it from scratch.

#### Anomaly detection
*--work in progress--*

#### Change point detection
*--work in progress--*

#### Object detection

*--work in progress--*

## Examples & Tutorials

A comprehensive tutorial will be available soon.
## Publications about FEDOT.Industrial

Our plan for publication activity is to publish papers related to
framework's usability and its applications.

First article: `AUTOMATED MACHINE LEARNING APPROACH FOR TIME SERIES
CLASSIFICATION PIPELINES USING EVOLUTIONARY OPTIMISATION` by Ilya E. Revin,
Vadim A. Potemkin, Nikita R. Balabanov, Nikolay O. Nikitin is under review.

Second article: `AUTOMATED ROCKBURST FORECASTING USING COMPOSITE MODELLING FOR SEISMIC SENSORS DATA`
by Ilya E. Revin, Vadim A. Potemkin, and Nikolay O. Nikitin is under review.

Stay tuned!

## Project structure

The latest stable release of FEDOT.Industrial is in the [**main
branch**](<https://github.com/ITMO-NSS-team/Fedot.Industrial>).

The repository includes the following directories:

- Package `core` contains the main classes and scripts
- Package `cases` includes several how-to-use-cases where you can start to discover how the framework works
- All unit and integration tests are in the `test` directory
- The sources of the documentation are in `docs`

## Current R&D and future plans

- [x] Implement feature space caching for feature generators
- [ ] Development of meta-knowledge storage for data obtained from the experiments
- [ ] Research on time series clusterization

## Documentation

Comprehensive docs are available [**here**](<https://fedotindustrial.readthedocs.io>).

## Supported by

The study is supported by the Research Center
[**Strong Artificial Intelligence in Industry**](<https://sai.itmo.ru/>)
of [**ITMO University**](https://itmo.ru) (Saint Petersburg, Russia)

## Citation

Here will be provided a list of citations for the project.

So far you can use citation for this repository:

    @online{fedot_industrial,
      author = {Revin, Ilya and Potemkin, Vadim and Balabanov, Nikita and Nikitin, Nikolay},
      title = {FEDOT.Industrial - Framework for automated time series analysis},
      year = 2022,
      url = {https://github.com/ITMO-NSS-team/Fedot.Industrial},
      urldate = {2022-05-05}
    }
