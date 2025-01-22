import torchvision.datasets as datasets
import torchvision.transforms as transforms
from fedot_ind.core.architecture.pipelines.abstract_pipeline import ApiTemplate
from fedot_ind.core.repository.config_repository import DEFAULT_COMPUTE_CONFIG, \
    DEFAULT_AUTOML_LEARNING_CONFIG


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST train and test dataset
train_data = (datasets.MNIST(
    root="./examples/data",
    train=True,
    download=True,
    transform=transform), 'torchvision_dataset')

test_data = (datasets.MNIST(
    root="./examples/data",
    train=False,
    download=True,
    transform=transform), 'torchvision_dataset')

metric_names = ('f1', 'accuracy', 'precision', 'roc_auc')

lora_params = dict(rank=2,
                   sampling_share=0.5,
                   lora_init='random',
                   epochs=1,
                   batch_size=10
                   )

api_config = dict(problem='classification',
                  metric='accuracy',
                  timeout=0.1,
                  with_tuning=False,
                  industrial_strategy='lora_strategy',
                  industrial_strategy_params=lora_params,
                  logging_level=20)

AUTOML_LEARNING_STRATEGY = DEFAULT_AUTOML_LEARNING_CONFIG
COMPUTE_CONFIG = DEFAULT_COMPUTE_CONFIG
AUTOML_CONFIG = {'task': 'classification',
                 'use_automl': True,
                 'optimisation_strategy': {'optimisation_strategy': {'mutation_agent': 'bandit',
                                                                     'mutation_strategy': 'growth_mutation_strategy'},
                                           'optimisation_agent': 'Industrial'}}

LEARNING_CONFIG = {'learning_strategy': 'from_scratch',
                   'learning_strategy_params': AUTOML_LEARNING_STRATEGY,
                   'optimisation_loss': {'quality_loss': 'accuracy'}}

INDUSTRIAL_PARAMS = {'rank': 2,
                     'sampling_share': 0.5,
                     'lora_init': 'random',
                     'epochs': 1,
                     'batch_size': 10,
                     'data_type': 'tensor'
                     }

INDUSTRIAL_CONFIG = {'problem': 'classification',
                     'strategy': 'lora_strategy',
                     'strategy_params': INDUSTRIAL_PARAMS
                     }

API_CONFIG = {'industrial_config': INDUSTRIAL_CONFIG,
              'automl_config': AUTOML_CONFIG,
              'learning_config': LEARNING_CONFIG,
              'compute_config': COMPUTE_CONFIG}

dataset = dict(test_data=test_data, train_data=train_data)

industrial = ApiTemplate(api_config=API_CONFIG,
                         metric_list=metric_names).eval(dataset=dataset)
industrial.fit(train_data)
predict = industrial.predict(test_data)
_ = 1
