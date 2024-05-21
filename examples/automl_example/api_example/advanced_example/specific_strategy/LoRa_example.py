from fedot_ind.api.main import FedotIndustrial
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
                  timeout=15,
                  with_tuning=False,
                  industrial_strategy='lora_strategy',
                  industrial_strategy_params=lora_params,
                  logging_level=20)

industrial = FedotIndustrial(**api_config)
industrial.fit(train_data)
predict = industrial.predict(test_data)
_ = 1
