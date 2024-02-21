import numpy as np
from matplotlib import pyplot as plt

from fedot_ind.api.utils.data import init_input_data
from fedot_ind.core.models.recurrence.reccurence_extractor import RecurrenceExtractor
from fedot_ind.tools.loader import DataLoader


def plot_recurrence_matrix(dataset_name: str = 'Herring', save: bool = False, show: bool = True):
    train_data, test_data = DataLoader(dataset_name=dataset_name).load_data()
    input_data = init_input_data(train_data[0], train_data[1])

    strides = [1, 5, 10]
    windows = [5, 10, 30]

    cls_dict = {f'class_{cls}': np.where(input_data.target == cls)[0] for cls in np.unique(input_data.target)}

    for cls in np.unique(input_data.target):
        fig, axs = plt.subplots(len(windows), len(strides), figsize=(20, 20))
        fig.suptitle(f'{dataset_name}, class {cls}', fontsize=40)
        for window_idx, window in enumerate(windows):
            for stride_idx, stride in enumerate(strides):
                params = {'window_size': window,
                          'stride': stride,
                          'image_mode': True}
                recur = RecurrenceExtractor(params)

                random_sample_idx = np.random.choice(cls_dict[f'class_{cls}'].flatten(), 1)[0]
                mtrx, _ = recur.generate(input_data.features[random_sample_idx])
                axs[window_idx, stride_idx].imshow(mtrx)
                axs[window_idx, stride_idx].set_title(f'window: {window}, stride: {stride}', fontsize=30)

    plt.tight_layout()
    if save:
        plt.savefig(f'{dataset_name}_recurrence_matrix.png')
    if show:
        plt.show()
