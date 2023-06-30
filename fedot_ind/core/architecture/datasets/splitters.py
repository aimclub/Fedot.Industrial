"""
This module contains functions for splitting a torch dataset into parts.
"""
from typing import List, Tuple, Generator, Optional, Dict

import numpy as np
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


def train_test_split(dataset: Dataset, p: float = 0.2) -> Tuple[Subset, Subset]:
    """
    Splits the data into two parts, keeping the proportions of the classes.

    Args:
        dataset: Torch dataset object.
        p: Proportion of the test sample, must be from 0 to 1.

    Returns:
        A tuple ``(train_ds, test_ds)``, where train_ds and test_ds
            are subsets.
    """
    n = int(1./p)
    return next(k_fold(dataset=dataset, n=n))


def k_fold(dataset: Dataset, n: int) -> Generator[Tuple[Subset, Subset], None, None]:
    """
    K-Folds cross-validator.

    Args:
        dataset: Torch dataset object.
        n: Number of folds.

    Yields:
        A tuple ``(train_ds, test_ds)``, where train_ds and test_ds
            are subsets.
    """
    fold_indices = split_data(dataset, n)
    for i in range(n):
        test_indices = fold_indices[i]
        test_ds = Subset(dataset, test_indices)
        train_indices = np.concatenate([fold_indices[j] for j in range(n) if j != i])
        train_ds = Subset(dataset, train_indices)
        yield train_ds, test_ds

def split_data(dataset: Dataset, n: int, verbose: bool = False) -> List[np.ndarray]:
    """
    Splits the data into n parts, keeping the proportions of the classes.

    Args:
        dataset: Torch dataset object.
        n: Number of parts.
        verbose: If `True` prints information about splitting.

    Returns:
        A list of indices for each part.
    """
    classes_of_imgs = _extract_classes(dataset)
    classes = np.unique(classes_of_imgs)
    fold_indices = [[] for _ in range(n)]
    for cl in classes:
        indices_of_class = np.random.permutation(np.nonzero(classes_of_imgs==cl)[0])
        if verbose:
            print(f"Class {cl} contains {indices_of_class.size} samples.")
        lengths = [int(indices_of_class.size / n) for _ in range(n)]
        remainder = indices_of_class.size - sum(lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            lengths[i % n] += 1
        start = 0
        if verbose:
            print(' '.join(map(str, lengths)))
        for i, length in enumerate(lengths):
            fold_indices[i].append(indices_of_class[start : start + length])
            start += length
    return [np.concatenate(fold) for fold in fold_indices]


def undersampling(dataset: Dataset, n: Optional[int] = None, verbose: bool = False) -> Subset:
    """
    A method for balancing uneven datasets by keeping all data in the
    minority class and decreasing the size of the majority class.

    Args:
        dataset: Torch dataset object.
        n: Number of samples in each class.
        verbose: If `True` prints information about undersampling.

    Returns:
        Balanced subset.
    """
    classes_of_imgs = _extract_classes(dataset)
    classes = np.unique(classes_of_imgs)
    indices_of_classes = []
    min_size = len(classes_of_imgs)
    for cl in classes:
        indices_of_class = np.random.permutation(np.nonzero(classes_of_imgs==cl)[0])
        indices_of_classes.append(indices_of_class)
        if verbose:
            print(f"Class {cl} contains {indices_of_class.size} samples.")
        if indices_of_class.size < min_size:
            min_size = indices_of_class.size
    min_size = min_size if n is None else n
    print(f"New size of any class {min_size} samples.")
    indices = np.concatenate([x[:min_size] for x in indices_of_classes])
    return Subset(dataset, indices)


def _extract_classes(dataset: Dataset) -> np.ndarray:
    """
    Returns the class for each sample.

    Args:
        dataset: Torch dataset object.
    """
    classes_of_imgs = []
    for i in tqdm(range(len(dataset)), desc='prepare dataset'):
        img, target = dataset.__getitem__(i)
        classes_of_imgs.append(target)
    return np.array(classes_of_imgs)


def dataset_info(dataset: Dataset, verbose: bool = False) -> Dict[int, int]:
    """
    Returns number of samples in each class

    Args:
        dataset: Torch dataset object.
        verbose: If `True` prints information about classes.

    Returns:
        Dictionary `{class_id: number_of_samples}`.
    """
    classes_of_imgs = _extract_classes(dataset)
    classes = np.unique(classes_of_imgs)
    class_samples = {}
    for cl in classes:
        indices_of_class = np.nonzero(classes_of_imgs==cl)[0]
        class_samples[cl] = indices_of_class.size
        if verbose:
            print(f"Class {cl} contains {indices_of_class.size} samples.")
    return class_samples


def get_dataset_mean_std(dataset: Dataset) -> Tuple[Tuple, Tuple]:
    """
    Compute mean and std of dataset.

    Args:
        dataset: Torch dataset object.

    Returns:
          Tuple(mean, std)
    """
    shape = dataset.__getitem__(0)[0].shape
    one_channel = len(shape) == 2

    if one_channel:
        psum = np.zeros(1)
        psum_sq = np.zeros(1)
        pixels = np.zeros(1)
        for i in tqdm(range(len(dataset)), desc='computing mean and std'):
            img, target = dataset.__getitem__(i)
            psum += img.sum().numpy()
            psum_sq += (img ** 2).sum().numpy()
            pixels += img.numel()
    else:
        psum = np.zeros(shape[0])
        psum_sq = np.zeros(shape[0])
        pixels = np.zeros(1)
        for i in tqdm(range(len(dataset)), desc='computing mean and std'):
            img, target = dataset.__getitem__(i)
            psum += img.sum(dim=[1, 2]).numpy()
            psum_sq += (img ** 2).sum(dim=[1, 2]).numpy()
            c, w, h = img.size()
            pixels += w * h
    mean = psum / pixels
    var = (psum_sq / pixels) - (mean ** 2)
    std = np.sqrt(var)
    return tuple(mean), tuple(std)
