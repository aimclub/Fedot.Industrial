"""
This module contains functions for splitting a torch dataset into parts.
"""
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


def k_fold(dataset: Dataset, n: int) -> Tuple[Subset]:
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

def split_data(dataset: Dataset, n: int) -> List[np.ndarray]:
    """
    Splits the data into n parts, keeping the proportions of the classes.

    Args:
        dataset: Torch dataset object.
        n: Number of parts.

    Returns:
        A list of indices for each part.
    """
    classes_of_imgs = []
    for i in range(len(dataset)):
        img, target = dataset.__getitem__(i)
        classes_of_imgs.append(target)
    classes_of_imgs = np.array(classes_of_imgs)
    classes = np.unique(classes_of_imgs)

    fold_indices = [[] for _ in range(n)]
    for cl in classes:
        indices_of_class = np.random.permutation(np.nonzero(classes_of_imgs==cl)[0])
        print(f"Class {cl} contains {indices_of_class.size} samples.")
        lengths = [int(indices_of_class.size / n) for _ in range(n)]
        remainder = indices_of_class.size - sum(lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            lengths[i % n] += 1
        start = 0
        print(' '.join(map(str, lengths)))
        for i, length in enumerate(lengths):
            fold_indices[i].append(indices_of_class[start : start + length])
            start += length
    return [np.concatenate(fold) for fold in fold_indices]
