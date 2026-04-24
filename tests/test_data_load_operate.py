import numpy as np

from utils import data_load_operate as dlo


def test_sampling_uses_available_samples_for_small_classes():
    # class 1 has only 2 samples; class 2 has 5 samples
    gt_reshape = np.array([1, 1, 2, 2, 2, 2, 2])

    train_idx, val_idx, test_idx, _ = dlo.sampling(
        ratio_list=[0.1, 0.1],
        num_list=[4, 1],
        gt_reshape=gt_reshape,
        class_count=2,
        Flag=1,
    )

    train_labels = gt_reshape[train_idx]
    assert np.sum(train_labels == 1) == 2
    assert np.sum(train_labels == 2) == 4

    # Ensure splits are disjoint.
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


def test_backward_compatible_aliases_exist():
    assert dlo.HSI_create_pathes is dlo.HSI_create_patches
    assert dlo.generate_auxilary_iter is dlo.generate_auxiliary_iter
