import pytest
import torch

from industrial_forecasting.neural.runtime import TensorDevicePolicy, WeightedAverageHead


@pytest.mark.parametrize(('branches', 'target', 'weights'), [
    ([[0.], [2.]], [2.], [0., 1.]),
    ([[1., 0.], [2., 3.]], [1., 0.], [1., 0.]),
])
def test_ensemble_calibrates_along_the_branch_axis(branches, target, weights):
    matrix, expected = torch.tensor(branches), torch.tensor(target)
    head = WeightedAverageHead(TensorDevicePolicy(device='cpu')).fit(matrix, expected)
    torch.testing.assert_close(head.weights_, torch.tensor(weights))
    torch.testing.assert_close(head.predict(matrix), expected)


def test_ensemble_rejects_wrong_axes_and_target_shape():
    matrix = torch.tensor([[2., 1., 3.], [1., 2., 4.]])
    target = matrix[0]
    head = WeightedAverageHead(TensorDevicePolicy(device='cpu'))
    with pytest.raises(ValueError, match='horizon must match'):
        head.fit(matrix.T, target)
    with pytest.raises(ValueError, match='target vector length'):
        head.fit(matrix[:1, :2], torch.eye(2))
    head.fit(matrix, target)
    with pytest.raises(ValueError, match='count must match'):
        head.predict(matrix.T)
