"""Test the L1 absolute loss metric."""
from jaxtyping import Float
import pytest
from torch import Tensor, allclose, ones, tensor, zeros

from sparse_autoencoder.metrics.loss.prod1p_loss import Prod1PLoss
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.parametrize(
    # Each source/decoded tensor is of the form (batch_size, num_components, num_features)
    ("learned_activations", "expected_loss"),
    [
        pytest.param(
            zeros(2, 3),
            tensor(1.0),
            id="All zero activations -> 1.0 loss (single component)",
        ),
        pytest.param(
            zeros(2, 2, 3),
            tensor([1.0, 1.0]),
            id="All zero activations -> 1.0 loss (2 components)",
        ),
        pytest.param(
            ones(2, 3),  # 3 features -> 8.0 loss
            tensor(8.0),
            id="All 1.0 activations -> 8.0 loss (single component)",
        ),
        pytest.param(
            ones(2, 2, 3),
            tensor([8.0, 8.0]),
            id="All 1.0 activations -> 8.0 loss (2 components)",
        ),
        pytest.param(
            ones(2, 2, 3) * -1,  # Loss is absolute so the same as +ve 1s
            tensor([8.0, 8.0]),
            id="All -ve 1.0 activations -> 8.0 loss (2 components)",
        ),
    ],
)
def test_prod1p_loss(
    learned_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
    ],
    expected_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL],
) -> None:
    """Test the prod1p loss."""
    num_components: int = learned_activations.shape[1] if learned_activations.ndim == 3 else 1  # noqa: PLR2004
    loss = Prod1PLoss(num_components)

    res = loss.forward(learned_activations=learned_activations)

    assert allclose(res, expected_loss)
