"""Test the L1 absolute loss metric."""
from jaxtyping import Float
import pytest
from torch import Tensor, allclose, ones, tensor, zeros

from sparse_autoencoder.metrics.loss.log1p_loss import Log1PLoss
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.parametrize(
    # Each source/decoded tensor is of the form (batch_size, num_components, num_features)
    ("learned_activations", "expected_loss"),
    [
        pytest.param(
            zeros(2, 3),
            tensor(0.0),
            id="All zero activations -> zero loss (single component)",
        ),
        pytest.param(
            zeros(2, 2, 3),
            tensor([0.0, 0.0]),
            id="All zero activations -> zero loss (2 components)",
        ),
        pytest.param(
            ones(2, 3),  # 3 features -> 2.0794 loss
            tensor(8.0).log(),
            id="All 1.0 activations -> 2.0794 loss (single component)",
        ),
        pytest.param(
            ones(2, 2, 3),
            tensor([8.0, 8.0]).log(),
            id="All 1.0 activations -> 2.0794 loss (2 components)",
        ),
        pytest.param(
            ones(2, 2, 3) * -1,  # Loss is absolute so the same as +ve 1s
            tensor([8.0, 8.0]).log(),
            id="All -ve 1.0 activations -> 2.0794 loss (2 components)",
        ),
    ],
)
def test_log1p_loss(
    learned_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
    ],
    expected_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL],
) -> None:
    """Test the log1p loss."""
    num_components: int = learned_activations.shape[1] if learned_activations.ndim == 3 else 1  # noqa: PLR2004
    loss = Log1PLoss(num_components)

    res = loss.forward(learned_activations=learned_activations)

    assert allclose(res, expected_loss)
