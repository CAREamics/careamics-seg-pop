from typing import Annotated, Literal

from pydantic import BaseModel, AfterValidator

from careamics.config.data import NGDataConfig
from careamics.config import TrainingConfig
from careamics.config.algorithms.unet_algorithm_model import UNetBasedAlgorithm
from careamics.config.architectures import UNetModel
from careamics.config.validators import (
    model_without_final_activation,
    model_without_n2v2,
)


# TODO UNetBasedAlgorithm is too restrictive, should accept str as algorithm and loss
class SegAlgorithm(UNetBasedAlgorithm):
    """Configuration for segmentation algorithm."""

    algorithm: Literal["seg"] = "seg"
    """Segmentation Algorithm name."""

    loss: Literal["dice", "ce", "dicece"] = "dice"
    """Segmentation-compatible loss function."""

    model:  Annotated[
        UNetModel,
        AfterValidator(model_without_n2v2),
        AfterValidator(model_without_final_activation),
    ]
    """UNet without a final activation function and without the `n2v2` modifications."""


class SegConfig(BaseModel):
    """Configuration for segmentation tasks."""
    algorithm_config: SegAlgorithm
    data_config: NGDataConfig
    training_config: TrainingConfig
