"""UNet for segmentation."""

from typing import Any

from torchmetrics import MetricCollection
from torchmetrics.segmentation import GeneralizedDiceScore

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.lightning.dataset_ng.lightning_modules.unet_module import UnetModule

from .configuration import SegAlgorithm
from .loss import get_loss


# TODO:
# - AdamW
# - CosineAnnealingLR
# - More augmentations
# - Add tests

class SegModule(UnetModule):

    def __init__(self, algorithm_config: SegAlgorithm) -> None:
        super().__init__(algorithm_config)

        self.loss_func = get_loss(algorithm_config.loss)

        # override metrics
        self.metrics = MetricCollection(
            GeneralizedDiceScore(num_classes=self.config.model.num_classes)
        )

    def training_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
    ) -> Any:

        x, target = batch

        prediction = self.model(x.data)
        loss = self.loss_func(prediction, target.data)

        self._log_training_stats(loss, batch_size=x.data.shape[0])

        return loss

    def validation_step(
        self,
        batch: tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
    ) -> None:

        x, target = batch[0], batch[1]

        prediction = self.model(x.data)
        val_loss = self.loss_func(prediction, target.data)
        
        ###### Log validation images
        self.logger.experiment.add_images(
            "val/images",
            x.data,
            self.current_epoch,
    
        )


        # ##### Metrics
        # convert predictions to class indices for metrics
        # for binary (1 channel): apply sigmoid and threshold
        # for multi-class (>1 channels): apply argmax
        if prediction.shape[1] == 1:
            pred_classes = (prediction.sigmoid() > 0.5).long()
        else:
            pred_classes = prediction.argmax(dim=1, keepdim=True)

        # # ensure targets are long type for metrics
        # target_long = target.data.long()
        # self.metrics(pred_classes.squeeze(), target_long.squeeze())
        self._log_validation_stats(val_loss, batch_size=x.data.shape[0])
    
    def predict_step(
        self,
        batch: tuple[ImageRegionData] | tuple[ImageRegionData, ImageRegionData],
        batch_idx: Any,
        load_best_checkpoint=False,
    ) -> Any:

        x = batch[0]
        # TODO: add TTA
        prediction = self.model(x.data)
        
        # apply appropriate activation function based on number of classes
        # for binary (1 channel): apply sigmoid to get probabilities
        # for multi-class (>1 channels): apply softmax to get class probabilities
        if prediction.shape[1] == 1:
            prediction = prediction.sigmoid()
        else:
            prediction = prediction.softmax(dim=1)
        
        prediction = prediction.cpu().numpy()

        output_batch = ImageRegionData(
            data=prediction,
            source=x.source,
            data_shape=x.data_shape,
            dtype=x.dtype,
            axes=x.axes,
            region_spec=x.region_spec,
            chunks=x.chunks,
        )
        return output_batch