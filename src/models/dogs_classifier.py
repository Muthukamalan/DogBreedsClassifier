import lightning as pl
import timm
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy
from typing import Dict
from typing import List


class DogsBreedClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        dims: List,
        depths: List,
        head_fn: str,
        conv_ratio: float,
        num_classes: int,
        pretrained: bool,
        trainable: bool,
        lr: float,
        weight_decay: float,
        scheduler_factor: float,
        scheduler_patience: int,
        min_lr: float,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        print(type(self.hparams.dims), self.hparams)
        # self.model:timm.models.resnest.ResNet = timm.create_model(model_name=self.hparams.model_name,pretrained=self.hparams.pretrained,num_classes=self.hparams.num_classes,global_pool = 'avg')
        self.model: timm.models.mambaout.MambaOut = timm.models.mambaout.MambaOut(
            in_chans=self.hparams.in_chans,
            num_classes=self.hparams.num_classes,
            depths=list(self.hparams.depths),
            dims=list(self.hparams.dims),
            head_fn=self.hparams.head_fn,
            conv_ratio=self.hparams.conv_ratio,
            act_layer=torch.nn.ReLU,
        )

        # for p in self.model.parameters():
        #     p.requires_grad=self.hparams.trainable

        self.train_acc: Accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.test_acc: Accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.valid_acc: Accuracy = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=-1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log(
            "train/acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=True
        )
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=-1)
        self.valid_acc(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("val/acc", self.valid_acc, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=-1)
        self.test_acc(preds, y)
        self.log("test/loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("test/acc", self.test_acc, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #                 optimizer=optimizer,
        #                 factor=self.hparams.scheduler_factor,
        #                 patience=self.hparams.scheduler_patience,
        #                 min_lr=self.hparams.min_lr
        # )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams.lr * 10,
            three_phase=True,
            final_div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            # "monitor":"train/loss_epoch"
        }
