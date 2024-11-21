import sys
import os
from functools import wraps
import lightning as pl
import torch
from loguru import logger
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from rich.progress import Progress, SpinnerColumn, TextColumn


def setup_logger(log_file):
    logger.remove()
    logger.add(
        sys.stderr,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
    )
    logger.add(log_file, rotation='10 MB')


def task_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f'Starting {func_name}')
        try:
            result = func(*args, **kwargs)
            logger.info(f'Finished {func_name}')
            return result
        except Exception as e:
            logger.exception(f'Error in {func_name}: {str(e)}')
            raise

    return wrapper


def get_rich_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        transient=True,
    )


def plot_confusion_matrix(
    model: pl.LightningModule, datamodule: pl.LightningDataModule, path: str = '.'
):
    model.eval()
    os.makedirs(path, exist_ok=True)

    def fn(mode: str, loader: torch.utils.data.DataLoader):
        y_pred = []
        y_true = []
        for batch in loader():
            x, y = batch
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            preds = F.softmax(logits, dim=-1)
            # preds,true comes in batch(32)
            preds = torch.argmax(preds, dim=-1)
            for i, j in zip(preds, y):
                # print(y.shape,preds.shape,type(y),type(preds))
                y_true.append(j.item())
                y_pred.append(i.item())

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=loader().dataset.classes
        )
        disp.plot(xticks_rotation='vertical', colorbar=False).figure_.savefig(
            os.path.join(path, f'{mode}_confusion_matrix.png')
        )  # f'{path}{mode}_confusion_matrix.png')

    for mode, loader in zip(
        ['train', 'test', 'val'],
        [
            datamodule.train_dataloader,
            datamodule.test_dataloader,
            datamodule.val_dataloader,
        ],
    ):
        fn(mode=mode, loader=loader)
