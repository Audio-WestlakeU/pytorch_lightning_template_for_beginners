import sys
from typing import Set

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar
from torch import Tensor


class MyRichProgressBar(RichProgressBar):
    """A progress bar prints metrics at the end of each epoch
    """

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        sys.stdout.flush()
        if trainer.is_global_zero:
            metrics = trainer.logged_metrics
            infos = f"Epoch {trainer.current_epoch} metrics: "
            for k, v in metrics.items():
                if not k.startswith('val/'):
                    continue
                value = v
                if isinstance(v, Tensor):
                    value = v.item()
                if isinstance(value, float):
                    infos += k + f"={value:.4f}  "
                else:
                    infos += k + f"={value}  "
            if len(metrics) > 0:
                sys.stdout.write(f'{infos}\n')
            sys.stdout.flush()
