import os
from typing import Tuple

os.environ["OMP_NUM_THREADS"] = str(8)  # limit the threads to reduce cpu usage, will speed up when there are lots of CPU cores on the running machine

import torch
from jsonargparse import lazy_instance
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import (LightningArgumentParser, LightningCLI)
from torch.utils.data import DataLoader, Dataset

from utils import MyRichProgressBar as RichProgressBar
from utils import MyLogger as TensorBoardLogger


class RandomDataset(Dataset):
    """一个随机数组成的数据集，此处用于展示其他模块的功能
    """

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class MyDataModule(LightningDataModule):
    """定义了如何生成训练、验证、测试、以及推理时的DataLoader。此处使用RandomDataset来生成数据。
    LightningDataModule相关资料：https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningDataModule.html
    """

    def __init__(self, input_size: int = 10, num_workers: int = 5, batch_size: Tuple[int, int] = (2, 4)):
        super().__init__()
        self.input_size = input_size
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(self.input_size, 640), batch_size=self.batch_size[0], num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(self.input_size, 640), batch_size=self.batch_size[1], num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(self.input_size, 640), batch_size=1, num_workers=self.num_workers)


class MyArch(torch.nn.Module):
    """这个类定义了网络结构，此处是Linear。也可以将网络结构写到LightningModule里面。
    """

    def __init__(self, input_size: int = 10, output_size: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear.forward(x)


class MyModel(LightningModule):
    """LightningModule控制了模型训练的各个方面，即定义了如何使用一个mini-batch的数据进行训练、验证、测试、推理，以及训练使用的optimizer
    
    LightningModule的相关资料：https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    def __init__(self, arch: MyArch = lazy_instance(MyArch)):
        super().__init__()
        self.arch = arch

    def forward(self, x):
        return self.arch(x)

    def training_step(self, batch, batch_idx):
        """如何使用一个mini-batch的数据得到train/loss。其他step同理。

        Args:
            batch: train DataLoader给出一个mini-batch的数据
        """
        loss = self.forward(batch).sum()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch).sum()
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch).sum()
        self.log("test/loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.arch.parameters(), lr=0.1)


class MyCLI(LightningCLI):
    """命令行接口（Command Line Interface）：命令行参数分析、Trainer的构建、Trainer命令（训练、测试、推理）的执行等等
    Trainer的相关资料: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    CLI的相关资料：https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """添加、设置默认的参数，需要使用的callback（此处callback的意思是在LightningModule各个step函数开始执行、结束执行等时机想要被调用的具有特殊功能的函数，这些函数被封装到了不同的callbacks类里面，如EarlyStopping、ModelCheckpoint）。  
        callback的相关资料: https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html
        """
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

        parser.set_defaults({"trainer.strategy": "ddp_find_unused_parameters_false"})

        # 添加早停策略默认参数
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults({
            "early_stopping.monitor": "val/loss",
            "early_stopping.min_delta": 0.01,
            "early_stopping.patience": 10,
            "early_stopping.mode": "min",
        })

        # 添加模型保存默认参数
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_valid_loss{val/loss:.4f}",
            "model_checkpoint.monitor": "val/loss",
            "model_checkpoint.mode": "min",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 5,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # RichProgressBar
        parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        parser.set_defaults({
            "progress_bar.console_kwargs": {
                "force_terminal": True,
                "no_color": True,  # 去除颜色，节省nohup保存下来的log文件的大小
                "width": 200,  # 设置足够的宽度，防止将进度条分成两行
            }
        })

    def before_fit(self):
        # 训练开始前，会被执行。下面代码的功能是如果是从last.ckpt恢复训练，则输出到同一个目录（version_x）;否则，输出到logs/{model_name}
        resume_from_checkpoint: str = self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # 如果是从last.ckpt恢复训练，则输出到同一个目录
            # resume_from_checkpoint example: /home/zhangsan/logs/MyModel/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = str(self.model_class).split('\'')[1].split('.')[-1]
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def before_test(self):
        # 测试开始前，会被执行。下面代码实现的功能是将测试时的log目录设置为/home/zhangsan/logs/MyModel/version_X/epochN/version_Y
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch)

        import time
        # add 10 seconds for threads to simultaneously detect the next version
        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)
        time.sleep(10)

    def after_test(self):
        # 测试结束之后，会被执行。下面代码实现的功能是将测试时生成的tensorboard log文件删除，防止tensorboard看着混乱。
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = MyCLI(MyModel, MyDataModule, seed_everything_default=None, save_config_overwrite=True, parser_kwargs={"parser_mode": "omegaconf"})
