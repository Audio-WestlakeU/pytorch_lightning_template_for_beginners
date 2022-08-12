import os

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # 允许使用BF16精度进行训练。enable bf16 precision in pytorch 1.12, see https://github.com/Lightning-AI/lightning/issues/11933#issuecomment-1181590004
os.environ["OMP_NUM_THREADS"] = str(8)  # 限制进程数量，放在import torch和numpy之前。不加会导致程序占用特别多的CPU资源，使得服务器变卡。
# limit the threads to reduce cpu overloads, will speed up when there are lots of CPU cores on the running machine

from typing import Tuple

import torch

# torch 1.12开始，TF32默认关闭，下面的参数会打开TF32。对于A100，使用TF32会使得速度得到很大的提升，同时不影响训练结果【或轻微影响】。
torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
# torch.set_float32_matmul_precision('medium')

from jsonargparse import lazy_instance
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import (LightningArgumentParser, LightningCLI)
from torch.utils.data import DataLoader, Dataset

from utils import MyRichProgressBar as RichProgressBar
from utils import MyLogger as TensorBoardLogger
from utils import tag_and_log_git_status


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
        self.batch_size = batch_size  # train: batch_size[0]; test: batch_size[1]

    def prepare_data(self) -> None:
        return super().prepare_data()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(RandomDataset(self.input_size, 640), batch_size=self.batch_size[0], num_workers=self.num_workers, shuffle=True)

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
    """LightningModule控制了模型训练的各个方面，即定义了如何使用一个mini-batch的数据进行训练、验证、测试、推理，以及训练使用的optimizer。
    
    LightningModule内部具有特定名字的函数（如下面涉及的on_train_start、training_step等）会被lightning框架在函数名称对应的阶段自动调用。
    
    LightningModule的相关资料：https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    def __init__(self, arch: MyArch = lazy_instance(MyArch), exp_name: str = "exp"):
        super().__init__()
        self.arch = arch

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['arch'])

    def forward(self, x):
        return self.arch(x)

    def on_train_start(self):
        if self.current_epoch == 0:
            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in self.hparams.exp_name:
                # 在当前训练的程序代码树上添加Git标签，使得代码版本可以与训练version对应起来。【注意先commit内容修改，然后再训练。测试的时候exp_name设置为notag】
                # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version, self.hparams.exp_name, model_name=type(self).__name__)

            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir'):
                # 输出模型到 model.txt
                with open(self.logger.log_dir + '/model.txt', 'a') as f:
                    f.write(str(self))
                    f.write('\n\n\n')

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
        self.log("val/loss", loss, sync_dist=True)  # 设置sync_dist=True，使得val/loss在多卡训练的时候能够同步，用于选择最佳的checkpoint等任务。train/loss不需要设置这个，因为训练步需要同步的是梯度，而不是指标，梯度会自动同步

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

        # 设置profiler寻找代码最耗时的位置。去除下面的注释把profiler打开
        # from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
        # parser.set_defaults({"trainer.profiler": lazy_instance(AdvancedProfiler, filename="profiler")})
        # parser.set_defaults({"trainer.max_epochs": 1, "trainer.limit_train_batches": 100, "trainer.limit_val_batches": 100})

    def before_fit(self):
        # 训练开始前，会被执行。下面代码的功能是如果是从last.ckpt恢复训练，则输出到同一个目录（如version_10）;否则，输出到logs/{model_name}/version_NEW
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
            model_name = type(self.model).__name__
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
