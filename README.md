# TODO
正在等待pytorch-lightning的CLI功能稳定，稳定后将出一个统一的模板。

内容将包括：Dataset、DataLoader、训练、验证、测试等基于pytorch-lightning框架的最佳（近似）实践。

涉及的领域将包括实验室的全部领域：语音分离、语音增强、TTS、定位、说话人分离、去混响等

# 安装

**pytorch** 参考[pytorch](https://pytorch.org/get-started/locally/)官网。注意cuda版本：

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

**pytorch_lightning**：pytorch项目的**最佳**实践
```
pip install pytorch-lightning
```

**torch_metrics**：用于计算各种指标，包括: sisdr, sisnr, snr, nb-pesq, wb-pesq, stoi, 以及其他深度学习领域的指标
```
pip uninstall torchmetrics -y
pip install git+https://github.com/quancs/metrics.git@personal
```

# 训练
```
python train.py --model xxx --other_args
```

# 测试
```
python test.py --model xxx --other_args
```
