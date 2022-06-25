# 简介
一套适合初学者的pytorch lightning代码框架，能够满足DL时需要用到的很多内容，如实验版本管理、配置文件管理、命令行解析、多卡训练、EarlyStopping策略、累计梯度、梯度clipping等等。

# 安装
```
pip install -r requirements.txt
```

# 生成配置文件
```
python boring.py fit --print_config=skip_null > boring.yaml
```

# 训练
```
python boring.py fit --config boring.yaml --data.batch_size=[24,48]
```

## GPU训练
多卡采用DDP模式训练时，需要保持batch_size一致。下面的例子的训练时的batch size = 4 * 2 * 3，其中4是train dataloader的batch size，2是累计的梯度数目，3是gpu的数目。
```
python boring.py fit --config boring.yaml --data.batch_size=[4,8] --trainer.accumulate_grad_batches=2 --trainer.gpus=0,1,3
```


## 恢复训练
恢复训练时，使用对应version的配置文件，以及对应的checkpoint
```
python boring.py fit --config logs/MyModel/version_x/config.yaml --ckpt_path logs/MyModel/version_x/checkpoints/last.ckpt
```

# 测试
测试时，使用训练时使用的配置文件，以及想要测试的checkpoint
```
python boring.py test --config logs/MyModel/version_x/config.yaml --ckpt_path logs/MyModel/version_x/checkpoints/epoch2_valid_loss-1576.2192.ckpt
```
