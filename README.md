# 简介
一套适合初学者的pytorch lightning代码框架，能够满足DL时需要用到的很多内容，如实验版本管理、配置文件管理、命令行解析、多卡训练、EarlyStopping策略、累计梯度、梯度clipping等等。

# 安装
```
pip install -r requirements.txt
```

# 生成配置文件
命令行运行以下指令会生成配置文件`boring.yaml`.
```
python boring.py fit --print_config > boring.yaml
```
配置文件主要包含：
```
seed_everything: 2 # 此处2是自己给定的随机数种子
ckpt_path: null # null表示从头开始训练，给定checkpoint的时候表示从checkpoint开始训练
trainer:
  # trainer的参数，如：
  gpus: 1,
  max_epochs: 100
model:
  # 模型的参数，对应代码里面的MyModule。如：
  arch:
    ...
  exp_name: exp
data:
  # 数据集的参数，对应代码里面的MyDataModule。如：
  batch_size:
    - 1
    - 2
# 此处省略了其他配置，如checkpoint保存，EarlyStopping也在这个配置文件里面
```

# 训练
命令行运行以下指令会使用生成的配置文件`boring.yaml`来进行训练（fit）。
```
python boring.py fit --config boring.yaml --data.batch_size=[24,48]
```
其中`--data.batch_size=[24,48]`会修改配置文件里面的`data`部分的`batch_size`为`[24,48]`（24是训练集的batch size，48是验证集的）。配置文件里面其他的参数，如`trainer.gpus`，都可以通过这种方式来修改，如`--trainer.gpus=2,`（表示使用2号卡，不要忘记在卡号后面加逗号，不加会被认为使用2张卡训练，而非2号卡）。

## GPU训练
多卡采用DDP模式训练时，注意保持不同实验间的batch_size一致。下面的例子的训练时总的batch size = 4 * 3，其中4是train dataloader的batch size，3是gpu的数目。
```
python boring.py fit --config boring.yaml --data.batch_size=[4,8] --trainer.gpus=0,1,3
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

# 批量训练
批量训练功能会不断地读取`tasks`目录下的任务，当任务启动要求满足时会使用`nohup`指令启动任务（任务配置参考`tasks/template.yaml`）。任务启动完成之后，任务配置文件会被移动到`tasks/started`目录下，任务输出会被保存为`tasks/started`目录下同名的log文件。**注意事项**：不要让`run_tasks.py`停止运行（可以在byobu里面运行，或者使用nohup指令运行）。

**安装**: 只使用`批量训练`功能时，可以不安装`requirements.txt`中的依赖。只需安装`pip install gpustat`。

**批量训练**：可以使用`run_tasks.py`:
```shell
python run_tasks.py --gpus 0 1 2 3 --email=zhangsan@qq.com --email_task_started --email_task_list_empty --email_task_all_ended --endless
```


**参数**：`--gpus` 需要监控的GPU编号；`--email`：通知邮箱；`--email_task_started`：给定这个参数时，任务启动时会邮件通知；`--email_task_list_empty`：给定这个参数时，任务列表为空时会邮件通知；`--email_task_all_ended`：给定这个参数时，任务全部结束时会邮件通知；`--endless`：是否无限循环。

**自定义**：如果觉得配置文件比较麻烦，可以直接修改`run_tasks.py`中的`read_tasks`方法，实现自己的需求。

