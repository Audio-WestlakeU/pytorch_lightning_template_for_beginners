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

# 批量训练
批量训练功能会不断地读取`tasks`目录下的任务，当任务启动要求满足时会使用`nohup`指令启动任务（任务配置参考`tasks/template.yaml`）。任务启动完成之后，任务配置文件会被移动到`tasks/started`目录下，任务输出会被保存为`tasks/started`目录下同名的log文件。**注意事项**：不要让`run_tasks.py`停止运行（可以在byobu里面运行，或者使用nohup指令运行）。

**安装**: 只使用`批量训练`功能时，可以不安装`requirements.txt`中的依赖。只需安装`pip install gpustat`。

**批量训练**：可以使用`run_tasks.py`:
```shell
python run_tasks.py --gpus 0 1 2 3 --email=zhangsan@qq.com --email_task_started --email_task_list_empty --email_task_all_ended --endless
```


**参数**：`--gpus` 需要监控的GPU编号；`--email`：通知邮箱；`--email_task_started`：给定这个参数时，任务启动时会邮件通知；`--email_task_list_empty`：给定这个参数时，任务列表为空时会邮件通知；`--email_task_all_ended`：给定这个参数时，任务全部结束时会邮件通知；`--endless`：是否无限循环。

**自定义**：如果觉得配置文件比较麻烦，可以直接修改`run_tasks.py`中的`read_tasks`方法，实现自己的需求。

